import json
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def proxy_bin_path() -> Path:
    if "LLAMA_PROXY_BIN_PATH" in os.environ:
        return Path(os.environ["LLAMA_PROXY_BIN_PATH"])
    if "LLAMA_SERVER_PROXY_BIN_PATH" in os.environ:
        return Path(os.environ["LLAMA_SERVER_PROXY_BIN_PATH"])
    return repo_root() / "build" / "bin" / "llama-server-proxy"


@dataclass
class StubState:
    summary_requests: int = 0
    main_requests: list[dict] = field(default_factory=list)
    template_inputs: list[dict] = field(default_factory=list)
    tokenize_inputs: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def token_count(text: str) -> int:
    return len(text.split())


def flatten_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts)
    return json.dumps(content, sort_keys=True)


def render_prompt(messages: list[dict]) -> str:
    parts: list[str] = []
    for message in messages:
        parts.append(f"{message.get('role', 'unknown')}:{flatten_content(message.get('content', ''))}")
        if message.get("tool_calls"):
            parts.append(json.dumps(message["tool_calls"], sort_keys=True))
        if message.get("tool_call_id"):
            parts.append(f"tool_call_id:{message['tool_call_id']}")
    return "\n".join(parts)


def make_stub_handler(state: StubState):
    class StubHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            return

        def _send_json(self, status: int, body: dict):
            payload = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path in ("/health", "/v1/health"):
                self._send_json(200, {"status": "ok"})
                return
            if self.path in ("/models", "/v1/models"):
                self._send_json(200, {
                    "object": "list",
                    "data": [{"id": "backend-model", "object": "model"}],
                })
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(content_length) or "{}")

            if self.path == "/apply-template":
                with state.lock:
                    state.template_inputs.append(body)
                self._send_json(200, {"prompt": render_prompt(body.get("messages", []))})
                return

            if self.path == "/tokenize":
                content = body.get("content", "")
                with state.lock:
                    state.tokenize_inputs.append(content)
                self._send_json(200, {"tokens": list(range(token_count(content)))})
                return

            if self.path == "/v1/chat/completions":
                messages = body.get("messages", [])
                if messages and messages[0].get("role") == "system" and "compress earlier chat history" in messages[0].get("content", "").lower():
                    with state.lock:
                        state.summary_requests += 1
                    self._send_json(200, {
                        "id": "cmpl-summary",
                        "object": "chat.completion",
                        "model": "backend-model",
                        "choices": [{
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "User Preferences:\n- Keep answers concise.\n\nRelevant Code/File Context:\n- Conversation history was compacted.",
                            },
                        }],
                    })
                    return

                with state.lock:
                    state.main_requests.append(body)

                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()

                    events = [
                        {
                            "id": "cmpl-stream",
                            "object": "chat.completion.chunk",
                            "model": body.get("model", "backend-model"),
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": "proxy"},
                                "finish_reason": None,
                            }],
                        },
                        {
                            "id": "cmpl-stream",
                            "object": "chat.completion.chunk",
                            "model": body.get("model", "backend-model"),
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }],
                        },
                        {
                            "id": "cmpl-stream",
                            "object": "chat.completion.chunk",
                            "model": body.get("model", "backend-model"),
                            "choices": [],
                            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                            "timings": {},
                        },
                    ]
                    for event in events:
                        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return

                self._send_json(200, {
                    "id": "cmpl-main",
                    "object": "chat.completion",
                    "model": body.get("model", "backend-model"),
                    "choices": [{
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                })
                return

            self._send_json(404, {"error": "not found"})

    return StubHandler


class ProxyProcess:
    def __init__(self,
                 backend_port: int,
                 proxy_port: int,
                 *,
                 trigger: int = 120,
                 hard_cap: int = 160,
                 tail: int = 2,
                 model_alias: str | None = "proxy-model"):
        self.backend_port = backend_port
        self.proxy_port = proxy_port
        self.trigger = trigger
        self.hard_cap = hard_cap
        self.tail = tail
        self.model_alias = model_alias
        self.process: subprocess.Popen | None = None

    def start(self):
        cmd = [
            str(proxy_bin_path()),
            "--host", "127.0.0.1",
            "--port", str(self.proxy_port),
            "--backend-base-url", f"http://127.0.0.1:{self.backend_port}",
            "--compaction-trigger", str(self.trigger),
            "--hard-prompt-cap", str(self.hard_cap),
            "--recent-raw-tail", str(self.tail),
            "--first-summary-target", "40",
            "--second-summary-target", "20",
        ]
        if self.model_alias is not None:
            cmd.extend(["--model-alias", self.model_alias])
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                res = requests.get(f"http://127.0.0.1:{self.proxy_port}/health", timeout=0.5)
                if res.status_code == 200:
                    return
            except Exception:
                pass
            if self.process.poll() is not None:
                stderr = self.process.stderr.read().decode("utf-8") if self.process.stderr else ""
                raise RuntimeError(f"proxy exited early: {stderr}")
            time.sleep(0.1)
        raise TimeoutError("proxy failed to start")

    def stop(self):
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.process = None


@pytest.fixture
def proxy_env():
    state = StubState()
    backend_port = find_free_port()
    proxy_port = find_free_port()

    server = ThreadingHTTPServer(("127.0.0.1", backend_port), make_stub_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    proxy = ProxyProcess(backend_port, proxy_port)
    proxy.start()

    try:
        yield state, proxy
    finally:
        proxy.stop()
        server.shutdown()
        thread.join(timeout=5)


def long_text(prefix: str, n: int) -> str:
    return " ".join([prefix] * n)


def test_models_endpoint_uses_proxy_alias(proxy_env):
    _, proxy = proxy_env
    res = requests.get(f"http://127.0.0.1:{proxy.proxy_port}/v1/models", timeout=5)
    assert res.status_code == 200
    body = res.json()
    assert body["data"][0]["id"] == "proxy-model"


def test_proxy_starts_without_template_file_or_backend_key(proxy_env):
    state, proxy = proxy_env
    res = requests.get(f"http://127.0.0.1:{proxy.proxy_port}/health", timeout=5)
    assert res.status_code == 200

    with state.lock:
        assert state.template_inputs == []
        assert state.main_requests == []


def test_small_chat_passes_through_unchanged(proxy_env):
    state, proxy = proxy_env
    messages = [
        {"role": "system", "content": "Keep answers short."},
        {"role": "user", "content": "Hello there"},
    ]
    res = requests.post(
        f"http://127.0.0.1:{proxy.proxy_port}/v1/chat/completions",
        json={"model": "proxy-model", "messages": messages},
        timeout=5,
    )
    assert res.status_code == 200
    assert res.headers["X-Llama-Proxy-Compacted"] == "0"
    with state.lock:
        assert state.summary_requests == 0
        assert state.template_inputs
        assert state.main_requests[-1]["messages"] == messages


def test_large_chat_compacts_and_reuses_cached_summary(proxy_env):
    state, proxy = proxy_env
    messages = [
        {"role": "system", "content": "Keep answers concise and stay on task."},
        {"role": "user", "content": long_text("alpha", 30)},
        {"role": "assistant", "content": long_text("beta", 26)},
        {"role": "user", "content": long_text("gamma", 26)},
        {"role": "assistant", "content": long_text("delta", 26)},
        {"role": "user", "content": long_text("epsilon", 26)},
        {"role": "assistant", "content": long_text("zeta", 26)},
    ]
    payload = {"model": "proxy-model", "messages": messages}

    for _ in range(2):
        res = requests.post(
            f"http://127.0.0.1:{proxy.proxy_port}/v1/chat/completions",
            json=payload,
            timeout=5,
        )
        assert res.status_code == 200
        assert res.headers["X-Llama-Proxy-Compacted"] == "1"

    with state.lock:
        assert state.summary_requests == 1
        assert state.template_inputs
        forwarded = state.main_requests[-1]["messages"]

    assert forwarded[0]["role"] == "system"
    assert "[Compacted Conversation Memory]" in forwarded[0]["content"]
    assert forwarded[-2:] == messages[-2:]


def test_recent_tool_chain_is_preserved_during_compaction(proxy_env):
    state, proxy = proxy_env
    messages = [
        {"role": "system", "content": "Use tools when needed."},
        {"role": "user", "content": long_text("history", 28)},
        {"role": "assistant", "content": long_text("done", 24)},
        {"role": "user", "content": "Look up the build file."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call00001",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{\"path\":\"CMakeLists.txt\"}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call00001", "content": "{\"path\":\"CMakeLists.txt\",\"exists\":true}"},
        {"role": "assistant", "content": "I found the build file."},
    ]

    res = requests.post(
        f"http://127.0.0.1:{proxy.proxy_port}/v1/chat/completions",
        json={"model": "proxy-model", "messages": messages},
        timeout=5,
    )
    assert res.status_code == 200
    assert res.headers["X-Llama-Proxy-Compacted"] == "1"

    with state.lock:
        forwarded = state.main_requests[-1]["messages"]

    assert any(msg.get("tool_calls") for msg in forwarded)
    assert any(msg.get("tool_call_id") == "call00001" for msg in forwarded)
    assert forwarded[-3:] == messages[-3:]


def test_large_recent_tool_result_is_compacted_to_fit_budget():
    state = StubState()
    backend_port = find_free_port()
    proxy_port = find_free_port()

    server = ThreadingHTTPServer(("127.0.0.1", backend_port), make_stub_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    proxy = ProxyProcess(backend_port, proxy_port, trigger=40, hard_cap=55, tail=2)
    proxy.start()

    try:
        messages = [
            {"role": "system", "content": "Use the tool results."},
            {"role": "user", "content": "Read the log file."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call00002",
                    "type": "function",
                    "function": {"name": "read_log", "arguments": "{\"path\":\"/tmp/log\"}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call00002", "content": long_text("blob", 120)},
            {"role": "assistant", "content": "The log is noisy but readable."},
        ]
        res = requests.post(
            f"http://127.0.0.1:{proxy.proxy_port}/v1/chat/completions",
            json={"model": "proxy-model", "messages": messages},
            timeout=5,
        )
        assert res.status_code == 200
        assert res.headers["X-Llama-Proxy-Compacted"] == "1"

        with state.lock:
            forwarded = state.main_requests[-1]["messages"]
            assert state.summary_requests == 0

        tool_msg = next(msg for msg in forwarded if msg.get("role") == "tool")
        assert tool_msg["tool_call_id"] == "call00002"
        assert "[compacted historical tool result]" in tool_msg["content"]
    finally:
        proxy.stop()
        server.shutdown()
        thread.join(timeout=5)


def test_models_and_chat_passthrough_when_proxy_alias_is_unset():
    state = StubState()
    backend_port = find_free_port()
    proxy_port = find_free_port()

    server = ThreadingHTTPServer(("127.0.0.1", backend_port), make_stub_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    proxy = ProxyProcess(backend_port, proxy_port, model_alias=None)
    proxy.start()

    try:
        models = requests.get(f"http://127.0.0.1:{proxy.proxy_port}/v1/models", timeout=5)
        assert models.status_code == 200
        assert models.json()["data"][0]["id"] == "backend-model"

        messages = [{"role": "user", "content": "hello"}]
        res = requests.post(
            f"http://127.0.0.1:{proxy.proxy_port}/v1/chat/completions",
            json={"model": "backend-model", "messages": messages},
            timeout=5,
        )
        assert res.status_code == 200

        with state.lock:
            assert state.main_requests[-1]["model"] == "backend-model"
    finally:
        proxy.stop()
        server.shutdown()
        thread.join(timeout=5)
