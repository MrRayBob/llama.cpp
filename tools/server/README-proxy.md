# llama-server-proxy

`llama-server-proxy` is a small OpenAI-compatible HTTP proxy that sits in front of a private `llama-server` instance and compacts oversized chat histories before forwarding them upstream.

The split is intentional: the backend remains a plain inference server, while the proxy owns prompt rewriting, budgeting, and summary caching. You can run both processes in one container with a supervisor, but that still leaves you with two long-lived processes and more brittle startup/logging. A two-service Docker Compose setup is the simpler operational shape.

## Scope

- Public endpoints:
  - `GET /health`
  - `GET /v1/health`
  - `GET /v1/models`
  - `POST /v1/chat/completions`
- Non-chat server routes are forwarded upstream unchanged.
- Chat requests are only rewritten when the rendered prompt crosses the configured compaction trigger.
- This route is intended for a text-only Mistral Nemo profile. Multimodal chat requests are out of scope.

## Runtime topology

Run the backend `llama-server` on a private address and point the proxy at it:

```sh
./build/bin/llama-server \
  --host 127.0.0.1 \
  --port 8081 \
  --model /path/to/Mistral-Nemo-Instruct-2407-IQ3_M.gguf \
  --alias mistral-nemo-32k \
  -ctk pq3_5 \
  -ctv q8_0 \
  --ctx-size 32768 \
  --parallel 1 \
  --context-shift \
  -fa on \
  -fit on
```

Then expose the proxy instead of the backend:

```sh
./build/bin/llama-server-proxy \
  --host 0.0.0.0 \
  --port 8080 \
  --backend-base-url http://127.0.0.1:8081 \
  --model-alias mistral-nemo-32k \
  --chat-template-file models/templates/mistralai-Mistral-Nemo-Instruct-2407.jinja \
  --compaction-trigger 24000 \
  --hard-prompt-cap 32000 \
  --recent-raw-tail 8 \
  --first-summary-target 800 \
  --second-summary-target 400
```

## Docker Compose

For Docker-based deployment, use:

- [docker-compose.proxy.yml](./docker-compose.proxy.yml)
- [proxy-compose.env.example](./proxy-compose.env.example)

Quick start:

```sh
cd tools/server
cp proxy-compose.env.example .env
# edit .env
docker compose -f docker-compose.proxy.yml up -d --build
```

This starts:

- `backend`: private `llama-server` on the internal Compose network
- `proxy`: public `llama-server-proxy` on `${TS_IP}:8080`

Useful commands:

```sh
docker compose -f docker-compose.proxy.yml logs -f backend
docker compose -f docker-compose.proxy.yml logs -f proxy
docker compose -f docker-compose.proxy.yml down
```

## Notes

- The proxy renders the Nemo prompt shape locally for budgeting and falls back to backend `/apply-template` if local rendering fails.
- Token counts come from backend `/tokenize`, so the prompt budget is measured against the same tokenizer the model uses.
- Older context is collapsed into a synthetic memory block, while recent raw turns stay verbatim.
- Summary results are cached in an in-memory LRU keyed by the compacted prefix.
