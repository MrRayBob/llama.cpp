# llama-server-proxy

`llama-server-proxy` is a small OpenAI-compatible HTTP proxy that sits in front of a private `llama-server` instance and compacts oversized chat histories before forwarding them upstream.

The split is intentional: the backend remains a plain inference server, while the proxy owns prompt rewriting, budgeting, and summary caching. The simplest operational shape is still two services in one Compose file.

## Default profile

The default deployment now runs `llama-server` in router mode on an 8 GB class GPU with three on-demand models:

- `gemma4-4b-instruct` -> `ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M`
- `mistral-nemo-12b-instruct` -> `bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q2_K`
- `qwen-coder-7b-instruct` -> `bartowski/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M`
- one public API key on the proxy only

The backend uses router autoload. The first request for a model loads it automatically, `MODELS_MAX=1` ensures only one model stays active at a time, and `SLEEP_IDLE_SECONDS=1800` makes the active child destroy its model/context after 30 minutes idle so VRAM returns to empty.

Prompt budgeting still uses backend `POST /apply-template` and `POST /tokenize`, so the backend remains the single source of truth for template rendering. The mixed bundle now uses model-specific contexts and proxy caps:

- `gemma4-4b-instruct`: `131072` ctx, with proxy compaction and proxy prompt caps disabled
- `mistral-nemo-12b-instruct`: `32768` ctx
- `qwen-coder-7b-instruct`: `32768` ctx

The proxy is now model-aware, so Gemma can accept larger prompts without forcing the 32k models to do the same.

For 8 GB VRAM systems, the compose defaults are tuned to maximize fit reliability while keeping decode work on GPU:

- `gemma4-4b-instruct` gets a larger `batch-size`, `ubatch-size`, and dedicated thread settings in the generated router preset.
- shared defaults use `q4_0` / `q8_0`, while Gemma overrides to `q4_0` / `q4_0` to leave more headroom for `128k`.
- `-fit on -fitt 256` allows tighter VRAM packing than the default 1024 MiB margin.
- shared defaults still keep `n-gpu-layers = all`, `parallel = 1`, and `flash-attn = true`.

## Docker Compose

Use:

- [docker-compose.proxy.yml](./docker-compose.proxy.yml)
- [proxy-compose.env.example](./proxy-compose.env.example)
- [README-gemma64.md](./README-gemma64.md) for a dedicated single-model Gemma 4 64k profile

Quick start:

```sh
cd tools/server
cp proxy-compose.env.example .env
# edit .env
docker compose -f docker-compose.proxy.yml up -d --build
```

Edit only these values in `.env` for the default path:

- `HF_CACHE_DIR`
- `PUBLIC_API_KEY`
- optional: `WEBUI_MCP_PROXY`
- optional: `HF_TOKEN`
- optional: `TS_IP`
- optional proxy budgeting knobs (`*_HARD_PROMPT_CAP`, `*_COMPACTION_TRIGGER`)
- optional per-model router knobs (`GEMMA4_CTX_SIZE`, `MISTRAL_NEMO_CTX_SIZE`, `QWEN_CODER_CTX_SIZE`, etc.)
- optional backend-wide fit knobs (`FIT_MODE`, `FIT_TARGET_MIB`)

This starts:

- `backend`: private router-mode `llama-server` on the internal Compose network
- `proxy`: public `llama-server-proxy` on `${TS_IP}:8080`

The compose file passes the proxy prompt-budget limits from `.env`:

- `DEFAULT_HARD_PROMPT_CAP` / `DEFAULT_COMPACTION_TRIGGER` remain the fallback proxy limits
- `GEMMA4_*`, `MISTRAL_NEMO_*`, and `QWEN_CODER_*` prompt caps are passed as per-model overrides
- the backend renders `/tmp/router-models.ini` from `.env` before startup, so per-model ctx and batch settings live in `.env`, not just `router-models.ini`

The shipped defaults expose full `128k` only for Gemma while keeping Nemo and Qwen at `32k`.

```env
DEFAULT_HARD_PROMPT_CAP=32000
DEFAULT_COMPACTION_TRIGGER=24000
GEMMA4_HARD_PROMPT_CAP=0
GEMMA4_COMPACTION_TRIGGER=0
GEMMA4_CTX_SIZE=131072
MISTRAL_NEMO_CTX_SIZE=32768
QWEN_CODER_CTX_SIZE=32768
```

Set a model's `*_COMPACTION_TRIGGER=0` to disable proxy rewriting for that model, and `*_HARD_PROMPT_CAP=0` to let backend context limits be the only cap.

Useful commands:

```sh
docker compose -f docker-compose.proxy.yml logs -f backend
docker compose -f docker-compose.proxy.yml logs -f proxy
docker compose -f docker-compose.proxy.yml ps
docker compose -f docker-compose.proxy.yml down
```

## Web UI MCP servers

If you add browser-based MCP servers in the Web UI and enable the card's proxy toggle, the public process serving the UI must expose `/cors-proxy`.

In this mixed deployment, the public process is `llama-server-proxy`, not the private backend. Enable it by setting:

```env
WEBUI_MCP_PROXY=1
```

Then recreate only the proxy service:

```sh
docker compose -f docker-compose.proxy.yml up -d --build proxy
```

That restarts only the proxy container. In the current Compose layout both services share the same Docker image, so `--build proxy` still recompiles the shared image, but the backend container, including any currently loaded model, stays up unless you recreate it too.

If your MCP server already sends browser-compatible CORS headers for the Web UI origin, you can leave `WEBUI_MCP_PROXY=0` and turn the proxy toggle off in the UI instead.

## Manual fallback

If you want to run the two services without Compose, keep the same split:

```sh
./build/bin/llama-server \
  --models-preset ./tools/server/router-models.ini \
  --models-max 1 \
  --sleep-idle-seconds 1800 \
  -fit on \
  -fitt 256 \
  --context-shift \
  --host 127.0.0.1 \
  --port 8080
```

```sh
./build/bin/llama-server-proxy \
  --host 0.0.0.0 \
  --backend-base-url http://127.0.0.1:8080 \
  --api-key replace-with-a-long-random-public-key \
  --hard-prompt-cap 32000 \
  --compaction-trigger 24000 \
  --model-hard-prompt-cap gemma4-4b-instruct=0 \
  --model-compaction-trigger gemma4-4b-instruct=0
```

Expose the proxy, not the backend.

Example request:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer replace-with-a-long-random-public-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-coder-7b-instruct",
    "messages": [{"role": "user", "content": "Write a tiny HTTP server in C."}]
  }'
```

The same `model` field can be set to `gemma4-4b-instruct` or `mistral-nemo-12b-instruct`. The first request autoloads that model. `GET /models` then reports `loaded`, `unloaded`, or `sleeping` for each preset.

## Memory-first tuning for 8 GB VRAM

If the backend still fails health checks:

1. Keep `DEFAULT_PARALLEL=1`.
2. Lower `FIT_TARGET_MIB` from `256` to `128` (more aggressive fitting).
3. Reduce `GEMMA4_BATCH_SIZE` to `1536` and `GEMMA4_UBATCH_SIZE` to `384` if Gemma becomes unstable.
4. Reduce the per-model `*_CTX_SIZE` values in `.env` if needed.
5. If Gemma does not fit at `128k`, keep Nemo and Qwen at `32k` and lower only `GEMMA4_CTX_SIZE`.
6. As a last resort, lower `DEFAULT_N_GPU_LAYERS` from `all`.

This path favors GPU-backed decode performance while keeping the single loaded model small enough to fit and unload cleanly.

## Notes

- Chat requests are only rewritten when the rendered prompt crosses the configured compaction trigger. Setting that trigger to `0` disables rewriting for that model.
- Older context is collapsed into a synthetic memory block, while recent raw turns stay verbatim.
- Summary results are cached in an in-memory LRU keyed by the compacted prefix.
- `--hf-repo` defaults to `Q4_K_M` and automatically downloads `mmproj` files when present, which is why the Gemma 4 preset does not need a separate projector path.
- Gemma 4 uses `n_embd_head_k = 512` in some layers, so it cannot run with `pq3_5` K-cache. The compose profile therefore defaults to `q4_0`/`q8_0` for the shared path and overrides Gemma to `q4_0`/`q4_0`.
