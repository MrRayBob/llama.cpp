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

Prompt budgeting still uses backend `POST /apply-template` and `POST /tokenize`, so the backend remains the single source of truth for template rendering. The mixed bundle now uses 32k router contexts across all three models, which is enough for larger editor prompts while staying much more practical on 8 GB hardware than a blanket 128k default.

For 8 GB VRAM systems, the compose defaults are tuned to maximize fit reliability while keeping decode work on GPU:

- `-ctk q4_0 -ctv q8_0` keeps KV cache compact while remaining compatible with Gemma 4.
- `-fit on -fitt 256` allows tighter VRAM packing than the default 1024 MiB margin.
- `-ngl all` keeps as many layers on GPU as possible (subject to fit safety checks).
- `-np 1` avoids extra slot-related KV allocations.

## Docker Compose

Use:

- [docker-compose.proxy.yml](./docker-compose.proxy.yml)
- [proxy-compose.env.example](./proxy-compose.env.example)

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
- optional: `HF_TOKEN`
- optional: `TS_IP`
- optional proxy budgeting knobs (`HARD_PROMPT_CAP`, `COMPACTION_TRIGGER`)
- optional runtime knobs (`MODELS_MAX`, `SLEEP_IDLE_SECONDS`, `N_BATCH`, etc.)

This starts:

- `backend`: private router-mode `llama-server` on the internal Compose network
- `proxy`: public `llama-server-proxy` on `${TS_IP}:8080`

The compose file passes the proxy prompt-budget limits from `.env`:

- `HARD_PROMPT_CAP` maps to `--hard-prompt-cap`
- `COMPACTION_TRIGGER` maps to `--compaction-trigger`

The default `32000` / `24000` pair is safe for the current 32k bundle. If you only serve Nemo or Gemma and want to experiment with a longer context profile, raise these together and increase the matching `ctx-size` in `router-models.ini`.

```env
HARD_PROMPT_CAP=32000
COMPACTION_TRIGGER=24000
```

Useful commands:

```sh
docker compose -f docker-compose.proxy.yml logs -f backend
docker compose -f docker-compose.proxy.yml logs -f proxy
docker compose -f docker-compose.proxy.yml ps
docker compose -f docker-compose.proxy.yml down
```

## Manual fallback

If you want to run the two services without Compose, keep the same split:

```sh
./build/bin/llama-server \
  --models-preset ./tools/server/router-models.ini \
  --models-max 1 \
  --sleep-idle-seconds 1800 \
  -fa on \
  -np 1 \
  -fit on \
  -fitt 256 \
  -ngl all \
  -b 1024 \
  -ub 256 \
  --context-shift \
  -ctk q4_0 \
  -ctv q8_0 \
  --host 127.0.0.1 \
  --port 8080
```

```sh
./build/bin/llama-server-proxy \
  --host 0.0.0.0 \
  --backend-base-url http://127.0.0.1:8080 \
  --api-key replace-with-a-long-random-public-key
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

1. Keep `N_PARALLEL=1`.
2. Lower `FIT_TARGET_MIB` from `256` to `128` (more aggressive fitting).
3. Reduce `N_BATCH` to `512` and `N_UBATCH` to `128`.
4. Reduce the per-model `ctx-size` values in `router-models.ini` if needed.
5. If fitting still pushes any layer off CUDA, switch to `KV_CACHE_K_TYPE=q4_0` and `KV_CACHE_V_TYPE=q4_0`.
6. As a last resort, set `N_GPU_LAYERS=auto`.

This path favors GPU-backed decode performance while keeping the single loaded model small enough to fit and unload cleanly.

## Notes

- Chat requests are only rewritten when the rendered prompt crosses the configured compaction trigger.
- Older context is collapsed into a synthetic memory block, while recent raw turns stay verbatim.
- Summary results are cached in an in-memory LRU keyed by the compacted prefix.
- `--hf-repo` defaults to `Q4_K_M` and automatically downloads `mmproj` files when present, which is why the Gemma 4 preset does not need a separate projector path.
- Gemma 4 uses `n_embd_head_k = 512` in some layers, so it cannot run with `pq3_5` K-cache. The compose profile therefore defaults to `q4_0`/`q8_0` for mixed-router compatibility.
