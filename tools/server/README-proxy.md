# llama-server-proxy

`llama-server-proxy` is a small OpenAI-compatible HTTP proxy that sits in front of a private `llama-server` instance and compacts oversized chat histories before forwarding them upstream.

The split is intentional: the backend remains a plain inference server, while the proxy owns prompt rewriting, budgeting, and summary caching. The simplest operational shape is still two services in one Compose file.

## Default profile

The default deployment now targets `Mistral Nemo` on an 8 GB class GPU with a performance-first 32k profile:

- backend model: `bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q2_K`
- public model alias: `mistral-nemo-32k`
- one public API key on the proxy only

The backend uses the model's built-in chat template. The proxy does not need any mounted template file. Prompt budgeting uses backend `POST /apply-template` and `POST /tokenize`, so the backend is the single source of truth for template rendering.
For 8 GB VRAM systems, the compose defaults are tuned to maximize usable context while keeping decode speed on GPU:

- `-ctk pq3_5 -ctv q8_0` compresses KV cache heavily while preserving GPU KV execution.
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
- optional tuning knobs (`MODEL_REF`, `N_CTX`, `N_BATCH`, etc.)

This starts:

- `backend`: private `llama-server` on the internal Compose network
- `proxy`: public `llama-server-proxy` on `${TS_IP}:8080`

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
  -hf bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q2_K \
  -a mistral-nemo-32k \
  -fa on \
  -c 32768 \
  -np 1 \
  -fit on \
  -fitt 256 \
  -ngl all \
  -b 1024 \
  -ub 256 \
  --context-shift \
  -ctk pq3_5 \
  -ctv q8_0 \
  --host 127.0.0.1
```

```sh
./build/bin/llama-server-proxy \
  --host 0.0.0.0 \
  --backend-base-url http://127.0.0.1:8080 \
  --api-key replace-with-a-long-random-public-key
```

Expose the proxy, not the backend.

## Alternative profile

If you want a different Nemo quant, change `MODEL_REF` in `.env` (or the backend model string in the Compose file / manual command). If you later switch back to a model that needs a template override, that is a backend-only change; the proxy does not need to change.

## Memory-first tuning for 8 GB VRAM

If the backend still fails health checks:

1. Keep `N_PARALLEL=1`.
2. Lower `FIT_TARGET_MIB` from `256` to `128` (more aggressive fitting).
3. Reduce `N_BATCH` to `512` and `N_UBATCH` to `128`.
4. Reduce `N_CTX` to `24576` or `16384` if needed.
5. If fitting still pushes any layer off CUDA, switch to `KV_CACHE_K_TYPE=q4_0` and `KV_CACHE_V_TYPE=q4_0`.
6. As a last resort, set `N_GPU_LAYERS=auto`.

This path favors GPU-backed decode performance and relies on proxy compaction to keep long conversations usable.

## Notes

- Chat requests are only rewritten when the rendered prompt crosses the configured compaction trigger.
- Older context is collapsed into a synthetic memory block, while recent raw turns stay verbatim.
- Summary results are cached in an in-memory LRU keyed by the compacted prefix.
- This repository does not currently include a `TurboQuant` runtime flag in `llama-server`; the compose profile therefore uses the most VRAM-efficient upstream KV settings currently available (`pq3_5`/`q8_0`).
