# llama-server-proxy

`llama-server-proxy` is a small OpenAI-compatible HTTP proxy that sits in front of a private `llama-server` instance and compacts oversized chat histories before forwarding them upstream.

The split is intentional: the backend remains a plain inference server, while the proxy owns prompt rewriting, budgeting, and summary caching. The simplest operational shape is still two services in one Compose file.

## Default profile

The default deployment now targets `Mistral Nemo`:

- backend model: `bartowski/Mistral-Nemo-Instruct-2407-GGUF:IQ3_M`
- public model alias: `mistral-nemo-32k`
- one public API key on the proxy only

The backend uses the model's built-in chat template. The proxy does not need any mounted template file. Prompt budgeting uses backend `POST /apply-template` and `POST /tokenize`, so the backend is the single source of truth for template rendering.

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
  -hf bartowski/Mistral-Nemo-Instruct-2407-GGUF:IQ3_M \
  -a mistral-nemo-32k \
  -fa on \
  -c 32768 \
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

If `IQ3_M` is unavailable for your hardware or preferred repo, switch the backend model to `Q3_K_M` or another Nemo quant. If you later switch back to a model that needs a template override, that is a backend-only change; the proxy does not need to change.

## Notes

- Chat requests are only rewritten when the rendered prompt crosses the configured compaction trigger.
- Older context is collapsed into a synthetic memory block, while recent raw turns stay verbatim.
- Summary results are cached in an in-memory LRU keyed by the compacted prefix.
