# Gemma 4 64k Profile

This is a dedicated `Gemma 4 E4B` deployment tuned separately from the mixed router stack in `docker-compose.proxy.yml`.

Use this profile when you want:

- only `gemma4-4b-instruct`
- a larger `65536` token context window
- better Gemma throughput than the mixed 3-model router
- the model to unload itself after 30 minutes idle

This profile keeps the public OpenAI-compatible proxy in front, but removes the router entirely. That avoids model-switching overhead and lets Gemma use more aggressive single-model settings.

## Important tradeoff

`64k` context is not inherently faster than `32k`. It is slower and heavier. The performance win here comes from:

- a dedicated single-model backend
- larger batch and ubatch defaults
- thread settings tuned for the 6C/12T host seen in recent logs
- a KV cache profile chosen to make `64k` practical on an 8 GB class GPU

If you want the absolute fastest Gemma decode speed, reduce `CTX_SIZE` below `65536`.

## Files

- [docker-compose.gemma64.yml](./docker-compose.gemma64.yml)
- [gemma64-compose.env.example](./gemma64-compose.env.example)

## Quick start

```sh
cd tools/server
cp gemma64-compose.env.example .env.gemma64
# edit .env.gemma64
docker compose --env-file .env.gemma64 -f docker-compose.gemma64.yml up -d --build
```

This starts:

- `backend`: private `llama-server` with `ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M`
- `proxy`: public `llama-server-proxy` on `${TS_IP}:${HOST_PORT}`

The request model id remains:

```text
gemma4-4b-instruct
```

Example request:

```sh
curl http://127.0.0.1:8082/v1/chat/completions \
  -H "Authorization: Bearer replace-with-a-long-random-public-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4-4b-instruct",
    "messages": [{"role": "user", "content": "Write a tiny HTTP server in C."}]
  }'
```

## Tuning

If the model does not fit or starts unstable:

1. Lower `N_BATCH` to `1536`.
2. Lower `N_UBATCH` to `384`.
3. Lower `CTX_SIZE` to `49152`.
4. Lower `FIT_TARGET_MIB` to `64`.

If you care more about quality than speed, change:

```env
KV_CACHE_V_TYPE=q8_0
```

## Idle unload

`SLEEP_IDLE_SECONDS=1800` keeps Gemma loaded for 30 minutes after the last request, then destroys the context and frees VRAM.
