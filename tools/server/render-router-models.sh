#!/bin/sh
set -eu

out_file="/tmp/router-models.ini"

emit_optional_kv() {
    key="$1"
    value="$2"
    if [ -n "$value" ]; then
        printf '%s = %s\n' "$key" "$value"
    fi
}

{
    printf 'version = 1\n\n'

    printf '[*]\n'
    printf '; Generated from tools/server/.env via render-router-models.sh\n'
    printf 'n-gpu-layers = %s\n' "${DEFAULT_N_GPU_LAYERS:-all}"
    printf 'parallel = %s\n' "${DEFAULT_PARALLEL:-1}"
    printf 'batch-size = %s\n' "${DEFAULT_BATCH_SIZE:-1024}"
    printf 'ubatch-size = %s\n' "${DEFAULT_UBATCH_SIZE:-256}"
    printf 'flash-attn = %s\n' "${DEFAULT_FLASH_ATTN:-true}"
    printf 'cache-type-k = %s\n' "${DEFAULT_KV_CACHE_K_TYPE:-q4_0}"
    printf 'cache-type-v = %s\n' "${DEFAULT_KV_CACHE_V_TYPE:-q8_0}"
    emit_optional_kv "threads" "${DEFAULT_THREADS:-}"
    emit_optional_kv "threads-batch" "${DEFAULT_THREADS_BATCH:-}"
    printf '\n'

    printf '[gemma4-4b-instruct]\n'
    printf 'hf-repo = ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M\n'
    printf 'ctx-size = %s\n' "${GEMMA4_CTX_SIZE:-131072}"
    printf 'batch-size = %s\n' "${GEMMA4_BATCH_SIZE:-2048}"
    printf 'ubatch-size = %s\n' "${GEMMA4_UBATCH_SIZE:-512}"
    emit_optional_kv "threads" "${GEMMA4_THREADS:-6}"
    emit_optional_kv "threads-batch" "${GEMMA4_THREADS_BATCH:-12}"
    printf 'cache-type-k = %s\n' "${GEMMA4_KV_CACHE_K_TYPE:-q4_0}"
    printf 'cache-type-v = %s\n' "${GEMMA4_KV_CACHE_V_TYPE:-q4_0}"
    printf 'load-on-startup = false\n\n'

    printf '[mistral-nemo-12b-instruct]\n'
    printf 'hf-repo = bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q2_K\n'
    printf 'ctx-size = %s\n' "${MISTRAL_NEMO_CTX_SIZE:-32768}"
    printf 'load-on-startup = false\n\n'

    printf '[qwen-coder-7b-instruct]\n'
    printf 'hf-repo = bartowski/Qwen2.5-Coder-7B-Instruct-GGUF:Q4_K_M\n'
    printf 'ctx-size = %s\n' "${QWEN_CODER_CTX_SIZE:-32768}"
    printf 'load-on-startup = false\n'
} > "$out_file"

exec "$@"
