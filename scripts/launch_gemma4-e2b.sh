#!/bin/bash
# Gemma-4-E2B optimal settings for Jetson Orin Nano 8GB
# 32K context, q4_0 KV, 23.7 tok/s (39.1 with MTP)
# With MTP: --spec-type draft-mtp --model-draft ~/models/gemma-4-e2b-mtp/mtp-gemma-4-E2B-it.gguf --spec-draft-n-max 3
# Usage: bash launch_gemma4-e2b.sh [port] [mtp] (default 8090, no mtp)

PORT=${1:-8090}
MTP=${2:-no}
MODEL=~/models/zoo-v2/models/gemma-4-e2b-q40.gguf
MTP_DRAFT=~/models/gemma-4-e2b-mtp/mtp-gemma-4-E2B-it.gguf

MTP_ARGS=""
if [ "$MTP" = "mtp" ]; then
  MTP_ARGS="--spec-type draft-mtp --model-draft $MTP_DRAFT --spec-draft-n-max 3"
fi

GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
GGML_CUDA_NO_PINNED=1 \
~/llama.cpp/build/bin/llama-server \
  -m "$MODEL" \
  --alias gemma-4-e2b \
  --host 127.0.0.1 --port $PORT \
  -ngl 99 -c 32768 \
  -ctk q4_0 -ctv q4_0 \
  -b 512 -ub 512 \
  -fa on --jinja -np 1 -t 6 \
  --temp 0.0 --top-k 1 --repeat-penalty 1.0 \
  --fit off $MTP_ARGS
