#!/bin/bash
# Qwen3.5-2B optimal settings for Jetson Orin Nano 8GB
# 48K context, f16 KV, 27.7 tok/s (41.0 with MTP!)
# Embedded MTP: --spec-type draft-mtp --spec-draft-n-max 3 (no draft model needed)
# Usage: bash launch_qwen3.5-2b.sh [port] [mtp] (default 8090, no mtp)

PORT=${1:-8090}
MTP=${2:-no}
MODEL=~/models/zoo-v2/models/qwen3.5-2b-q4km.gguf

MTP_ARGS=""
if [ "$MTP" = "mtp" ]; then
  MTP_ARGS="--spec-type draft-mtp --spec-draft-n-max 3"
fi

GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
GGML_CUDA_NO_PINNED=1 \
~/llama.cpp/build/bin/llama-server \
  -m "$MODEL" \
  --alias qwen3.5-2b \
  --host 127.0.0.1 --port $PORT \
  -ngl 99 -c 49152 \
  -ctk f16 -ctv f16 \
  -b 512 -ub 512 \
  -fa on --jinja -np 1 -t 6 \
  --temp 0.0 --top-k 1 --repeat-penalty 1.0 \
  --fit off $MTP_ARGS
