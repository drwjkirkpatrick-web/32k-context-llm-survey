#!/bin/bash
# Hermes-3-3B Q5_K_M optimal settings for Jetson Orin Nano 8GB
# 32K context, q4_0 KV, 18.6 tok/s, all 12 tests stop
# No MTP support
# Usage: bash launch_hermes-3-3b.sh [port] (default 8090)

PORT=${1:-8090}
MODEL=~/models/zoo-v2/models/hermes-3-3b-q5km.gguf

GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
GGML_CUDA_NO_PINNED=1 \
~/llama.cpp/build/bin/llama-server \
  -m "$MODEL" \
  --alias hermes-3-3b \
  --host 127.0.0.1 --port $PORT \
  -ngl 99 -c 32768 \
  -ctk q4_0 -ctv q4_0 \
  -b 512 -ub 512 \
  -fa on --jinja -np 1 -t 6 \
  --temp 0.0 --top-k 1 --repeat-penalty 1.0 \
  --fit off
