#!/bin/bash
# Gemma-4-E4B-UD (Unsloth Dynamic) optimal settings for Jetson Orin Nano 8GB
# BFCL 92.3% champion. 32K q4_0 for full BFCL (coding 2/2). 15.6 tok/s (31.7 with MTP!)
# NOTE: f16 KV crashes on 2nd coding test; use q4_0 for multi-turn BFCL
# With MTP: --spec-type draft-mtp --model-draft ~/models/mtp-gemma-4-E4B-it.gguf --spec-draft-n-max 3
# Usage: bash launch_gemma4-e4b-ud.sh [port] [mtp] (default 8090, no mtp)

PORT=${1:-8090}
MTP=${2:-no}
MODEL=~/models/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf
MTP_DRAFT=~/models/mtp-gemma-4-E4B-it.gguf

MTP_ARGS=""
if [ "$MTP" = "mtp" ]; then
  MTP_ARGS="--spec-type draft-mtp --model-draft $MTP_DRAFT --spec-draft-n-max 3"
fi

GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
GGML_CUDA_NO_PINNED=1 \
~/llama.cpp/build/bin/llama-server \
  -m "$MODEL" \
  --alias gemma-4-e4b-ud \
  --host 127.0.0.1 --port $PORT \
  -ngl 99 -c 32768 \
  -ctk q4_0 -ctv q4_0 \
  -b 512 -ub 512 \
  -fa on --jinja -np 1 -t 6 \
  --temp 0.0 --top-k 1 --repeat-penalty 1.0 \
  --fit off $MTP_ARGS
