#!/bin/bash
# Spark-X2.5-4B optimal settings for Jetson Orin Nano 8GB
# BFCL 90.4% tool-calling accuracy
# 32K context, q8_0 KV, 14.4 tok/s, no -nkvo needed
# Usage: bash launch_spark.sh [port] (default 8090)

PORT=${1:-8090}
MODEL=~/models/new-zoo/Spark-X2.5-4B-Q4_K_M.gguf

GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
GGML_CUDA_NO_PINNED=1 \
~/llama.cpp/build/bin/llama-server \
  -m "$MODEL" \
  --alias spark-x2.5-4b \
  --host 127.0.0.1 --port $PORT \
  -ngl 99 -c 32768 \
  -ctk q8_0 -ctv q8_0 \
  -b 512 -ub 512 \
  -fa on --jinja -np 1 -t 6 \
  --temp 0.0 --top-k 1 --repeat-penalty 1.0 \
  --fit off
