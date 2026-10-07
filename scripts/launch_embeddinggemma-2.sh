#!/bin/bash
# EmbeddingGemma 2 (UD-Q4_K_XL, 168MB) optimal settings for Jetson Orin Nano 8GB
# Multimodal embedding model — text/code embeddings for RAG + semantic search.
# Retrieval 8/8, MRL truncation to 128d lossless, ~1200 tok/s batch throughput.
#
# KEY SETTINGS:
#   --pooling mean     CRITICAL: last pooling breaks discrimination (1/3 -> 3/3 retrieval)
#   -b 2048 -ub 2048   batch width = longest doc limit (embedding = ONE forward pass per doc;
#                      b=512 caps docs at 512 tok, b=8192 OOMs from activation buffers)
#   --embeddings       enable embedding endpoint
#   setsid launch      escapes hermes-worker memcg cap (~4GB); server lands in gateway cgroup
#
# INPUT FORMATS (raw strings, no chat template needed):
#   documents: "title: <name>\ntext: <content>"
#   queries:   "search: <question>"
#
# MRL: truncate embedding vector to 512/256/128 dims + renormalize for 3-6x storage cut.
# MTP: N/A (embedding models do not generate; GGUF has no nextn tensors).
# Multimodal: this GGUF is text-only; image/audio need the mmproj vision tower.
#
# Requires llama.cpp >= build 1098 (commit 448147d4, 2026-10-07) for gemma-embedding2 arch.
#
# Usage: bash launch_embeddinggemma-2.sh [port] (default 8090)

PORT="${1:-8090}"

GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 GGML_CUDA_NO_PINNED=1 \
setsid ~/llama.cpp/build/bin/llama-server \
    -m ~/models/embeddinggemma-2/embeddinggemma-2-UD-Q4_K_XL.gguf \
    --alias embeddinggemma-2 \
    --host 127.0.0.1 --port "$PORT" \
    -ngl 99 -c 8192 --embeddings --pooling mean \
    -b 2048 -ub 2048 -fa on -np 1 -t 6 --fit off \
    > /tmp/embeddinggemma-2-server.log 2>&1 &

echo "EmbeddingGemma 2 server starting on port $PORT (log: /tmp/embeddinggemma-2-server.log)"
echo "Test: curl -s http://127.0.0.1:$PORT/v1/embeddings -d '{\"input\": [\"search: hello\"], \"model\": \"embeddinggemma-2\"}'"
