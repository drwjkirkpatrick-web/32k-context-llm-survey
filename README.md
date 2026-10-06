# 32K Context LLM Survey — Jetson Orin Nano 8GB

Benchmarking small LLMs (1B–7B) for local Hermes agent tool-calling on a Jetson Orin Nano 8GB.
Tests which models can run at 32K context with q4_0 KV cache inside the memory budget,
and how they perform on 12 standardized prompts (HTML, Python, poetry, math, creative, function calls).

## Hardware

- **Platform:** NVIDIA Jetson Orin Nano 8GB (unified memory)
- **Total RAM:** 7.6 GB
- **Available for model:** ~5.2 GB (after Hermes gateway ~600MB + system overhead)
- **Realistic budget:** ~4.5 GB (accounting for CUDA overhead, flash attention buffers, compute graphs)
- **GPU:** Orin (7607 MiB, 5017 MiB free with GDM off)
- **Swap:** 6 × zram partitions (~3.7 GB total, compressed)

## Software

- **llama.cpp:** 0.6.0-dev, build 1051, commit 8345f333 (Oct 5 2026)
- **155 architectures** supported
- **GDM/GUI off** for maximum RAM availability
- **GGML_CUDA_ENABLE_UNIFIED_MEMORY=1** required

## Standard Launch Flags

```
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 ~/llama.cpp/build/bin/llama-server   -m <model.gguf> --alias <name> --host 127.0.0.1 --port 8090   -ngl 99 -c <ctx> -ctk q4_0 -ctv q4_0   -b 512 -ub 512 -fa on --jinja -np 1 -t 6   --temp 0.0 --top-k 1 --repeat-penalty 1.0 --fit off
```

## Test Suite

12 prompts from `~/projects/jetson-model-zoo/test_prompts.json`:

| Category | Count | Max tokens |
|---|---|---|
| HTML | 2 | 4096 |
| Python | 1 | 4096 |
| Poetry (iambic pentameter) | 1 | 1024 |
| Math proof | 1 | 2048 |
| Creative writing | 1 | 1024 |
| Function call | 6 | 1024 |

Sampling: greedy (temp=0.0, top_k=1, repeat_penalty=1.0)

## KV Cache Formula

```
KV_per_token_bytes = 2 * n_layers * n_kv_heads * head_dim
KV_total = KV_per_token * ctx_size * bytes_per_element
Total_memory = model_weight + KV_total + ~400MB CUDA overhead

bytes_per_element: f16=2.0, q8_0=1.0, q4_0=0.5
```

## Key Findings

### What fits at 32K q4_0 (realistic 4500MB budget)

- **All sub-4B models** fit at 32K q4_0 (25 of 27 models)
- **4B models** (Qwen3-4B, Spark-X2.5-4B, Nemotron-3N-4B) fit at 32K q4_0 but NOT at q8_0
- **Phi-3-3.8B** does NOT fit (32 kv_heads, no GQA → 196,608 bytes/token KV)
- **Mistral-7B** does NOT fit at any context (4.37GB weight alone)

### Only 2 models achieve 48K f16

- **Llama-3.2-1B** (810MB weight, 16 layers, 8 kv_heads, 128 head_dim)
- **Qwen3.5-2B** (1400MB, SSM hybrid, 4 kv_heads, 256 head_dim)

### Architecture matters more than parameter count

- **Gemma-4-E2B** (2.84GB, 1 kv_head, head_dim=512): extreme GQA → only 35,840 bytes/token KV
- **Qwen2.5-3B** (1.93GB, 2 kv_heads, 128 head_dim): GQA → only 18,432 bytes/token KV
- **Phi-3-3.8B** (2.18GB, 32 kv_heads, 96 head_dim): no GQA → 196,608 bytes/token KV (5.4x more than Gemma!)

### Best models for Hermes agent use

| Criterion | Model | tok/s | ctx | Notes |
|---|---|---|---|---|
| Fastest overall | Llama-3.2-1B | 45.4 | 48K f16 | All 12 stop |
| Best 48K f16 + tools | Qwen3.5-2B | 27.7 | 48K f16 | SSM hybrid |
| Best 3-4B class | Gemma-4-E2B | 23.7 | 32K q4 | All 12 stop |
| Most concise | Gemma-3n-E2B | 20.5 | 32K q4 | All 12 stop, avg 138 tok func |
| Best tool accuracy (4B) | Spark-X2.5-4B | 14.5 | 16K→32K | BFCL 90.4% |
| Best 48K q4_0 all-stop | Granite 3.0-2B | 24.8 | 48K q4 | All 12 stop |

### Models that never stop generating (bad for agent use)

- **Ministral-3B-Instruct**: 11/12 hit max_tokens
- **Agents-A1-4B**: 11/12 hit max_tokens

### Failed to load (OOM at all contexts)

- **Gemma-4-E4B-QAT** (4.22GB weight, BFCL 92.3% champion — cannot run in this budget)
- **Mistral-7B** (4.37GB weight)

## Repository Structure

```
├── README.md                          # This file
├── llm_48k_context_survey.xlsx        # Master 5-sheet survey (80 models)
├── llm_48k_f16_hermes_toolcall.xlsx   # Filtered 39-model list + Sweep Results sheet
├── results/
│   ├── sweep_summary.json             # All model results summary
│   └── test_results_*.json            # Per-model detailed test results
├── architecture_data/
│   └── model_architecture.json        # Real GGUF architecture params (from llama-server -lv 4)
└── scripts/
    └── run_prompt_tests.py            # Reusable 12-prompt test runner
```

## Methodology

1. GDM stopped: `sudo systemctl stop gdm.service`
2. One model at a time, launched via `terminal background=true`
3. Context fallback ladder: 49152 → 32768 → 24576 → 16384 → 8192 → 4096
4. KV fallback ladder: f16 → q8_0 → q4_0
5. Server killed between models: `pkill -9 -f '[l]lama-server'`
6. Results saved after EVERY test (kernel/gateway timeouts can't lose data)
7. If a model's native context < 48K, use native context (per user directive)

## 32K Retest Candidates

Models originally tested below 32K that the corrected KV math says should fit at 32K q4_0:

| Model | Original ctx | Math (32K q4_0) | Status |
|---|---|---|---|
| Gemma-4-E2B | 16K | 3,827 MB | ✅ Retested: 23.7 t/s, all stop |
| Gemma-3n-E2B | 16K | 3,933 MB | Testing |
| Hermes-3-3B | 24K | 3,660 MB | Pending |
| Phi-4-mini | 16K | 3,695 MB | Pending |
| Qwen3-4B | 16K | 4,108 MB | Pending (tight) |
| Spark-X2.5-4B | 16K | 4,208 MB | Pending (tight) |
| Nemotron-3N-4B | 16K | 3,904 MB | Pending |
| xLAM-2-3B Q8 | 8K | 4,294 MB | Pending (very tight) |

## Author

Walker Kirkpatrick — Oregon naturopathic physician, Jetson edge AI researcher
GitHub: drwjkirkpatrick-web
