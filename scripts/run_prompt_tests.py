#!/usr/bin/env python3
"""
Reusable 12-prompt test runner for Jetson 8GB LLM sweep.
Saves results to JSON after EVERY test so kernel/gateway timeouts can't lose data.

Usage: python3 run_prompt_tests.py <model_alias> <port> <output_json_path> [prompts_path]

Prompts: 12 total (html x2, python x1, poetry x1, math x1, creative x1, function_call x6)
Per-request max_tokens: html=4096, python=4096, poetry=1024, math=2048, creative=1024, function_call=1024
Sampling: temp=0.0, top_k=1 (greedy), repeat_penalty=1.0
"""
import json, sys, time, urllib.request, urllib.error

ALIAS = sys.argv[1]
PORT  = int(sys.argv[2])
OUT   = sys.argv[3]
PROMPTS_PATH = sys.argv[4] if len(sys.argv) > 4 else "/home/walker/projects/jetson-model-zoo/test_prompts.json"

with open(PROMPTS_PATH) as f:
    PROMPTS = json.load(f)

MAX_TOKENS = {
    "html": 4096, "python": 4096, "poetry": 1024,
    "math": 2048, "creative": 1024, "function_call": 1024,
}

def send_request(prompt_text, max_tokens, category):
    payload = json.dumps({
        "model": ALIAS,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.0,
        "top_k": 1,
        "repeat_penalty": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        "http://127.0.0.1:%d/v1/chat/completions" % PORT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        elapsed = time.time() - start
        return {
            "id": "", "category": category, "model": ALIAS,
            "prompt_tokens": 0, "completion_tokens": 0,
            "elapsed": round(elapsed, 1), "tok_s": 0.0,
            "max_tokens": max_tokens, "finish_reason": "error",
            "output_chars": 0, "error": str(e),
        }
    elapsed = time.time() - start
    choice = data.get("choices", [{}])[0]
    usage = data.get("usage", {})
    content = choice.get("message", {}).get("content", "")
    finish = choice.get("finish_reason", "unknown")
    comp_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    tok_s = comp_tokens / elapsed if elapsed > 0 and comp_tokens > 0 else 0.0
    return {
        "prompt_tokens": prompt_tokens, "completion_tokens": comp_tokens,
        "elapsed": round(elapsed, 1), "tok_s": round(tok_s, 1),
        "max_tokens": max_tokens, "finish_reason": finish,
        "output_chars": len(content), "error": None,
    }

def run_all():
    results = []
    for p in PROMPTS:
        pid = p.get("id", p.get("name", ""))
        cat = p.get("category", "")
        prompt_text = p.get("prompt", p.get("text", ""))
        max_tok = MAX_TOKENS.get(cat, 2048)
        result = send_request(prompt_text, max_tok, cat)
        result["id"] = pid
        result["category"] = cat
        result["model"] = ALIAS
        status = "PASS" if result["error"] is None else "FAIL"
        print("  %-20s %-15s %5d tok  %6.1fs  %5.1f t/s  fin=%-8s  %s" % (
            pid, cat, result["completion_tokens"], result["elapsed"],
            result["tok_s"], result["finish_reason"], status))
        results.append(result)
        with open(OUT, "w") as f:
            json.dump({"model": ALIAS, "results": results}, f, indent=2)
    tps_vals = [r["tok_s"] for r in results if r["tok_s"] > 0]
    avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else 0
    n_pass = sum(1 for r in results if r["error"] is None)
    print("=== %s SUMMARY ===" % ALIAS)
    print("Average tok/s: %.1f" % avg_tps)
    print("Tests completed: %d/%d" % (n_pass, len(results)))
    with open(OUT, "w") as f:
        json.dump({"model": ALIAS, "results": results, "avg_tps": avg_tps,
                    "pass": n_pass, "total": len(results)}, f, indent=2)

if __name__ == "__main__":
    run_all()
