# Cache-Aware vLLM Router — Operator Guide

One-stop path: install → two workers + router → synthetic bench → clean metrics.
Also covers a cold DP baseline for comparison.

Canonical helpers in this directory (or `benchmarks/` in the router repo):

```text
benchmarks/README.md
benchmarks/chat_prefix_repetition.py
benchmarks/router_metrics_summary.sh   ← prefer this
benchmarks/router_metrics_summary.py   ← implementation
benchmarks/CACHE_AWARE_OPERATOR_GUIDE.md
```

**Recommended model for demos:** [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
(or your local checkout of that checkpoint). It is small enough for 2-device
smoke runs while still supporting long context for prefix-cache experiments.

---

## 1. Install

Run on a GPU/NPU node (or in your CUDA/Ascend container), not a bare login node
with a mismatched Python.

```bash
cd /path/to/workspace
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# vLLM workers (separate from the router binary)
uv venv env_vllm && source env_vllm/bin/activate
uv pip install "vllm==0.26.0" --torch-backend=auto   # pin as needed

git clone https://github.com/vllm-project/router.git
# Checkout a branch that includes chat routing key modes if needed
cd router
git checkout <branch-with-chat-routing>
cargo build --release
./target/release/vllm-router --help
# expect --chat-routing-key-mode when that feature is in your checkout
```

This guide uses the **native Rust binary** only:

```bash
./target/release/vllm-router …
# equivalent while developing:
cargo run --release -- …
```

A Python `vllm-router` wheel entrypoint also exists; it is not required here.

Verify workers:

```bash
python - <<'PY'
import torch, vllm
print(torch.__version__, getattr(torch, "cuda", None) and torch.cuda.is_available(), vllm.__version__)
PY
```

---

## 2. Topology (read this once)

| Setup | What you start | Role |
|-------|----------------|------|
| **Router demo (this guide)** | N× independent `vllm serve` (no DP) + `./target/release/vllm-router --policy cache_aware` | Router sticks sessions to a worker that already has the prefix |
| **DP baseline** | One `vllm serve --data-parallel-size N` (no router) | Native DP / shared prefix inside one engine |

On repeated-prefix / multi-session chat workloads, **cache-aware + `lb_mid` usually
beats the DP baseline** on TTFT and prefix hit rate (router keeps a session on the
worker that already holds the KV). DP remains a useful control: same GPU count,
no external router.

---

## 3. Start workers + cache-aware router

```bash
# From the router repo root (after cargo build --release)
# Recommend Qwen3.5-4B: small weights, long context, good for 2-device demos
export MODEL_PATH=/path/to/Qwen3.5-4B
export SERVED=qwen35-4b

# Worker 0 / 1 (CUDA example; NPU: set your device visibility env instead)
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port 18100 \
  --served-model-name "$SERVED" \
  --tensor-parallel-size 1 \
  --enable-prefix-caching --trust-remote-code \
  > vllm_w0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port 18101 \
  --served-model-name "$SERVED" \
  --tensor-parallel-size 1 \
  --enable-prefix-caching --trust-remote-code \
  > vllm_w1.log 2>&1 &

curl -fsS http://127.0.0.1:18100/health && curl -fsS http://127.0.0.1:18101/health

./target/release/vllm-router \
  --host 0.0.0.0 --port 18180 \
  --worker-urls http://127.0.0.1:18100 http://127.0.0.1:18101 \
  --policy cache_aware \
  --prometheus-port 29400 \
  --cache-threshold 0.3 \
  --balance-abs-threshold 2 \
  --balance-rel-threshold 1.5 \
  --chat-routing-key-mode full-history \
  > router.log 2>&1 &

curl -fsS http://127.0.0.1:18180/health
```

CLI key-mode spelling uses hyphens: `full-history`, `session-id`,
`session-id-full-history-fallback`.

### Threshold parameters

| Flag | Meaning |
|------|---------|
| `--cache-threshold` | Min prefix-tree match rate to treat as a strong hit and prefer that worker |
| `--balance-abs-threshold` | Absolute load gap that allows breaking affinity |
| `--balance-rel-threshold` | Relative load gap that allows breaking affinity |
| `--chat-routing-key-mode` | Which string keys the cache-aware tree for chat |

Presets:

```text
lb_mid   = cache=0.3,   abs=2, rel=1.5   # fallback / full_history demos (usual winner vs DP)
lb_aggr  = cache=0.3,   abs=0, rel=1.0   # more even workers, lower hit rate
sid999   = cache=0.999, abs=2, rel=1.5   # pure session_id (near-exact)
```

Stickier gates → higher hit rate, risk of one-worker hotspot. Aggressive LB →
better balance / RPS, worse hit rate.

---

## 4. Run benchmarks

Cold-start each case (kill previous workers/router, or use fresh ports) so absolute
Prometheus counters match that case.

### 4.1 Synthetic chat (cache-aware router)

```bash
cd /path/to/router

python3 benchmarks/chat_prefix_repetition.py \
  --base-url http://127.0.0.1:18180 \
  --model qwen35-4b \
  --num-prompts 100 \
  --session-groups 16 \
  --unique-prefixes 12 \
  --prefix-len 256 \
  --suffix-len 16 \
  --output-len 16 \
  --max-concurrency 4 \
  --label full_history_lb_mid
```

While still up:

```bash
bash benchmarks/router_metrics_summary.sh 127.0.0.1:29400 \
  --out summary_router_lb_mid.json --brief-only --label full_history_lb_mid
```

### 4.2 DP baseline (no router)

Same model and device count, one process with `--data-parallel-size 2`. Point the
bench at the DP API port (not the router).

```bash
# Tear down router workers first, or use different GPUs/ports
pkill -f 'vllm serve' || true
pkill -f 'target/release/vllm-router' || true
sleep 5

export MODEL_PATH=/path/to/Qwen3.5-4B
export SERVED=qwen35-4b-dp

CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port 18000 \
  --served-model-name "$SERVED" \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --enable-prefix-caching --trust-remote-code \
  > vllm_dp.log 2>&1 &

curl -fsS http://127.0.0.1:18000/health

python3 benchmarks/chat_prefix_repetition.py \
  --base-url http://127.0.0.1:18000 \
  --model "$SERVED" \
  --num-prompts 100 \
  --session-groups 16 \
  --unique-prefixes 12 \
  --prefix-len 256 \
  --suffix-len 16 \
  --output-len 16 \
  --max-concurrency 4 \
  --label dp_baseline
```

DP has no router decision metrics. For prefix hit rate, scrape the DP `/metrics`
(or sum DP-rank endpoints if your build exposes them separately):

```bash
curl -fsS http://127.0.0.1:18000/metrics > metrics_dp.prom
# Inspect vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total
```

**Expectation:** on this synthetic repeated-prefix chat shape, **`full_history` +
`lb_mid` (and often `session_id` / fallback + `lb_mid`) should beat DP** on TTFT
and prefix hit rate. If DP wins, check that router workers had `--enable-prefix-caching`,
that you used cold starts, and that `cache_threshold` / LB gates were not set to
aggressive `lb_aggr` (which trades hit rate for balance).

---

## 5. Metrics: use the `.sh`

`router_metrics_summary.sh` is enough for operators — it wraps the `.py`.

**After the router bench, before teardown:**

```bash
cd /path/to/router

bash benchmarks/router_metrics_summary.sh 127.0.0.1:29400 --brief

bash benchmarks/router_metrics_summary.sh 127.0.0.1:29400 \
  --out summary.json --brief-only --label full_history_lb_mid
```

Raw curls are optional backup (messy; do not paste into chat logs):

```bash
curl -fsS http://127.0.0.1:29400/metrics > metrics_router.prom
curl -fsS http://127.0.0.1:18100/metrics > metrics_w0.prom
curl -fsS http://127.0.0.1:18101/metrics > metrics_w1.prom

bash benchmarks/router_metrics_summary.sh metrics_router.prom \
  --workers metrics_w0.prom,metrics_w1.prom --brief-only
```

### Decision labels (short)

**Fallback:** `session_id_match`, `session_id_fallback`, `full_history_match`,
`full_history_low_match`.

**Single-key:** `cache_affinity`, `load_balance`, `low_match_min_load`.

Full definitions: `benchmarks/README.md`.

---

## 6. Minimal checklist

1. Install vLLM + `cargo build --release` (branch with chat modes).  
2. Prefer **Qwen3.5-4B** for a small long-context demo.  
3. Start 2 workers with `--enable-prefix-caching` + router `cache_aware` + `lb_mid`.  
4. Run `benchmarks/chat_prefix_repetition.py` against the router.  
5. `bash benchmarks/router_metrics_summary.sh … --brief` while still up.  
6. Cold-start DP baseline (§4.2) on the same GPU count; compare TTFT / hit rate.  
7. Expect **lb_mid cache-aware > DP** on this workload; then tear down.  
