# Cache-Aware vLLM Router — Operator Guide

One-stop path: install → bring up topology → synthetic bench → clean metrics.

Canonical helpers:

```text
benchmarks/README.md
benchmarks/chat_prefix_repetition.py      ← synthetic client
benchmarks/chat_jsonl_bench.py            ← Codex / OpenAI chat JSONL client
benchmarks/router_metrics_summary.sh      ← prefer this
benchmarks/router_metrics_summary.py
benchmarks/run_dp_cache_aware_demo.sh     ← synthetic DP+router demo
benchmarks/run_codex_dp_cache_aware.sh            ← Codex JSONL DP+router (NPU/CUDA)
benchmarks/CACHE_AWARE_OPERATOR_GUIDE.md
```

**Recommended model for demos:** [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
(or your local checkout). Small enough for 2-device smoke runs, long context for
prefix-cache experiments.

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
# NPU: use your vllm + vllm-ascend env instead (e.g. 0.23) and source CANN/ATB.

git clone https://github.com/vllm-project/router.git
cd router
git checkout <branch-with-chat-routing>
cargo build --release
./target/release/vllm-router --help
# expect --chat-routing-key-mode and --intra-node-data-parallel-size
```

This guide uses the **native Rust binary**:

```bash
./target/release/vllm-router …
# or while developing:
cargo run --release -- …
```

---

## 2. Two supported topologies (keep both)

Cache-aware routing is **not** tied to DP. Pick the topology that matches your
deployment story:

| Topology | What you start | Router flags | When to use |
|----------|----------------|--------------|-------------|
| **A. Independent workers** | N× `vllm serve` (no DP), one URL each | `--worker-urls u0 u1 …` (no intra-node DP) | Clearest cache-affinity demo; each process owns its KV |
| **B. Intra-node DP + router** | One `vllm serve --data-parallel-size N` | `--worker-urls <one-url> --intra-node-data-parallel-size N` | Closer to many production “one DP server” setups; router sends `X-data-parallel-rank` |
| **C. DP baseline (control)** | One `vllm serve --data-parallel-size N` **without** router | — | Same GPU count, no external routing |

```text
A) Client → router(cache_aware) → worker0
                                 → worker1

B) Client → router(cache_aware, intra-node-dp=N)
                 → one vLLM API (DP ranks 0..N-1 via X-data-parallel-rank)

C) Client → one vLLM API (--data-parallel-size N)   # no router
```

Neither A nor B is deprecated. A is still the simplest mental model; B is the
customer-like DP+router path (`benchmarks/run_dp_cache_aware_demo.sh`).

---

## 3A. Independent workers + cache-aware router

```bash
export MODEL_PATH=/path/to/Qwen3.5-4B
export SERVED=qwen35-4b

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

## 3B. Intra-node DP + cache-aware router

One backend process, DP ranks selected by the router:

```bash
export MODEL_PATH=/path/to/Qwen3.5-4B
export SERVED=qwen35-4b-dp

CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port 18100 \
  --served-model-name "$SERVED" \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --enable-prefix-caching --trust-remote-code \
  > vllm_dp.log 2>&1 &

curl -fsS http://127.0.0.1:18100/health

./target/release/vllm-router \
  --host 0.0.0.0 --port 18180 \
  --worker-urls http://127.0.0.1:18100 \
  --policy cache_aware \
  --prometheus-port 29400 \
  --cache-threshold 0.3 \
  --balance-abs-threshold 2 \
  --balance-rel-threshold 1.5 \
  --chat-routing-key-mode session-id-full-history-fallback \
  --intra-node-data-parallel-size 2 \
  > router.log 2>&1 &

curl -fsS http://127.0.0.1:18180/health
```

Or run the packaged **synthetic** demo (backend + router + bench + summary):

```bash
MODEL_PATH=/path/to/Qwen3.5-4B \
DEVICE_ENV_NAME=CUDA_VISIBLE_DEVICES DEVICES=0,1 \
ROUTER_BIN=./target/release/vllm-router \
bash benchmarks/run_dp_cache_aware_demo.sh
```

For **Codex JSONL** on the same DP+router topology (NPU defaults):

```bash
# Source Ascend env first (torch_npu must import)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh   # if present

MODEL_PATH=/path/to/Qwen3.5-4B \
DATASET=/path/to/01_codex_swebenchpro_128k_filter_25s4t_chat.jsonl \
DEVICE_ENV_NAME=ASCEND_RT_VISIBLE_DEVICES DEVICES=0,1 \
ROUTER_BIN=./target/release/vllm-router \
RUN_DP_BASELINE=1 \
CONFIGS=lb_mid:0.3:2:1.5 \
MAX_TOKENS=256 \
bash benchmarks/run_codex_dp_cache_aware.sh
```

`CONFIGS` can list several `label:cache:abs:rel` entries (cold-started sequentially).
CUDA: `DEVICE_ENV_NAME=CUDA_VISIBLE_DEVICES`.

CLI key-mode spelling uses hyphens: `full-history`, `session-id`,
`session-id-full-history-fallback`.

### Threshold parameters

| Flag | Meaning |
|------|---------|
| `--cache-threshold` | Min prefix-tree match rate to treat as a strong hit and prefer that worker |
| `--balance-abs-threshold` | Absolute load gap that allows breaking affinity |
| `--balance-rel-threshold` | Relative load gap that allows breaking affinity |
| `--chat-routing-key-mode` | Which string keys the cache-aware tree for chat |
| `--intra-node-data-parallel-size` | Expand one backend URL into DP-rank virtual workers (topology B only) |

Presets:

```text
lb_mid   = cache=0.3,   abs=2, rel=1.5   # usual winner vs plain DP
lb_aggr  = cache=0.3,   abs=0, rel=1.0   # more even workers, lower hit rate
sid999   = cache=0.999, abs=2, rel=1.5   # pure session_id (near-exact)
```

---

## 4. Run benchmarks

Cold-start each case (kill previous processes, or use fresh ports) so absolute
Prometheus counters match that case.

### 4.1 Synthetic chat (against the router)

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
  --label cache_aware_lb_mid
```

### 4.2 DP baseline without router (control)

```bash
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

For DP-only (no router decisions), scrape the DP `/metrics` and pass it as
`--workers` to the summary helper (APC / Prompt / latency still work; decisions
will be empty).

---

## 5. Metrics: use the `.sh`

`router_metrics_summary.sh` wraps `router_metrics_summary.py`. Prefer the `.sh`.

It supports **both** topologies:

- Independent workers: discovers each worker URL and scrapes them.
- DP+router: router workers look like `http://host:port@0`, `@1`, … — the helper
  **strips `@rank`, dedupes**, and scrapes the real backend once (sums `engine`
  labels).

### Hit-rate names (both always reported)

| JSON / brief name | Formula | Meaning |
|-------------------|---------|---------|
| **`apc_prefix_cache`** / `apc_hit_rate` | `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total` | Engine APC block/query reuse |
| **`prompt_token_cache`** / `prompt_hit_rate` | `vllm:prompt_tokens_cached_total / vllm:prompt_tokens_total` | Share of prompt tokens from cache (closer to prefill saved) |

`prefix_cache` remains as a **legacy alias of APC only** for older parsers.

### Latency means (from worker histograms)

| Brief key | Metric |
|-----------|--------|
| `queue_mean_s` | `vllm:request_queue_time_seconds` |
| `prefill_mean_s` | `vllm:request_prefill_time_seconds` |
| `decode_mean_s` | `vllm:request_decode_time_seconds` |
| `ttft_mean_s` | `vllm:time_to_first_token_seconds` |
| `e2e_mean_s` | `vllm:e2e_request_latency_seconds` |
| `inference_mean_s` | `vllm:request_inference_time_seconds` |

Mean = histogram `sum / count` across scraped backends.

### Commands

While still up (or from saved `.prom` files):

```bash
# Live router + auto worker discovery
bash benchmarks/router_metrics_summary.sh 127.0.0.1:29400 --brief

# Saved scrapes — independent workers
bash benchmarks/router_metrics_summary.sh metrics_router.prom \
  --workers metrics_w0.prom,metrics_w1.prom \
  --out summary.json --brief-only --label independent_workers

# Saved scrapes — DP backend (one file is enough)
bash benchmarks/router_metrics_summary.sh metrics_router.prom \
  --workers metrics_backend_dp.prom \
  --out summary.json --brief-only --label dp2_cache_aware
```

Brief line example:

```text
METRICS_SUMMARY label=dp2_cache_aware apc_hit_rate=62.93% ... prompt_hit_rate=57.96% ...
queue_mean_s=0.869 prefill_mean_s=6.739 decode_mean_s=3.380 ttft_mean_s=8.468 e2e_mean_s=11.847 ...
```

### Decision labels (short)

**Fallback:** `session_id_match`, `session_id_fallback`, `full_history_match`,
`full_history_low_match`.

**Single-key:** `cache_affinity`, `load_balance`, `low_match_min_load`.

Full definitions: `benchmarks/README.md`.

---

## 6. Minimal checklist

1. Install vLLM (+ Ascend if NPU) and `cargo build --release`.
2. Prefer **Qwen3.5-4B** for a small long-context demo.
3. Choose topology **A** (independent workers) or **B** (DP + `--intra-node-data-parallel-size`).
4. Enable `--enable-prefix-caching` on every backend; use `lb_mid` for demos.
5. Run `benchmarks/chat_prefix_repetition.py` against the router (or `run_dp_cache_aware_demo.sh` for B).
6. `bash benchmarks/router_metrics_summary.sh … --brief` — check **APC + Prompt hit** and **queue/prefill/decode/TTFT/E2E**.
7. Optional: cold DP baseline (§4.2) on the same GPU count for comparison.
