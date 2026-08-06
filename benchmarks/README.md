# Cache-Aware Chat Benchmarks

Small, dependency-free helpers for `/v1/chat/completions` prefix-cache experiments
with vLLM Router.

| File | Role |
|------|------|
| `CACHE_AWARE_OPERATOR_GUIDE.md` | End-to-end install → workers+router → bench → DP baseline → metrics |
| `chat_prefix_repetition.py` | Synthetic chat prefix-repetition client (TTFT / E2E / RPS) |
| `router_metrics_summary.sh` | **User entrypoint** for a clean post-run metrics summary |
| `router_metrics_summary.py` | Implementation behind the `.sh` (also callable from other tools) |

**Recommended demo model:** Qwen3.5-4B (small weights, long context). See the operator guide.

Cold-start each case (fresh workers + router) so absolute Prometheus counters are
the case counters.

## Mental model

```text
Client  →  vllm-router (cache_aware)  →  independent vLLM workers
                │                              │
                │  picks worker likely to      │  each owns its own
                │  already hold this prefix    │  KV / prefix cache
```

For these demos, start **N independent** `vllm serve` processes (TP=1, no DP)
and pass each URL to the router.

## Bring up 2 workers + cache-aware router

```bash
# Recommend Qwen3.5-4B for a small long-context demo
export MODEL_PATH=/path/to/Qwen3.5-4B
export SERVED=qwen35-4b

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port 18100 \
  --served-model-name "$SERVED" \
  --tensor-parallel-size 1 \
  --enable-prefix-caching \
  --trust-remote-code \
  > vllm_w0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port 18101 \
  --served-model-name "$SERVED" \
  --tensor-parallel-size 1 \
  --enable-prefix-caching \
  --trust-remote-code \
  > vllm_w1.log 2>&1 &

curl -fsS http://127.0.0.1:18100/health
curl -fsS http://127.0.0.1:18101/health

# Preferred: native Rust binary (from repo root after cargo build --release)
# Chat key modes: full-history | session-id | session-id-full-history-fallback
./target/release/vllm-router \
  --host 0.0.0.0 \
  --port 18180 \
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

Build the binary with `cargo build --release` (from the router repo root).  
`cargo run --release -- …` is equivalent. A `vllm-router` console script from a
Python wheel exists and wraps the same engine, but this guide uses the Rust
binary only.

### Threshold knobs

| Flag | Role |
|------|------|
| `--cache-threshold` | Min prefix-tree match rate for cache affinity (use `~0.3` for text keys; `~0.999` for pure `session-id` to avoid false matches) |
| `--balance-abs-threshold` / `--balance-rel-threshold` | Load-balance gates; if imbalance exceeds them, router may skip affinity (`load_balance`) |

Common presets: `lb_mid` = `0.3 / 2 / 1.5`, `lb_aggr` = `0.3 / 0 / 1.0`, `sid999` = `0.999 / 2 / 1.5`.

## Run the synthetic chat benchmark

From the router repo root:

```bash
python3 benchmarks/chat_prefix_repetition.py \
  --base-url http://127.0.0.1:18180 \
  --model qwen35-4b \
  --num-prompts 100 \
  --prefix-len 256 \
  --suffix-len 16 \
  --session-groups 16 \
  --unique-prefixes 12 \
  --output-len 16 \
  --max-concurrency 4 \
  --label full_history_lb_mid
```

Env vars `BASE_URL` / `MODEL` / `NUM_PROMPTS` / … still work as fallbacks.

The client sends `session_params.session_id` on every request so `session-id` and
fallback key modes have a real sticky key.

## Get metrics (prefer the `.sh`)

**Why both `.sh` and `.py`?**  
`router_metrics_summary.sh` is the one-liner you should run; it only forwards
args to `router_metrics_summary.py`. The `.py` is the real implementation (also
imported/invoked by automation). Prefer the `.sh` in docs and shells.

While router and workers are **still alive**:

```bash
# Clean JSON + optional one-liner
bash benchmarks/router_metrics_summary.sh 127.0.0.1:29400 --brief

# Save JSON to a file, print only the brief line
bash benchmarks/router_metrics_summary.sh 127.0.0.1:29400 \
  --out summary.json --brief-only --label full_history_lb_mid
```

Target may be `host:port`, a URL, or a saved `.prom` file:

```bash
curl -fsS http://127.0.0.1:29400/metrics > metrics_router.prom
curl -fsS http://127.0.0.1:18100/metrics > metrics_w0.prom
curl -fsS http://127.0.0.1:18101/metrics > metrics_w1.prom

bash benchmarks/router_metrics_summary.sh metrics_router.prom \
  --workers metrics_w0.prom,metrics_w1.prom \
  --out summary.json --brief-only
```

Raw `/metrics` dumps are huge and messy; use the summary helper for decisions +
prefix hit rate. Do not rely on scraping after teardown — worker hit rate needs
a live (or saved) worker scrape.

### JSON fields

- `cache_aware_decisions` — routing decision counts
- `workers_balance` — policy decisions by worker URL
- `prefix_cache` — aggregate / per-worker hits, queries, hit rate

### Decision labels

**Fallback mode** (`session_id_full_history_fallback`):

- `session_id_match` — session key strong → sticky worker
- `session_id_fallback` — session weak/stale → try full history (path counter)
- `full_history_match` — after fallback, history strong → affinity
- `full_history_low_match` — after fallback still weak → min-load

**Single-key modes** (`full_history`, `session_id`):

- `cache_affinity` — match ≥ `cache_threshold`
- `load_balance` — imbalance past abs/rel gates
- `low_match_min_load` — match below threshold → min-load

### Required Prometheus series

Router:

```text
vllm_router_cache_aware_decisions_total{decision="..."}
vllm_router_policy_decisions_total{policy="cache_aware",worker="..."}
```

Workers:

```text
vllm:prefix_cache_hits_total
vllm:prefix_cache_queries_total
```

### Experimental delta mode

```bash
bash benchmarks/router_metrics_summary.sh \
  --pre logs/case_pre.prom \
  --post logs/case_post.prom
```

Marked `delta_experimental_untested` in JSON. Prefer cold-start absolute snapshots.

## DP baseline and expectations

See `CACHE_AWARE_OPERATOR_GUIDE.md` §4.2 for a same-GPU-count
`--data-parallel-size 2` control (no router). On this synthetic repeated-prefix
chat shape, **cache-aware + `lb_mid` should generally beat DP** on TTFT and
prefix hit rate.
