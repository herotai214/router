# Cache-Aware Chat Benchmarks

Small, dependency-free helpers for `/v1/chat/completions` prefix-cache experiments
with vLLM Router.

| File | Role |
|------|------|
| `CACHE_AWARE_OPERATOR_GUIDE.md` | End-to-end install → topologies → bench → metrics |
| `chat_prefix_repetition.py` | Synthetic chat prefix-repetition client (TTFT / E2E / RPS) |
| `chat_jsonl_bench.py` | Replay OpenAI chat JSONL (Codex-style) with TTFT / E2E |
| `router_metrics_summary.sh` | **User entrypoint** for post-run metrics summary |
| `router_metrics_summary.py` | Implementation behind the `.sh` |
| `run_dp_cache_aware_demo.sh` | Synthetic **DP + cache-aware router** demo |
| `run_codex_dp_cache_aware.sh` | Codex JSONL **DP + cache-aware** runner (NPU/CUDA) |

**Recommended demo model:** Qwen3.5-4B. See the operator guide.

Cold-start each case (fresh backends + router) so absolute Prometheus counters
are the case counters.

## Two topologies (both supported)

Cache-aware does **not** require DP. Keep both setups:

```text
A) Independent workers (no DP)
   Client → router(cache_aware) → vllm:18100
                                → vllm:18101

B) Intra-node DP + router  (customer-like)
   Client → router(cache_aware, --intra-node-data-parallel-size 2)
                 → one vllm serve --data-parallel-size 2
                   (router sends X-data-parallel-rank)
```

| | Topology A | Topology B |
|--|--|--|
| Backends | N independent `vllm serve` | One `vllm serve --data-parallel-size N` |
| Router | `--worker-urls u0 u1 …` | `--worker-urls u0 --intra-node-data-parallel-size N` |
| Demo script | manual (below) | `run_dp_cache_aware_demo.sh` |

Neither is deprecated. A is the clearest affinity demo; B matches many “one DP
server behind a router” deployments.

## Topology A — 2 independent workers + router

```bash
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
```

## Topology B — DP + cache-aware router (scripted)

```bash
# From router repo root after: cargo build --release
MODEL_PATH=/path/to/Qwen3.5-4B \
DEVICE_ENV_NAME=CUDA_VISIBLE_DEVICES DEVICES=0,1 \
ROUTER_BIN=./target/release/vllm-router \
bash benchmarks/run_dp_cache_aware_demo.sh
```

NPU:

```bash
MODEL_PATH=/path/to/Qwen3.5-4B \
DEVICE_ENV_NAME=ASCEND_RT_VISIBLE_DEVICES DEVICES=0,1 \
VLLM_BIN=vllm \
ROUTER_BIN=./target/release/vllm-router \
bash benchmarks/run_dp_cache_aware_demo.sh
```

The synthetic script starts DP backend + router, runs `chat_prefix_repetition.py`,
scrapes `.prom`s, and writes `summary.json` via `router_metrics_summary.sh`.

### Codex JSONL on DP + router (NPU-friendly)

For the realistic Codex 100-req JSONL (same shape as `npu_codex_100req_runner`),
use `run_codex_dp_cache_aware.sh`. Defaults assume Ascend
(`ASCEND_RT_VISIBLE_DEVICES`); override for CUDA.

```bash
# NPU — source CANN/ATB first so torch_npu imports
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh   # if present

MODEL_PATH=/path/to/Qwen3.5-4B \
DATASET=/path/to/01_codex_swebenchpro_128k_filter_25s4t_chat.jsonl \
DEVICE_ENV_NAME=ASCEND_RT_VISIBLE_DEVICES DEVICES=0,1 \
ROUTER_BIN=./target/release/vllm-router \
RUN_DP_BASELINE=1 RUN_CACHE_AWARE=1 \
CONFIGS=lb_mid:0.3:2:1.5 \
MAX_TOKENS=256 \
bash benchmarks/run_codex_dp_cache_aware.sh
```

Run configs **one at a time** on a single DP pair (set `DEVICES` / `CONFIGS` per case):

```bash
DEVICES=0,1 CONFIGS="lb_mid:0.3:2:1.5" bash benchmarks/run_codex_dp_cache_aware.sh
```

CUDA:

```bash
DEVICE_ENV_NAME=CUDA_VISIBLE_DEVICES DEVICES=0,1 ... bash benchmarks/run_codex_dp_cache_aware.sh
```

Artifacts per `LOG_DIR`: `vllm_*.log`, `router_*.log`, `bench_*.log`,
`metrics/*.prom`, `summary_*.json` (APC + Prompt hit + queue/prefill/decode/…).

Manual equivalent (no script):

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port 18100 \
  --served-model-name qwen35-4b-dp \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --enable-prefix-caching --trust-remote-code \
  > vllm_dp.log 2>&1 &

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
```

### Threshold knobs

| Flag | Role |
|------|------|
| `--cache-threshold` | Min prefix-tree match for affinity (`~0.3` text; `~0.999` pure session-id) |
| `--balance-abs-threshold` / `--balance-rel-threshold` | Load-balance gates (request-count mode) |
| `--cache-aware-load-metric` | `request` (default) or `token` predicted-load routing |
| `--token-abs-req-equiv` / `--token-balance-rel` | Token-mode abs (incoming-request equivalents) and relative slack |
| `--intra-node-data-parallel-size` | Topology B only — expand one URL into DP-rank workers |

Presets: `lb_mid` = `0.3 / 2 / 1.5`, `lb_aggr` = `0.3 / 0 / 1.0`, `sid999` = `0.999 / 2 / 1.5`.

## Run the synthetic chat benchmark

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
  --label cache_aware_lb_mid
```

The client sends `session_params.session_id` so `session-id` / fallback modes
have a real sticky key.

## Get metrics (prefer the `.sh`)

**Why both `.sh` and `.py`?**
`.sh` is the one-liner; `.py` is the implementation. Prefer `.sh` in docs/shells.

### What it reports

**Hit rates (distinct names):**

| Name | Formula | Meaning |
|------|---------|---------|
| `apc_prefix_cache` / `apc_hit_rate` | `prefix_cache_hits_total / prefix_cache_queries_total` | APC block/query reuse |
| `prompt_token_cache` / `prompt_hit_rate` | `prompt_tokens_cached_total / prompt_tokens_total` | Prompt tokens from cache |

Legacy JSON field `prefix_cache` = **APC only** (alias). Prefer the two names above.

**Latency means** (`latency_seconds` / brief `*_mean_s`):

| Key | Prometheus histogram |
|-----|----------------------|
| queue | `vllm:request_queue_time_seconds` |
| prefill | `vllm:request_prefill_time_seconds` |
| decode | `vllm:request_decode_time_seconds` |
| ttft | `vllm:time_to_first_token_seconds` |
| e2e | `vllm:e2e_request_latency_seconds` |
| inference | `vllm:request_inference_time_seconds` |

**DP support:** router worker labels like `http://127.0.0.1:18100@0` are stripped
to the backend base URL, deduped, and scraped once (`engine` series summed).

### Commands

```bash
# Live
bash benchmarks/router_metrics_summary.sh 127.0.0.1:29400 --brief

# Independent workers from files
bash benchmarks/router_metrics_summary.sh metrics_router.prom \
  --workers metrics_w0.prom,metrics_w1.prom \
  --out summary.json --brief-only

# DP backend from files (one backend scrape)
bash benchmarks/router_metrics_summary.sh metrics_router.prom \
  --workers metrics_backend_dp.prom \
  --out summary.json --brief-only --label dp2_cache_aware
```

### JSON fields

- `cache_aware_decisions` — routing decision counts
- `workers_balance` — policy decisions by worker URL (`@rank` for DP)
- `apc_prefix_cache` — APC hit stats
- `prompt_token_cache` — prompt-token cache stats
- `latency_seconds` — queue / prefill / decode / ttft / e2e / inference means
- `backends.per_endpoint` — per scraped backend breakdown
- `prefix_cache` — legacy APC alias

### Decision labels

**Fallback** (`session_id_full_history_fallback`):

- `session_id_match` — session key strong → sticky
- `session_id_fallback` — session weak → try full history
- `full_history_match` — history strong → affinity
- `full_history_low_match` — still weak → min-load

**Single-key** (`full_history`, `session_id`):

- `cache_affinity` — match ≥ `cache_threshold`
- `load_balance` — imbalance past abs/rel gates
- `low_match_min_load` — match below threshold → min-load

### Required Prometheus series

Router:

```text
vllm_router_cache_aware_decisions_total{decision="..."}
vllm_router_policy_decisions_total{policy="cache_aware",worker="..."}
```

Workers / DP backend:

```text
vllm:prefix_cache_hits_total
vllm:prefix_cache_queries_total
vllm:prompt_tokens_cached_total
vllm:prompt_tokens_total
vllm:request_queue_time_seconds_{sum,count}
vllm:request_prefill_time_seconds_{sum,count}
vllm:request_decode_time_seconds_{sum,count}
vllm:time_to_first_token_seconds_{sum,count}
vllm:e2e_request_latency_seconds_{sum,count}
vllm:request_inference_time_seconds_{sum,count}
```

### Experimental delta mode

```bash
bash benchmarks/router_metrics_summary.sh \
  --pre logs/case_pre.prom \
  --post logs/case_post.prom
```

Marked `delta_experimental_untested`. Prefer cold-start absolute snapshots.

## DP baseline (no router)

Same GPU count, `--data-parallel-size 2`, point the bench at the DP API port.
Details: `CACHE_AWARE_OPERATOR_GUIDE.md` §4.2. On repeated-prefix chat,
**cache-aware + `lb_mid` should generally beat plain DP** on TTFT and hit rate
when the stack’s prefix cache is healthy.
