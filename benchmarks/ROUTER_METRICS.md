# Router / worker metrics summary

Prefer `router_metrics_summary.sh` (wraps `router_metrics_summary.py`).

Install and topologies: [`CACHE_AWARE_OPERATOR_GUIDE.md`](../docs/load_balancing/CACHE_AWARE_OPERATOR_GUIDE.md).
How to run smoke / Codex: [`CACHE_AWARE_BENCHMARKS.md`](CACHE_AWARE_BENCHMARKS.md).

## Quick summary (what we actually do)

**Quoted numbers:** scrape **after the client finishes, while router + workers
are still up**, then kill. Prometheus counters are cumulative since process
start; we cold-start each case so that snapshot is the case. Packaged wrappers
already do this (`run_dp_cache_aware_demo.sh` /
`run_codex_dp_cache_aware.sh`: bench → curl `.prom` → summary → teardown).

**Mid-run:** the same live command works while the bench is still going — a
partial snapshot (APC climbing? decisions showing?). Use that as a sanity
peek only; do not quote it as the case result.

Live scrape (stack still up; discovers workers from the router). `29400` is
`--prometheus-port` in the operator guide:

```bash
bash benchmarks/router_metrics_summary.sh 127.0.0.1:29400 --brief
```

After kill, use the saved `.prom` files the wrappers wrote (see Commands).

---

## Topologies

- Independent workers: discovers each worker URL and scrapes them.
- DP + router: worker labels look like `http://host:port@0`, `@1`, … — the
  helper **strips `@rank`, dedupes**, and scrapes the real backend once (sums
  `engine` labels).

---

## Hit-rate names (both always reported)

| JSON / brief name | Formula | Meaning |
|-------------------|---------|---------|
| **`apc_prefix_cache`** / `apc_hit_rate` | `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total` | Engine APC block/query reuse |
| **`prompt_token_cache`** / `prompt_hit_rate` | `vllm:prompt_tokens_cached_total / vllm:prompt_tokens_total` | Share of prompt tokens from cache (closer to prefill saved) |

`prefix_cache` remains as a **legacy alias of APC only** for older parsers.

---

## Latency means

Histogram `sum / count` across scraped backends.

| Brief key | Metric |
|-----------|--------|
| `queue_mean_s` | `vllm:request_queue_time_seconds` |
| `prefill_mean_s` | `vllm:request_prefill_time_seconds` |
| `decode_mean_s` | `vllm:request_decode_time_seconds` |
| `ttft_mean_s` | `vllm:time_to_first_token_seconds` |
| `e2e_mean_s` | `vllm:e2e_request_latency_seconds` |
| `inference_mean_s` | `vllm:request_inference_time_seconds` |

---

## Commands (saved `.prom` after the live scrape)

Same helper; pass files instead of `host:port` when the processes are already
gone. Wrappers write these under `LOG_DIR/metrics/`.

```bash
# Independent workers
bash benchmarks/router_metrics_summary.sh metrics_router.prom \
  --workers metrics_w0.prom,metrics_w1.prom \
  --out summary.json --brief-only --label independent_workers

# DP backend (one worker file is enough; helper strips `@rank`)
bash benchmarks/router_metrics_summary.sh metrics_router.prom \
  --workers metrics_backend_dp.prom \
  --out summary.json --brief-only --label dp2_cache_aware
```

JSON fields: `cache_aware_decisions`, `workers_balance`, `apc_prefix_cache`,
`prompt_token_cache`, `latency_seconds`, `backends.per_endpoint`, plus legacy
`prefix_cache` (APC alias).

---

## Decision labels

Match vs threshold uses **`match_rate > cache_threshold`** (strict `>`).

**Fallback** (`session_id_full_history_fallback`):

- `session_id_match` — session key strong → sticky
- `session_id_fallback` — session weak → try full history
- `full_history_match` — history strong → affinity
- `full_history_low_match` — still weak → min-load

**Single-key** (`full_history`, `session_id`):

- `cache_affinity` — match `>` `cache_threshold`
- `load_balance` — imbalance past abs/rel gates
- `low_match_min_load` — match at or below threshold → min-load

---

## Required Prometheus series

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

Experimental delta mode (`--pre` / `--post`) is marked
`delta_experimental_untested`. Prefer cold-start absolute snapshots.
