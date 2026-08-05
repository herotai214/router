# Router Metrics Summary

This directory contains a small helper for summarizing router Prometheus metrics
after chat-completions benchmark runs.

This is a rough benchmark utility for the current `/v1/chat/completions`
prefix-cache experiments. It assumes the router and workers are cold-started for
one benchmark case, then queried right after the case finishes.

## One-command Usage

After a benchmark finishes, while the router and workers are still alive, run:

```bash
bash router/benchmarks/router_metrics_summary.sh 127.0.0.1:29400
```

The target may be either `host:port`, a full URL, or a saved `.prom` file:

```bash
bash router/benchmarks/router_metrics_summary.sh http://127.0.0.1:29400/metrics
bash router/benchmarks/router_metrics_summary.sh logs/my_run/metrics_router_case_post.prom
```

The helper curls router `/metrics`, discovers worker URLs from the router's
worker-balance metrics, curls each worker `/metrics` once, and computes prefix
hit rate from worker counters. Nothing runs in the router background.

The output JSON uses the benchmark-facing names:

- `cache_aware_decisions`: cache-aware routing decision counts.
- `workers_balance`: policy decision split by worker.
- `prefix_cache`: aggregate and per-worker prefix-cache hits, queries, and hit
  rate.

Example shape:

```json
{
  "window": "absolute_cold_start",
  "scope": "chat_completions_benchmark_rough",
  "cache_aware_decisions": {
    "session_id_match": 73,
    "session_id_fallback": 27,
    "full_history_match": 1,
    "full_history_low_match": 26,
    "cache_affinity": 0,
    "load_balance": 0,
    "low_match_min_load": 0,
    "total": 127
  },
  "workers_balance": {
    "by_worker": {
      "http://127.0.0.1:28100": 52,
      "http://127.0.0.1:28101": 48
    },
    "total": 100
  },
  "prefix_cache": {
    "hits": 4405104,
    "queries": 6491110,
    "hit_rate": 0.6786,
    "hit_rate_pct": 67.86,
    "per_worker": {}
  }
}
```

## Router `/metrics` Requirements

The helper expects router `/metrics` to contain the existing router metrics:

```text
vllm_router_cache_aware_decisions_total{decision="..."}
vllm_router_policy_decisions_total{policy="cache_aware",worker="..."}
```

The helper expects worker `/metrics` to contain the existing vLLM prefix-cache
counters:

```text
vllm:prefix_cache_hits_total
vllm:prefix_cache_queries_total
```

If worker discovery from router metrics is not enough, pass explicit workers:

```bash
bash router/benchmarks/router_metrics_summary.sh 127.0.0.1:29400 \
  --workers http://127.0.0.1:28100,http://127.0.0.1:28101
```

## Cold-start Semantics

For the main path, use one fresh router and fresh workers per benchmark case.
Then one post-benchmark command is enough:

```bash
bash router/benchmarks/router_metrics_summary.sh 127.0.0.1:29400
```

The JSON field `window` is `absolute_cold_start` in this mode. Since the router
and workers start from zero for each case, absolute counters are the case
counters. Run the helper before the benchmark cleanup kills the workers, because
hit rate comes from one on-demand worker `/metrics` scrape.

The helper also has an experimental pre/post delta mode:

```bash
bash router/benchmarks/router_metrics_summary.sh \
  --pre logs/my_run/metrics_router_case_pre.prom \
  --post logs/my_run/metrics_router_case_post.prom
```

This is marked `delta_experimental_untested` in JSON. Keep it as a convenience
for old logs; the cold-start snapshot path is the supported benchmark path.

## Decision Labels

### Fallback Mode

`session_id_full_history_fallback` probes the session key first, then falls back
to full chat history only when the session key is weak or stale.

- `session_id_match`: the `session_id` key had a strong match and the router
  chose the matched healthy worker.
- `session_id_fallback`: the `session_id` key was weak or stale, so the router
  tried the full-history key next. This is a path counter and can be counted in
  addition to the final fallback result.
- `full_history_match`: after session fallback, full history had a strong match
  and the router chose that matched worker.
- `full_history_low_match`: after session fallback, full history was still below
  threshold, so the router chose the minimum-load healthy worker.

In Codex-style multi-turn runs, `full_history_match` can be rare because later
turns usually hit `session_id_match`; full-history fallback is mostly evaluated
on cold session starts.

### Single-key Modes

Pure `full_history` and pure `session_id` modes emit the generic labels:

- `cache_affinity`: the selected routing key matched above `cache_threshold`, so
  the router used cache affinity.
- `load_balance`: load imbalance exceeded the configured gates, so the router
  chose a lower-load worker instead of following cache affinity.
- `low_match_min_load`: the selected routing key did not match above threshold,
  so the router chose the minimum-load healthy worker.

`load_balance` and `low_match_min_load` both route away from cache affinity, but
for different reasons: one is load pressure, the other is weak prefix match.
