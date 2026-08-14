# Cache-Aware Chat Benchmarks

Index of helpers under `benchmarks/`. Do not duplicate how-tos here.

| Doc | Contents |
|-----|----------|
| [`CACHE_AWARE_OPERATOR_GUIDE.md`](../docs/load_balancing/CACHE_AWARE_OPERATOR_GUIDE.md) | Install, topologies (DP+router first), router flags |
| [`CACHE_AWARE_BENCHMARKS.md`](CACHE_AWARE_BENCHMARKS.md) | Smoke vs Codex, fire mode, demo / Codex scripts |
| [`ROUTER_METRICS.md`](ROUTER_METRICS.md) | Prometheus scrape, hit rates, decision labels |
| [`dataset/CODEX_SWEBENCHPRO.md`](dataset/CODEX_SWEBENCHPRO.md) | Codex JSONL build (convert, then sample) |
| [`docs/load_balancing/README.md`](../docs/load_balancing/README.md) | Policy semantics (`cache_aware` vs others) |

| File | Role |
|------|------|
| `chat_prefix_repetition.py` | Smoke client (constructed prefix-repetition; must show cache hit) |
| `chat_jsonl_bench.py` | Quoted eval client (Codex JSONL). Not `vllm bench serve` |
| `run_dp_cache_aware_demo.sh` | Smoke: DP + cache-aware + `chat_prefix_repetition.py` |
| `run_codex_dp_cache_aware.sh` | Codex JSONL: DP + cache-aware (NPU/CUDA) |
| `dataset/` | Codex JSONL builders (see [`dataset/CODEX_SWEBENCHPRO.md`](dataset/CODEX_SWEBENCHPRO.md)) |
| `router_metrics_summary.sh` | Scrape while stack is up (after bench / before kill; mid-run peek OK) |
| `router_metrics_summary.py` | Implementation behind the `.sh` |
