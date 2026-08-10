#!/usr/bin/env bash
# Thin wrapper around router_metrics_summary.py (prefer this from shells/docs).
#
# Examples:
#   bash benchmarks/router_metrics_summary.sh 127.0.0.1:29400 --brief
#   bash benchmarks/router_metrics_summary.sh metrics_router.prom \
#     --workers metrics_w0.prom,metrics_w1.prom --brief-only
#   # DP+router: pass the single backend scrape (or let discovery strip @rank)
#   bash benchmarks/router_metrics_summary.sh metrics_router.prom \
#     --workers metrics_backend_dp.prom --brief-only --label dp2_cache_aware
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${here}/router_metrics_summary.py" "$@"
