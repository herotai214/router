#!/usr/bin/env python3
"""Summarize router + worker Prometheus metrics for chat-completions benchmarks.

Dependency-free. Prefer the wrapper:

    bash benchmarks/router_metrics_summary.sh 127.0.0.1:29400 --brief

Supports:
  - N independent workers (scrape each worker URL)
  - One DP backend behind router ``--intra-node-data-parallel-size``
    (router workers look like ``http://host:port@0``; this tool strips ``@rank``,
    dedupes, and scrapes the real backend once — engine labels are summed)

Hit-rate naming (both reported):
  - apc_prefix_cache  = prefix_cache_hits_total / prefix_cache_queries_total
  - prompt_token_cache = prompt_tokens_cached_total / prompt_tokens_total

Latency means from worker histograms (sum/count, all engines):
  queue / prefill / decode / ttft / e2e / inference
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

DECISION_KEYS = [
    "session_id_match",
    "session_id_fallback",
    "full_history_match",
    "full_history_low_match",
    "cache_affinity",
    "load_balance",
    "low_match_min_load",
    "no_tree_random",
    "stale_tenant_fallback",
    "first_healthy_fallback",
]

LATENCY_HISTOGRAMS = {
    "queue_seconds": "vllm:request_queue_time_seconds",
    "prefill_seconds": "vllm:request_prefill_time_seconds",
    "decode_seconds": "vllm:request_decode_time_seconds",
    "ttft_seconds": "vllm:time_to_first_token_seconds",
    "e2e_seconds": "vllm:e2e_request_latency_seconds",
    "inference_seconds": "vllm:request_inference_time_seconds",
}


def normalize_target(target: str) -> str:
    if target.startswith(("http://", "https://")):
        url = target
    else:
        url = f"http://{target}"
    if not url.endswith("/metrics"):
        url = url.rstrip("/") + "/metrics"
    return url


def strip_dp_rank(url: str) -> str:
    """http://host:port@1 -> http://host:port (router DP-aware worker form)."""
    if "@" not in url:
        return url
    # Keep scheme://userinfo@host intact; only strip trailing @dp_rank.
    # Router DP URLs look like http://127.0.0.1:18100@0
    head, _, maybe_rank = url.rpartition("@")
    if maybe_rank.isdigit():
        return head
    return url


def dedupe_scrape_targets(worker_urls: list[str]) -> list[str]:
    seen: list[str] = []
    for url in worker_urls:
        base = strip_dp_rank(url.strip())
        if not base:
            continue
        if base not in seen:
            seen.append(base)
    return seen


def read_prometheus(target: str) -> str:
    path = Path(target)
    if path.exists():
        return path.read_text(errors="ignore")
    url = normalize_target(target)
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read().decode("utf-8", errors="ignore")


def parse_labels(label_text: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    if not label_text:
        return labels
    current = []
    in_quote = False
    parts = []
    for ch in label_text:
        if ch == '"':
            in_quote = not in_quote
        if ch == "," and not in_quote:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        labels[key.strip()] = value.strip().strip('"')
    return labels


def parse_prometheus(body: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    metrics: dict[str, list[tuple[dict[str, str], float]]] = defaultdict(list)
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            series, value_s = line.rsplit(None, 1)
            value = float(value_s)
        except ValueError:
            continue
        if "{" in series and series.endswith("}"):
            name, label_text = series.split("{", 1)
            labels = parse_labels(label_text[:-1])
        else:
            name = series
            labels = {}
        metrics[name].append((labels, value))
    return metrics


def by_label(
    metrics: dict[str, list[tuple[dict[str, str], float]]], name: str, label: str
) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for labels, value in metrics.get(name, []):
        key = labels.get(label)
        if key is not None:
            out[key] += value
    return dict(out)


def subtract_metrics(
    post: dict[str, list[tuple[dict[str, str], float]]],
    pre: dict[str, list[tuple[dict[str, str], float]]],
) -> dict[str, list[tuple[dict[str, str], float]]]:
    keyed: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
    label_maps: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, str]] = {}
    for sign, metrics in ((1.0, post), (-1.0, pre)):
        for name, samples in metrics.items():
            for labels, value in samples:
                key_labels = tuple(sorted(labels.items()))
                key = (name, key_labels)
                keyed[key] += sign * value
                label_maps[key] = labels

    out: dict[str, list[tuple[dict[str, str], float]]] = defaultdict(list)
    for (name, key_labels), value in keyed.items():
        if abs(value) < 1e-12:
            continue
        out[name].append((dict(key_labels), value))
    return dict(out)


def json_number(value: float) -> int | float:
    if math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return int(round(value))
    return value


def metric_total(metrics: dict[str, list[tuple[dict[str, str], float]]], name: str) -> float:
    return sum(value for _labels, value in metrics.get(name, []))


def histogram_mean(
    metrics: dict[str, list[tuple[dict[str, str], float]]], base: str
) -> dict[str, Any] | None:
    total_sum = metric_total(metrics, f"{base}_sum")
    total_count = metric_total(metrics, f"{base}_count")
    if total_count <= 0:
        return None
    return {
        "mean": total_sum / total_count,
        "sum": json_number(total_sum),
        "count": json_number(total_count),
        "metric": base,
    }


def hit_stats(hits: float, queries: float) -> dict[str, Any]:
    rate = hits / queries if queries > 0 else 0.0
    return {
        "hits": json_number(hits),
        "queries": json_number(queries),
        "hit_rate": rate,
        "hit_rate_pct": rate * 100.0,
    }


def prompt_cache_stats(cached: float, total: float) -> dict[str, Any]:
    rate = cached / total if total > 0 else 0.0
    return {
        "cached_tokens": json_number(cached),
        "prompt_tokens_total": json_number(total),
        "hit_rate": rate,
        "hit_rate_pct": rate * 100.0,
    }


def discover_worker_urls(metrics: dict[str, list[tuple[dict[str, str], float]]]) -> list[str]:
    urls = set(by_label(metrics, "vllm_router_policy_decisions_total", "worker"))
    urls.update(by_label(metrics, "vllm_router_processed_requests_total", "worker"))
    return sorted(urls)


def extract_backend_stats(
    metrics: dict[str, list[tuple[dict[str, str], float]]],
) -> dict[str, Any]:
    apc_hits = metric_total(metrics, "vllm:prefix_cache_hits_total")
    apc_queries = metric_total(metrics, "vllm:prefix_cache_queries_total")
    prompt_cached = metric_total(metrics, "vllm:prompt_tokens_cached_total")
    prompt_total = metric_total(metrics, "vllm:prompt_tokens_total")

    latency: dict[str, Any] = {}
    for key, base in LATENCY_HISTOGRAMS.items():
        mean = histogram_mean(metrics, base)
        if mean is not None:
            latency[key] = mean

    engines = sorted(
        {
            labels.get("engine")
            for labels, _ in metrics.get("vllm:request_success_total", [])
            if labels.get("engine") is not None
        }
    )

    return {
        "apc_prefix_cache": {
            "name": "APC hit% (engine prefix-cache block/query reuse)",
            "formula": "vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total",
            **hit_stats(apc_hits, apc_queries),
        },
        "prompt_token_cache": {
            "name": "Prompt hit% (prompt tokens served from cache)",
            "formula": "vllm:prompt_tokens_cached_total / vllm:prompt_tokens_total",
            **prompt_cache_stats(prompt_cached, prompt_total),
        },
        "latency_seconds": latency,
        "engines_seen": engines,
    }


def scrape_worker_metrics(
    worker_urls: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[str]]:
    """Scrape unique backend endpoints (DP @rank URLs deduped)."""
    targets = dedupe_scrape_targets(worker_urls)
    per_endpoint: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for endpoint in targets:
        try:
            metrics = parse_prometheus(read_prometheus(endpoint))
            per_endpoint[endpoint] = extract_backend_stats(metrics)
        except Exception as exc:  # noqa: BLE001 - best-effort summary
            errors[endpoint] = repr(exc)
    return per_endpoint, errors, targets


def aggregate_backend_stats(per_endpoint: dict[str, dict[str, Any]]) -> dict[str, Any]:
    apc_hits = 0.0
    apc_queries = 0.0
    prompt_cached = 0.0
    prompt_total = 0.0
    lat_sums: dict[str, float] = defaultdict(float)
    lat_counts: dict[str, float] = defaultdict(float)

    for stats in per_endpoint.values():
        apc = stats.get("apc_prefix_cache") or {}
        prompt = stats.get("prompt_token_cache") or {}
        apc_hits += float(apc.get("hits") or 0.0)
        apc_queries += float(apc.get("queries") or 0.0)
        prompt_cached += float(prompt.get("cached_tokens") or 0.0)
        prompt_total += float(prompt.get("prompt_tokens_total") or 0.0)
        for key, item in (stats.get("latency_seconds") or {}).items():
            lat_sums[key] += float(item.get("sum") or 0.0)
            lat_counts[key] += float(item.get("count") or 0.0)

    latency: dict[str, Any] = {}
    for key, base in LATENCY_HISTOGRAMS.items():
        count = lat_counts.get(key, 0.0)
        if count <= 0:
            continue
        total_sum = lat_sums[key]
        latency[key] = {
            "mean": total_sum / count,
            "sum": json_number(total_sum),
            "count": json_number(count),
            "metric": base,
        }

    return {
        "apc_prefix_cache": {
            "name": "APC hit% (engine prefix-cache block/query reuse)",
            "formula": "vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total",
            **hit_stats(apc_hits, apc_queries),
        },
        "prompt_token_cache": {
            "name": "Prompt hit% (prompt tokens served from cache)",
            "formula": "vllm:prompt_tokens_cached_total / vllm:prompt_tokens_total",
            **prompt_cache_stats(prompt_cached, prompt_total),
        },
        "latency_seconds": latency,
    }


def build_summary(
    metrics: dict[str, list[tuple[dict[str, str], float]]],
    window: str,
    per_endpoint: dict[str, dict[str, Any]] | None = None,
    worker_scrape_errors: dict[str, str] | None = None,
    scrape_targets: list[str] | None = None,
    discovered_worker_urls: list[str] | None = None,
) -> dict[str, Any]:
    decisions_raw = by_label(
        metrics, "vllm_router_cache_aware_decisions_total", "decision"
    )
    decisions = {key: json_number(decisions_raw.get(key, 0.0)) for key in DECISION_KEYS}
    extra_decisions = {
        key: json_number(value) for key, value in decisions_raw.items() if key not in decisions
    }
    decisions_total = sum(decisions_raw.values())

    workers = by_label(metrics, "vllm_router_policy_decisions_total", "worker")
    worker_total = sum(workers.values())
    workers_json = {worker: json_number(value) for worker, value in workers.items()}

    per_endpoint = per_endpoint or {}
    agg = aggregate_backend_stats(per_endpoint)

    # Backward-compatible alias: old "prefix_cache" == APC only.
    apc = agg["apc_prefix_cache"]
    prefix_cache_compat = {
        "hits": apc.get("hits", 0),
        "queries": apc.get("queries", 0),
        "hit_rate": apc.get("hit_rate", 0.0),
        "hit_rate_pct": apc.get("hit_rate_pct", 0.0),
        "note": "Alias of apc_prefix_cache (kept for older parsers). Prefer apc_prefix_cache / prompt_token_cache.",
    }

    dp_like = any("@" in u for u in (discovered_worker_urls or []))

    return {
        "window": window,
        "scope": "chat_completions_benchmark_rough",
        "topology_hint": (
            "router_cache_aware_over_dp_backend"
            if dp_like
            else "router_cache_aware_over_independent_workers"
        ),
        "cache_aware_decisions": {
            **decisions,
            **extra_decisions,
            "total": json_number(decisions_total),
        },
        "workers_balance": {
            "by_worker": workers_json,
            "total": json_number(worker_total),
            "note": (
                "For DP+router, keys look like http://host:port@rank "
                "(virtual DP ranks). Backend scrape targets are deduped without @rank."
            ),
        },
        "apc_prefix_cache": agg["apc_prefix_cache"],
        "prompt_token_cache": agg["prompt_token_cache"],
        "latency_seconds": agg["latency_seconds"],
        "prefix_cache": prefix_cache_compat,
        "backends": {
            "scrape_targets": scrape_targets or list(per_endpoint.keys()),
            "discovered_worker_urls": discovered_worker_urls or [],
            "per_endpoint": per_endpoint,
            "worker_scrape_errors": worker_scrape_errors or {},
        },
        "evidence_metrics": {
            "router_decisions": "vllm_router_cache_aware_decisions_total",
            "router_workers": "vllm_router_policy_decisions_total",
            "apc_hits": "vllm:prefix_cache_hits_total",
            "apc_queries": "vllm:prefix_cache_queries_total",
            "prompt_cached": "vllm:prompt_tokens_cached_total",
            "prompt_total": "vllm:prompt_tokens_total",
            "latency": {k: v for k, v in LATENCY_HISTOGRAMS.items()},
        },
        "notes": [
            "Designed for rough chat-completions endpoint benchmarks.",
            "Single-snapshot mode assumes router/workers were cold-started for the benchmark.",
            "APC hit% = engine prefix-cache block/query reuse; Prompt hit% = prompt tokens from cache.",
            "Latency means are histogram sum/count across scraped backends (all engine labels).",
            "DP-aware router worker URLs (host:port@rank) are stripped/deduped before scrape.",
            "Delta mode is experimental and lightly tested.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize vLLM Router + worker metrics (APC/prompt hit, latency, decisions)."
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Router metrics target, e.g. 127.0.0.1:29400, http://host:port, or a .prom file",
    )
    parser.add_argument(
        "--workers",
        help=(
            "Optional comma-separated worker URLs or .prom files. "
            "Defaults to workers discovered from router metrics. "
            "DP URLs with @rank are auto-deduped to the backend base URL."
        ),
    )
    parser.add_argument(
        "--no-worker-scrape",
        action="store_true",
        help="Only summarize router metrics; cache/latency will be empty/zero.",
    )
    parser.add_argument("--pre", help="Experimental: pre-benchmark router .prom file")
    parser.add_argument("--post", help="Experimental: post-benchmark router .prom file")
    parser.add_argument(
        "--out",
        help="Optional path to write the JSON summary (stdout still gets JSON unless --brief-only).",
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Also print a one-line human summary to stderr.",
    )
    parser.add_argument(
        "--brief-only",
        action="store_true",
        help="Print only the one-line human summary to stdout (implies --brief).",
    )
    parser.add_argument(
        "--label",
        default="",
        help="Optional label included in --brief output.",
    )
    args = parser.parse_args()

    discovered: list[str] = []
    if args.pre or args.post:
        if not (args.pre and args.post):
            parser.error("--pre and --post must be provided together")
        pre = parse_prometheus(read_prometheus(args.pre))
        post = parse_prometheus(read_prometheus(args.post))
        metrics = subtract_metrics(post, pre)
        summary = build_summary(metrics, "delta_experimental_untested")
    else:
        if not args.target:
            parser.error("target is required unless --pre/--post are provided")
        metrics = parse_prometheus(read_prometheus(args.target))
        discovered = (
            [url.strip() for url in args.workers.split(",") if url.strip()]
            if args.workers
            else discover_worker_urls(metrics)
        )
        if args.no_worker_scrape:
            per_endpoint, worker_errors, targets = {}, {}, []
        else:
            per_endpoint, worker_errors, targets = scrape_worker_metrics(discovered)
        summary = build_summary(
            metrics,
            "absolute_cold_start",
            per_endpoint=per_endpoint,
            worker_scrape_errors=worker_errors,
            scrape_targets=targets,
            discovered_worker_urls=discovered,
        )

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    def brief_line() -> str:
        decisions = summary.get("cache_aware_decisions") or {}
        apc = summary.get("apc_prefix_cache") or {}
        prompt = summary.get("prompt_token_cache") or {}
        latency = summary.get("latency_seconds") or {}
        workers = (summary.get("workers_balance") or {}).get("by_worker") or {}
        label = args.label or "case"
        parts = [
            f"METRICS_SUMMARY label={label}",
            f"apc_hit_rate={float(apc.get('hit_rate_pct') or 0.0):.2f}%",
            f"apc_hits={apc.get('hits', 0)}",
            f"apc_queries={apc.get('queries', 0)}",
            f"prompt_hit_rate={float(prompt.get('hit_rate_pct') or 0.0):.2f}%",
            f"prompt_cached={prompt.get('cached_tokens', 0)}",
            f"prompt_total={prompt.get('prompt_tokens_total', 0)}",
            f"decisions_total={decisions.get('total', 0)}",
        ]
        for lat_key in (
            "queue_seconds",
            "prefill_seconds",
            "decode_seconds",
            "ttft_seconds",
            "e2e_seconds",
            "inference_seconds",
        ):
            item = latency.get(lat_key)
            if item and item.get("mean") is not None:
                short = lat_key.replace("_seconds", "")
                parts.append(f"{short}_mean_s={float(item['mean']):.3f}")
        for key in DECISION_KEYS:
            val = decisions.get(key, 0)
            if val:
                parts.append(f"{key}={val}")
        if workers:
            balance = ",".join(f"{k.split(':')[-1]}={v}" for k, v in sorted(workers.items()))
            parts.append(f"workers={balance}")
        return " ".join(parts)

    if args.brief or args.brief_only:
        line = brief_line()
        if args.brief_only:
            print(line)
        else:
            print(line, file=sys.stderr)

    if not args.brief_only:
        json.dump(summary, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
