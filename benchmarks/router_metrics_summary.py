#!/usr/bin/env python3
"""Summarize router Prometheus metrics for chat-completions benchmarks.

This is intentionally small and dependency-free so it can be used on benchmark
nodes right after a run finishes:

    python3 router/benchmarks/router_metrics_summary.py 127.0.0.1:29400

The primary path is a single post-benchmark snapshot from the router /metrics
endpoint plus one on-demand scrape of each worker URL discovered from router
metrics.  Experimental pre/post delta parsing is included for old benchmark
logs, but the current design is the single-snapshot path.
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


def normalize_target(target: str) -> str:
    if target.startswith(("http://", "https://")):
        url = target
    else:
        url = f"http://{target}"
    if not url.endswith("/metrics"):
        url = url.rstrip("/") + "/metrics"
    return url


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


def sum_unlabeled(metrics: dict[str, list[tuple[dict[str, str], float]]], name: str) -> float:
    return sum(value for _labels, value in metrics.get(name, []))


def first_unlabeled(metrics: dict[str, list[tuple[dict[str, str], float]]], name: str) -> float:
    values = metrics.get(name, [])
    return values[0][1] if values else 0.0


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


def hit_stats(hits: float, queries: float) -> dict[str, float]:
    rate = hits / queries if queries > 0 else 0.0
    return {
        "hits": json_number(hits),
        "queries": json_number(queries),
        "hit_rate": rate,
        "hit_rate_pct": rate * 100.0,
    }


def json_number(value: float) -> int | float:
    if math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return int(round(value))
    return value


def metric_total(metrics: dict[str, list[tuple[dict[str, str], float]]], name: str) -> float:
    return sum(value for _labels, value in metrics.get(name, []))


def discover_worker_urls(metrics: dict[str, list[tuple[dict[str, str], float]]]) -> list[str]:
    urls = set(by_label(metrics, "vllm_router_policy_decisions_total", "worker"))
    urls.update(by_label(metrics, "vllm_router_processed_requests_total", "worker"))
    return sorted(urls)


def scrape_worker_metrics(worker_urls: list[str]) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    per_worker: dict[str, dict[str, float]] = {}
    errors: dict[str, str] = {}
    for worker_url in worker_urls:
        try:
            metrics = parse_prometheus(read_prometheus(worker_url))
            hits = metric_total(metrics, "vllm:prefix_cache_hits_total")
            queries = metric_total(metrics, "vllm:prefix_cache_queries_total")
            per_worker[worker_url] = hit_stats(hits, queries)
        except Exception as exc:  # noqa: BLE001 - benchmark summary should be best-effort
            errors[worker_url] = repr(exc)
    return per_worker, errors


def build_summary(
    metrics: dict[str, list[tuple[dict[str, str], float]]],
    window: str,
    worker_prefix_cache: dict[str, dict[str, float]] | None = None,
    worker_scrape_errors: dict[str, str] | None = None,
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

    per_worker = worker_prefix_cache or {}
    hits = sum(float(item["hits"]) for item in per_worker.values())
    queries = sum(float(item["queries"]) for item in per_worker.values())

    return {
        "window": window,
        "scope": "chat_completions_benchmark_rough",
        "cache_aware_decisions": {
            **decisions,
            **extra_decisions,
            "total": json_number(decisions_total),
        },
        "workers_balance": {
            "by_worker": workers_json,
            "total": json_number(worker_total),
        },
        "prefix_cache": {
            **hit_stats(hits, queries),
            "per_worker": per_worker,
            "worker_scrape_errors": worker_scrape_errors or {},
            "evidence_metrics": {
                "router_decisions": "vllm_router_cache_aware_decisions_total",
                "router_workers": "vllm_router_policy_decisions_total",
                "worker_hits": "vllm:prefix_cache_hits_total",
                "worker_queries": "vllm:prefix_cache_queries_total",
            },
        },
        "notes": [
            "Designed for rough chat-completions endpoint benchmarks.",
            "Single-snapshot mode assumes the router/workers were cold-started for the benchmark.",
            "Delta mode is included for old pre/post logs but is currently unused and lightly tested.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        nargs="?",
        help="Router metrics target, e.g. 127.0.0.1:29400, http://host:port, or a .prom file",
    )
    parser.add_argument(
        "--workers",
        help="Optional comma-separated worker URLs. Defaults to workers discovered from router metrics.",
    )
    parser.add_argument(
        "--no-worker-scrape",
        action="store_true",
        help="Only summarize router metrics; prefix_cache will be zero.",
    )
    parser.add_argument("--pre", help="Experimental: pre-benchmark router .prom file")
    parser.add_argument("--post", help="Experimental: post-benchmark router .prom file")
    args = parser.parse_args()

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
        worker_urls = (
            [url.strip() for url in args.workers.split(",") if url.strip()]
            if args.workers
            else discover_worker_urls(metrics)
        )
        if args.no_worker_scrape:
            per_worker, worker_errors = {}, {}
        else:
            per_worker, worker_errors = scrape_worker_metrics(worker_urls)
        summary = build_summary(
            metrics,
            "absolute_cold_start",
            worker_prefix_cache=per_worker,
            worker_scrape_errors=worker_errors,
        )

    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
