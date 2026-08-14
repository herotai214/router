#!/usr/bin/env python3
"""Synthetic /v1/chat/completions prefix-repetition smoke client.

Constructed dataset (not a real agent trace). Defaults: 100 requests, 16
sessions, 12 unique long prefixes, short changing suffix, plus a stable
``session_params.session_id``. After the first request of each prefix, later
requests that share it **must** show prefix-cache hits if they land on the
same worker. Treat this as a unit-test-like sanity check of cache-aware
routing + ``--enable-prefix-caching``, not as the realistic eval (use
``chat_jsonl_bench.py`` / Codex JSONL for that).

Chat analogue of vLLM's completions ``prefix_repetition`` shape. Router chat
key default is ``session_id_full_history_fallback``; this client still sends
session id so ``session_id`` / fallback can stick.

Assumptions:
  - Workers and (optionally) the router are already up and healthy.
  - Cold start (fresh workers + router) per case so Prometheus counters match.
  - Summarize hit rate while processes are still alive:

      bash benchmarks/router_metrics_summary.sh 127.0.0.1:29400
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def env_or_none(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--base-url",
        default=env_or_none("BASE_URL"),
        help="Router or worker base URL (env BASE_URL). Required.",
    )
    p.add_argument(
        "--model",
        default=env_or_none("MODEL"),
        help="Served model name (env MODEL). Required.",
    )
    p.add_argument("--num-prompts", type=int, default=int(os.environ.get("NUM_PROMPTS", "100")))
    p.add_argument("--prefix-len", type=int, default=int(os.environ.get("PREFIX_LEN", "256")))
    p.add_argument("--suffix-len", type=int, default=int(os.environ.get("SUFFIX_LEN", "16")))
    p.add_argument("--num-prefixes", type=int, default=int(os.environ.get("NUM_PREFIXES", "16")))
    p.add_argument(
        "--session-groups",
        type=int,
        default=int(os.environ.get("SESSION_GROUPS", os.environ.get("NUM_PREFIXES", "16"))),
    )
    p.add_argument(
        "--unique-prefixes",
        type=int,
        default=int(os.environ.get("UNIQUE_PREFIXES", os.environ.get("NUM_PREFIXES", "16"))),
    )
    p.add_argument("--output-len", type=int, default=int(os.environ.get("OUTPUT_LEN", "16")))
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.environ.get("MAX_CONCURRENCY", "4")),
    )
    p.add_argument(
        "--disable-shuffle",
        action="store_true",
        default=os.environ.get("DISABLE_SHUFFLE", "0") == "1",
    )
    p.add_argument(
        "--no-stream",
        action="store_true",
        default=os.environ.get("STREAM", "1") == "0",
        help="Disable streaming (TTFT will be n/a).",
    )
    p.add_argument("--timeout", type=float, default=float(os.environ.get("TIMEOUT", "1800")))
    p.add_argument("--label", default=os.environ.get("LABEL", ""), help="Optional run label.")
    p.add_argument(
        "--metrics-hint-port",
        default=os.environ.get("ROUTER_PROM_PORT", "29400"),
        help="Printed in the post-run metrics hint (default 29400).",
    )
    return p.parse_args(argv)


def make_repeated_text(prefix_id: int, length: int, kind: str) -> str:
    words = [f"{kind}{prefix_id:02d}_{i:03d}" for i in range(length)]
    return " ".join(words)


def build_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.session_groups < 1:
        raise ValueError("--session-groups must be positive")
    if args.unique_prefixes < 1:
        raise ValueError("--unique-prefixes must be positive")
    if args.session_groups < args.unique_prefixes:
        raise ValueError("--session-groups must be >= --unique-prefixes")

    prefixes = [
        make_repeated_text(i, args.prefix_len, "prefix") for i in range(args.unique_prefixes)
    ]
    session_to_prefix = [
        session_idx
        if session_idx < args.unique_prefixes
        else (session_idx - args.unique_prefixes) % args.unique_prefixes
        for session_idx in range(args.session_groups)
    ]
    stream = not args.no_stream
    requests: list[dict[str, Any]] = []
    for i in range(args.num_prompts):
        session_idx = i % args.session_groups
        prefix_id = session_to_prefix[session_idx]
        suffix = make_repeated_text(i, args.suffix_len, "suffix")
        payload: dict[str, Any] = {
            "model": args.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a concise benchmark assistant. Reply briefly.",
                },
                {
                    "role": "developer",
                    "content": "Preserve the provided prefix context when answering.",
                },
                {
                    "role": "user",
                    "content": f"{prefixes[prefix_id]}\n{suffix}\nReturn a short answer.",
                },
            ],
            "session_params": {"session_id": f"session-{session_idx:02d}"},
            "temperature": 0,
            "max_tokens": args.output_len,
            "ignore_eos": True,
        }
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        requests.append(payload)
    if not args.disable_shuffle:
        random.Random(0).shuffle(requests)
    return requests


def parse_streaming_response(resp: Any, start: float) -> dict[str, Any]:
    first_token_time = None
    usage: dict[str, Any] = {}
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="ignore").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue
        if parsed.get("usage"):
            usage = parsed["usage"]
        for choice in parsed.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if first_token_time is None and content:
                first_token_time = time.perf_counter()
    latency = time.perf_counter() - start
    return {
        "ok": True,
        "latency": latency,
        "ttft": (first_token_time - start) if first_token_time else None,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
    }


def post_chat(base_url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if payload.get("stream"):
                return parse_streaming_response(resp, start)
            body = resp.read()
            latency = time.perf_counter() - start
            parsed = json.loads(body)
            usage = parsed.get("usage") or {}
            return {
                "ok": True,
                "latency": latency,
                "ttft": None,
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")[:500]
        return {
            "ok": False,
            "latency": time.perf_counter() - start,
            "error": f"HTTP {exc.code}: {body}",
        }
    except Exception as exc:  # noqa: BLE001 - benchmark must keep going
        return {"ok": False, "latency": time.perf_counter() - start, "error": repr(exc)}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((pct / 100.0) * (len(values) - 1)))))
    return values[idx]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.base_url or not args.model:
        print(
            "ERROR: --base-url and --model are required "
            "(or set env BASE_URL and MODEL).",
            file=sys.stderr,
        )
        return 2

    base_url = args.base_url.rstrip("/")
    try:
        requests = build_requests(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    label = args.label or "chat_prefix"
    print(
        "CHAT_BENCH_CONFIG "
        f"label={label} base={base_url} model={args.model} n={args.num_prompts} "
        f"sessions={args.session_groups} unique_prefixes={args.unique_prefixes} "
        f"prefix={args.prefix_len} suffix={args.suffix_len} output={args.output_len} "
        f"concurrency={args.max_concurrency} shuffle={int(not args.disable_shuffle)} "
        f"stream={int(not args.no_stream)}",
        flush=True,
    )

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
        results = list(
            pool.map(lambda payload: post_chat(base_url, payload, args.timeout), requests)
        )
    duration = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    latencies = [r["latency"] * 1000 for r in ok]
    ttfts = [r["ttft"] * 1000 for r in ok if r.get("ttft") is not None]
    prompt_tokens = sum(r["prompt_tokens"] for r in ok)
    completion_tokens = sum(r["completion_tokens"] for r in ok)

    print("============ Chat Prefix Benchmark Result ============")
    print(f"Label:                                   {label}")
    print(f"Successful requests:                     {len(ok)}")
    print(f"Failed requests:                         {len(failed)}")
    print(f"Maximum request concurrency:             {args.max_concurrency}")
    print(f"Benchmark duration (s):                  {duration:.2f}")
    print(f"Total input tokens:                      {prompt_tokens}")
    print(f"Total generated tokens:                  {completion_tokens}")
    print(f"Request throughput (req/s):              {len(ok) / duration if duration else 0:.2f}")
    print(
        f"Output token throughput (tok/s):         "
        f"{completion_tokens / duration if duration else 0:.2f}"
    )
    if ttfts:
        print(f"Mean TTFT (ms):                          {statistics.mean(ttfts):.2f}")
        print(f"P50 TTFT (ms):                           {percentile(ttfts, 50):.2f}")
        print(f"P90 TTFT (ms):                           {percentile(ttfts, 90):.2f}")
    else:
        print("Mean TTFT (ms):                          n/a")
    print(f"Mean E2EL (ms):                          {statistics.mean(latencies) if latencies else 0:.2f}")
    print(f"P50 E2EL (ms):                           {percentile(latencies, 50):.2f}")
    print(f"P90 E2EL (ms):                           {percentile(latencies, 90):.2f}")
    print("======================================================")
    print(
        "METRICS_HINT: while router/workers are still up, run:\n"
        f"  bash benchmarks/router_metrics_summary.sh 127.0.0.1:{args.metrics_hint_port}"
    )
    if failed:
        print("FAILED_SAMPLE", failed[0])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
