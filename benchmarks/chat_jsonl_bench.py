#!/usr/bin/env python3
"""Replay OpenAI chat JSONL against /v1/chat/completions.

Prints duration, RPS, TTFT, and E2E. Uses streaming by default so TTFT works.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((pct / 100.0) * (len(values) - 1)))))
    return values[idx]


def parse_streaming_response(resp, start: float) -> dict[str, Any]:
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


def load_requests(
    path: Path,
    model: str | None,
    max_tokens: int | None,
    force_stream: bool,
    limit: int,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if model:
                payload["model"] = model
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            if force_stream:
                payload["stream"] = True
                payload.setdefault("stream_options", {"include_usage": True})
            requests.append(payload)
            if limit > 0 and len(requests) >= limit:
                break
    if not requests:
        raise SystemExit(f"no requests loaded from {path}")
    return requests


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://127.0.0.1:18180"))
    p.add_argument("--model", default=os.environ.get("MODEL"))
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--max-concurrency", type=int, default=int(os.environ.get("MAX_CONCURRENCY", "4")))
    p.add_argument("--limit", type=int, default=int(os.environ.get("NUM_PROMPTS", "0")))
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--label", default="jsonl")
    args = p.parse_args()

    base_url = args.base_url.rstrip("/")
    requests = load_requests(
        args.input,
        model=args.model,
        max_tokens=args.max_tokens,
        force_stream=not args.no_stream,
        limit=args.limit,
    )

    print(
        "CHAT_JSONL_BENCH_CONFIG "
        f"label={args.label} base={base_url} model={args.model or 'from_jsonl'} "
        f"n={len(requests)} concurrency={args.max_concurrency} "
        f"stream={int(not args.no_stream)} "
        f"max_tokens={args.max_tokens if args.max_tokens is not None else 'from_jsonl'} "
        f"input={args.input}",
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

    print("============ Chat JSONL Benchmark Result ============")
    print(f"Label:                                   {args.label}")
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
    if failed:
        print("FAILED_SAMPLE", failed[0])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
