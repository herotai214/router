#!/usr/bin/env python3
"""Replay OpenAI chat JSONL against /v1/chat/completions.

This is the **quoted** cache-aware performance client (Codex SWE-bench Pro
JSONL). Official ``vllm bench serve`` cannot do this: it fires independent
ShareGPT/random/completions requests with no ``session_params.session_id``,
no growing per-session history, and no session-serial fire mode. See
``CACHE_AWARE_BENCHMARKS.md``.

Prints duration, RPS, TTFT, TPOT, and E2E. Uses streaming by default so TTFT works.

Default fire mode is ``session_serial``: global concurrency is honored, but at
most one turn per ``session_params.session_id`` is in flight, and turns within a
session run in ``_trace_turn`` order. That avoids launching all turns of one
session together (common when the JSONL is session-blocked and concurrency>=turns).

Per-request traces (``--per-request-jsonl``) record:

* Client: HTTP start / first-token / finish, client TTFT/TPOT/E2E
* Server (needs ``vllm serve --enable-per-request-metrics``): response
  ``metrics`` with ``queue_time_ms``, ``time_to_first_token_ms`` (scheduled→first
  token ≈ prefill), ``mean_itl_ms`` (TPOT), ``generation_time_ms`` — same
  internals as the Prometheus histogram means, but one sample per request.

Absolute server timestamps are not in the API; we approximate
``queued/scheduled`` unix times by anchoring on the client's first-token time.
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
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((pct / 100.0) * (len(values) - 1)))))
    return values[idx]


def mean_or_zero(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


ROUTER_TRACE_HEADERS = {
    "routed_worker": "x-vllm-router-worker",
    "routed_base_worker": "x-vllm-router-base-worker",
    "routed_dp_rank": "x-vllm-router-dp-rank",
    "router_decision": "x-vllm-router-decision",
}


def extract_router_trace_headers(resp: Any) -> dict[str, str | int | None]:
    values: dict[str, str | int | None] = {}
    for field, header in ROUTER_TRACE_HEADERS.items():
        value = resp.headers.get(header)
        if field == "routed_dp_rank" and value is not None:
            try:
                values[field] = int(value)
            except ValueError:
                values[field] = value
        else:
            values[field] = value
    return values


def cached_tokens_from_usage(usage: dict[str, Any]) -> tuple[int | None, bool]:
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict) or "cached_tokens" not in details:
        return None, False
    value = details.get("cached_tokens")
    if value is None:
        return None, True
    return int(value), True


def parse_streaming_response(resp, start_perf: float) -> dict[str, Any]:
    first_token_perf = None
    first_token_unix = None
    usage: dict[str, Any] = {}
    server_metrics: dict[str, Any] | None = None
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
        # Final usage chunk carries metrics when --enable-per-request-metrics.
        if parsed.get("metrics"):
            server_metrics = parsed["metrics"]
        for choice in parsed.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if first_token_perf is None and content:
                first_token_perf = time.perf_counter()
                first_token_unix = time.time()
    finish_perf = time.perf_counter()
    finish_unix = time.time()
    e2e_s = finish_perf - start_perf
    ttft_s = (first_token_perf - start_perf) if first_token_perf is not None else None
    cached_tokens, cached_tokens_present = cached_tokens_from_usage(usage)
    return {
        "ok": True,
        "e2e_s": e2e_s,
        "ttft_s": ttft_s,
        "first_token_unix": first_token_unix,
        "finish_unix": finish_unix,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cached_tokens": cached_tokens,
        "cached_tokens_present": cached_tokens_present,
        "server_metrics": server_metrics,
    }


def post_chat(base_url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    http_start_perf = time.perf_counter()
    http_start_unix = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            router_trace = extract_router_trace_headers(resp)
            if payload.get("stream"):
                out = parse_streaming_response(resp, http_start_perf)
            else:
                body = resp.read()
                finish_perf = time.perf_counter()
                finish_unix = time.time()
                parsed = json.loads(body)
                usage = parsed.get("usage") or {}
                cached_tokens, cached_tokens_present = cached_tokens_from_usage(usage)
                out = {
                    "ok": True,
                    "e2e_s": finish_perf - http_start_perf,
                    "ttft_s": None,
                    "first_token_unix": None,
                    "finish_unix": finish_unix,
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                    "cached_tokens": cached_tokens,
                    "cached_tokens_present": cached_tokens_present,
                    "server_metrics": parsed.get("metrics"),
                }
            out.update(router_trace)
            out["http_start_unix"] = http_start_unix
            out["http_start_perf"] = http_start_perf
            return out
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")[:500]
        finish_unix = time.time()
        return {
            "ok": False,
            "e2e_s": time.perf_counter() - http_start_perf,
            "ttft_s": None,
            "http_start_unix": http_start_unix,
            "http_start_perf": http_start_perf,
            "first_token_unix": None,
            "finish_unix": finish_unix,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": None,
            "cached_tokens_present": False,
            "server_metrics": None,
            "error": f"HTTP {exc.code}: {body}",
        }
    except Exception as exc:  # noqa: BLE001 - benchmark must keep going
        finish_unix = time.time()
        return {
            "ok": False,
            "e2e_s": time.perf_counter() - http_start_perf,
            "ttft_s": None,
            "http_start_unix": http_start_unix,
            "http_start_perf": http_start_perf,
            "first_token_unix": None,
            "finish_unix": finish_unix,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": None,
            "cached_tokens_present": False,
            "server_metrics": None,
            "error": repr(exc),
        }


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def enrich_record(meta: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    """Build one per-request trace row (client + optional server metrics)."""
    e2e_s = float(raw.get("e2e_s") or 0.0)
    ttft_s = raw.get("ttft_s")
    completion = int(raw.get("completion_tokens") or 0)
    decode_s = None
    tpot_s = None
    if ttft_s is not None:
        decode_s = max(0.0, e2e_s - float(ttft_s))
        # TPOT over tokens after the first; if only 1 token, TPOT = decode_s.
        denom = max(completion - 1, 1) if completion > 0 else 0
        if denom > 0:
            tpot_s = decode_s / denom

    http_start_unix = raw.get("http_start_unix")
    first_token_unix = raw.get("first_token_unix")
    finish_unix = raw.get("finish_unix")
    eligible_unix = meta.get("eligible_unix")

    sm = raw.get("server_metrics") or {}
    server_queue_ms = _as_float(sm.get("queue_time_ms"))
    # vLLM: scheduled_ts -> first_token_ts (same as request_prefill_time_seconds)
    server_prefill_ms = _as_float(sm.get("time_to_first_token_ms"))
    server_generation_ms = _as_float(sm.get("generation_time_ms"))
    server_mean_itl_ms = _as_float(sm.get("mean_itl_ms"))  # ≈ TPOT
    server_tps = _as_float(sm.get("tokens_per_second"))

    # Approximate absolute server timestamps by walking back from client first-token.
    # API only returns durations, not engine wall clocks.
    approx_scheduled_unix = None
    approx_queued_unix = None
    approx_prefill_start_unix = None
    if first_token_unix is not None and server_prefill_ms is not None:
        approx_scheduled_unix = float(first_token_unix) - server_prefill_ms / 1000.0
        approx_prefill_start_unix = approx_scheduled_unix
        if server_queue_ms is not None:
            approx_queued_unix = approx_scheduled_unix - server_queue_ms / 1000.0

    prompt_tokens = int(raw.get("prompt_tokens") or 0)
    cached_tokens = raw.get("cached_tokens")
    prompt_cache_hit_pct = (
        (float(cached_tokens) / float(prompt_tokens) * 100.0)
        if cached_tokens is not None and prompt_tokens > 0
        else None
    )

    return {
        "req_index": meta["req_index"],
        "session_id": meta["session_id"],
        "trace_turn": meta.get("trace_turn"),
        "ok": bool(raw.get("ok")),
        "error": raw.get("error"),
        # Client wall-clock (unix epoch)
        "eligible_unix": eligible_unix,
        "submit_unix": meta.get("submit_unix"),
        "http_start_unix": http_start_unix,
        "first_token_unix": first_token_unix,
        "finish_unix": finish_unix,
        "eligible_rel_s": meta.get("eligible_rel_s"),
        "submit_rel_s": meta.get("submit_rel_s"),
        "http_start_rel_s": (
            (float(raw["http_start_perf"]) - float(meta["bench_t0_perf"]))
            if raw.get("http_start_perf") is not None
            else None
        ),
        # Client-observed latency
        "client_ttft_ms": None if ttft_s is None else float(ttft_s) * 1000.0,
        "client_decode_ms": None if decode_s is None else decode_s * 1000.0,
        "client_tpot_ms": None if tpot_s is None else tpot_s * 1000.0,
        "client_e2e_ms": e2e_s * 1000.0,
        # Compat aliases used by aggregate printers
        "ttft_ms": None if ttft_s is None else float(ttft_s) * 1000.0,
        "tpot_ms": None if tpot_s is None else tpot_s * 1000.0,
        "e2e_ms": e2e_s * 1000.0,
        "ttft_s": ttft_s,
        "tpot_s": tpot_s,
        "e2e_s": e2e_s,
        "decode_s": decode_s,
        "decode_ms": None if decode_s is None else decode_s * 1000.0,
        # Server per-request metrics (vllm --enable-per-request-metrics)
        # Same source as Prometheus queue/prefill/ITL histogram observations.
        "server_queue_ms": server_queue_ms,
        "server_prefill_ms": server_prefill_ms,
        "server_generation_ms": server_generation_ms,
        "server_mean_itl_ms": server_mean_itl_ms,
        "server_tokens_per_second": server_tps,
        "server_metrics_raw": sm or None,
        # Approx absolute times (derived; not engine clocks)
        "approx_queued_unix": approx_queued_unix,
        "approx_scheduled_unix": approx_scheduled_unix,
        "approx_prefill_start_unix": approx_prefill_start_unix,
        "approx_timestamp_note": (
            "approx_*_unix derived from client first_token_unix - server durations; "
            "enable with: vllm serve --enable-per-request-metrics"
        ),
        # Router trace headers (requires router support; absent for direct DP baseline).
        "routed_worker": raw.get("routed_worker"),
        "routed_base_worker": raw.get("routed_base_worker"),
        "routed_dp_rank": raw.get("routed_dp_rank"),
        "router_decision": raw.get("router_decision"),
        # Tokens
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "cached_tokens": cached_tokens,
        "cached_tokens_present": bool(raw.get("cached_tokens_present")),
        "prompt_cache_hit_pct": prompt_cache_hit_pct,
    }


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


def session_id_of(payload: dict[str, Any], fallback: int) -> str:
    sid = (payload.get("session_params") or {}).get("session_id")
    if sid is None or sid == "":
        return f"__anon_{fallback}"
    return str(sid)


def run_one(
    base_url: str,
    payload: dict[str, Any],
    timeout: float,
    meta: dict[str, Any],
) -> dict[str, Any]:
    meta = dict(meta)
    meta["submit_unix"] = time.time()
    meta["submit_rel_s"] = time.perf_counter() - float(meta["bench_t0_perf"])
    raw = post_chat(base_url, payload, timeout)
    return enrich_record(meta, raw)


def run_jsonl_order(
    base_url: str,
    requests: list[dict[str, Any]],
    max_concurrency: int,
    timeout: float,
    bench_t0_perf: float,
) -> list[dict[str, Any]]:
    # File-order map: mark eligible at submit time (when a worker picks it up).
    def _task(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        idx, payload = item
        now_unix = time.time()
        now_perf = time.perf_counter()
        meta = {
            "req_index": idx,
            "session_id": session_id_of(payload, idx),
            "trace_turn": payload.get("_trace_turn"),
            "bench_t0_perf": bench_t0_perf,
            "eligible_unix": now_unix,
            "eligible_rel_s": now_perf - bench_t0_perf,
        }
        return run_one(base_url, payload, timeout, meta)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        return list(pool.map(_task, enumerate(requests)))


def run_session_serial(
    base_url: str,
    requests: list[dict[str, Any]],
    max_concurrency: int,
    timeout: float,
    bench_t0_perf: float,
) -> list[dict[str, Any]]:
    """Global concurrency with <=1 in-flight turn per session_id."""
    by_sid: dict[str, deque[tuple[int, dict[str, Any]]]] = defaultdict(deque)
    session_order: list[str] = []
    for idx, payload in enumerate(requests):
        sid = session_id_of(payload, idx)
        if sid not in by_sid:
            session_order.append(sid)
        by_sid[sid].append((idx, payload))

    for sid in by_sid:
        by_sid[sid] = deque(
            sorted(
                by_sid[sid],
                key=lambda item: (
                    int(item[1].get("_trace_turn") or 0),
                    item[0],
                ),
            )
        )

    t0_unix = time.time()
    # First turn of every session is eligible at bench start.
    eligible_unix: dict[str, float] = {sid: t0_unix for sid in session_order}
    eligible_rel: dict[str, float] = {sid: 0.0 for sid in session_order}

    ready: deque[str] = deque(sid for sid in session_order if by_sid[sid])
    results: list[dict[str, Any] | None] = [None] * len(requests)
    inflight_sid: dict[concurrent.futures.Future, str] = {}
    inflight_idx: dict[concurrent.futures.Future, int] = {}
    workers = max(1, max_concurrency)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:

        def submit_sid(sid: str) -> None:
            idx, payload = by_sid[sid].popleft()
            meta = {
                "req_index": idx,
                "session_id": sid,
                "trace_turn": payload.get("_trace_turn"),
                "bench_t0_perf": bench_t0_perf,
                "eligible_unix": eligible_unix[sid],
                "eligible_rel_s": eligible_rel[sid],
            }
            fut = pool.submit(run_one, base_url, payload, timeout, meta)
            inflight_sid[fut] = sid
            inflight_idx[fut] = idx

        while ready and len(inflight_sid) < workers:
            submit_sid(ready.popleft())

        while inflight_sid:
            done, _ = concurrent.futures.wait(
                inflight_sid,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                sid = inflight_sid.pop(fut)
                idx = inflight_idx.pop(fut)
                row = fut.result()
                results[idx] = row
                # Next turn of this session becomes eligible when this turn finishes.
                fin = row.get("finish_unix")
                if fin is None:
                    fin = time.time()
                eligible_unix[sid] = float(fin)
                eligible_rel[sid] = time.perf_counter() - bench_t0_perf
                if by_sid[sid]:
                    ready.append(sid)
            while ready and len(inflight_sid) < workers:
                submit_sid(ready.popleft())

    assert all(r is not None for r in results)
    return [r for r in results if r is not None]


def write_per_request_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_trace_lines(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(
            "REQ_TRACE "
            f"idx={row['req_index']} sid={row['session_id']} turn={row.get('trace_turn')} "
            f"ok={int(row['ok'])} "
            f"http_start_unix={row.get('http_start_unix')} "
            f"approx_queued_unix={row.get('approx_queued_unix')} "
            f"approx_prefill_start_unix={row.get('approx_prefill_start_unix')} "
            f"first_token_unix={row.get('first_token_unix')} "
            f"finish_unix={row.get('finish_unix')} "
            f"worker={row.get('routed_worker')} "
            f"dp_rank={row.get('routed_dp_rank')} "
            f"decision={row.get('router_decision')} "
            f"server_queue_ms={row.get('server_queue_ms')} "
            f"server_prefill_ms={row.get('server_prefill_ms')} "
            f"server_mean_itl_ms={row.get('server_mean_itl_ms')} "
            f"client_ttft_ms={row.get('client_ttft_ms')} "
            f"client_tpot_ms={row.get('client_tpot_ms')} "
            f"client_e2e_ms={row.get('client_e2e_ms')} "
            f"prompt_tok={row.get('prompt_tokens')} "
            f"completion_tok={row.get('completion_tokens')} "
            f"cached_tok={row.get('cached_tokens')} "
            f"prompt_cache_hit_pct={row.get('prompt_cache_hit_pct')}",
            flush=True,
        )


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
    p.add_argument(
        "--fire-mode",
        choices=("session_serial", "jsonl"),
        default=os.environ.get("CHAT_JSONL_FIRE_MODE", "session_serial"),
        help=(
            "session_serial (default): <=1 in-flight turn per session_id, "
            "turns in _trace_turn order, global concurrency still applies. "
            "jsonl: legacy file-order pool.map."
        ),
    )
    p.add_argument(
        "--per-request-jsonl",
        type=Path,
        default=(
            Path(os.environ["PER_REQUEST_JSONL"])
            if os.environ.get("PER_REQUEST_JSONL")
            else None
        ),
        help="Write one JSON object per request with arrival/TTFT/TPOT/E2E fields.",
    )
    p.add_argument(
        "--trace-requests",
        action="store_true",
        default=os.environ.get("CHAT_JSONL_TRACE_REQUESTS", "").lower() in ("1", "true", "yes"),
        help="Also print REQ_TRACE lines for each request to stdout.",
    )
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
        f"fire_mode={args.fire_mode} "
        f"stream={int(not args.no_stream)} "
        f"max_tokens={args.max_tokens if args.max_tokens is not None else 'from_jsonl'} "
        f"per_request_jsonl={args.per_request_jsonl or ''} "
        f"input={args.input}",
        flush=True,
    )

    bench_t0_perf = time.perf_counter()
    if args.fire_mode == "session_serial":
        results = run_session_serial(
            base_url, requests, args.max_concurrency, args.timeout, bench_t0_perf
        )
    else:
        results = run_jsonl_order(
            base_url, requests, args.max_concurrency, args.timeout, bench_t0_perf
        )
    duration = time.perf_counter() - bench_t0_perf

    if args.trace_requests:
        print_trace_lines(results)
    if args.per_request_jsonl is not None:
        write_per_request_jsonl(args.per_request_jsonl, results)
        print(f"PER_REQUEST_JSONL={args.per_request_jsonl}", flush=True)

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    latencies = [float(r["e2e_ms"]) for r in ok]
    ttfts = [float(r["ttft_ms"]) for r in ok if r.get("ttft_ms") is not None]
    tpots = [float(r["tpot_ms"]) for r in ok if r.get("tpot_ms") is not None]
    server_queues = [
        float(r["server_queue_ms"]) for r in ok if r.get("server_queue_ms") is not None
    ]
    server_prefills = [
        float(r["server_prefill_ms"]) for r in ok if r.get("server_prefill_ms") is not None
    ]
    server_itls = [
        float(r["server_mean_itl_ms"]) for r in ok if r.get("server_mean_itl_ms") is not None
    ]
    prompt_tokens = sum(int(r["prompt_tokens"]) for r in ok)
    completion_tokens = sum(int(r["completion_tokens"]) for r in ok)

    print("============ Chat JSONL Benchmark Result ============")
    print(f"Label:                                   {args.label}")
    print(f"Successful requests:                     {len(ok)}")
    print(f"Failed requests:                         {len(failed)}")
    print(f"Maximum request concurrency:             {args.max_concurrency}")
    print(f"Fire mode:                               {args.fire_mode}")
    print(f"Benchmark duration (s):                  {duration:.2f}")
    print(f"Total input tokens:                      {prompt_tokens}")
    print(f"Total generated tokens:                  {completion_tokens}")
    print(f"Request throughput (req/s):              {len(ok) / duration if duration else 0:.2f}")
    print(
        f"Output token throughput (tok/s):         "
        f"{completion_tokens / duration if duration else 0:.2f}"
    )
    if server_queues:
        print(f"Mean server_queue (ms):                  {mean_or_zero(server_queues):.2f}")
        print(f"P50 server_queue (ms):                   {percentile(server_queues, 50):.2f}")
        print(f"P90 server_queue (ms):                   {percentile(server_queues, 90):.2f}")
    else:
        print("Mean server_queue (ms):                  n/a (need --enable-per-request-metrics)")
    if server_prefills:
        print(f"Mean server_prefill (ms):                {mean_or_zero(server_prefills):.2f}")
        print(f"P50 server_prefill (ms):                 {percentile(server_prefills, 50):.2f}")
        print(f"P90 server_prefill (ms):                 {percentile(server_prefills, 90):.2f}")
    if server_itls:
        print(f"Mean server_ITL/TPOT (ms):               {mean_or_zero(server_itls):.2f}")
        print(f"P50 server_ITL/TPOT (ms):                {percentile(server_itls, 50):.2f}")
        print(f"P90 server_ITL/TPOT (ms):                {percentile(server_itls, 90):.2f}")
    if ttfts:
        print(f"Mean client TTFT (ms):                   {mean_or_zero(ttfts):.2f}")
        print(f"P50 client TTFT (ms):                    {percentile(ttfts, 50):.2f}")
        print(f"P90 client TTFT (ms):                    {percentile(ttfts, 90):.2f}")
    else:
        print("Mean client TTFT (ms):                   n/a")
    if tpots:
        print(f"Mean client TPOT (ms):                   {mean_or_zero(tpots):.2f}")
        print(f"P50 client TPOT (ms):                    {percentile(tpots, 50):.2f}")
        print(f"P90 client TPOT (ms):                    {percentile(tpots, 90):.2f}")
    else:
        print("Mean client TPOT (ms):                   n/a")
    print(f"Mean client E2EL (ms):                   {mean_or_zero(latencies):.2f}")
    print(f"P50 client E2EL (ms):                    {percentile(latencies, 50):.2f}")
    print(f"P90 client E2EL (ms):                    {percentile(latencies, 90):.2f}")
    print("======================================================")
    if failed:
        print("FAILED_SAMPLE", failed[0])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
