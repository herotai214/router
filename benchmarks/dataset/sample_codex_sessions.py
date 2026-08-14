#!/usr/bin/env python3
"""Sample N sessions × T turns from an already-converted Codex chat JSONL.

This is step 2 of the Codex bench pipeline. It does **not** download HuggingFace
traces or convert them. Step 1 is `convert_codex_swebenchpro_traces.py`.
See `dataset/CODEX_SWEBENCHPRO.md`.

Input rows must be OpenAI chat-completions payloads with:
  session_params.session_id
  _trace_turn
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--sessions", type=int, default=25)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument(
        "--order",
        choices=("first", "sorted_sid", "stratified_size"),
        default="stratified_size",
        help=(
            "Which N eligible sessions to keep (see dataset/CODEX_SWEBENCHPRO.md). "
            "stratified_size (default): even ranks by turn-(T-1) prompt size "
            "(25×4 recipe). first: pool file order. sorted_sid: lexicographic "
            "session_id (32×8 cut)."
        ),
    )
    args = ap.parse_args()

    by_sid: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = (row.get("session_params") or {}).get("session_id")
            if not sid:
                continue
            if sid not in by_sid:
                order.append(sid)
            by_sid[sid].append(row)

    # Normalize turn order inside each session.
    for sid in by_sid:
        by_sid[sid].sort(key=lambda r: int(r.get("_trace_turn") or 0))

    eligible_sids: list[str] = []
    if args.order == "stratified_size":
        # Same recipe used to build 01_codex_swebenchpro_128k_filter_25s4t_chat.jsonl:
        # sessions with >=T turns, sort by prompt tokens of turn (T-1), then take
        # evenly spaced indices across that size distribution.
        eligible_items = [
            (sid, rows)
            for sid, rows in by_sid.items()
            if len(rows) >= args.turns
        ]
        eligible_items.sort(
            key=lambda item: int(item[1][args.turns - 1].get("_prompt_tokens") or 0)
        )
        if len(eligible_items) < args.sessions:
            raise SystemExit(
                f"only {len(eligible_items)} sessions have >= {args.turns} turns; "
                f"need {args.sessions}"
            )
        if args.sessions == 1:
            idxs = [0]
        else:
            idxs = [
                round(i * (len(eligible_items) - 1) / (args.sessions - 1))
                for i in range(args.sessions)
            ]
        # Preserve selection order (small → large by turn-T size).
        eligible_sids = [eligible_items[i][0] for i in idxs]
    else:
        sid_iter = order if args.order == "first" else sorted(by_sid)
        for sid in sid_iter:
            if len(by_sid[sid]) >= args.turns:
                eligible_sids.append(sid)
            if len(eligible_sids) >= args.sessions:
                break
        if len(eligible_sids) < args.sessions:
            raise SystemExit(
                f"only {len(eligible_sids)} sessions have >= {args.turns} turns; "
                f"need {args.sessions}"
            )

    out_rows: list[dict] = []
    for sid in eligible_sids:
        out_rows.extend(by_sid[sid][: args.turns])
    eligible = eligible_sids

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    toks = [
        int(r.get("_prompt_tokens") or r.get("_approx_prompt_tokens") or 0)
        for r in out_rows
    ]
    per_sess = Counter((r.get("session_params") or {}).get("session_id") for r in out_rows)
    print(f"wrote {args.output}")
    print(f"  requests={len(out_rows)} sessions={len(per_sess)}")
    print(f"  turns={dict(Counter(per_sess.values()))}")
    print(
        f"  prompt_tokens min={min(toks)} med={int(statistics.median(toks))} "
        f"mean={int(statistics.mean(toks))} max={max(toks)}"
    )
    print(f"  first_sessions={eligible[:5]}")


if __name__ == "__main__":
    main()
