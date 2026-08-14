#!/usr/bin/env python3
"""Build the Codex N×T chat JSONL used by run_codex_dp_cache_aware.sh.

Two sequential tools — they are **not** the same script:

1. convert_codex_swebenchpro_traces.py
   HuggingFace Inferact/codex_swebenchpro_traces → OpenAI chat JSONL pool
   (optionally token-filtered). The filtered pool is large (hundreds of MB).

2. sample_codex_sessions.py
   Subsample that pool to N sessions × T turns (default 25×4,
   --order stratified_size).

This wrapper only orchestrates both. It does not vendor the JSONL.
See dataset/CODEX_SWEBENCHPRO.md.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
CONVERT = BENCH_DIR / "convert_codex_swebenchpro_traces.py"
SAMPLE = BENCH_DIR / "sample_codex_sessions.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert HF Codex traces, then sample N×T sessions. "
            "Not a substitute for either inner script."
        )
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HF tokenizer / model path for filter-only token counts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Final sampled JSONL (e.g. 01_codex_swebenchpro_128k_filter_25s4t_chat.jsonl).",
    )
    parser.add_argument(
        "--pool-output",
        type=Path,
        default=None,
        help="Intermediate converted pool JSONL. Default: <output>.pool.jsonl",
    )
    parser.add_argument("--model", default="qwen35-4b-prefix-datasets")
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=131072,
        help="Filter-only cap for prompt_tokens + max_tokens (128k recipe default).",
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--sessions", type=int, default=25)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument(
        "--order",
        choices=("first", "sorted_sid", "stratified_size"),
        default="stratified_size",
        help=(
            "Passed to sample_codex_sessions.py. Default stratified_size "
            "(even ranks by prompt size). See dataset/CODEX_SWEBENCHPRO.md."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Local HF traces JSON instead of downloading.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        default=True,
        help="Download Inferact/codex_swebenchpro_traces (default on if --input unset).",
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool_output = args.pool_output or Path(str(args.output) + ".pool.jsonl")
    download = args.download and not args.no_download and args.input is None

    convert_cmd = [
        args.python,
        str(CONVERT),
        "--model",
        args.model,
        "--limit-traces",
        "0",
        "--max-calls-per-trace",
        "8",
        "--max-prompt-tokens",
        "0",
        "--max-tokens",
        str(args.max_tokens),
        "--expand-turns",
        "--filter-only",
        "--tokenizer",
        args.tokenizer,
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--output",
        str(pool_output),
    ]
    if args.input is not None:
        convert_cmd.extend(["--input", str(args.input)])
    elif download:
        convert_cmd.append("--download")
    else:
        raise SystemExit("pass --input or allow download of the HF traces")

    sample_cmd = [
        args.python,
        str(SAMPLE),
        "--input",
        str(pool_output),
        "--output",
        str(args.output),
        "--sessions",
        str(args.sessions),
        "--turns",
        str(args.turns),
        "--order",
        args.order,
    ]

    print("+", " ".join(convert_cmd), flush=True)
    subprocess.run(convert_cmd, check=True)
    print("+", " ".join(sample_cmd), flush=True)
    subprocess.run(sample_cmd, check=True)
    print(f"pool={pool_output}")
    print(f"sampled={args.output}")


if __name__ == "__main__":
    main()
