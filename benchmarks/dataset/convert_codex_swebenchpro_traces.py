#!/usr/bin/env python3
"""Convert Codex/SWE-bench Pro traces into OpenAI chat request JSONL.

This is step 1 of the Codex bench pipeline: HuggingFace traces → a (possibly
token-filtered) `/v1/chat/completions` JSONL **pool**. That pool is large and
is not the 25×4 eval file. Step 2 is `sample_codex_sessions.py`. See
`dataset/CODEX_SWEBENCHPRO.md`.

Target dataset:
  https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces

The dataset has changed format at least once, so this converter is deliberately
defensive: it looks for common message containers and emits best-effort
`/v1/chat/completions` payloads. Use `--inspect-only` first to print the
top-level schema before converting a full file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REPO_ID = "Inferact/codex_swebenchpro_traces"
DEFAULT_FILENAME = "codex_swebenchpro.json"


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((pct / 100.0) * (len(values) - 1))))
    return values[idx]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def download_from_hf(repo_id: str, filename: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise SystemExit(
            "huggingface_hub is required for download. Install it or pass --input."
        ) from exc

    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))


def iter_records(data: Any) -> Iterable[dict[str, Any]]:
    data = maybe_json(data)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(data, dict):
        for key in ("data", "rows", "traces", "examples"):
            value = maybe_json(data.get(key))
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        yield data


def normalize_role(role: str) -> str:
    role = role.lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "agent"}:
        return "assistant"
    if role in {"system", "developer", "tool", "function"}:
        return role
    return "user"


def normalize_message(message: Any) -> dict[str, Any] | None:
    message = maybe_json(message)
    if isinstance(message, str):
        return {"role": "user", "content": message}
    if not isinstance(message, dict):
        return None

    role = (
        message.get("role")
        or message.get("from")
        or message.get("source")
        or message.get("speaker")
        or "user"
    )
    content = (
        message.get("content")
        or message.get("value")
        or message.get("message")
        or message.get("text")
        or ""
    )

    normalized = {"role": normalize_role(str(role)), "content": content}
    if normalized["role"] == "tool" and "tool_call_id" in message:
        normalized["tool_call_id"] = message["tool_call_id"]
    if "name" in message and normalized["role"] in {"system", "user", "assistant"}:
        normalized["name"] = message["name"]
    return normalized


def find_message_lists(record: dict[str, Any]) -> list[list[dict[str, Any]]]:
    candidates = []
    for key in (
        "messages",
        "conversation",
        "conversations",
        "chat",
        "trajectory",
        "trace",
        "turns",
        "steps",
    ):
        value = maybe_json(record.get(key))
        if isinstance(value, list):
            normalized = [normalize_message(item) for item in value]
            normalized = [item for item in normalized if item and item.get("content")]
            if normalized:
                candidates.append(normalized)
    return candidates


def normalize_tools(record: dict[str, Any]) -> list[dict[str, Any]]:
    raw_tools = maybe_json(
        record.get("tools")
        or record.get("tool_definitions")
        or record.get("functions")
        or []
    )
    if not isinstance(raw_tools, list):
        return []

    tools = []
    for tool in raw_tools:
        tool = maybe_json(tool)
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            tools.append(tool)
            continue
        name = tool.get("name") or tool.get("function_name")
        if not name:
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object"}),
                },
            }
        )
    return tools


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += max(1, len(content.split()))
        else:
            total += 8
    return total


class PromptTokenizer:
    def __init__(self, tokenizer_path: str):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise SystemExit(
                "transformers is required for --tokenizer. Run inside the router venv."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

    def token_ids(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> list[int]:
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        if tools:
            kwargs["tools"] = tools
        try:
            rendered = self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            # Some tokenizers do not accept tool schemas.
            kwargs.pop("tools", None)
            rendered = self.tokenizer.apply_chat_template(messages, **kwargs)

        if hasattr(rendered, "keys") and "input_ids" in rendered:
            rendered = rendered.get("input_ids") or []
        if hasattr(rendered, "tolist"):
            rendered = rendered.tolist()
        if rendered and isinstance(rendered[0], list):
            rendered = rendered[0]
        return [int(x) for x in rendered]

    def count(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> int:
        return len(self.token_ids(messages, tools))

    def prefix_digest(
        self,
        messages: list[dict[str, Any]],
        prefix_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> str | None:
        ids = self.token_ids(messages, tools)
        if len(ids) < prefix_tokens:
            return None
        prefix = ",".join(str(x) for x in ids[:prefix_tokens])
        return hashlib.sha1(prefix.encode("utf-8")).hexdigest()


def word_prefix_digest(messages: list[dict[str, Any]], prefix_words: int) -> str | None:
    words: list[str] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            words.extend(content.split())
        else:
            words.append(str(content))
        if len(words) >= prefix_words:
            break
    if len(words) < prefix_words:
        return None
    return hashlib.sha1(" ".join(words[:prefix_words]).encode("utf-8")).hexdigest()


def _clip_text(content: str, max_words: int) -> str:
    if max_words <= 0:
        return ""
    words = content.split()
    if len(words) <= max_words:
        return content
    return " ".join(words[:max_words])


def truncate_messages(
    messages: list[dict[str, Any]], max_prompt_tokens: int
) -> list[dict[str, Any]]:
    """Keep a leading shared prefix, then as many later turns as fit.

    Codex/ShareGPT traces often put a huge system+skills blob in the first
    human turn. If we naively take only that blob, multi-turn windows collapse
    to one message under small max-model-len caps. Reserve headroom for later
    turns so expanding windows still exercise prefix reuse.
    """
    if max_prompt_tokens <= 0 or not messages:
        return messages

    kept: list[dict[str, Any]] = []
    used = 0

    first = dict(messages[0])
    first_content = first.get("content")
    if isinstance(first_content, str):
        # Leave ~40% budget for later turns when history has more than one message.
        first_budget = (
            max_prompt_tokens
            if len(messages) == 1
            else max(64, int(max_prompt_tokens * 0.6))
        )
        first["content"] = _clip_text(first_content, first_budget)
        used = max(1, len(first["content"].split()))
    else:
        used = 8
    kept.append(first)

    for message in messages[1:]:
        msg = dict(message)
        content = msg.get("content")
        if isinstance(content, str):
            remain = max_prompt_tokens - used
            if remain <= 0:
                break
            clipped = _clip_text(content, remain)
            if not clipped:
                break
            msg["content"] = clipped
            cost = max(1, len(clipped.split()))
        else:
            cost = 8
            if used + cost > max_prompt_tokens:
                break
        kept.append(msg)
        used += cost
    return kept


def expand_turn_windows(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Emit growing histories ending at each user/tool turn (multi-round prefix reuse)."""
    windows: list[list[dict[str, Any]]] = []
    for idx, message in enumerate(messages):
        role = message.get("role")
        if role in {"user", "tool"} and idx + 1 < len(messages):
            # Prefer windows that end just before the next assistant reply.
            if messages[idx + 1].get("role") == "assistant" or role == "user":
                windows.append(messages[: idx + 1])
        elif role == "user" and idx == len(messages) - 1:
            windows.append(messages[: idx + 1])
    if not windows and messages:
        windows.append(messages)
    # Deduplicate identical lengths.
    uniq: list[list[dict[str, Any]]] = []
    seen = set()
    for window in windows:
        key = len(window)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(window)
    return uniq


def convert_record(
    record: dict[str, Any],
    model: str,
    max_tokens: int,
    max_calls_per_trace: int,
    expand_turns: bool,
    max_prompt_tokens: int,
    session_id: str,
    tokenizer: PromptTokenizer | None,
    filter_only: bool,
    min_prompt_tokens: int,
    max_total_tokens: int,
    min_turn_index: int,
    max_turn_index: int,
) -> list[dict[str, Any]]:
    message_lists = find_message_lists(record)
    if not message_lists:
        return []

    tools = normalize_tools(record)
    outputs = []
    for messages in message_lists:
        windows = expand_turn_windows(messages) if expand_turns else [messages]
        for turn_idx, window in enumerate(windows):
            if turn_idx < min_turn_index:
                continue
            if max_turn_index >= 0 and turn_idx > max_turn_index:
                continue

            clipped = [dict(message) for message in window]
            if not filter_only:
                clipped = truncate_messages(clipped, max_prompt_tokens)

            prompt_tokens = (
                tokenizer.count(clipped, tools) if tokenizer else estimate_tokens(clipped)
            )
            total_tokens = prompt_tokens + max_tokens
            if filter_only:
                if min_prompt_tokens > 0 and prompt_tokens < min_prompt_tokens:
                    continue
                if max_total_tokens > 0 and total_tokens > max_total_tokens:
                    continue

            payload = {
                "model": model,
                "messages": clipped,
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": True,
                "stream_options": {"include_usage": True},
                # Expanding turns share one session_id for sticky routing.
                "session_params": {"session_id": session_id},
                "_approx_prompt_tokens": estimate_tokens(clipped),
                "_prompt_tokens": prompt_tokens,
                "_total_tokens": total_tokens,
                "_trace_turn": turn_idx,
            }
            if tools:
                payload["tools"] = tools
            outputs.append(payload)
            if len(outputs) >= max_calls_per_trace:
                return outputs
    return outputs


def inspect_schema(records: list[dict[str, Any]]) -> None:
    print(f"records: {len(records)}")
    for idx, record in enumerate(records[:3]):
        print(f"record {idx} keys: {sorted(record.keys())}")
        for key, value in record.items():
            value = maybe_json(value)
            if isinstance(value, list):
                print(f"  {key}: list len={len(value)}")
            elif isinstance(value, dict):
                print(f"  {key}: dict keys={list(value)[:10]}")
            else:
                print(f"  {key}: {type(value).__name__}")


def exact_prompt_key(payload: dict[str, Any]) -> str:
    body = {
        "messages": payload.get("messages", []),
        "tools": payload.get("tools", []),
    }
    return json.dumps(body, sort_keys=True, ensure_ascii=False)


def audit_payloads(
    payloads: list[dict[str, Any]],
    tokenizer: PromptTokenizer | None,
    prefix_sizes: list[int],
) -> None:
    print("== Codex chat JSONL audit ==")
    print(f"requests={len(payloads)}")
    if not payloads:
        return

    by_session: dict[str, int] = defaultdict(int)
    for payload in payloads:
        session_id = (
            (payload.get("session_params") or {}).get("session_id") or "<missing>"
        )
        by_session[str(session_id)] += 1
    group_sizes = sorted(by_session.values(), reverse=True)
    print(
        "sessions="
        f"{len(by_session)} group_size_min/p50/p90/max="
        f"{min(group_sizes)}/{percentile(group_sizes, 50)}/"
        f"{percentile(group_sizes, 90)}/{max(group_sizes)} "
        f"top10={group_sizes[:10]}"
    )

    prompt_tokens = [
        int(payload.get("_prompt_tokens") or payload.get("_approx_prompt_tokens") or 0)
        for payload in payloads
    ]
    total_tokens = [int(payload.get("_total_tokens") or 0) for payload in payloads]
    print(
        "prompt_tokens_min/p50/p90/max="
        f"{min(prompt_tokens)}/{percentile(prompt_tokens, 50)}/"
        f"{percentile(prompt_tokens, 90)}/{max(prompt_tokens)}"
    )
    if any(total_tokens):
        print(
            "total_tokens_min/p50/p90/max="
            f"{min(total_tokens)}/{percentile(total_tokens, 50)}/"
            f"{percentile(total_tokens, 90)}/{max(total_tokens)}"
        )

    exact_counts = Counter(exact_prompt_key(payload) for payload in payloads)
    exact_duplicate_reqs = sum(count for count in exact_counts.values() if count > 1)
    print(
        f"exact_duplicate_reqs={exact_duplicate_reqs} "
        f"max_exact_group={max(exact_counts.values())}"
    )

    for size in prefix_sizes:
        digests = []
        for payload in payloads:
            messages = payload.get("messages") or []
            tools = payload.get("tools") or []
            if tokenizer:
                digest = tokenizer.prefix_digest(messages, size, tools)
            else:
                digest = word_prefix_digest(messages, size)
            if digest:
                digests.append(digest)
        counts = Counter(digests)
        repeated = sum(count for count in counts.values() if count > 1)
        max_group = max(counts.values()) if counts else 0
        print(
            f"prefix_{size}_{'tokens' if tokenizer else 'words'}: "
            f"eligible={len(digests)} repeated_reqs={repeated} "
            f"repeated_pct={(repeated / len(payloads) * 100.0):.1f}% "
            f"groups_gt1={sum(1 for count in counts.values() if count > 1)} "
            f"max_group={max_group}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/codex_swebenchpro_chat.jsonl"))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--download", action="store_true", help="Force HF download even if --input unset")
    parser.add_argument("--model", default="qwen3-32b-chat-routing")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--limit-traces", type=int, default=20)
    parser.add_argument("--max-calls-per-trace", type=int, default=8)
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=1536,
        help="Approx word cap so prompts fit local max-model-len; 0 disables",
    )
    parser.add_argument(
        "--filter-only",
        action="store_true",
        help="Do not truncate. Keep only natural windows that pass token filters.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HF tokenizer/model path for true chat-template token counts.",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=0,
        help="Filter-only cap for prompt_tokens + max_tokens. 0 disables.",
    )
    parser.add_argument(
        "--min-prompt-tokens",
        type=int,
        default=0,
        help="Filter-only lower bound for natural prompt length. 0 disables.",
    )
    parser.add_argument(
        "--min-turn-index",
        type=int,
        default=0,
        help="0-based expanded window index to start from; 1 skips the first call.",
    )
    parser.add_argument(
        "--max-turn-index",
        type=int,
        default=-1,
        help="0-based expanded window index to end at; -1 disables.",
    )
    parser.add_argument(
        "--drop-exact-duplicates",
        action="store_true",
        help="Drop exact duplicate message/tools payloads. Default keeps and audits them.",
    )
    parser.add_argument(
        "--audit-prefix-sizes",
        default="1024,4096,8192,16384",
        help="Comma-separated token prefix sizes for audit; word sizes if no tokenizer.",
    )
    parser.add_argument("--no-audit", action="store_true")
    parser.add_argument(
        "--expand-turns",
        action="store_true",
        default=True,
        help="Emit growing multi-turn windows (default on)",
    )
    parser.add_argument("--no-expand-turns", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expand_turns = args.expand_turns and not args.no_expand_turns
    tokenizer = PromptTokenizer(args.tokenizer) if args.tokenizer else None
    if args.input is None or args.download:
        input_path = download_from_hf(args.repo_id, args.filename)
    else:
        input_path = args.input
    data = load_json(input_path)
    records = list(iter_records(data))
    if args.limit_traces > 0:
        records = records[: args.limit_traces]

    if args.inspect_only:
        inspect_schema(records)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, Any]] = []
    exact_seen: set[str] = set()
    with args.output.open("w", encoding="utf-8") as f:
        for trace_idx, record in enumerate(records):
            session_id = f"codex-session-{trace_idx:04d}"
            for payload in convert_record(
                record,
                model=args.model,
                max_tokens=args.max_tokens,
                max_calls_per_trace=args.max_calls_per_trace,
                expand_turns=expand_turns,
                max_prompt_tokens=args.max_prompt_tokens,
                session_id=session_id,
                tokenizer=tokenizer,
                filter_only=args.filter_only,
                min_prompt_tokens=args.min_prompt_tokens,
                max_total_tokens=args.max_total_tokens,
                min_turn_index=args.min_turn_index,
                max_turn_index=args.max_turn_index,
            ):
                if args.drop_exact_duplicates:
                    key = exact_prompt_key(payload)
                    if key in exact_seen:
                        continue
                    exact_seen.add(key)
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                payloads.append(payload)

    print(
        f"wrote {len(payloads)} chat requests to {args.output} "
        f"(traces={len(records)} expand_turns={int(expand_turns)} "
        f"max_prompt_tokens={args.max_prompt_tokens} "
        f"filter_only={int(args.filter_only)} "
        f"max_total_tokens={args.max_total_tokens})"
    )
    if not args.no_audit:
        prefix_sizes = [
            int(value)
            for value in args.audit_prefix_sizes.split(",")
            if value.strip()
        ]
        audit_payloads(payloads, tokenizer, prefix_sizes)


if __name__ == "__main__":
    main()
