# Codex SWE-bench Pro JSONL

JSONL is **not** checked in (`*.jsonl` here is gitignored). Point `DATASET=` at
the **sampled** file, not the convert pool.

Needs `huggingface_hub` (download) and a local tokenizer path (same model you
will serve, e.g. Qwen3.5-4B).

---

## Get the 25×4 dataset we actually ran

To reproduce the stable Codex eval (25 sessions × 4 turns = 100 requests,
128k-filter pool, `stratified_size`), just run the one-shot wrapper. Defaults
already match that recipe.

```bash
python3 benchmarks/dataset/prepare_codex_swebenchpro_jsonl.py \
  --tokenizer /path/to/Qwen3.5-4B \
  --output /path/to/01_codex_swebenchpro_128k_filter_25s4t_chat.jsonl
```

That downloads HuggingFace traces, converts a 128k-filter pool, then samples
25×4. It also writes an intermediate `*.pool.jsonl` (large). Use only the
`--output` file with `run_codex_dp_cache_aware.sh`.

On this 25×4 file, every session pair shares the Codex agent preamble (~8% of
turn-0 text, so all pairs are `> 1%`) but **no pair exceeds 10%**, so
`--cache-threshold 0.3` does not treat that as a cross-session history hit.
Later turns of the *same* session still reuse history (~51% LCP vs turn 0).

---

## Source (why this dataset)

[Inferact/codex_swebenchpro_traces](https://huggingface.co/datasets/Inferact/codex_swebenchpro_traces)
is **recorded Codex agent runs on SWE-bench Pro**: multi-turn GitHub-issue
solving across 11 Python repos. Original use was agent eval (pass/fail on the
issue) plus characterizing that serving shape — not router load-balancing.
ShareGPT-style conversations; MIT.

**Source size (HF card, successful trials only):**

| | |
|--|--|
| Traces (sessions) | **610** (of 731 trials; 120 failed, 1 skipped) |
| LLM calls | **20,230** total; **mean 33 / trial** (p50 30, p90 57, p99 90) |
| First-call context | ~12k tokens (p50 12,278) |
| Last-call context | p50 ~80k, p90 ~130k, p99 ~181k |
| Shape | Prefill-heavy (~131:1 input:output; mean ~520 output tokens/call) |

The original Codex serving run was extremely cache-friendly (~94% of input
tokens cached on the HF card). That is the workload shape cache-aware routing
is meant to preserve; we do **not** quote that 94% as our result.

### Trace replay (cold transcript)

We do **not** re-run Codex, tools, or the SWE-bench harness. Convert expands
each trace into `/v1/chat/completions` rows:

- one `session_params.session_id` per trial (`codex-session-NNNN`)
- turn *t* is the **recorded** message prefix up to that LLM call
- the live model’s new tokens are **not** fed into the next prompt

So history is a **frozen transcript**: the same JSONL every run. Prefix cache
and session sticky routing see growing multi-turn prompts independently of
what Qwen actually generates. That is what makes DP vs cache_aware comparable.
`session_serial` then keeps at most one in-flight turn per session (like a
real agent). Convert also caps `--max-calls-per-trace` (default 8 in the 128k
recipe); the quoted eval then samples further.

**Source** (one HF row = one trial; ShareGPT `from` / `value`; truncated):

```json
{
  "conversations": [
    {"from": "human", "value": "<permissions instructions>\nFilesystem sandboxing...\n<issue / repo context>..."},
    {"from": "gpt",    "value": "...first assistant reply / tool call..."},
    {"from": "human", "value": "Command: /bin/bash -lc 'git status --short'\n...tool output..."},
    {"from": "gpt",    "value": "...next assistant reply..."},
    {"from": "human", "value": "Command: /bin/bash -lc \"sed -n '751,840p' ...\"\n..."}
  ]
}
```

**Converted chat body** (`human`→`user`, `gpt`→`assistant`). One JSONL line per
turn; same `session_id`; `messages` grows. Truncated:

```json
{"model": "qwen35-4b-prefix-datasets", "session_params": {"session_id": "codex-session-0000"},
 "_trace_turn": 0, "max_tokens": 32, "temperature": 0, "stream": true,
 "messages": [
   {"role": "user", "content": "<permissions instructions>\nFilesystem sandboxing..."}
 ]}

{"model": "qwen35-4b-prefix-datasets", "session_params": {"session_id": "codex-session-0000"},
 "_trace_turn": 1, "max_tokens": 32, "temperature": 0, "stream": true,
 "messages": [
   {"role": "user",      "content": "<permissions instructions>\nFilesystem sandboxing..."},
   {"role": "assistant", "content": "...first assistant reply / tool call..."},
   {"role": "user",      "content": "Command: /bin/bash -lc 'git status --short'\n..."}
 ]}
```

Turn 0 is the first human message only. Turn 1 is that prefix plus the recorded
assistant reply plus the next human/tool chunk. The client POSTs this body to
`/v1/chat/completions`; it does not append the live completion onto turn 2.

Our quoted cut is **not** the full 33-turn trials: 128k-filter pool, then
**25 sessions × 4 turns** (`stratified_size`) = 100 requests.

### Why not `vllm bench serve`

`vllm bench serve` (ShareGPT / random / sonnet / prefix_repetition) fires
**independent** requests: no stable `session_id`, no growing per-session
history. The **client** for this JSONL is `chat_jsonl_bench.py`, not
`vllm bench` — see
[`CACHE_AWARE_BENCHMARKS.md`](../CACHE_AWARE_BENCHMARKS.md).

---

## Custom cuts (bigger file or different filters)

Only needed if you want a larger N×T, a 256k pool, or a different sample order.
`prepare` is convert then sample; you can also run the two jobs yourself.

```text
HuggingFace Inferact/codex_swebenchpro_traces
        │
        ▼  convert_codex_swebenchpro_traces.py     (pool, hundreds of MB)
OpenAI chat JSONL pool
        │
        ▼  sample_codex_sessions.py                (N sessions × T turns)
eval JSONL
```

`--limit-traces` defaults to `20` in the converter (inspect-sized). Pass `0`
for the full corpus.

### Convert — 128k-filter pool (or 256k)

```bash
python3 benchmarks/dataset/convert_codex_swebenchpro_traces.py \
  --download \
  --model qwen35-4b-prefix-datasets \
  --limit-traces 0 \
  --max-calls-per-trace 8 \
  --max-prompt-tokens 0 \
  --max-tokens 32 \
  --expand-turns \
  --filter-only \
  --tokenizer /path/to/Qwen3.5-4B \
  --max-total-tokens 131072 \
  --output /path/to/01_codex_swebenchpro_128k_filter_chat.jsonl
```

For a 256k-context pool, use `--max-total-tokens 262144`. Do not reuse an old
32k filter.

### Sample — different N×T or order

`--sessions N` / `--turns T` keep the first T turns (`_trace_turn` 0..T−1) of N
sessions. A session is eligible only if it has ≥ T turns. `--order` only
changes **which** N sessions are chosen; turn order inside a session is always
normalized by `_trace_turn`.

**`first`** — walk `session_id`s in **pool file order** (first time each id
appears in the convert JSONL). Take the first N eligible. Fast, but the cut
depends on convert write order and is usually whatever landed at the top of
the file — not a mix of prompt sizes.

**`sorted_sid`** — same walk, after a **lexicographic sort of `session_id`**.
Deterministic even if convert rewrites the pool in a different order. Still a
prefix of that sorted list (this is how the 32×8 cut was built). Not
size-balanced.

**`stratified_size`** (default; 25×4 recipe) — size-balanced subsample, **not**
“first N in the file”:

1. Keep sessions with ≥ T turns.
2. Sort them by `_prompt_tokens` of turn **T−1** (the last turn you will keep;
   for T=4 that is turn 3).
3. Take N **evenly spaced ranks** on that sorted list:
   `round(i * (M − 1) / (N − 1))` for `i = 0 .. N − 1` (`M` = eligible count).
   If N=1, take the smallest. Chosen sessions are emitted small → large.
4. Keep turns 0..T−1 of each chosen session.

Example: 9 eligible sessions, N=5 → ranks `0, 2, 4, 6, 8` (smallest, then
steps toward largest). So 25 sessions from a large pool is a spread across
the turn-T−1 prompt-size distribution, not the 25 smallest or the first 25
ids.

```bash
python3 benchmarks/dataset/sample_codex_sessions.py \
  --input /path/to/01_codex_swebenchpro_128k_filter_chat.jsonl \
  --output /path/to/01_codex_swebenchpro_128k_filter_25s4t_chat.jsonl \
  --sessions 25 \
  --turns 4 \
  --order stratified_size
```

Example of a non-default cut: `--sessions 32 --turns 8 --order sorted_sid`.

`prepare_codex_swebenchpro_jsonl.py` accepts the same knobs (`--sessions`,
`--turns`, `--order`, `--max-total-tokens`) if you still want one command for a
custom cut.
