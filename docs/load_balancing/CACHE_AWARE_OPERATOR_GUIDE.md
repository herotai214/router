# Cache-Aware vLLM Router — Operator Guide

This guide is for **`cache_aware` routing of `/v1/chat/completions`**.

**How to run the smoke / Codex benches:**
[`CACHE_AWARE_BENCHMARKS.md`](../../benchmarks/CACHE_AWARE_BENCHMARKS.md).
**Metrics scrape / hit-rate names:** [`ROUTER_METRICS.md`](../../benchmarks/ROUTER_METRICS.md).
**JSONL build:** [`CODEX_SWEBENCHPRO.md`](../../benchmarks/dataset/CODEX_SWEBENCHPRO.md).
Policy semantics vs other policies:
[`README.md`](README.md).

**Recommended model for demos:** [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
(or your local checkout). Small enough for 2-device smoke runs, long context for
prefix-cache experiments.

Cache-aware chat routing was exercised stably with:

| Device | Serving stack |
|--------|----------------|
| GPU | vLLM **0.26.0** |
| NPU | vLLM **0.23.0** + vLLM Ascend **v0.19.1rc1** |

---

## 1. Install

Run on a GPU/NPU node (or in your CUDA/Ascend container), not a bare login node
with a mismatched Python.

**Rust is required.** The router is a Rust binary. The optional Python
`vllm-router` launcher is only a thin argparse wrapper around the same PyO3
extension — `pip install -e .` still invokes `cargo` / `rustc` via
`setuptools-rust`. PyPI `vllm-router` does **not** include this branch’s
chat-routing changes. Do not install from a local wheel (`python -m build`,
`dist/*.whl`); this tree is not shipped that way.

```bash
cd /path/to/workspace
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustc --version && cargo --version

# vLLM workers (separate Python env from the router)
uv venv env_vllm && source env_vllm/bin/activate
uv pip install "vllm==0.26.0" --torch-backend=auto   # pin as needed
# NPU: use your vllm + vllm-ascend env instead (0.23 + ascend 0.19.1rc1) and source CANN/ATB.

cd /path/to/router          # this tree / branch
cargo build --release
./target/release/vllm-router --help
# expect --chat-routing-key-mode (default: session-id-full-history-fallback)
# and --intra-node-data-parallel-size
```

Preferred: the **native Rust binary**

```bash
./target/release/vllm-router …
# or while developing:
cargo run --release -- …
```

Optional: Python launcher from **this source tree** (still needs Rust on PATH):

```bash
source env_vllm/bin/activate
pip install -e .          # compiles vllm_router_rs; not a wheel
vllm-router --help
```

Use one or the other. Both must be built from this branch.

---

## 2. Topologies

Cache-aware routing is **not** tied to DP, but **this repo’s demos and Codex
runs almost always use router + intra-node DP** (topology A). Independent
workers remain supported.

| Topology | What you start | Router flags | When to use |
|----------|----------------|--------------|-------------|
| **A. Intra-node DP + router** (usual) | One `vllm serve --data-parallel-size N` | `--worker-urls <one-url> --intra-node-data-parallel-size N` | What we run; closer to “one DP server behind a router”; router sends `X-data-parallel-rank` |
| **B. Independent workers** | N× `vllm serve` (no DP), one URL each | `--worker-urls u0 u1 …` (no intra-node DP) | Clearest per-process KV picture |
| **C. DP baseline (control)** | One `vllm serve --data-parallel-size N` **without** router | — | Same GPU/NPU count, no external routing |

```text
A) Client → router(cache_aware, intra-node-dp=N)
                 → one vLLM API (DP ranks 0..N-1 via X-data-parallel-rank)

B) Client → router(cache_aware) → worker0
                                 → worker1

C) Client → one vLLM API (--data-parallel-size N)   # no router
```

---

## 3. Intra-node DP + cache-aware router (topology A)

Packaged smoke (starts DP backend + router + `chat_prefix_repetition.py`):

```bash
MODEL_PATH=/path/to/Qwen3.5-4B \
DEVICE_ENV_NAME=CUDA_VISIBLE_DEVICES DEVICES=0,1 \
ROUTER_BIN=./target/release/vllm-router \
bash benchmarks/run_dp_cache_aware_demo.sh
```

NPU: `DEVICE_ENV_NAME=ASCEND_RT_VISIBLE_DEVICES` and source CANN/ATB first.
Codex JSONL on the same topology: `benchmarks/run_codex_dp_cache_aware.sh`
(see the bench doc).

Manual equivalent:

```bash
export MODEL_PATH=/path/to/Qwen3.5-4B
export SERVED=qwen35-4b-dp

CUDA_VISIBLE_DEVICES=0,1 vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port 18100 \
  --served-model-name "$SERVED" \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --enable-prefix-caching --trust-remote-code \
  > vllm_dp.log 2>&1 &

curl -fsS http://127.0.0.1:18100/health

./target/release/vllm-router \
  --host 0.0.0.0 --port 18180 \
  --worker-urls http://127.0.0.1:18100 \
  --policy cache_aware \
  --prometheus-port 29400 \
  --cache-threshold 0.3 \
  --balance-abs-threshold 2 \
  --balance-rel-threshold 1.5 \
  --chat-routing-key-mode session-id-full-history-fallback \
  --intra-node-data-parallel-size 2 \
  > router.log 2>&1 &

curl -fsS http://127.0.0.1:18180/health
```

Then run a client from [`CACHE_AWARE_BENCHMARKS.md`](../../benchmarks/CACHE_AWARE_BENCHMARKS.md).

<details>
<summary>Independent workers (topology B) — expand if you are not using DP</summary>

Each process owns its KV. No `--intra-node-data-parallel-size`.

```bash
export MODEL_PATH=/path/to/Qwen3.5-4B
export SERVED=qwen35-4b

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port 18100 \
  --served-model-name "$SERVED" \
  --tensor-parallel-size 1 \
  --enable-prefix-caching --trust-remote-code \
  > vllm_w0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 --port 18101 \
  --served-model-name "$SERVED" \
  --tensor-parallel-size 1 \
  --enable-prefix-caching --trust-remote-code \
  > vllm_w1.log 2>&1 &

curl -fsS http://127.0.0.1:18100/health && curl -fsS http://127.0.0.1:18101/health

./target/release/vllm-router \
  --host 0.0.0.0 --port 18180 \
  --worker-urls http://127.0.0.1:18100 http://127.0.0.1:18101 \
  --policy cache_aware \
  --prometheus-port 29400 \
  --cache-threshold 0.3 \
  --balance-abs-threshold 2 \
  --balance-rel-threshold 1.5 \
  --chat-routing-key-mode session-id-full-history-fallback \
  > router.log 2>&1 &

curl -fsS http://127.0.0.1:18180/health
```

NPU: use `ASCEND_RT_VISIBLE_DEVICES` and source CANN/ATB first.

</details>

---

## 4. Flags and defaults

**Default `--chat-routing-key-mode` is `session-id-full-history-fallback`**
(Rust binary, Python `vllm-router`, and `RouterArgs`). Session id first, then
full chat history. Override with `full-history` or `session-id`. CLI spelling
uses hyphens; the Python launcher also accepts underscores.

That fallback is the point of the **smoke** demo
(`chat_prefix_repetition.py` / `run_dp_cache_aware_demo.sh`): 16 sessions share
only 12 long prefixes. Different `session_id`s → session probe fails → full
history still matches → prefix cache hit. Codex JSONL is the opposite shape
(mostly sticky per session; see the bench / dataset notes).

Binary / Python **software** defaults for the load gates are
`cache_threshold=0.3`, `balance_abs_threshold=64`, `balance_rel_threshold=1.5`.
Demo / Codex snippets use the **`lb_mid` preset** `0.3 / 2 / 1.5`. The stickier
`abs=2, rel=1.5` pair is what we kept after Codex runs with
`run_codex_dp_cache_aware.sh` at `MAX_TOKENS=32` and `256`.

| Flag | Meaning |
|------|---------|
| `--cache-threshold` | Prefix-tree match rate must be **`>` this value** (strict) to treat as a strong hit |
| `--balance-abs-threshold` | Absolute load gap that allows breaking affinity |
| `--balance-rel-threshold` | Relative load gap that allows breaking affinity |
| `--chat-routing-key-mode` | Chat key. Default: `session-id-full-history-fallback` |
| `--intra-node-data-parallel-size` | Expand one backend URL into DP-rank virtual workers (topology A) |

Presets:

```text
lb_mid   = cache=0.3,   abs=2, rel=1.5   # recommended vs plain DP (Codex mt32/mt256)
lb_aggr  = cache=0.3,   abs=0, rel=1.0   # more even workers, lower hit rate
sid999   = cache=0.999, abs=2, rel=1.5   # near-exact session_id (see below)
```

Enable `--enable-prefix-caching` on every backend.

### Why `sid999` is 0.999, not 1.0

The policy uses **`match_rate > cache_threshold`**, not `>=`. Match rate is in
`[0, 1]`, so `--cache-threshold 1` can **never** count as a cache hit (nothing
is strictly greater than 1). Changing `>` to `>=` would be a wider behavior
change for every cache-aware mode, including `full_history` / fallback at 0.3.
Until that is an intentional API change, near-exact session stickiness uses
`0.999`.

`session_id` alone is close to **`consistent_hash`**: sticky by session key.
We still implement it inside `cache_aware` so one policy can do **session then
full-history fallback** (session miss → history probe) without switching
policies mid-request.

---

## 5. Minimal checklist

1. Install Rust + vLLM (+ Ascend if NPU). `cargo build --release` (or
   `pip install -e .` from this tree — still compiles Rust). Not PyPI, not a wheel.
2. Prefer **Qwen3.5-4B** for a small long-context demo.
3. Usual topology: **A** (DP + `--intra-node-data-parallel-size`), via
   `run_dp_cache_aware_demo.sh` or the commands in §3.
4. Enable `--enable-prefix-caching` on every backend. Chat key default is
   fallback; use `lb_mid` (`0.3 / 2 / 1.5`) for demos.
5. Measure: [`CACHE_AWARE_BENCHMARKS.md`](../../benchmarks/CACHE_AWARE_BENCHMARKS.md) (smoke must
   show hit rate; Codex JSONL is the realistic eval). Scrape after the client
   finishes, before kill:
   [`ROUTER_METRICS.md`](../../benchmarks/ROUTER_METRICS.md).
