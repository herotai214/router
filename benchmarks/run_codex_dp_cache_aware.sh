#!/usr/bin/env bash
# Codex JSONL bench: one vLLM DP backend + cache-aware router (intra-node DP).
#
# Topology (cache-aware does NOT require DP; this script demos the DP+router path):
#   client -> vllm-router (cache_aware, --intra-node-data-parallel-size=N)
#          -> one `vllm serve --data-parallel-size N`
#             (router sends X-data-parallel-rank)
#
# NPU-friendly defaults (override for CUDA):
#   DEVICE_ENV_NAME=ASCEND_RT_VISIBLE_DEVICES
# Source CANN/ATB in the shell first so `import torch_npu` works.
#
# Examples (from router repo root after cargo build --release).
# Dataset build: benchmarks/dataset/CODEX_SWEBENCHPRO.md
# Python `vllm-router` also works if you `pip install -e .` from this tree
# (still needs rustc/cargo; not PyPI / not a wheel).
#
# Chat key default is session-id-full-history-fallback. Codex client default
# fire mode is session_serial.
#
#   # NPU: DP baseline + one fallback lb_mid case
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   source /usr/local/Ascend/nnal/atb/set_env.sh   # if present
#   MODEL_PATH=/path/to/Qwen3.5-4B \
#   DATASET=/path/to/01_codex_swebenchpro_128k_filter_25s4t_chat.jsonl \
#   DEVICE_ENV_NAME=ASCEND_RT_VISIBLE_DEVICES DEVICES=0,1 \
#   ROUTER_BIN=./target/release/vllm-router \
#   RUN_DP_BASELINE=1 RUN_CACHE_AWARE=1 \
#   CONFIGS=lb_mid:0.3:2:1.5 \
#   MAX_TOKENS=256 \
#   bash benchmarks/run_codex_dp_cache_aware.sh
#
#   # Four router knobs (sequential cold starts), no DP baseline:
#   RUN_DP_BASELINE=0 RUN_CACHE_AWARE=1 \
#   CONFIGS="lb_rel15:0.3:2:1.5 lb_rel20:0.3:2:2.0 lb_rel25:0.3:2:2.5 lb_c02_rel20:0.2:2:2.0" \
#   ... bash benchmarks/run_codex_dp_cache_aware.sh
#
#   # CUDA:
#   DEVICE_ENV_NAME=CUDA_VISIBLE_DEVICES DEVICES=0,1 ...
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${BENCH_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-}"
SERVED_MODEL="${SERVED_MODEL:-qwen-codex-dp-cache-aware}"

# NPU default; set CUDA_VISIBLE_DEVICES for GPU boxes.
DEVICE_ENV_NAME="${DEVICE_ENV_NAME:-ASCEND_RT_VISIBLE_DEVICES}"
DEVICES="${DEVICES:-0,1}"
DP_SIZE="${DP_SIZE:-2}"

DATASET="${DATASET:-}"
NUM_PROMPTS="${NUM_PROMPTS:-100}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
# Empty = keep each JSONL row's max_tokens (dataset default is often 32).
MAX_TOKENS="${MAX_TOKENS:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
# Per-request server queue/prefill/ITL in chat response `metrics` (vLLM >= ~0.26).
# Set ENABLE_PER_REQUEST_METRICS=0 on older stacks (e.g. Ascend v0.23) that reject the flag.
ENABLE_PER_REQUEST_METRICS="${ENABLE_PER_REQUEST_METRICS:-1}"
# Per-request reused prefix tokens in response `usage.prompt_tokens_details.cached_tokens`.
# Set ENABLE_PROMPT_TOKENS_DETAILS=0 if an older backend rejects the flag.
ENABLE_PROMPT_TOKENS_DETAILS="${ENABLE_PROMPT_TOKENS_DETAILS:-1}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---max-num-seqs 4}"
if [ "${ENABLE_PER_REQUEST_METRICS}" = "1" ]; then
  case " ${VLLM_EXTRA_ARGS} " in
    *" --enable-per-request-metrics "*) ;;
    *) VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS} --enable-per-request-metrics" ;;
  esac
fi
if [ "${ENABLE_PROMPT_TOKENS_DETAILS}" = "1" ]; then
  case " ${VLLM_EXTRA_ARGS} " in
    *" --enable-prompt-tokens-details "*) ;;
    *) VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS} --enable-prompt-tokens-details" ;;
  esac
fi

VLLM_BIN="${VLLM_BIN:-vllm}"
ROUTER_BIN="${ROUTER_BIN:-${ROOT_DIR}/target/release/vllm-router}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

BACKEND_PORT="${BACKEND_PORT:-18100}"
DP_BASELINE_PORT="${DP_BASELINE_PORT:-18000}"
ROUTER_PORT="${ROUTER_PORT:-18180}"
ROUTER_PROM_PORT="${ROUTER_PROM_PORT:-29400}"

RUN_DP_BASELINE="${RUN_DP_BASELINE:-1}"
RUN_CACHE_AWARE="${RUN_CACHE_AWARE:-1}"
# label:cache_threshold:balance_abs:balance_rel
CONFIGS="${CONFIGS:-lb_mid:0.3:2:1.5}"
CHAT_ROUTING_KEY_MODE="${CHAT_ROUTING_KEY_MODE:-session-id-full-history-fallback}"
CHAT_JSONL_FIRE_MODE="${CHAT_JSONL_FIRE_MODE:-session_serial}"

TS="$(date +%Y%m%d_%H%M%S)"
MT_TAG="${MAX_TOKENS:-jsonl}"
LOG_DIR="${LOG_DIR:-${PWD}/logs_codex_dp${DP_SIZE}_cache_aware_mt${MT_TAG}_rp${ROUTER_PORT}_${TS}}"
mkdir -p "${LOG_DIR}/metrics"

PIDS=()

log() {
  echo "[$(date -Is)] $*" | tee -a "${LOG_DIR}/driver.log"
}

require() {
  if [ -x "$1" ]; then
    return 0
  fi
  if command -v "$1" >/dev/null 2>&1; then
    return 0
  fi
  echo "ERROR: missing command/binary: $1" >&2
  exit 1
}

kill_pid_tree() {
  local pid="$1"
  if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
    pkill -P "${pid}" 2>/dev/null || true
    kill "${pid}" 2>/dev/null || true
  fi
}

cleanup_case() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill_pid_tree "${pid}"
  done
  PIDS=()
  sleep 5
}

cleanup_all() {
  cleanup_case || true
}
trap cleanup_all EXIT

wait_health() {
  local name="$1"
  local base_url="$2"
  local timeout_s="${3:-900}"
  local start
  start="$(date +%s)"
  log "WAIT_HEALTH ${name} ${base_url}"
  while true; do
    if curl -fsS "${base_url}/health" >/dev/null 2>&1; then
      log "HEALTHY ${name}"
      return 0
    fi
    if [ $(( $(date +%s) - start )) -ge "${timeout_s}" ]; then
      echo "ERROR: ${name} not healthy after ${timeout_s}s" >&2
      return 1
    fi
    sleep 5
  done
}

curl_metrics() {
  local name="$1"
  local url="$2"
  local out="$3"
  mkdir -p "$(dirname "${out}")"
  log "SCRAPE_METRICS ${name} ${url}/metrics -> ${out}"
  if curl -fsS "${url}/metrics" >"${out}" 2>"${out}.err"; then
    : >"${out}.err"
  else
    log "WARN scrape failed name=${name} url=${url}; see ${out}.err"
  fi
}

start_vllm_dp() {
  local label="$1"
  local port="$2"
  local served="$3"
  local log_file="${LOG_DIR}/vllm_${label}_port${port}.log"
  log "START_DP label=${label} devices=${DEVICES} dp_size=${DP_SIZE} port=${port} served=${served}"
  env "${DEVICE_ENV_NAME}=${DEVICES}" \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --host 0.0.0.0 \
      --port "${port}" \
      --served-model-name "${served}" \
      --tensor-parallel-size 1 \
      --data-parallel-size "${DP_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --enable-prefix-caching \
      --trust-remote-code \
      ${VLLM_EXTRA_ARGS} \
      >"${log_file}" 2>&1 &
  PIDS+=("$!")
  wait_health "${label}" "http://127.0.0.1:${port}"
}

start_router_dp_aware() {
  local label="$1"
  local cache="$2"
  local abs="$3"
  local rel="$4"
  local log_file="${LOG_DIR}/router_${label}.log"
  if [ ! -x "${ROUTER_BIN}" ] && ! command -v "${ROUTER_BIN}" >/dev/null 2>&1; then
    echo "ERROR: ROUTER_BIN not found/executable: ${ROUTER_BIN}" >&2
    echo "Build with: (cd ${ROOT_DIR} && cargo build --release)" >&2
    exit 1
  fi
  log "START_ROUTER label=${label} mode=${CHAT_ROUTING_KEY_MODE} cache=${cache} abs=${abs} rel=${rel} intra_dp=${DP_SIZE}"
  "${ROUTER_BIN}" \
    --host 0.0.0.0 \
    --port "${ROUTER_PORT}" \
    --worker-urls "http://127.0.0.1:${BACKEND_PORT}" \
    --policy cache_aware \
    --prometheus-port "${ROUTER_PROM_PORT}" \
    --cache-threshold "${cache}" \
    --balance-abs-threshold "${abs}" \
    --balance-rel-threshold "${rel}" \
    --chat-routing-key-mode "${CHAT_ROUTING_KEY_MODE}" \
    --intra-node-data-parallel-size "${DP_SIZE}" \
    >"${log_file}" 2>&1 &
  PIDS+=("$!")
  wait_health "router_${label}" "http://127.0.0.1:${ROUTER_PORT}" 300
}

run_bench() {
  local label="$1"
  local base_url="$2"
  local model="$3"
  local bench_log="${LOG_DIR}/bench_${label}.log"
  local max_tokens_args=()
  if [ -n "${MAX_TOKENS}" ]; then
    max_tokens_args+=(--max-tokens "${MAX_TOKENS}")
  fi
  local per_req_jsonl="${LOG_DIR}/per_request_${label}.jsonl"
  log "BENCH_START label=${label} MAX_TOKENS=${MAX_TOKENS:-from_jsonl} base=${base_url}"
  "${PYTHON_BIN}" "${BENCH_DIR}/chat_jsonl_bench.py" \
    --input "${DATASET}" \
    --base-url "${base_url}" \
    --model "${model}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --limit "${NUM_PROMPTS}" \
    --label "${label}" \
    --fire-mode "${CHAT_JSONL_FIRE_MODE}" \
    --per-request-jsonl "${per_req_jsonl}" \
    "${max_tokens_args[@]}" \
    2>&1 | tee "${bench_log}"
  log "BENCH_DONE label=${label} per_request_jsonl=${per_req_jsonl}"
}

summarize_dp_only() {
  local label="$1"
  local backend_prom="$2"
  local per_req_jsonl="${LOG_DIR}/per_request_${label}.jsonl"
  local out_json="${LOG_DIR}/summary_${label}.json"
  # No router decisions; still get APC / Prompt / latency via --workers and empty router scrape.
  # Pass backend as both target (ignored decisions) and workers.
  bash "${BENCH_DIR}/router_metrics_summary.sh" "${backend_prom}" \
    --workers "${backend_prom}" \
    --per-request-jsonl "${per_req_jsonl}" \
    --out "${out_json}" \
    --brief-only \
    --label "${label}" \
    | tee -a "${LOG_DIR}/driver.log" || true
  log "SUMMARY_JSON=${out_json}"
}

summarize_router_case() {
  local label="$1"
  local router_prom="$2"
  local backend_prom="$3"
  local per_req_jsonl="${LOG_DIR}/per_request_${label}.jsonl"
  local out_json="${LOG_DIR}/summary_${label}.json"
  bash "${BENCH_DIR}/router_metrics_summary.sh" "${router_prom}" \
    --workers "${backend_prom}" \
    --per-request-jsonl "${per_req_jsonl}" \
    --out "${out_json}" \
    --brief-only \
    --label "${label}" \
    | tee -a "${LOG_DIR}/driver.log"
  log "SUMMARY_JSON=${out_json}"
}

run_dp_baseline_case() {
  local label="dp_baseline"
  local served="${SERVED_MODEL}-${label}"
  cleanup_case
  log "CASE_START ${label}"
  start_vllm_dp "${label}" "${DP_BASELINE_PORT}" "${served}"
  run_bench "${label}" "http://127.0.0.1:${DP_BASELINE_PORT}" "${served}"
  local backend_prom="${LOG_DIR}/metrics/${label}_backend.prom"
  curl_metrics "${label}" "http://127.0.0.1:${DP_BASELINE_PORT}" "${backend_prom}"
  summarize_dp_only "${label}" "${backend_prom}"
  log "CASE_DONE ${label}"
  cleanup_case
}

run_cache_aware_case() {
  local cfg="$1"
  local name cache abs rel
  IFS=: read -r name cache abs rel <<<"${cfg}"
  local label="cache_aware_${name}"
  local served="${SERVED_MODEL}-${label}"
  cleanup_case
  log "CASE_START ${label} cfg=${cfg}"
  start_vllm_dp "${label}_backend" "${BACKEND_PORT}" "${served}"
  start_router_dp_aware "${label}" "${cache}" "${abs}" "${rel}"
  run_bench "${label}" "http://127.0.0.1:${ROUTER_PORT}" "${served}"
  local router_prom="${LOG_DIR}/metrics/${label}_router.prom"
  local backend_prom="${LOG_DIR}/metrics/${label}_backend.prom"
  curl_metrics "${label}_router" "http://127.0.0.1:${ROUTER_PROM_PORT}" "${router_prom}"
  curl_metrics "${label}_backend" "http://127.0.0.1:${BACKEND_PORT}" "${backend_prom}"
  summarize_router_case "${label}" "${router_prom}" "${backend_prom}"
  log "CASE_DONE ${label}"
  cleanup_case
}

preflight_npu_hint() {
  if [ "${DEVICE_ENV_NAME}" = "ASCEND_RT_VISIBLE_DEVICES" ]; then
    log "NPU mode: DEVICE_ENV_NAME=${DEVICE_ENV_NAME} DEVICES=${DEVICES}"
    if ! "${PYTHON_BIN}" -c "import torch_npu" >/dev/null 2>&1; then
      log "WARN: python cannot import torch_npu. Source CANN/ATB in this shell first, e.g.:"
      log "  source /usr/local/Ascend/ascend-toolkit/set_env.sh"
      log "  source /usr/local/Ascend/nnal/atb/set_env.sh"
      log "Do NOT set TORCH_DEVICE_BACKEND_AUTOLOAD=0 for real NPU serving."
    else
      log "torch_npu import: ok"
    fi
  else
    log "Device env: ${DEVICE_ENV_NAME}=${DEVICES}"
  fi
}

main() {
  require curl
  require "${PYTHON_BIN}"
  require "${VLLM_BIN}"

  if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: set MODEL_PATH=/path/to/model" >&2
    exit 1
  fi
  if [ -z "${DATASET}" ] || [ ! -f "${DATASET}" ]; then
    echo "ERROR: set DATASET=/path/to/01_codex_swebenchpro_128k_filter_25s4t_chat.jsonl" >&2
    echo "Tip: build it with benchmarks/dataset/prepare_codex_swebenchpro_jsonl.py (see benchmarks/dataset/CODEX_SWEBENCHPRO.md)." >&2
    exit 1
  fi

  log "LOG_DIR=${LOG_DIR}"
  log "MODEL_PATH=${MODEL_PATH}"
  log "SERVED_MODEL=${SERVED_MODEL}"
  log "DATASET=${DATASET}"
  log "DEVICE_ENV_NAME=${DEVICE_ENV_NAME} DEVICES=${DEVICES} DP_SIZE=${DP_SIZE}"
  log "NUM_PROMPTS=${NUM_PROMPTS} MAX_CONCURRENCY=${MAX_CONCURRENCY} MAX_TOKENS=${MAX_TOKENS:-from_jsonl}"
  log "ENABLE_PER_REQUEST_METRICS=${ENABLE_PER_REQUEST_METRICS} ENABLE_PROMPT_TOKENS_DETAILS=${ENABLE_PROMPT_TOKENS_DETAILS} VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS}"
  log "PORTS baseline=${DP_BASELINE_PORT} backend=${BACKEND_PORT} router=${ROUTER_PORT} prom=${ROUTER_PROM_PORT}"
  log "RUN_DP_BASELINE=${RUN_DP_BASELINE} RUN_CACHE_AWARE=${RUN_CACHE_AWARE}"
  log "CONFIGS=${CONFIGS}"
  log "CHAT_ROUTING_KEY_MODE=${CHAT_ROUTING_KEY_MODE}"
  log "ROUTER_BIN=${ROUTER_BIN} VLLM_BIN=${VLLM_BIN}"
  log "NOTE: cache-aware also works with N independent workers (no DP); this script is DP+router only."
  preflight_npu_hint

  if [ "${RUN_DP_BASELINE}" = "1" ]; then
    run_dp_baseline_case
  fi

  if [ "${RUN_CACHE_AWARE}" = "1" ]; then
    local cfg
    for cfg in ${CONFIGS}; do
      run_cache_aware_case "${cfg}"
    done
  fi

  log "ALL_DONE ${LOG_DIR}"
  log "Artifacts: vllm_*.log router_*.log bench_*.log per_request_*.jsonl metrics/*.prom summary_*.json driver.log"
}

main "$@"
