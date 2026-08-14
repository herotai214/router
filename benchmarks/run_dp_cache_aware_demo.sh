#!/usr/bin/env bash
# Lightweight smoke demo: one vLLM DP backend + cache-aware router.
#
# Workload is chat_prefix_repetition.py — a constructed prefix-repetition
# dataset (not Codex). If routing + prefix cache work, Prompt/APC hit rate
# MUST show. Realistic eval: run_codex_dp_cache_aware.sh.
#
# Topology:
#   client -> vllm-router (cache_aware, --intra-node-data-parallel-size=N)
#          -> one `vllm serve --data-parallel-size N`
#             (router sends X-data-parallel-rank)
#
# Cache-aware does NOT require DP. The alternative (also documented) is
# N independent `vllm serve` workers + router without --intra-node-data-parallel-size.
# This script only automates the DP+router topology that many deployments use.
#
# Usage (from router repo root, after cargo build --release).
# Full how-to: benchmarks/CACHE_AWARE_BENCHMARKS.md
# Python launcher: pip install -e . from this tree (needs rustc/cargo; not a wheel).
# Chat key default: session-id-full-history-fallback.
#   MODEL_PATH=/path/to/Qwen3.5-4B \
#   DEVICE_ENV_NAME=CUDA_VISIBLE_DEVICES DEVICES=0,1 \
#   bash benchmarks/run_dp_cache_aware_demo.sh
#
# NPU example:
#   MODEL_PATH=/path/to/Qwen3.5-4B \
#   DEVICE_ENV_NAME=ASCEND_RT_VISIBLE_DEVICES DEVICES=0,1 \
#   VLLM_BIN=vllm ROUTER_BIN=./target/release/vllm-router \
#   bash benchmarks/run_dp_cache_aware_demo.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-}"
SERVED_MODEL="${SERVED_MODEL:-qwen35-4b-dp-cache-aware}"

DEVICE_ENV_NAME="${DEVICE_ENV_NAME:-CUDA_VISIBLE_DEVICES}"
DEVICES="${DEVICES:-0,1}"
DP_SIZE="${DP_SIZE:-2}"

NUM_PROMPTS="${NUM_PROMPTS:-100}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-4}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
PREFIX_LEN="${PREFIX_LEN:-256}"
SUFFIX_LEN="${SUFFIX_LEN:-16}"
SESSION_GROUPS="${SESSION_GROUPS:-16}"
UNIQUE_PREFIXES="${UNIQUE_PREFIXES:-12}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---max-num-seqs 4}"

VLLM_BIN="${VLLM_BIN:-vllm}"
ROUTER_BIN="${ROUTER_BIN:-${ROOT_DIR}/target/release/vllm-router}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

BACKEND_PORT="${BACKEND_PORT:-18100}"
ROUTER_PORT="${ROUTER_PORT:-18180}"
ROUTER_PROM_PORT="${ROUTER_PROM_PORT:-29400}"

CACHE_THRESHOLD="${CACHE_THRESHOLD:-0.3}"
BALANCE_ABS_THRESHOLD="${BALANCE_ABS_THRESHOLD:-2}"
BALANCE_REL_THRESHOLD="${BALANCE_REL_THRESHOLD:-1.5}"
CHAT_ROUTING_KEY_MODE="${CHAT_ROUTING_KEY_MODE:-session-id-full-history-fallback}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-${PWD}/logs_dp${DP_SIZE}_cache_aware_demo_rp${ROUTER_PORT}_${TS}}"
mkdir -p "${LOG_DIR}/metrics"

PIDS=()

log() {
  echo "[$(date -Is)] $*" | tee -a "${LOG_DIR}/driver.log"
}

require() {
  if ! command -v "$1" >/dev/null 2>&1 && [ ! -x "$1" ]; then
    echo "ERROR: missing command/binary: $1" >&2
    exit 1
  fi
}

kill_pid_tree() {
  local pid="$1"
  if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
    pkill -P "${pid}" 2>/dev/null || true
    kill "${pid}" 2>/dev/null || true
  fi
}

cleanup_all() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill_pid_tree "${pid}"
  done
  PIDS=()
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

start_backend_dp() {
  local log_file="${LOG_DIR}/vllm_backend_dp${DP_SIZE}_port${BACKEND_PORT}.log"
  log "START_BACKEND_DP devices=${DEVICES} dp_size=${DP_SIZE} port=${BACKEND_PORT}"
  env "${DEVICE_ENV_NAME}=${DEVICES}" \
    "${VLLM_BIN}" serve "${MODEL_PATH}" \
      --host 0.0.0.0 \
      --port "${BACKEND_PORT}" \
      --served-model-name "${SERVED_MODEL}" \
      --tensor-parallel-size 1 \
      --data-parallel-size "${DP_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --enable-prefix-caching \
      --trust-remote-code \
      ${VLLM_EXTRA_ARGS} \
      >"${log_file}" 2>&1 &
  PIDS+=("$!")
  wait_health "backend_dp${DP_SIZE}" "http://127.0.0.1:${BACKEND_PORT}"
}

start_router_dp_aware() {
  local log_file="${LOG_DIR}/router_cache_aware_dp${DP_SIZE}.log"
  log "START_ROUTER cache_aware mode=${CHAT_ROUTING_KEY_MODE} cache=${CACHE_THRESHOLD} abs=${BALANCE_ABS_THRESHOLD} rel=${BALANCE_REL_THRESHOLD} intra_dp=${DP_SIZE}"
  if [ ! -x "${ROUTER_BIN}" ] && ! command -v "${ROUTER_BIN}" >/dev/null 2>&1; then
    echo "ERROR: ROUTER_BIN not found/executable: ${ROUTER_BIN}" >&2
    echo "Build with: (cd ${ROOT_DIR} && cargo build --release)" >&2
    exit 1
  fi
  "${ROUTER_BIN}" \
    --host 0.0.0.0 \
    --port "${ROUTER_PORT}" \
    --worker-urls "http://127.0.0.1:${BACKEND_PORT}" \
    --policy cache_aware \
    --prometheus-port "${ROUTER_PROM_PORT}" \
    --cache-threshold "${CACHE_THRESHOLD}" \
    --balance-abs-threshold "${BALANCE_ABS_THRESHOLD}" \
    --balance-rel-threshold "${BALANCE_REL_THRESHOLD}" \
    --chat-routing-key-mode "${CHAT_ROUTING_KEY_MODE}" \
    --intra-node-data-parallel-size "${DP_SIZE}" \
    >"${log_file}" 2>&1 &
  PIDS+=("$!")
  wait_health "router_cache_aware_dp${DP_SIZE}" "http://127.0.0.1:${ROUTER_PORT}" 300
}

run_bench() {
  local bench_log="${LOG_DIR}/bench_chat_prefix_repetition_dp${DP_SIZE}.log"
  local label="dp${DP_SIZE}_cache_aware_${CHAT_ROUTING_KEY_MODE//-/_}_c${CACHE_THRESHOLD}_a${BALANCE_ABS_THRESHOLD}_r${BALANCE_REL_THRESHOLD}"
  log "BENCH_START label=${label} log=${bench_log}"
  "${PYTHON_BIN}" "${BENCH_DIR}/chat_prefix_repetition.py" \
    --base-url "http://127.0.0.1:${ROUTER_PORT}" \
    --model "${SERVED_MODEL}" \
    --num-prompts "${NUM_PROMPTS}" \
    --session-groups "${SESSION_GROUPS}" \
    --unique-prefixes "${UNIQUE_PREFIXES}" \
    --prefix-len "${PREFIX_LEN}" \
    --suffix-len "${SUFFIX_LEN}" \
    --output-len "${OUTPUT_LEN}" \
    --max-concurrency "${MAX_CONCURRENCY}" \
    --label "${label}" \
    2>&1 | tee "${bench_log}"
  log "BENCH_DONE"
}

main() {
  require curl
  require "${PYTHON_BIN}"
  require "${VLLM_BIN}"

  if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: set MODEL_PATH=/path/to/model" >&2
    exit 1
  fi

  log "LOG_DIR=${LOG_DIR}"
  log "MODEL_PATH=${MODEL_PATH}"
  log "SERVED_MODEL=${SERVED_MODEL}"
  log "DEVICE_ENV_NAME=${DEVICE_ENV_NAME} DEVICES=${DEVICES} DP_SIZE=${DP_SIZE}"
  log "BACKEND_PORT=${BACKEND_PORT} ROUTER_PORT=${ROUTER_PORT} ROUTER_PROM_PORT=${ROUTER_PROM_PORT}"
  log "CACHE_THRESHOLD=${CACHE_THRESHOLD} ABS=${BALANCE_ABS_THRESHOLD} REL=${BALANCE_REL_THRESHOLD}"
  log "CHAT_ROUTING_KEY_MODE=${CHAT_ROUTING_KEY_MODE}"
  log "NOTE: cache-aware also works with N independent workers (no DP); this script demos DP+router only."

  start_backend_dp
  start_router_dp_aware
  run_bench

  local router_prom="${LOG_DIR}/metrics/router.prom"
  local backend_prom="${LOG_DIR}/metrics/backend_dp${DP_SIZE}.prom"
  local summary_json="${LOG_DIR}/summary.json"

  curl_metrics "router" "http://127.0.0.1:${ROUTER_PROM_PORT}" "${router_prom}"
  curl_metrics "backend" "http://127.0.0.1:${BACKEND_PORT}" "${backend_prom}"

  log "SUMMARY via router_metrics_summary.sh (APC + Prompt hit + latency)"
  bash "${BENCH_DIR}/router_metrics_summary.sh" "${router_prom}" \
    --workers "${backend_prom}" \
    --out "${summary_json}" \
    --brief-only \
    --label "dp${DP_SIZE}_cache_aware" \
    | tee -a "${LOG_DIR}/driver.log"

  log "SUMMARY_JSON=${summary_json}"
  log "ALL_DONE ${LOG_DIR}"
}

main "$@"
