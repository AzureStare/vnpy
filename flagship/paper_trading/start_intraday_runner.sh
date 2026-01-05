#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/app"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

LOG_DATE="$(date +%Y%m%d)"
INTRADAY_LOG="${LOG_DIR}/intraday_runner_${LOG_DATE}.log"
PID_FILE="${LOG_DIR}/intraday_runner.pid"

# Prefer venv python if present
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

# Avoid duplicate runners
if [ -f "${PID_FILE}" ] && ps -p "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
  echo "[start_intraday_runner] already running (pid=$(cat "${PID_FILE}")), skip."
  exit 0
fi

cd "${PROJECT_ROOT}"

SIGNAL_TOP_N="${FLAGSHIP_INTRADAY_SIGNAL_TOPN:-10}"
nohup ${PYTHON_BIN} flagship/paper_trading/intraday_runner.py \
  --mode exit-only \
  --use-polygon-ws \
  --rth-only \
  --symbols-source both \
  --signal-top-n "${SIGNAL_TOP_N}" \
  >> "${INTRADAY_LOG}" 2>&1 &

echo $! > "${PID_FILE}"
echo "[start_intraday_runner] started (pid=$!), log=${INTRADAY_LOG}, signal_top_n=${SIGNAL_TOP_N}"


