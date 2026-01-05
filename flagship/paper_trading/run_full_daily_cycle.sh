#!/bin/bash
# Flagship Full Automated Paper Trading Cycle
# Run this via cron before market open (e.g., 09:00 ET)

# Navigate to project root
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Pick python executable (prefer project venv if available)
PYTHON_BIN="python"
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
fi

# --- Date handling ---
# TRADING_DATE: the date we want to trade (orders executed at open)
# DATA_DATE: previous trading day close used for features/signal generation (上一交易日收盘，不是简单 T-1)
TRADING_DATE="${1:-$($PYTHON_BIN - <<'PY'
from datetime import datetime
from zoneinfo import ZoneInfo
print(datetime.now(ZoneInfo("America/New_York")).date().isoformat())
PY
)}"

# Determine holiday mode (market closed day) using Polygon calendar tool
HOLIDAY_MODE="$(
  TRADING_DATE="$TRADING_DATE" $PYTHON_BIN - <<'PY'
import os
from datetime import date
from flagship.paper_trading.trading_calendar import is_market_closed_day
d = date.fromisoformat(os.environ["TRADING_DATE"])
print("1" if is_market_closed_day(d) else "0")
PY
)"

# Activate Virtual Environment
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -f "$HOME/.poetry/env" ]; then
    source "$HOME/.poetry/env"
elif [ -f "$HOME/Library/Application Support/pypoetry/venv/bin/activate" ]; then
    source "$HOME/Library/Application Support/pypoetry/venv/bin/activate"
fi

LOG_DATE=$(date +%Y%m%d)
LOG_FILE="$PROJECT_ROOT/logs/paper_trading_$LOG_DATE.log"

# --- Metrics (Prometheus textfile collector) ---
METRICS_DIR="${FLAGSHIP_TEXTFILE_DIR:-$PROJECT_ROOT/logs/metrics}"
METRICS_FILE="$METRICS_DIR/flagship_daily_cycle.prom"
mkdir -p "$METRICS_DIR" >/dev/null 2>&1 || true

DATA_DATE="${DATA_DATE:-unknown}"
CYCLE_START_TS="$(date +%s)"
CYCLE_RUNNING=1
CYCLE_SUCCESS=0
CYCLE_LAST_STEP="start"
CYCLE_LAST_STEP_DURATION=0
CYCLE_TOTAL_DURATION=0
CYCLE_LAST_UPDATE_TS="$CYCLE_START_TS"

write_metrics() {
  local tmp="$METRICS_FILE.tmp"
  CYCLE_LAST_UPDATE_TS="$(date +%s)"
  CYCLE_TOTAL_DURATION=$((CYCLE_LAST_UPDATE_TS - CYCLE_START_TS))
  {
    echo "# TYPE flagship_daily_cycle_running gauge"
    echo "# TYPE flagship_daily_cycle_success gauge"
    echo "# TYPE flagship_daily_cycle_last_update_timestamp_seconds gauge"
    echo "# TYPE flagship_daily_cycle_last_step_duration_seconds gauge"
    echo "# TYPE flagship_daily_cycle_total_duration_seconds gauge"
    echo "flagship_daily_cycle_running{trading_date=\"${TRADING_DATE}\",data_date=\"${DATA_DATE}\",holiday_mode=\"${HOLIDAY_MODE}\"} ${CYCLE_RUNNING}"
    echo "flagship_daily_cycle_success{trading_date=\"${TRADING_DATE}\",data_date=\"${DATA_DATE}\"} ${CYCLE_SUCCESS}"
    echo "flagship_daily_cycle_last_update_timestamp_seconds ${CYCLE_LAST_UPDATE_TS}"
    echo "flagship_daily_cycle_last_step_duration_seconds{step=\"${CYCLE_LAST_STEP}\"} ${CYCLE_LAST_STEP_DURATION}"
    echo "flagship_daily_cycle_total_duration_seconds ${CYCLE_TOTAL_DURATION}"
  } > "$tmp" && mv "$tmp" "$METRICS_FILE"
}

write_metrics

echo "=== Starting Full Daily Cycle: $(date) ===" | tee -a "$LOG_FILE"
echo "TRADING_DATE=${TRADING_DATE} HOLIDAY_MODE=${HOLIDAY_MODE}" | tee -a "$LOG_FILE"

# 1. Update Market Indices (VIX, SPY, QQQ)
echo "[1/8] Updating Market Indices..." | tee -a "$LOG_FILE"
STEP_START_TS="$(date +%s)"
$PYTHON_BIN flagship/paper_trading/update_market_indices.py --lookback 5 >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
  CYCLE_RUNNING=0; CYCLE_SUCCESS=0; CYCLE_LAST_STEP="1_update_market_indices"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics
  echo "ERROR: Index update failed." | tee -a "$LOG_FILE"; exit 1
fi
CYCLE_LAST_STEP="1_update_market_indices"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics

# 2. Incremental Data Update (Daily Full Market) - must happen before Daily Selection
echo "[2/8] Incrementally Updating Daily Bars (full market)..." | tee -a "$LOG_FILE"
STEP_START_TS="$(date +%s)"
$PYTHON_BIN flagship/scripts/update_lab_data_incremental.py \
  --end-date "$TRADING_DATE" \
  --interval daily \
  --universe ref_tickers_cs \
  --overlap-days 1 \
  >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
  CYCLE_RUNNING=0; CYCLE_SUCCESS=0; CYCLE_LAST_STEP="2_update_daily_full_market"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics
  echo "ERROR: Daily full market incremental update failed." | tee -a "$LOG_FILE"; exit 1
fi
CYCLE_LAST_STEP="2_update_daily_full_market"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics

# Infer DATA_DATE (previous trading day close) if not already set
if [ -z "$DATA_DATE" ] || [ "$DATA_DATE" = "unknown" ]; then
  DATA_DATE="$($PYTHON_BIN - <<'PY'
from flagship.paper_trading.config import LAB_PATH
from flagship.paper_trading.trading_calendar import infer_data_date_from_lab
print(infer_data_date_from_lab(LAB_PATH).isoformat())
PY
)"
fi
echo "TRADING_DATE=${TRADING_DATE} DATA_DATE=${DATA_DATE} HOLIDAY_MODE=${HOLIDAY_MODE}" | tee -a "$LOG_FILE"
write_metrics

# Holiday/Weekend: update-only mode (no selection/train/inference/orders/runner)
if [ "$HOLIDAY_MODE" = "1" ]; then
  echo "[holiday] Market closed on TRADING_DATE=${TRADING_DATE}. Run update-only and exit." | tee -a "$LOG_FILE"
  echo "[holiday] Checking data freshness (bars only) for DATA_DATE=${DATA_DATE}..." | tee -a "$LOG_FILE"
  STEP_START_TS="$(date +%s)"
  $PYTHON_BIN flagship/paper_trading/check_lab_freshness.py \
    --expected-date "$DATA_DATE" \
    --no-check-model \
    --no-check-signals \
    --no-check-datasets \
    >> "$LOG_FILE" 2>&1
  if [ $? -ne 0 ]; then
    CYCLE_RUNNING=0; CYCLE_SUCCESS=0; CYCLE_LAST_STEP="holiday_check_freshness"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics
    echo "ERROR: Holiday freshness check failed." | tee -a "$LOG_FILE"; exit 1
  fi
  CYCLE_RUNNING=0; CYCLE_SUCCESS=1; CYCLE_LAST_STEP="holiday_check_freshness"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics
  echo "=== Holiday update-only complete: $(date) ===" | tee -a "$LOG_FILE"
  exit 0
fi

# 3. Run Daily Selection (U_t based on DATA_DATE close)
echo "[3/8] Running Daily Selection (v7)..." | tee -a "$LOG_FILE"
STEP_START_TS="$(date +%s)"
$PYTHON_BIN flagship/paper_trading/run_daily_selection.py --date "$DATA_DATE" --strategy v7 >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
  CYCLE_RUNNING=0; CYCLE_SUCCESS=0; CYCLE_LAST_STEP="3_daily_selection"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics
  echo "ERROR: Daily selection failed." | tee -a "$LOG_FILE"; exit 1
fi
CYCLE_LAST_STEP="3_daily_selection"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics

# 4. Ensure Data Completeness for Selection (backfill history if needed)
echo "[4/8] Verifying Data Completeness..." | tee -a "$LOG_FILE"
STEP_START_TS="$(date +%s)"
$PYTHON_BIN flagship/paper_trading/ensure_data_completeness.py --date "$DATA_DATE" >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
  CYCLE_RUNNING=0; CYCLE_SUCCESS=0; CYCLE_LAST_STEP="4_ensure_data_completeness"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics
  echo "ERROR: Data check failed." | tee -a "$LOG_FILE"; exit 1
fi
CYCLE_LAST_STEP="4_ensure_data_completeness"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics

# 4.8 Minute bars: only daily_selection universe for DATA_DATE, update to TRADING_DATE (premarket included if available)
echo "[4.8/8] Incrementally Updating Minute Bars (daily_selection)..." | tee -a "$LOG_FILE"
STEP_START_TS="$(date +%s)"
$PYTHON_BIN flagship/scripts/update_lab_data_incremental.py \
  --end-date "$TRADING_DATE" \
  --interval minute \
  --universe daily_selection \
  --selection-start "$DATA_DATE" \
  --selection-end "$DATA_DATE" \
  --overlap-days 1 \
  >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
  CYCLE_RUNNING=0; CYCLE_SUCCESS=0; CYCLE_LAST_STEP="4_8_update_minute_selection"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics
  echo "ERROR: Minute selection incremental update failed." | tee -a "$LOG_FILE"; exit 1
fi
CYCLE_LAST_STEP="4_8_update_minute_selection"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics

# 4.5 Freshness Check & Auto-fix (Bars/Model/Signal/Dataset)
echo "[4.5/8] Checking Lab Freshness (auto-fix)..." | tee -a "$LOG_FILE"
STEP_START_TS="$(date +%s)"
$PYTHON_BIN flagship/paper_trading/check_lab_freshness.py --expected-date "$DATA_DATE" >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
  CYCLE_RUNNING=0; CYCLE_SUCCESS=0; CYCLE_LAST_STEP="4_5_check_lab_freshness"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics
  echo "ERROR: Lab freshness check failed." | tee -a "$LOG_FILE"; exit 1
fi
CYCLE_LAST_STEP="4_5_check_lab_freshness"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics

# 5. Train Model (Weekly on Mondays, or if missing)
MODEL_FILE="$PROJECT_ROOT/lab/flagship_alpha_momentum/model/live_model.joblib"
DOW=$(date +%u)
SHOULD_TRAIN=0

if [ ! -f "$MODEL_FILE" ]; then
  echo "[5/8] Model file missing, forcing training..." | tee -a "$LOG_FILE"
  SHOULD_TRAIN=1
elif [ "$DOW" -eq 1 ]; then
  echo "[5/8] Today is Monday, scheduled weekly retraining..." | tee -a "$LOG_FILE"
  SHOULD_TRAIN=1
else
  echo "[5/8] Not Monday and model exists, skipping retraining today." | tee -a "$LOG_FILE"
fi

if [ "$SHOULD_TRAIN" -eq 1 ]; then
  echo "[5/8] Retraining Model (3-year window)..." | tee -a "$LOG_FILE"
  STEP_START_TS="$(date +%s)"
  $PYTHON_BIN flagship/paper_trading/train_daily_model.py --date "$TRADING_DATE" --strategy v7 >> "$LOG_FILE" 2>&1
  if [ $? -ne 0 ]; then
    CYCLE_RUNNING=0; CYCLE_SUCCESS=0; CYCLE_LAST_STEP="5_train_model"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics
    echo "ERROR: Model training failed." | tee -a "$LOG_FILE"; exit 1
  fi
  CYCLE_LAST_STEP="5_train_model"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics
fi

# 6. Run Inference
echo "[6/8] Generating Signals (v7)..." | tee -a "$LOG_FILE"
STEP_START_TS="$(date +%s)"
$PYTHON_BIN flagship/paper_trading/run_live_inference.py --date "$DATA_DATE" --strategy v7 >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
  CYCLE_RUNNING=0; CYCLE_SUCCESS=0; CYCLE_LAST_STEP="6_run_inference"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics
  echo "ERROR: Inference failed." | tee -a "$LOG_FILE"; exit 1
fi
CYCLE_LAST_STEP="6_run_inference"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics

# 6.5 Freshness Check (ensure signal up-to-date)
echo "[6.5/8] Re-checking Lab Freshness..." | tee -a "$LOG_FILE"
STEP_START_TS="$(date +%s)"
$PYTHON_BIN flagship/paper_trading/check_lab_freshness.py --expected-date "$DATA_DATE" --no-check-datasets >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
  CYCLE_RUNNING=0; CYCLE_SUCCESS=0; CYCLE_LAST_STEP="6_5_recheck_lab_freshness"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics
  echo "ERROR: Lab freshness re-check failed." | tee -a "$LOG_FILE"; exit 1
fi
CYCLE_LAST_STEP="6_5_recheck_lab_freshness"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics

# 7. Execute Orders (open rebalance)
echo "[7/8] Executing Orders (v7)..." | tee -a "$LOG_FILE"
STEP_START_TS="$(date +%s)"
$PYTHON_BIN flagship/paper_trading/alpaca_executor.py \
  --use-polygon-ws \
  --strategy v7 \
  --max-wait-seconds "${FLAGSHIP_EXECUTOR_MAX_WAIT_SECONDS:-259200}" \
  >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
  CYCLE_RUNNING=0; CYCLE_SUCCESS=0; CYCLE_LAST_STEP="7_execute_orders"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics
  echo "ERROR: Execution failed." | tee -a "$LOG_FILE"; exit 1
fi
CYCLE_LAST_STEP="7_execute_orders"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics

# 8. Start Intraday Runner (Plan B: minute-level exits during RTH)
ENABLE_INTRADAY_RUNNER="${ENABLE_INTRADAY_RUNNER:-1}"
if [ "$ENABLE_INTRADAY_RUNNER" = "1" ]; then
  echo "[8/8] Starting Intraday Runner (exit-only, RTH-only)..." | tee -a "$LOG_FILE"
  INTRADAY_LOG="$PROJECT_ROOT/logs/intraday_runner_$LOG_DATE.log"
  PID_FILE="$PROJECT_ROOT/logs/intraday_runner.pid"

  # Avoid duplicate runners
  if [ -f "$PID_FILE" ] && ps -p "$(cat "$PID_FILE")" >/dev/null 2>&1; then
    echo "[8/8] Intraday Runner already running (pid=$(cat "$PID_FILE")), skip." | tee -a "$LOG_FILE"
  else
    STEP_START_TS="$(date +%s)"
    nohup $PYTHON_BIN flagship/paper_trading/intraday_runner.py \
      --mode exit-only \
      --use-polygon-ws \
      --rth-only \
      >> "$INTRADAY_LOG" 2>&1 &
    echo $! > "$PID_FILE"
    echo "[8/8] Intraday Runner started (pid=$!), log=$INTRADAY_LOG" | tee -a "$LOG_FILE"
    CYCLE_LAST_STEP="8_start_intraday_runner"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics
  fi
else
  echo "[8/8] Intraday Runner disabled (ENABLE_INTRADAY_RUNNER=$ENABLE_INTRADAY_RUNNER)" | tee -a "$LOG_FILE"
fi

# 8.5 Refresh Ops Console snapshots (all) + generate report
echo "[8.5/9] Refreshing Ops Console snapshots..." | tee -a "$LOG_FILE"
STEP_START_TS="$(date +%s)"
if ! $PYTHON_BIN flagship/monitoring/app_console_snapshot.py all >> "$LOG_FILE" 2>&1; then
  echo "[warn] Ops Console snapshot failed (non-blocking)." | tee -a "$LOG_FILE"
fi
CYCLE_LAST_STEP="8_5_refresh_ops_console"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics

# 8.6 Generate Ops Console report (with GPT summary)
echo "[8.6/9] Generating Ops Console report..." | tee -a "$LOG_FILE"
STEP_START_TS="$(date +%s)"
if ! $PYTHON_BIN flagship/monitoring/app_console_report.py >> "$LOG_FILE" 2>&1; then
  echo "[warn] Ops Console report generation failed (non-blocking)." | tee -a "$LOG_FILE"
fi
CYCLE_LAST_STEP="8_6_generate_ops_report"; CYCLE_LAST_STEP_DURATION=$(( $(date +%s) - STEP_START_TS )); write_metrics

echo "=== Cycle Complete: $(date) ===" | tee -a "$LOG_FILE"
CYCLE_RUNNING=0
CYCLE_SUCCESS=1
CYCLE_LAST_STEP="done"
CYCLE_LAST_STEP_DURATION=0
write_metrics
