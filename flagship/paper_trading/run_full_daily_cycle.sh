#!/bin/bash
# Flagship Full Automated Paper Trading Cycle
# Run this via cron before market open (e.g., 09:00 ET)

# Navigate to project root
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# --- Date handling ---
# TRADING_DATE: the date we want to trade (orders executed at open)
# DATA_DATE: previous trading day close used for features/signal generation
TRADING_DATE="${1:-$(python - <<'PY'
from datetime import datetime
from zoneinfo import ZoneInfo
print(datetime.now(ZoneInfo("America/New_York")).date().isoformat())
PY
)}"

DATA_DATE="$(python - <<PY
from datetime import date, timedelta
td = date.fromisoformat("${TRADING_DATE}")
print((td - timedelta(days=1)).isoformat())
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

echo "=== Starting Full Daily Cycle: $(date) ===" | tee -a "$LOG_FILE"
echo "TRADING_DATE=${TRADING_DATE} DATA_DATE=${DATA_DATE}" | tee -a "$LOG_FILE"

# 1. Update Market Indices (VIX, SPY, QQQ)
echo "[1/8] Updating Market Indices..." | tee -a "$LOG_FILE"
python flagship/paper_trading/update_market_indices.py --lookback 5 >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then echo "ERROR: Index update failed." | tee -a "$LOG_FILE"; exit 1; fi

# 2. Incremental Data Update (Daily Full Market) - must happen before Daily Selection
echo "[2/8] Incrementally Updating Daily Bars (full market)..." | tee -a "$LOG_FILE"
python flagship/scripts/update_lab_data_incremental.py \
  --end-date "$DATA_DATE" \
  --interval daily \
  --universe ref_tickers_cs \
  --overlap-days 1 \
  >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then echo "ERROR: Daily full market incremental update failed." | tee -a "$LOG_FILE"; exit 1; fi

# 3. Run Daily Selection (U_t based on DATA_DATE close)
echo "[3/8] Running Daily Selection..." | tee -a "$LOG_FILE"
python flagship/paper_trading/run_daily_selection.py --date "$DATA_DATE" >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then echo "ERROR: Daily selection failed." | tee -a "$LOG_FILE"; exit 1; fi

# 4. Ensure Data Completeness for Selection (backfill history if needed)
echo "[4/8] Verifying Data Completeness..." | tee -a "$LOG_FILE"
python flagship/paper_trading/ensure_data_completeness.py --date "$DATA_DATE" >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then echo "ERROR: Data check failed." | tee -a "$LOG_FILE"; exit 1; fi

# 4.8 Minute bars: only daily_selection universe for DATA_DATE, update to TRADING_DATE (premarket included if available)
echo "[4.8/8] Incrementally Updating Minute Bars (daily_selection)..." | tee -a "$LOG_FILE"
python flagship/scripts/update_lab_data_incremental.py \
  --end-date "$TRADING_DATE" \
  --interval minute \
  --universe daily_selection \
  --selection-start "$DATA_DATE" \
  --selection-end "$DATA_DATE" \
  --overlap-days 1 \
  >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then echo "ERROR: Minute selection incremental update failed." | tee -a "$LOG_FILE"; exit 1; fi

# 4.5 Freshness Check & Auto-fix (Bars/Model/Signal/Dataset)
echo "[4.5/8] Checking Lab Freshness (auto-fix)..." | tee -a "$LOG_FILE"
python flagship/paper_trading/check_lab_freshness.py --expected-date "$DATA_DATE" >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then echo "ERROR: Lab freshness check failed." | tee -a "$LOG_FILE"; exit 1; fi

# 5. Train Daily Model (New Step)
echo "[5/8] Retraining Daily Model..." | tee -a "$LOG_FILE"
python flagship/paper_trading/train_daily_model.py --date "$TRADING_DATE" >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then echo "ERROR: Model training failed." | tee -a "$LOG_FILE"; exit 1; fi

# 6. Run Inference
echo "[6/8] Generating Signals..." | tee -a "$LOG_FILE"
python flagship/paper_trading/run_live_inference.py --date "$DATA_DATE" >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then echo "ERROR: Inference failed." | tee -a "$LOG_FILE"; exit 1; fi

# 6.5 Freshness Check (ensure signal up-to-date)
echo "[6.5/8] Re-checking Lab Freshness..." | tee -a "$LOG_FILE"
python flagship/paper_trading/check_lab_freshness.py --expected-date "$DATA_DATE" --no-check-datasets >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then echo "ERROR: Lab freshness re-check failed." | tee -a "$LOG_FILE"; exit 1; fi

# 7. Execute Orders (open rebalance)
echo "[7/8] Executing Orders..." | tee -a "$LOG_FILE"
python flagship/paper_trading/alpaca_executor.py --use-polygon-ws >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then echo "ERROR: Execution failed." | tee -a "$LOG_FILE"; exit 1; fi

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
    nohup python flagship/paper_trading/intraday_runner.py \
      --mode exit-only \
      --use-polygon-ws \
      --rth-only \
      >> "$INTRADAY_LOG" 2>&1 &
    echo $! > "$PID_FILE"
    echo "[8/8] Intraday Runner started (pid=$!), log=$INTRADAY_LOG" | tee -a "$LOG_FILE"
  fi
else
  echo "[8/8] Intraday Runner disabled (ENABLE_INTRADAY_RUNNER=$ENABLE_INTRADAY_RUNNER)" | tee -a "$LOG_FILE"
fi

echo "=== Cycle Complete: $(date) ===" | tee -a "$LOG_FILE"
