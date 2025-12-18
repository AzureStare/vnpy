#!/bin/bash
# Flagship Full Automated Paper Trading Cycle
# Run this via cron before market open (e.g., 09:00 ET)

# Navigate to project root
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Activate Virtual Environment
if [ -f "$HOME/.poetry/env" ]; then
    source "$HOME/.poetry/env"
elif [ -f "$HOME/Library/Application Support/pypoetry/venv/bin/activate" ]; then
    source "$HOME/Library/Application Support/pypoetry/venv/bin/activate"
fi

LOG_DATE=$(date +%Y%m%d)
LOG_FILE="$PROJECT_ROOT/logs/paper_trading_$LOG_DATE.log"

echo "=== Starting Full Daily Cycle: $(date) ===" | tee -a "$LOG_FILE"

# 1. Update Market Indices (VIX, SPY, QQQ)
echo "[1/8] Updating Market Indices..." | tee -a "$LOG_FILE"
python flagship/paper_trading/update_market_indices.py --lookback 5 >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then echo "ERROR: Index update failed." | tee -a "$LOG_FILE"; exit 1; fi

# 2. Run Daily Selection
echo "[2/8] Running Daily Selection..." | tee -a "$LOG_FILE"
# Assuming 'today' or let script default to yesterday if run before market close
python flagship/paper_trading/run_daily_selection.py >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then echo "ERROR: Daily selection failed." | tee -a "$LOG_FILE"; exit 1; fi

# 3. Ensure Data Completeness for Selection
echo "[3/8] Verifying Data Completeness..." | tee -a "$LOG_FILE"
python flagship/paper_trading/ensure_data_completeness.py >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then echo "ERROR: Data check failed." | tee -a "$LOG_FILE"; exit 1; fi

# 4. Update Live Data (Download Yesterday's Bars for everything else if needed)
echo "[4/8] Updating Live Data..." | tee -a "$LOG_FILE"
python flagship/paper_trading/update_live_data.py --lookback 5 >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then echo "ERROR: Live data update failed." | tee -a "$LOG_FILE"; exit 1; fi

# 4.5 Freshness Check & Auto-fix (Bars/Model/Signal/Dataset)
echo "[4.5/8] Checking Lab Freshness (auto-fix)..." | tee -a "$LOG_FILE"
python flagship/paper_trading/check_lab_freshness.py >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then echo "ERROR: Lab freshness check failed." | tee -a "$LOG_FILE"; exit 1; fi

# 5. Train Daily Model (New Step)
echo "[5/8] Retraining Daily Model..." | tee -a "$LOG_FILE"
python flagship/paper_trading/train_daily_model.py >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then echo "ERROR: Model training failed." | tee -a "$LOG_FILE"; exit 1; fi

# 6. Run Inference
echo "[6/8] Generating Signals..." | tee -a "$LOG_FILE"
python flagship/paper_trading/run_live_inference.py >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then echo "ERROR: Inference failed." | tee -a "$LOG_FILE"; exit 1; fi

# 6.5 Freshness Check (ensure signal up-to-date)
echo "[6.5/8] Re-checking Lab Freshness..." | tee -a "$LOG_FILE"
python flagship/paper_trading/check_lab_freshness.py --no-check-datasets >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then echo "ERROR: Lab freshness re-check failed." | tee -a "$LOG_FILE"; exit 1; fi

# 7. Execute Orders
echo "[7/8] Executing Orders..." | tee -a "$LOG_FILE"
python flagship/paper_trading/alpaca_executor.py >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then echo "ERROR: Execution failed." | tee -a "$LOG_FILE"; exit 1; fi

echo "=== Cycle Complete: $(date) ===" | tee -a "$LOG_FILE"

# Optional: Start Monitor Service in background (if not already running)
# nohup python flagship/paper_trading/monitor_service.py >> "$PROJECT_ROOT/logs/monitor_service.log" 2>&1 &
