#!/bin/bash
# Flagship Daily Paper Trading Cycle
# Run this via cron: 0 9 * * 1-5 (At 09:00 on every day-of-week from Monday through Friday)

# Navigate to project root
cd "$(dirname "$0")/../.."
PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Activate Virtual Environment (adjust path as needed)
# Assuming typical poetry location or user env
if [ -f "$HOME/.poetry/env" ]; then
    source "$HOME/.poetry/env"
elif [ -f "$HOME/Library/Application Support/pypoetry/venv/bin/activate" ]; then
    source "$HOME/Library/Application Support/pypoetry/venv/bin/activate"
fi

# Date for logging
LOG_DATE=$(date +%Y%m%d)
LOG_FILE="$PROJECT_ROOT/logs/paper_trading_$LOG_DATE.log"

echo "=== Starting Daily Cycle: $(date) ===" | tee -a "$LOG_FILE"

# 1. Update Data
echo "[1/3] Updating Live Data..." | tee -a "$LOG_FILE"
python flagship/paper_trading/update_live_data.py --lookback 5 >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Data update failed." | tee -a "$LOG_FILE"
    exit 1
fi

# 2. Run Inference
echo "[2/3] Running Inference..." | tee -a "$LOG_FILE"
python flagship/paper_trading/run_live_inference.py >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Inference failed." | tee -a "$LOG_FILE"
    exit 1
fi

# 3. Execute Orders
echo "[3/3] Executing Orders..." | tee -a "$LOG_FILE"
python flagship/paper_trading/alpaca_executor.py >> "$LOG_FILE" 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Execution failed." | tee -a "$LOG_FILE"
    exit 1
fi

echo "=== Cycle Complete: $(date) ===" | tee -a "$LOG_FILE"

