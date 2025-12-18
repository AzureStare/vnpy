"""
Configuration for Flagship Alpha-Momentum Paper Trading on Alpaca.
Reads credentials from vt_setting.json.
"""
import json
from pathlib import Path
from flagship.config import VT_SETTING_PATH

# Load settings from vt_setting.json
SETTINGS = {}
if VT_SETTING_PATH.exists():
    try:
        with open(VT_SETTING_PATH, "r", encoding="utf-8") as f:
            SETTINGS = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load vt_setting.json: {e}")

# --- Alpaca API Configuration ---
# Keys should be set in vt_setting.json as:
# "alpaca.api_key": "YOUR_KEY",
# "alpaca.secret_key": "YOUR_SECRET"
ALPACA_API_KEY = SETTINGS.get("alpaca.api_key", "")
ALPACA_SECRET_KEY = SETTINGS.get("alpaca.secret_key", "")
ALPACA_PAPER = True  # Set to True for paper trading

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    # Warn instead of raising error immediately so other modules can import config
    # The executor will fail if keys are missing
    print("WARNING: 'alpaca.api_key' or 'alpaca.secret_key' not found in vt_setting.json")

# Base URL for paper trading
ALPACA_BASE_URL = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"

# --- Strategy Configuration ---
# Current Regime ID for model selection (e.g., 10 for "2025 Autumn High Volatility Drop")
# Update this as market conditions change
CURRENT_REGIME_ID = 10

# Top N stocks to hold (should match strategy logic)
TOP_N = 5

# Minimum score threshold to buy
MIN_SCORE_THRESHOLD = 0.5

# Cash usage ratio (keep some buffer)
CASH_RATIO = 0.95

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAB_PATH = PROJECT_ROOT / "lab" / "flagship_alpha_momentum"
MODEL_DIR = LAB_PATH / "model"
SIGNAL_DIR = LAB_PATH / "signal"

# The static model file (backtest)
MODEL_FILE = f"flagship_alpha_mom_regime{CURRENT_REGIME_ID:02d}_lgb.pkl"
MODEL_PATH = MODEL_DIR / MODEL_FILE

# The live retrained model file
LIVE_MODEL_FILE = "flagship_alpha_mom_live_lgb.pkl"
LIVE_MODEL_PATH = MODEL_DIR / LIVE_MODEL_FILE

# Output path for daily signals
DAILY_SIGNAL_FILE = SIGNAL_DIR / "daily_signal.parquet"

# --- Universe ---
# Path to the static universe file if used, or None for dynamic
# If None, it will rely on what's in the lab or postgres
UNIVERSE_FILE = None 
