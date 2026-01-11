"""
Configuration for Flagship Alpha-Momentum trading.
Reads credentials from vt_setting.json.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

# --- IBKR Gateway Configuration ---
@dataclass(frozen=True)
class IbkrAccountConfig:
    """
    One IBKR account corresponds to one IB Gateway/TWS login in our current deployment model.
    """

    account_id: str
    display_name: str
    host: str
    port: int
    client_id: int


def _as_str(v: Any) -> str:
    return str(v or "").strip()


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def load_ibkr_accounts() -> list[IbkrAccountConfig]:
    """
    Load IBKR accounts from vt_setting.json:

      "ibkr.accounts": [
        {"account_id": "U123...", "display_name": "IBKR Paper 01", "host": "127.0.0.1", "port": 4002, "client_id": 11}
      ]
    """
    raw = SETTINGS.get("ibkr.accounts")
    if not isinstance(raw, list):
        return []

    out: list[IbkrAccountConfig] = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        account_id = _as_str(it.get("account_id"))
        if not account_id:
            continue
        display_name = _as_str(it.get("display_name")) or account_id
        host = _as_str(it.get("host")) or "127.0.0.1"
        port = _as_int(it.get("port"), 4002)
        client_id = _as_int(it.get("client_id"), 1)
        out.append(
            IbkrAccountConfig(
                account_id=account_id,
                display_name=display_name,
                host=host,
                port=port,
                client_id=client_id,
            )
        )
    return out

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

# The live logistic meta model file (2nd stage)
LIVE_LR_MODEL_FILE = "flagship_alpha_mom_live_lr.joblib"
LIVE_LR_MODEL_PATH = MODEL_DIR / LIVE_LR_MODEL_FILE

# Output path for daily signals
DAILY_SIGNAL_FILE = SIGNAL_DIR / "daily_signal.parquet"

# --- Universe ---
# Path to the static universe file if used, or None for dynamic
# If None, it will rely on what's in the lab or postgres
UNIVERSE_FILE = None 
