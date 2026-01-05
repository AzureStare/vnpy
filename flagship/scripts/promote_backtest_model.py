import shutil
import os
from pathlib import Path
from datetime import datetime
from vnpy.trader.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_MODEL_DIR = PROJECT_ROOT / "flagship" / "models" / "backtest_v7_rolling"
LIVE_MODEL_DIR = PROJECT_ROOT / "lab" / "flagship_alpha_momentum" / "model"
LIVE_MODEL_PATH = LIVE_MODEL_DIR / "live_model.joblib"

def promote_latest_backtest_model():
    """找到最新的回测模型并设置为实盘模型"""
    if not BACKTEST_MODEL_DIR.exists():
        print(f"Error: Backtest model directory not found: {BACKTEST_MODEL_DIR}")
        return False

    # 获取所有模型文件并按日期排序
    model_files = sorted(BACKTEST_MODEL_DIR.glob("model_*.pkl"))
    if not model_files:
        print("No backtest models found to promote.")
        return False

    latest_model = model_files[-1]
    print(f"Latest backtest model found: {latest_model.name}")

    # 确保目标目录存在
    LIVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 备份旧的实盘模型（如果有）
    if LIVE_MODEL_PATH.exists():
        backup_path = LIVE_MODEL_PATH.with_suffix(f".joblib.bak_{datetime.now().strftime('%Y%m%d')}")
        shutil.copy(LIVE_MODEL_PATH, backup_path)
        print(f"Backed up current live model to {backup_path.name}")

    # 复制回测模型到实盘路径
    shutil.copy(latest_model, LIVE_MODEL_PATH)
    print(f"SUCCESS: Promoted {latest_model.name} to {LIVE_MODEL_PATH}")
    return True

if __name__ == "__main__":
    promote_latest_backtest_model()

