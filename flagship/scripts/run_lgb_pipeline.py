"""
LightGBM训练和回测的端到端流程脚本。

完整流程：
1. 准备数据集（prepare_filtered_dataset_for_lgb.py）
2. 训练LightGBM模型（train_flagship_lgb.py）
3. 可选：运行回测（flagship_alpha_momentum_backtest.py）
"""
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from vnpy.trader.logger import logger
from flagship.backtest.index_regime_windows import get_regime_window


def run_command(cmd: list[str], description: str) -> bool:
    """运行命令并返回是否成功"""
    logger.info(f"[run_lgb_pipeline] {description}...")
    logger.info(f"[run_lgb_pipeline] 执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=False,
        )
        logger.info(f"[run_lgb_pipeline] {description}完成")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[run_lgb_pipeline] {description}失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="LightGBM训练和回测的端到端流程"
    )
    parser.add_argument(
        "--regime-id",
        type=int,
        required=True,
        help="Regime编号（1-10）",
    )
    parser.add_argument(
        "--lab-path",
        type=str,
        default="lab/flagship_alpha_momentum",
        help="AlphaLab数据根目录",
    )
    parser.add_argument(
        "--use-postgres-selection",
        action="store_true",
        default=True,
        help="使用PostgreSQL daily_selection过滤股票",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="跳过数据准备步骤（如果数据集已存在）",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="跳过训练步骤（如果模型已存在）",
    )
    parser.add_argument(
        "--run-backtest",
        action="store_true",
        help="训练完成后运行回测",
    )
    parser.add_argument(
        "--backtest-capital",
        type=float,
        default=1000000.0,
        help="回测初始资金",
    )
    args = parser.parse_args()
    
    regime = get_regime_window(args.regime_id)
    logger.info(f"[run_lgb_pipeline] 开始处理 Regime {args.regime_id}: {regime.label}")
    logger.info(f"[run_lgb_pipeline] 日期范围: {regime.start} ~ {regime.end}")
    
    dataset_name = f"flagship_alpha_mom_regime{args.regime_id:02d}_lgb"
    model_name = f"flagship_alpha_mom_regime{args.regime_id:02d}_lgb"
    signal_name = f"flagship_alpha_mom_regime{args.regime_id:02d}_lgb_signal"
    
    # Step 1: 准备数据集
    if not args.skip_prepare:
        cmd = [
            sys.executable,
            "flagship/scripts/prepare_filtered_dataset_for_lgb.py",
            "--lab-path", args.lab_path,
            "--regime-id", str(args.regime_id),
        ]
        if args.use_postgres_selection:
            cmd.append("--use-postgres-selection")
        
        if not run_command(cmd, "准备数据集"):
            logger.error("[run_lgb_pipeline] 数据准备失败，终止流程")
            return
    else:
        logger.info("[run_lgb_pipeline] 跳过数据准备步骤")
    
    # Step 2: 训练LightGBM模型
    if not args.skip_train:
        cmd = [
            sys.executable,
            "flagship/model/train_flagship_lgb.py",
            "--lab-path", args.lab_path,
            "--dataset-name", dataset_name,
            "--model-name", model_name,
            "--signal-name", signal_name,
            "--regime-id", str(args.regime_id),
            "--label-column", "rank_5d",
            "--signal-segment", "test",
        ]
        
        if not run_command(cmd, "训练LightGBM模型"):
            logger.error("[run_lgb_pipeline] 模型训练失败，终止流程")
            return
    else:
        logger.info("[run_lgb_pipeline] 跳过训练步骤")
    
    # Step 3: 可选回测
    if args.run_backtest:
        cmd = [
            sys.executable,
            "flagship/backtest/flagship_alpha_momentum_backtest.py",
            "--lab-path", args.lab_path,
            "--start", regime.start.isoformat(),
            "--end", regime.end.isoformat(),
            "--signal-name", signal_name,
            "--interval", "minute",
            "--capital", str(int(args.backtest_capital)),
        ]
        
        if args.use_postgres_selection:
            cmd.append("--use-postgres-selection")
        
        run_command(cmd, "运行回测")
    
    logger.info(f"[run_lgb_pipeline] Regime {args.regime_id} 流程完成")


if __name__ == "__main__":
    main()

