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

# This file lives in flagship/scripts/research/, so project root is 3 levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
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

    # 统一报告目录命名（与 backtest 一致）
    report_folder_name = f"{regime.start.strftime('%Y%m%d')}_{regime.end.strftime('%Y%m%d')}_regime{args.regime_id:02d}"
    
    # Step 1: 准备数据集
    if not args.skip_prepare:
        prepare_script = PROJECT_ROOT / "flagship" / "scripts" / "research" / "prepare_filtered_dataset_for_lgb.py"
        if not prepare_script.exists():
            logger.warning(
                "[run_lgb_pipeline] prepare_filtered_dataset_for_lgb.py not found. "
                "Skip prepare step (assume dataset already exists)."
            )
        else:
            cmd = [
                sys.executable,
                str(prepare_script),
                "--lab-path",
                args.lab_path,
                "--regime-id",
                str(args.regime_id),
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

    # Step 2.5: 因子诊断（相关性/重要性）
    # 目的：验证特征共线性（独立性）以及模型依赖的特征重要性
    cmd = [
        sys.executable,
        "flagship/model/diagnose_factors.py",
        "--lab-path", args.lab_path,
        "--dataset-name", dataset_name,
        "--model-name", model_name,
        "--segment", "train",
        "--corr-mode", "cross_sectional_mean",
        "--output-path", str(Path(args.lab_path) / "report" / report_folder_name / "model_diagnostics.html"),
        "--llm-summary",
        "--llm-model", "gpt-5.2",
        "--llm-max-completion-tokens", "2000",
    ]
    if not run_command(cmd, "生成因子诊断报告"):
        logger.error("[run_lgb_pipeline] 因子诊断失败，终止流程")
        return
    
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

