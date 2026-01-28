"""
LightGBM 训练与回测的端到端流程脚本（非 Regime）。

完整流程：
1. 构建 daily_selection（可选）
2. 准备 AlphaDataset（可选）
3. 训练 LightGBM 模型
4. 生成因子诊断报告
5. 可选：运行回测
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from flagship.config import PROJECT_ROOT
from vnpy.trader.logger import logger


def _run_command(cmd: list[str], description: str) -> None:
    """运行命令，失败直接抛错（fail fast）。"""
    logger.info(f"[run_lgb_pipeline] {description}...")
    logger.info(f"[run_lgb_pipeline] 执行命令: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, capture_output=False)
    logger.info(f"[run_lgb_pipeline] {description}完成")


def _build_selection_cmd(
    *,
    lab_path: str,
    start_date: str,
    end_date: str,
    strategy: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "flagship.universe.build_daily_selection",
        "--lab-path",
        lab_path,
        "--start",
        start_date,
        "--end",
        end_date,
        "--strategy",
        strategy,
    ]


def _build_dataset_cmd(
    *,
    lab_path: str,
    dataset_name: str,
    start_date: str,
    end_date: str,
    strategy: str,
    valid_days: int,
    test_days: int,
    gap_days: int,
    extended_days: int,
    max_workers: int | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "flagship/factors/prepare_alpha_momentum_dataset.py",
        "--lab-path",
        lab_path,
        "--dataset-name",
        dataset_name,
        "--strategy",
        strategy,
        "--start",
        start_date,
        "--end",
        end_date,
        "--valid-days",
        str(valid_days),
        "--test-days",
        str(test_days),
        "--gap-days",
        str(gap_days),
        "--extended-days",
        str(extended_days),
    ]
    if max_workers is not None:
        cmd.extend(["--max-workers", str(max_workers)])
    return cmd


def _build_train_cmd(
    *,
    lab_path: str,
    dataset_name: str,
    model_name: str,
    signal_name: str,
) -> list[str]:
    return [
        sys.executable,
        "flagship/model/train_flagship_lgb.py",
        "--lab-path",
        lab_path,
        "--dataset-name",
        dataset_name,
        "--model-name",
        model_name,
        "--signal-name",
        signal_name,
        "--label-column",
        "rank_5d",
        "--signal-segment",
        "test",
    ]


def _build_diagnose_cmd(
    *,
    lab_path: str,
    dataset_name: str,
    model_name: str,
    report_folder_name: str,
    llm_summary: bool,
    llm_model: str,
    llm_max_completion_tokens: int,
) -> list[str]:
    report_path = str(Path(lab_path) / "report" / report_folder_name / "model_diagnostics.html")
    cmd = [
        sys.executable,
        "flagship/model/diagnose_factors.py",
        "--lab-path",
        lab_path,
        "--dataset-name",
        dataset_name,
        "--model-name",
        model_name,
        "--segment",
        "train",
        "--corr-mode",
        "cross_sectional_mean",
        "--output-path",
        report_path,
    ]
    if llm_summary:
        cmd.extend(
            [
                "--llm-summary",
                "--llm-model",
                llm_model,
                "--llm-max-completion-tokens",
                str(llm_max_completion_tokens),
            ]
        )
    return cmd


def _build_backtest_cmd(
    *,
    lab_path: str,
    start_date: str,
    end_date: str,
    signal_name: str,
    dataset_name: str,
    backtest_capital: float,
    use_postgres_selection: bool,
    strategy: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "flagship/backtest/flagship_alpha_momentum_backtest.py",
        "--lab-path",
        lab_path,
        "--start",
        start_date,
        "--end",
        end_date,
        "--signal-name",
        signal_name,
        "--interval",
        "minute",
        "--capital",
        str(int(backtest_capital)),
        "--dataset-name",
        dataset_name,
        "--strategy",
        strategy,
    ]
    if use_postgres_selection:
        cmd.append("--use-postgres-selection")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM 训练与回测端到端流程")
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="训练/回测起始日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="训练/回测结束日期 (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--lab-path",
        type=str,
        default="lab/flagship_alpha_momentum",
        help="AlphaLab 数据根目录",
    )
    parser.add_argument(
        "--use-postgres-selection",
        action="store_true",
        default=True,
        help="使用 PostgreSQL daily_selection 过滤股票",
    )
    parser.add_argument(
        "--selection-strategy",
        type=str,
        choices=["v5", "v7"],
        default="v7",
        help="构建 daily_selection 时使用的策略版本",
    )
    parser.add_argument(
        "--dataset-strategy",
        type=str,
        choices=["v5", "v7"],
        default=None,
        help="生成 AlphaDataset 使用的策略版本（默认与 selection 相同）",
    )
    parser.add_argument(
        "--skip-selection",
        action="store_true",
        help="跳过 daily_selection 构建步骤",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="兼容旧参数：同 --skip-selection",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="跳过 AlphaDataset 生成步骤",
    )
    parser.add_argument("--valid-days", type=int, default=60, help="VALID 窗口天数")
    parser.add_argument("--test-days", type=int, default=60, help="TEST 窗口天数")
    parser.add_argument("--gap-days", type=int, default=7, help="VALID/TEST 前的间隔天数")
    parser.add_argument("--extended-days", type=int, default=120, help="额外加载历史天数")
    parser.add_argument("--max-workers", type=int, default=None, help="特征计算并行进程数")
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="跳过因子诊断报告生成",
    )
    parser.add_argument(
        "--llm-summary",
        action="store_true",
        default=False,
        help="在诊断报告中启用 LLM 总结（需要 OpenAI key）",
    )
    parser.add_argument("--llm-model", type=str, default="gpt-5.2", help="LLM 模型名")
    parser.add_argument("--llm-max-completion-tokens", type=int, default=2000)
    parser.add_argument(
        "--backtest-strategy",
        type=str,
        choices=["v5", "v7"],
        default=None,
        help="回测策略版本（默认与 dataset 相同）",
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
        default=1_000_000.0,
        help="回测初始资金",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="数据集名称（默认按日期范围生成）",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="模型名称（默认按日期范围生成）",
    )
    parser.add_argument(
        "--signal-name",
        type=str,
        default=None,
        help="信号名称（默认按日期范围生成）",
    )
    args = parser.parse_args()

    start_date = args.start.strip()
    end_date = args.end.strip()
    date_tag = f"{start_date.replace('-', '')}_{end_date.replace('-', '')}"

    dataset_name = args.dataset_name or f"flagship_alpha_momentum_{date_tag}_lgb"
    model_name = args.model_name or f"flagship_alpha_momentum_{date_tag}_lgb"
    signal_name = args.signal_name or f"flagship_alpha_momentum_{date_tag}_lgb_signal"
    report_folder_name = f"{date_tag}_backtest"

    skip_selection = bool(args.skip_selection or args.skip_prepare)
    selection_script = PROJECT_ROOT / "flagship" / "universe" / "build_daily_selection.py"
    dataset_script = PROJECT_ROOT / "flagship" / "factors" / "prepare_alpha_momentum_dataset.py"
    dataset_strategy = args.dataset_strategy or args.selection_strategy
    backtest_strategy = args.backtest_strategy or dataset_strategy

    if not skip_selection:
        if not selection_script.exists():
            raise FileNotFoundError(
                "[run_lgb_pipeline] build_daily_selection.py 不存在，无法构建 daily_selection"
            )
        selection_cmd = _build_selection_cmd(
            lab_path=args.lab_path,
            start_date=start_date,
            end_date=end_date,
            strategy=args.selection_strategy,
        )
        _run_command(selection_cmd, "构建 daily_selection")
    else:
        logger.info("[run_lgb_pipeline] 跳过 daily_selection 构建步骤")

    if not args.skip_dataset:
        if not dataset_script.exists():
            raise FileNotFoundError(
                "[run_lgb_pipeline] prepare_alpha_momentum_dataset.py 不存在，无法生成数据集"
            )
        dataset_cmd = _build_dataset_cmd(
            lab_path=args.lab_path,
            dataset_name=dataset_name,
            start_date=start_date,
            end_date=end_date,
            strategy=dataset_strategy,
            valid_days=int(args.valid_days),
            test_days=int(args.test_days),
            gap_days=int(args.gap_days),
            extended_days=int(args.extended_days),
            max_workers=args.max_workers,
        )
        _run_command(dataset_cmd, "生成 AlphaDataset")
    else:
        logger.info("[run_lgb_pipeline] 跳过 AlphaDataset 生成步骤")

    if not args.skip_train:
        train_cmd = _build_train_cmd(
            lab_path=args.lab_path,
            dataset_name=dataset_name,
            model_name=model_name,
            signal_name=signal_name,
        )
        _run_command(train_cmd, "训练 LightGBM 模型")
    else:
        logger.info("[run_lgb_pipeline] 跳过训练步骤")

    if not args.skip_diagnostics:
        diagnose_cmd = _build_diagnose_cmd(
            lab_path=args.lab_path,
            dataset_name=dataset_name,
            model_name=model_name,
            report_folder_name=report_folder_name,
            llm_summary=bool(args.llm_summary),
            llm_model=str(args.llm_model),
            llm_max_completion_tokens=int(args.llm_max_completion_tokens),
        )
        _run_command(diagnose_cmd, "生成因子诊断报告")
    else:
        logger.info("[run_lgb_pipeline] 跳过因子诊断报告")

    if args.run_backtest:
        backtest_cmd = _build_backtest_cmd(
            lab_path=args.lab_path,
            start_date=start_date,
            end_date=end_date,
            signal_name=signal_name,
            dataset_name=dataset_name,
            backtest_capital=args.backtest_capital,
            use_postgres_selection=args.use_postgres_selection,
            strategy=backtest_strategy,
        )
        _run_command(backtest_cmd, "运行回测")

    logger.info(f"[run_lgb_pipeline] Pipeline 完成：{start_date} ~ {end_date}")


if __name__ == "__main__":
    main()
