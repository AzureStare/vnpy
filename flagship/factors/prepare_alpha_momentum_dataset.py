"""
生成 Flagship Alpha-Momentum 数据集并保存到 AlphaLab.dataset。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from vnpy.alpha import AlphaLab
from vnpy.trader.constant import Interval
from vnpy.trader.logger import logger

from flagship.factors.alpha_momentum.v5_dataset import FlagshipAlphaMomentumV5Dataset
from flagship.factors.alpha_momentum.v7_dataset import FlagshipAlphaMomentumV7Dataset
from flagship.universe.pg_ticker_db import get_selected_symbols_in_range

DEFAULT_VALID_DAYS = 60
DEFAULT_TEST_DAYS = 60
DEFAULT_GAP_DAYS = 7
DEFAULT_EXTENDED_DAYS = 120


@dataclass(frozen=True)
class PrepareConfig:
    lab_path: Path
    dataset_name: str
    strategy: str
    start_date: date
    end_date: date
    valid_days: int
    test_days: int
    gap_days: int
    extended_days: int
    max_workers: int | None


def _parse_date(value: str) -> date:
    return datetime.fromisoformat(value).date()


def _build_periods(cfg: PrepareConfig) -> tuple[tuple[str, str], tuple[str, str], tuple[str, str]]:
    assert cfg.start_date <= cfg.end_date, "start_date must be <= end_date"
    valid_days = max(int(cfg.valid_days), 1)
    test_days = max(int(cfg.test_days), 1)
    gap_days = max(int(cfg.gap_days), 0)

    total_days = (cfg.end_date - cfg.start_date).days + 1
    min_required = 2 * gap_days + 2
    if total_days < min_required:
        raise ValueError(
            f"区间过短：总天数={total_days}，至少需要 {min_required} 天才能完成切分"
        )

    required_non_train = valid_days + test_days + 2 * gap_days
    if required_non_train >= total_days:
        usable = total_days - 2 * gap_days - 1
        total_default = valid_days + test_days
        valid_days = max(1, int(round(usable * valid_days / total_default)))
        test_days = max(1, usable - valid_days)
        logger.warning(
            f"[prepare_dataset] 训练区间不足，已自动缩短 valid/test 窗口为 {valid_days}/{test_days} 天"
        )

    test_end = cfg.end_date
    test_start = test_end - timedelta(days=test_days - 1)

    valid_end = test_start - timedelta(days=gap_days)
    valid_start = valid_end - timedelta(days=valid_days - 1)

    train_end = valid_start - timedelta(days=gap_days)
    train_start = cfg.start_date

    if train_start > train_end:
        raise ValueError(
            f"训练区间不足：train_start={train_start} > train_end={train_end}，"
            f"请缩小 valid/test 窗口或扩大 start/end 范围"
        )
    if valid_start > valid_end:
        raise ValueError(f"VALID 区间异常: {valid_start} > {valid_end}")
    if test_start > test_end:
        raise ValueError(f"TEST 区间异常: {test_start} > {test_end}")

    train_period = (train_start.isoformat(), train_end.isoformat())
    valid_period = (valid_start.isoformat(), valid_end.isoformat())
    test_period = (test_start.isoformat(), test_end.isoformat())
    return train_period, valid_period, test_period


def _get_dataset_class(strategy: str):
    if strategy == "v5":
        return FlagshipAlphaMomentumV5Dataset
    if strategy == "v7":
        return FlagshipAlphaMomentumV7Dataset
    raise ValueError(f"Unsupported strategy: {strategy}")


def prepare_dataset(cfg: PrepareConfig) -> None:
    train_period, valid_period, test_period = _build_periods(cfg)
    logger.info(f"[prepare_dataset] Train Period: {train_period}")
    logger.info(f"[prepare_dataset] Valid Period: {valid_period}")
    logger.info(f"[prepare_dataset] Test Period: {test_period}")

    lab = AlphaLab(str(cfg.lab_path))
    vt_symbols = get_selected_symbols_in_range(start_date=cfg.start_date, end_date=cfg.end_date)
    if not vt_symbols:
        raise RuntimeError(
            f"[prepare_dataset] daily_selection 在 {cfg.start_date}~{cfg.end_date} 没有任何选股记录，"
            f"无法构建数据集。"
        )

    if "SPY.NASDAQ" not in vt_symbols:
        vt_symbols.append("SPY.NASDAQ")

    logger.info(f"[prepare_dataset] Universe symbols: {len(vt_symbols)}")

    data_start = cfg.start_date - timedelta(days=cfg.extended_days)
    data_end = cfg.end_date + timedelta(days=max(cfg.extended_days // 10, 10))

    raw_df = lab.load_bar_df(
        vt_symbols=vt_symbols,
        interval=Interval.DAILY,
        start=data_start.isoformat(),
        end=data_end.isoformat(),
        extended_days=0,
    )
    if raw_df is None or raw_df.is_empty():
        raise RuntimeError("[prepare_dataset] 无法加载日线数据")

    DatasetClass = _get_dataset_class(cfg.strategy)
    dataset = DatasetClass(
        df=raw_df,
        train_period=train_period,
        valid_period=valid_period,
        test_period=test_period,
    )

    dataset.prepare_data(filters=None, max_workers=cfg.max_workers)
    dataset.process_data()

    lab.save_dataset(cfg.dataset_name, dataset)
    logger.info(f"[prepare_dataset] Dataset saved: {cfg.dataset_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="准备 Alpha-Momentum 数据集")
    parser.add_argument("--lab-path", type=str, default="lab/flagship_alpha_momentum")
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--strategy", type=str, choices=["v5", "v7"], default="v7")
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--valid-days", type=int, default=DEFAULT_VALID_DAYS)
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS)
    parser.add_argument("--gap-days", type=int, default=DEFAULT_GAP_DAYS)
    parser.add_argument("--extended-days", type=int, default=DEFAULT_EXTENDED_DAYS)
    parser.add_argument("--max-workers", type=int, default=None)

    args = parser.parse_args()

    cfg = PrepareConfig(
        lab_path=Path(args.lab_path),
        dataset_name=str(args.dataset_name),
        strategy=str(args.strategy),
        start_date=_parse_date(args.start),
        end_date=_parse_date(args.end),
        valid_days=int(args.valid_days),
        test_days=int(args.test_days),
        gap_days=int(args.gap_days),
        extended_days=int(args.extended_days),
        max_workers=args.max_workers if args.max_workers else None,
    )
    prepare_dataset(cfg)


if __name__ == "__main__":
    main()
