from datetime import date
from pathlib import Path

import pytest

from flagship.factors.prepare_alpha_momentum_dataset import PrepareConfig, _build_periods


def test_build_periods_fixed_default_windows_2025_full_year() -> None:
    """
    说明性用例：用你常跑的全年区间，验证 fixed 模式下切分边界与 gap 生效。
    默认：valid=60d, test=60d, gap=7d。
    """
    cfg = PrepareConfig(
        lab_path=Path("lab/flagship_alpha_momentum"),
        dataset_name="dummy",
        strategy="v7",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
        split_mode="fixed",
        valid_days=60,
        test_days=60,
        valid_ratio=0.15,
        test_ratio=0.15,
        gap_days=7,
        extended_days=120,
        max_workers=None,
    )

    train_period, valid_period, test_period = _build_periods(cfg)
    assert train_period == ("2025-01-01", "2025-08-21")
    assert valid_period == ("2025-08-28", "2025-10-26")
    assert test_period == ("2025-11-02", "2025-12-31")


def test_build_periods_ratio_default_ratios_2025_full_year() -> None:
    """
    说明性用例：ratio 模式默认 valid/test 各 15%，train 吃掉剩余。
    gap 仍然是 7 天（按日历天）。
    """
    cfg = PrepareConfig(
        lab_path=Path("lab/flagship_alpha_momentum"),
        dataset_name="dummy",
        strategy="v7",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
        split_mode="ratio",
        valid_days=60,
        test_days=60,
        valid_ratio=0.15,
        test_ratio=0.15,
        gap_days=7,
        extended_days=120,
        max_workers=None,
    )

    train_period, valid_period, test_period = _build_periods(cfg)
    # Python round() 的 .5 采用 bankers rounding，这里会得到 52 天
    assert train_period == ("2025-01-01", "2025-09-06")
    assert valid_period == ("2025-09-13", "2025-11-03")
    assert test_period == ("2025-11-10", "2025-12-31")


def test_build_periods_ratio_rejects_invalid_ratios() -> None:
    cfg = PrepareConfig(
        lab_path=Path("lab/flagship_alpha_momentum"),
        dataset_name="dummy",
        strategy="v7",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
        split_mode="ratio",
        valid_days=60,
        test_days=60,
        valid_ratio=0.0,
        test_ratio=0.15,
        gap_days=7,
        extended_days=120,
        max_workers=None,
    )
    with pytest.raises(ValueError):
        _build_periods(cfg)

