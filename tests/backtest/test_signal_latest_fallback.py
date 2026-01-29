from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import polars as pl

from vnpy.alpha import AlphaLab


def _write_signal_parquet(path: Path, *, vt_symbol: str, signal_value: float) -> None:
    df = pl.DataFrame(
        {
            # 用 Python datetime，避免 polars 表达式对象写 parquet 引发 dtype=object
            "datetime": [datetime(2025, 1, 2, 0, 0, 0)],
            "vt_symbol": [vt_symbol],
            "signal": [signal_value],
        }
    )
    df.write_parquet(path)


def test_load_signal_fallback_to_latest(tmp_path: Path) -> None:
    from flagship.backtest.flagship_alpha_momentum_backtest import load_signal

    lab = AlphaLab(str(tmp_path))
    assert lab.signal_path.exists()

    older = lab.signal_path / "older_signal.parquet"
    newer = lab.signal_path / "newer_signal.parquet"

    _write_signal_parquet(older, vt_symbol="AAA.NASDAQ", signal_value=1.0)
    _write_signal_parquet(newer, vt_symbol="BBB.NASDAQ", signal_value=2.0)

    # 强制设置 mtime，确保 newer 更新更晚
    now = time.time()
    os.utime(older, (now - 10, now - 10))
    os.utime(newer, (now, now))

    df = load_signal(lab, "does_not_exist")
    assert df.select(pl.col("vt_symbol").unique()).to_series().to_list() == ["BBB.NASDAQ"]
    assert df.select(pl.col("signal").unique()).to_series().to_list() == [2.0]


def test_load_signal_prefers_exact_name_when_exists(tmp_path: Path) -> None:
    from flagship.backtest.flagship_alpha_momentum_backtest import load_signal

    lab = AlphaLab(str(tmp_path))
    exact = lab.signal_path / "exact.parquet"
    other = lab.signal_path / "other.parquet"

    _write_signal_parquet(exact, vt_symbol="AAA.NASDAQ", signal_value=1.0)
    _write_signal_parquet(other, vt_symbol="BBB.NASDAQ", signal_value=2.0)

    # 让 other 更新更晚，但依然应该按 name 精确取 exact
    now = time.time()
    os.utime(exact, (now - 10, now - 10))
    os.utime(other, (now, now))

    df = load_signal(lab, "exact")
    assert df.select(pl.col("vt_symbol").unique()).to_series().to_list() == ["AAA.NASDAQ"]
    assert df.select(pl.col("signal").unique()).to_series().to_list() == [1.0]


def test_load_signal_raises_when_folder_empty(tmp_path: Path) -> None:
    import pytest

    from flagship.backtest.flagship_alpha_momentum_backtest import load_signal

    lab = AlphaLab(str(tmp_path))
    with pytest.raises(RuntimeError):
        load_signal(lab, None)

