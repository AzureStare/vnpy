"""
交易日历工具（Polygon 数据源）。

原则：
- **数据源统一**：交易日/节假日/提前收盘信息来自 Polygon（market status/holidays）。
- **Broker 只负责执行**：Alpaca/IBKR 等仅用于下单与账户信息，不用于交易日历判断。

本模块用于 Paper Trading 的日常编排：
- 判断某个日期是否为“市场关闭日”（周末或节假日 closed）
- 获取“提前收盘”信息（early close）
- 从 AlphaLab（Polygon 下载的指数日线）推断 DATA_DATE（上一交易日收盘）

说明：
- Polygon 的 holidays endpoint（/v1/marketstatus/upcoming）提供“近期”节假日/提前收盘，
  适合判断 **TRADING_DATE 是否休市/提前收盘**。
- DATA_DATE 不使用简单 T-1：我们用 AlphaLab 的 SPY/QQQ 日线 parquet 的 max(datetime) 推断，
  这天然兼容节假日/周末（并保持与 Polygon 数据一致）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dtime, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from vnpy.alpha.lab import AlphaLab
from vnpy.trader.logger import logger

from flagship.config.polygon_config import create_polygon_client

try:
    from polygon.rest.models.markets import MarketHoliday
except Exception:  # pragma: no cover
    MarketHoliday = None  # type: ignore


EASTERN = ZoneInfo("America/New_York")
DEFAULT_OPEN_ET = dtime(9, 30)
DEFAULT_CLOSE_ET = dtime(16, 0)


@dataclass(frozen=True)
class HolidayInfo:
    """
    Polygon market holiday entry.
    - status: usually 'closed' or 'early_close'/'early-close' etc.
    - open/close: time string (e.g. '09:30', '13:00') when provided
    """

    holiday_date: date
    exchange: str | None
    name: str | None
    status: str | None
    open_time: dtime | None
    close_time: dtime | None

    @property
    def is_closed(self) -> bool:
        s = (self.status or "").strip().lower()
        return s == "closed"

    @property
    def is_early_close(self) -> bool:
        s = (self.status or "").strip().lower().replace("-", "_").replace(" ", "_")
        return s in {"early_close", "earlyclose"}

    def session_open_eastern(self) -> datetime:
        t = self.open_time or DEFAULT_OPEN_ET
        return datetime.combine(self.holiday_date, t, tzinfo=EASTERN)

    def session_close_eastern(self) -> datetime:
        t = self.close_time or DEFAULT_CLOSE_ET
        return datetime.combine(self.holiday_date, t, tzinfo=EASTERN)


def _parse_hhmm(text: str | None) -> dtime | None:
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    # Common formats: HH:MM, HH:MM:SS
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).time()
        except Exception:
            continue
    return None


@lru_cache(maxsize=1)
def _load_upcoming_holidays() -> dict[date, HolidayInfo]:
    """
    Load upcoming holidays/early closes from Polygon.
    Cached per-process to avoid repeated API calls.
    """
    try:
        client = create_polygon_client()
    except Exception as exc:
        logger.warning(f"[trading_calendar] Polygon client unavailable, skip holidays: {exc}")
        return {}

    try:
        items = client.get_market_holidays()
    except Exception as exc:
        logger.warning(f"[trading_calendar] Polygon get_market_holidays failed: {exc}")
        return {}

    out: dict[date, HolidayInfo] = {}
    for it in items or []:
        try:
            d_str = getattr(it, "date", None)
            if not d_str:
                continue
            d = date.fromisoformat(str(d_str))
        except Exception:
            continue

        out[d] = HolidayInfo(
            holiday_date=d,
            exchange=getattr(it, "exchange", None),
            name=getattr(it, "name", None),
            status=getattr(it, "status", None),
            open_time=_parse_hhmm(getattr(it, "open", None)),
            close_time=_parse_hhmm(getattr(it, "close", None)),
        )
    return out


def get_holiday_info(d: date) -> HolidayInfo | None:
    return _load_upcoming_holidays().get(d)


def is_market_closed_day(d: date) -> bool:
    """
    True if market should be considered closed for trading execution:
    - weekend
    - Polygon holiday status=closed (if available)
    """
    if d.weekday() >= 5:
        return True
    info = get_holiday_info(d)
    if info and info.is_closed:
        return True
    return False


def infer_data_date_from_lab(lab_path: Path) -> date:
    """
    Infer DATA_DATE (上一交易日收盘) from index daily parquet.

    Strategy:
    - prefer SPY daily parquet max(datetime)
    - fallback to QQQ daily parquet max(datetime)
    - fallback to today-1 (calendar), with weekend adjustment
    """
    lab = AlphaLab(str(lab_path))

    def _find_index_file(symbol_root: str) -> Path | None:
        matches = sorted(lab.daily_path.glob(f"{symbol_root}.*.parquet"))
        return matches[0] if matches else None

    def _max_dt(p: Path) -> datetime | None:
        v = (
            pl.scan_parquet(p)
            .select(pl.col("datetime").max())
            .collect()
            .item()
        )
        return v if isinstance(v, datetime) else None

    for root in ("SPY", "QQQ"):
        p = _find_index_file(root)
        if not p or not p.exists():
            continue
        try:
            last_dt = _max_dt(p)
            if last_dt:
                return last_dt.date()
        except Exception as exc:
            logger.warning(f"[trading_calendar] infer DATA_DATE failed: {p}: {exc}")

    # Fallback: calendar yesterday, skip weekends
    d = date.today() - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def describe_trading_date(d: date) -> dict[str, str]:
    """
    Return a dict for logging/debug.
    """
    info = get_holiday_info(d)
    if info is None:
        return {
            "date": d.isoformat(),
            "market_closed": str(is_market_closed_day(d)),
            "holiday_status": "",
            "open": "",
            "close": "",
        }
    return {
        "date": d.isoformat(),
        "market_closed": str(is_market_closed_day(d)),
        "holiday_status": str(info.status or ""),
        "open": (info.open_time.strftime("%H:%M") if info.open_time else ""),
        "close": (info.close_time.strftime("%H:%M") if info.close_time else ""),
    }


