"""
Polygon data tools (download + quick checks).


"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl

from vnpy.alpha.lab import AlphaLab
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.logger import logger
from vnpy.trader.object import BarData
from vnpy.trader.setting import SETTINGS
from vnpy.trader.utility import ZoneInfo

from flagship.config import DEFAULT_LAB_DIR, DEFAULT_UNIVERSE_DIR, VT_SETTING_PATH, create_polygon_client

try:
    from flagship.universe.pg_ticker_db import get_ref_tickers

    PG_AVAILABLE = True
except Exception:
    PG_AVAILABLE = False


EXCHANGE_MAP = {
    "XNAS": Exchange.NASDAQ,
    "XNYS": Exchange.NYSE,
    "XASE": Exchange.AMEX,
    "ARCX": Exchange.NYSE,
    "BATS": Exchange.BATS,
    "IEXG": Exchange.IEX,
}


# VIX indices (Polygon uses "I:" prefix for indices)
VIX_INDICES = {
    "VIX": {
        "ticker": "I:VIX",
        "exchange": Exchange.CBOE,
        "name": "VIX 波动率指数",
    },
    "VIX3M": {
        "ticker": "I:VIX3M",
        "exchange": Exchange.CBOE,
        "name": "VIX 3个月期货指数",
    },
}


def _reload_vt_setting_if_present() -> None:
    if not VT_SETTING_PATH.exists():
        return
    try:
        setting_data = json.loads(VT_SETTING_PATH.read_text(encoding="utf-8"))
        SETTINGS.update(setting_data)
    except Exception as exc:
        logger.warning(f"[polygon_data_tools] 重新加载 vt_setting.json 失败: {exc}")


def download_vix_indices(
    start_date: date,
    end_date: date,
    lab_dir: Path = DEFAULT_LAB_DIR,
) -> None:
    """
    Download VIX and VIX3M daily bars from Polygon and save into AlphaLab.
    Output vt_symbols:
    - VIX.CBOE
    - VIX3M.CBOE
    """
    logger.info("[download_vix_indices] 开始下载 VIX 指数数据")
    logger.info(f"[download_vix_indices] 日期范围: {start_date} 到 {end_date}")

    lab = AlphaLab(str(lab_dir))
    client = create_polygon_client()

    success_count = 0
    fail_count = 0

    for idx_name, cfg in VIX_INDICES.items():
        ticker = str(cfg["ticker"])
        exchange: Exchange = cfg["exchange"]
        name = str(cfg["name"])
        vt_symbol = f"{idx_name}.{exchange.value}"

        try:
            logger.info(f"[download_vix_indices] 下载 {name} ({ticker})...")
            fetch_start = start_date - timedelta(days=30)

            try:
                aggs = client.get_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan="day",
                    from_=fetch_start.isoformat(),
                    to=end_date.isoformat(),
                    adjusted=False,
                    sort="asc",
                    limit=50000,
                )
            except Exception as exc:
                logger.warning(f"[download_vix_indices] {ticker}: Polygon API 调用失败: {exc}")
                if ticker.startswith("I:"):
                    alt_ticker = ticker[2:]
                    logger.info(f"[download_vix_indices] 尝试使用替代 ticker: {alt_ticker}")
                    try:
                        aggs = client.get_aggs(
                            ticker=alt_ticker,
                            multiplier=1,
                            timespan="day",
                            from_=fetch_start.isoformat(),
                            to=end_date.isoformat(),
                            adjusted=False,
                            sort="asc",
                            limit=50000,
                        )
                    except Exception as exc2:
                        logger.error(f"[download_vix_indices] {alt_ticker}: 也失败: {exc2}")
                        fail_count += 1
                        continue
                else:
                    fail_count += 1
                    continue

            if not aggs:
                logger.warning(f"[download_vix_indices] {ticker}: 未获取到数据")
                fail_count += 1
                continue

            utc_tz = timezone.utc
            eastern_tz = ZoneInfo("America/New_York")

            bars: list[BarData] = []
            for agg in aggs:
                utc_dt = datetime.fromtimestamp(agg.timestamp / 1000, tz=utc_tz)
                bar_dt = utc_dt.astimezone(eastern_tz).replace(tzinfo=None)
                d = bar_dt.date()
                if d < start_date or d > end_date:
                    continue

                bars.append(
                    BarData(
                        symbol=idx_name,
                        exchange=exchange,
                        datetime=bar_dt,
                        interval=Interval.DAILY,
                        open_price=float(getattr(agg, "open", 0) or 0.0),
                        high_price=float(getattr(agg, "high", 0) or 0.0),
                        low_price=float(getattr(agg, "low", 0) or 0.0),
                        close_price=float(getattr(agg, "close", 0) or 0.0),
                        volume=int(getattr(agg, "volume", 0) or 0),
                        turnover=0.0,
                        open_interest=0,
                        gateway_name="POLYGON",
                    )
                )

            if not bars:
                logger.warning(f"[download_vix_indices] {ticker}: 日期范围内无数据")
                fail_count += 1
                continue

            logger.info(f"[download_vix_indices] {ticker}: 保存 {len(bars)} 条数据到 {vt_symbol}")
            lab.save_bar_data(bars)
            success_count += 1
        except Exception as exc:
            logger.error(f"[download_vix_indices] {idx_name} 下载失败: {exc}", exc_info=True)
            fail_count += 1

    logger.info(f"[download_vix_indices] 下载完成: 成功={success_count}, 失败={fail_count}, 总计={len(VIX_INDICES)}")


def load_daily_universe(universe_file: Path) -> list[str]:
    if not universe_file.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_file}")
    data = json.loads(universe_file.read_text(encoding="utf-8"))
    symbols = [item["symbol"] for item in data.get("symbols", [])]
    logger.info(f"Loaded {len(symbols)} symbols from {universe_file}")
    return symbols


def get_exchange_for_symbol(symbol: str) -> Exchange:
    if PG_AVAILABLE:
        try:
            ref_tickers = get_ref_tickers(
                market="stocks",
                locale="us",
                ticker_type="CS",
                active=True,
            )
            for ticker in ref_tickers:
                if ticker.get("symbol") == symbol:
                    primary_exchange = ticker.get("primary_exchange", "")
                    return EXCHANGE_MAP.get(primary_exchange, Exchange.NASDAQ)
        except Exception as exc:
            logger.warning(f"Failed to query exchange for {symbol}: {exc}")
    return Exchange.NASDAQ


def download_bars_for_symbols(
    symbols: list[str],
    start_date: date,
    end_date: date,
    interval: Interval = Interval.DAILY,
    lab_dir: Path = DEFAULT_LAB_DIR,
) -> None:
    """
    Download equity bars for symbols and save into AlphaLab.
    """
    logger.info("[download_bars_for_symbols] 开始下载历史数据")
    logger.info(
        f"[download_bars_for_symbols] 股票数量: {len(symbols)}, 日期范围: {start_date} 到 {end_date}, 周期: {interval.value}"
    )

    lab = AlphaLab(str(lab_dir))
    client = create_polygon_client()

    success_count = 0
    fail_count = 0

    for idx, symbol in enumerate(symbols, start=1):
        try:
            exchange = get_exchange_for_symbol(symbol)
            # vt_symbol is decided by (symbol, exchange) inside BarData

            timespan = "day" if interval == Interval.DAILY else "minute"
            multiplier = 1

            fetch_start = start_date - timedelta(days=90)
            aggs = client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=fetch_start.isoformat(),
                to=end_date.isoformat(),
                adjusted=True,
                sort="asc",
                limit=50000,
            )

            if not aggs:
                logger.warning(f"[download_bars_for_symbols] {symbol}: 未获取到数据")
                fail_count += 1
                continue

            utc_tz = timezone.utc
            eastern_tz = ZoneInfo("America/New_York")

            bars: list[BarData] = []
            for agg in aggs:
                utc_dt = datetime.fromtimestamp(agg.timestamp / 1000, tz=utc_tz)
                bar_dt = utc_dt.astimezone(eastern_tz).replace(tzinfo=None)

                raw_open = getattr(agg, "open", None)
                raw_high = getattr(agg, "high", None)
                raw_low = getattr(agg, "low", None)
                raw_close = getattr(agg, "close", None)
                open_val = float(raw_open) if raw_open is not None else 0.0
                high_val = float(raw_high) if raw_high is not None else 0.0
                low_val = float(raw_low) if raw_low is not None else 0.0
                close_val = float(raw_close) if raw_close is not None else 0.0

                raw_volume = getattr(agg, "volume", 0) or 0
                try:
                    volume_int = int(raw_volume)
                except Exception:
                    volume_int = 0

                bars.append(
                    BarData(
                        symbol=symbol,
                        exchange=exchange,
                        datetime=bar_dt,
                        interval=interval,
                        open_price=open_val,
                        high_price=high_val,
                        low_price=low_val,
                        close_price=close_val,
                        volume=volume_int,
                        turnover=float(volume_int) * close_val,
                        open_interest=0,
                        gateway_name="POLYGON",
                    )
                )

            if not bars:
                logger.warning(f"[download_bars_for_symbols] {symbol}: 日期范围内无数据")
                fail_count += 1
                continue

            lab.save_bar_data(bars)
            success_count += 1
            if idx % 50 == 0:
                logger.info(f"[download_bars_for_symbols] 进度 {idx}/{len(symbols)}: 成功={success_count}, 失败={fail_count}")

        except Exception as exc:
            logger.error(f"[download_bars_for_symbols] {symbol} 下载失败: {exc}")
            fail_count += 1

    logger.info(f"[download_bars_for_symbols] 下载完成: 成功={success_count}, 失败={fail_count}, 总计={len(symbols)}")


def check_vix_data(lab_dir: Path = DEFAULT_LAB_DIR) -> dict[str, Any]:
    """
    Quick local sanity check for VIX/VIX3M parquet files (no API call).
    Returns a dict useful for printing/logging.
    """
    daily_dir = Path(lab_dir) / "daily"
    vix_file = daily_dir / "VIX.CBOE.parquet"
    vix3m_file = daily_dir / "VIX3M.CBOE.parquet"

    out: dict[str, Any] = {
        "daily_dir": str(daily_dir),
        "vix_exists": vix_file.exists(),
        "vix3m_exists": vix3m_file.exists(),
    }

    if vix_file.exists():
        df = pl.read_parquet(vix_file)
        out["vix_rows"] = len(df)
        out["vix_min_dt"] = str(df["datetime"].min())
        out["vix_max_dt"] = str(df["datetime"].max())
    if vix3m_file.exists():
        df = pl.read_parquet(vix3m_file)
        out["vix3m_rows"] = len(df)
        out["vix3m_min_dt"] = str(df["datetime"].min())
        out["vix3m_max_dt"] = str(df["datetime"].max())

    if vix_file.exists() and vix3m_file.exists():
        df_vix = pl.read_parquet(vix_file)
        df_vix3m = pl.read_parquet(vix3m_file)
        merged = (
            df_vix.join(df_vix3m, on="datetime", how="inner", suffix="_vix3m")
            .with_columns((pl.col("close") / pl.col("close_vix3m")).alias("vix_ratio"))
            .sort("datetime")
        )
        out["ratio_rows"] = len(merged)
        if len(merged) > 0:
            out["ratio_mean"] = float(merged["vix_ratio"].mean())
            out["ratio_min"] = float(merged["vix_ratio"].min())
            out["ratio_max"] = float(merged["vix_ratio"].max())
            out["ratio_median"] = float(merged["vix_ratio"].median())
    return out


def _cmd_download_vix(args: argparse.Namespace) -> None:
    _reload_vt_setting_if_present()
    download_vix_indices(
        start_date=datetime.fromisoformat(args.start).date(),
        end_date=datetime.fromisoformat(args.end).date(),
        lab_dir=Path(args.lab_dir),
    )


def _cmd_check_vix(args: argparse.Namespace) -> None:
    info = check_vix_data(lab_dir=Path(args.lab_dir))
    print(json.dumps(info, ensure_ascii=False, indent=2))


def _cmd_download_backtest(args: argparse.Namespace) -> None:
    _reload_vt_setting_if_present()

    start_date = datetime.fromisoformat(args.start).date()
    end_date = datetime.fromisoformat(args.end).date()
    interval = Interval.DAILY if args.interval == "daily" else Interval.MINUTE

    all_symbols: set[str] = set()
    if args.universe_file:
        all_symbols.update(load_daily_universe(Path(args.universe_file)))
    elif args.universe_dir:
        universe_dir = Path(args.universe_dir)
        files = sorted(universe_dir.glob("universe_*.json"))
        logger.info(f"Found {len(files)} universe files")
        for f in files:
            all_symbols.update(load_daily_universe(f))
    elif args.symbols:
        all_symbols.update([s.strip().upper() for s in str(args.symbols).split(",") if s.strip()])
    else:
        raise ValueError("Need one of --symbols / --universe-file / --universe-dir")

    logger.info(f"Total unique symbols to download: {len(all_symbols)}")
    download_bars_for_symbols(
        symbols=sorted(all_symbols),
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        lab_dir=Path(args.lab_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Flagship Polygon data tools (download/check).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_vix = sub.add_parser("download-vix", help="Download VIX/VIX3M daily bars into AlphaLab.")
    p_vix.add_argument("--start", required=True, help="起始日期 (YYYY-MM-DD)")
    p_vix.add_argument("--end", required=True, help="结束日期 (YYYY-MM-DD)")
    p_vix.add_argument("--lab-dir", default=str(DEFAULT_LAB_DIR), help="AlphaLab 数据目录")
    p_vix.set_defaults(func=_cmd_download_vix)

    p_check = sub.add_parser("check-vix", help="Check local VIX/VIX3M parquet files in AlphaLab.")
    p_check.add_argument("--lab-dir", default=str(DEFAULT_LAB_DIR), help="AlphaLab 数据目录")
    p_check.set_defaults(func=_cmd_check_vix)

    p_bt = sub.add_parser("download-backtest", help="Download bars for backtest universe/symbols.")
    p_bt.add_argument("--start", required=True, help="起始日期 (YYYY-MM-DD)")
    p_bt.add_argument("--end", required=True, help="结束日期 (YYYY-MM-DD)")
    p_bt.add_argument("--interval", choices=["daily", "minute"], default="daily", help="K线周期（默认 daily）")
    p_bt.add_argument("--lab-dir", default=str(DEFAULT_LAB_DIR), help="AlphaLab 数据目录")
    p_bt.add_argument("--symbols", help="逗号分隔股票列表（与 universe 参数互斥）")
    p_bt.add_argument("--universe-file", help="每日股票池 JSON 文件路径")
    p_bt.add_argument("--universe-dir", default=str(DEFAULT_UNIVERSE_DIR), help="股票池目录（默认 flagship/data/universe）")
    p_bt.set_defaults(func=_cmd_download_backtest)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


