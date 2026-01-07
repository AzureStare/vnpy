from __future__ import annotations

from flagship.trading.realtime import polygon_ws


def test_polygon_trade_price_cache_degrades_without_api_key() -> None:
    cache = polygon_ws.PolygonTradePriceCache(api_key="", symbols=["AAPL", "MSFT"])
    cache.start()  # should be a no-op, no exception
    assert cache.get_price("AAPL") is None


def test_polygon_trade_price_cache_degrades_with_empty_symbols() -> None:
    cache = polygon_ws.PolygonTradePriceCache(api_key="dummy", symbols=[])
    cache.start()  # should be a no-op, no exception
    assert cache.get_price("AAPL") is None


