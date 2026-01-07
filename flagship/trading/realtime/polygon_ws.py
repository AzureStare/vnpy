"""
Shared Polygon WebSocket helpers for paper trading scripts.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable

from vnpy.trader.logger import logger

_IMPORT_ERROR: str | None = None

try:
    from polygon.websocket import WebSocketClient  # type: ignore
    from polygon.websocket.models import Market  # type: ignore

    POLYGON_WS_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    WebSocketClient = None  # type: ignore
    Market = None  # type: ignore
    POLYGON_WS_AVAILABLE = False
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def polygon_ws_import_error() -> str | None:
    return _IMPORT_ERROR


@dataclass
class PolygonWS:
    """
    Small wrapper around polygon-api-client WebSocketClient.
    """

    client: Any

    @classmethod
    def create(cls, api_key: str, market: Any) -> "PolygonWS":
        if WebSocketClient is None:
            raise RuntimeError(f"Polygon WS not available: {_IMPORT_ERROR}")
        return cls(client=WebSocketClient(api_key=api_key, market=market, subscriptions=[]))

    def subscribe(self, *subs: str) -> None:
        self.client.subscribe(*subs)

    def run(self, handle_msg: Callable[[list[Any]], None]) -> None:
        self.client.run(handle_msg)

    def close(self) -> None:
        try:
            # polygon-api-client close is async, but .run() manages loop; best-effort.
            c = getattr(self.client, "close", None)
            if c:
                c()
        except Exception:
            pass


def log_ws_unavailable(prefix: str = "[PolygonWS]") -> None:
    detail = f" ({_IMPORT_ERROR})" if _IMPORT_ERROR else ""
    logger.warning(f"{prefix} polygon websocket client not available{detail}")


class PolygonTradePriceCache:
    """
    Lightweight Polygon WS trade price cache.

    - Subscribes `T.{symbol}` (trades) for a list of root tickers (e.g. "AAPL")
    - Maintains latest trade price per symbol
    - Best-effort: if Polygon WS client is unavailable, `start()` becomes a no-op
    """

    def __init__(self, api_key: str, symbols: list[str]) -> None:
        self.api_key = str(api_key or "").strip()
        self.symbols = sorted({str(s).strip() for s in symbols if str(s).strip()})
        self.latest_trade: dict[str, float] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._ws: PolygonWS | None = None

    def start(self) -> None:
        if self._running:
            return
        if not self.api_key:
            logger.warning("[PolygonWS] empty api_key, skipping ws subscription.")
            return
        if not self.symbols:
            logger.warning("[PolygonWS] empty symbols, skipping ws subscription.")
            return
        if (not POLYGON_WS_AVAILABLE) or (Market is None):
            log_ws_unavailable(prefix="[PolygonWS]")
            return

        self._running = True
        subs = [f"T.{sym}" for sym in self.symbols]
        self._ws = PolygonWS.create(api_key=self.api_key, market=Market.Stocks)  # type: ignore[arg-type]

        def _handle_msg(batch: list[Any]) -> None:
            if not self._running:
                return
            for m in batch:
                if getattr(m, "event_type", None) != "T":
                    continue
                sym = getattr(m, "symbol", None)
                px = getattr(m, "price", None)
                if not sym or px is None:
                    continue
                try:
                    self.latest_trade[str(sym)] = float(px)
                except Exception:
                    continue

        def _run() -> None:
            assert self._ws is not None
            try:
                self._ws.subscribe(*subs)
                logger.info(f"[PolygonWS] Subscribed {len(subs)} tickers (trades).")
                self._ws.run(_handle_msg)
            except Exception as exc:
                logger.warning(f"[PolygonWS] websocket stopped: {exc}")
            finally:
                try:
                    self._ws.close()
                except Exception:
                    pass
                logger.info("[PolygonWS] websocket closed.")

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass

    def get_price(self, symbol: str) -> float | None:
        return self.latest_trade.get(str(symbol or "").strip())



