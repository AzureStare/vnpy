"""
Shared Polygon WebSocket helpers for paper trading scripts.
"""

from __future__ import annotations

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



