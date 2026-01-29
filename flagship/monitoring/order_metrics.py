"""
订单错误分类与通用计数工具。
"""
from __future__ import annotations

try:
    from alpaca.common.exceptions import APIError as AlpacaAPIError  # type: ignore
except Exception:  # pragma: no cover
    AlpacaAPIError = None  # type: ignore


def classify_order_error(exc: Exception) -> tuple[str, bool]:
    """
    Returns: (reason, rejected)
    """
    if AlpacaAPIError and isinstance(exc, AlpacaAPIError):
        status = getattr(exc, "status_code", None)
        try:
            status_i = int(status) if status is not None else None
        except Exception:
            status_i = None
        if status_i is not None:
            if status_i >= 500:
                return "alpaca_5xx", False
            if status_i >= 400:
                return "alpaca_4xx", True
        return "alpaca_api_error", True
    return "exception", False
