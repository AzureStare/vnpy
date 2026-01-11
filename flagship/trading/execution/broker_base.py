from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class AccountInfo:
    cash: float
    equity: float
    buying_power: float


@runtime_checkable
class BrokerAdapter(Protocol):
    """
    Minimal broker interface needed by open rebalance + snapshots.

    Notes:
    - Positions are returned as root symbols (e.g. AAPL, MSFT).
    - All monetary values are USD floats.
    """

    def get_account_id(self) -> str: ...
    def get_display_name(self) -> str: ...
    def get_broker_name(self) -> str: ...  # e.g. "alpaca" / "ibkr"
    def get_env_name(self) -> str: ...  # e.g. "paper" / "live"

    def get_account_info(self) -> AccountInfo: ...
    def get_buying_power(self) -> float: ...
    def get_cash(self) -> float: ...
    def get_positions(self) -> dict[str, int]: ...

    def cancel_all_open_orders(self) -> None: ...
    def get_last_trade_price(self, symbol: str) -> float: ...
    def place_order(self, symbol: str, qty: int, side: OrderSide) -> None: ...

