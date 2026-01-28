from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import polars as pl

import flagship.trading.intraday.intraday_runner as intraday_runner
from flagship.strategy import registry as strategy_registry
from flagship.strategy.registry import StrategySpec
from flagship.trading.intraday.intraday_runner import Direction, LiveEngine, Offset


@dataclass(frozen=True)
class DummyOrder:
    symbol: str
    side: str
    status: str


class DummyClient:
    def __init__(self, orders: list[DummyOrder]) -> None:
        self._orders = orders

    def get_orders(self, **_kwargs):
        return self._orders


class DummyAdapter:
    def __init__(self, positions: dict[str, int], client: DummyClient) -> None:
        self._positions = positions
        self.client = client
        self.placed: list[tuple[str, int, object]] = []

    def get_positions(self) -> dict[str, int]:
        return self._positions

    def place_order(self, symbol: str, qty: int, side: object) -> None:
        self.placed.append((symbol, qty, side))


class DummyOrderSide:
    BUY = "buy"
    SELL = "sell"


class DummyStrategy:
    def __init__(self, strategy_engine, strategy_name, vt_symbols, setting) -> None:
        self.strategy_engine = strategy_engine
        self.strategy_name = strategy_name
        self.vt_symbols = vt_symbols
        self.setting = setting
        self.inited = False

    def on_init(self) -> None:
        self.inited = True


def test_send_order_skips_when_open_sell_order(monkeypatch) -> None:
    monkeypatch.setattr(intraday_runner, "OrderSide", DummyOrderSide)
    orders = [DummyOrder(symbol="FLY", side="sell", status="new")]
    adapter = DummyAdapter({"FLY": 100}, DummyClient(orders))
    engine = LiveEngine(adapter=adapter, signal_df=pl.DataFrame(), mode="exit-only")

    engine.send_order(None, "FLY.NASDAQ", Direction.SHORT, Offset.CLOSE, price=0.0, volume=50)

    assert adapter.placed == []
    assert "FLY" in engine.pending_exit_roots


def test_send_order_skips_when_in_cooldown(monkeypatch) -> None:
    monkeypatch.setattr(intraday_runner, "OrderSide", DummyOrderSide)
    adapter = DummyAdapter({"FLY": 100}, DummyClient([]))
    engine = LiveEngine(adapter=adapter, signal_df=pl.DataFrame(), mode="exit-only")
    engine.pending_exit_roots["FLY"] = datetime.now(timezone.utc)

    engine.send_order(None, "FLY.NASDAQ", Direction.SHORT, Offset.CLOSE, price=0.0, volume=50)

    assert adapter.placed == []


def test_send_order_places_sell_when_clear(monkeypatch) -> None:
    monkeypatch.setattr(intraday_runner, "OrderSide", DummyOrderSide)
    adapter = DummyAdapter({"FLY": 100}, DummyClient([]))
    engine = LiveEngine(adapter=adapter, signal_df=pl.DataFrame(), mode="exit-only")

    engine.send_order(None, "FLY.NASDAQ", Direction.SHORT, Offset.CLOSE, price=0.0, volume=50)

    assert adapter.placed == [("FLY", 50, DummyOrderSide.SELL)]


def test_build_strategy_uses_registry(monkeypatch) -> None:
    spec = StrategySpec(
        strategy_class=DummyStrategy,
        default_name="DefaultName",
        default_setting={"top_n": 5},
    )

    monkeypatch.setattr(strategy_registry, "get_strategy_spec", lambda _: spec)
    strategy = strategy_registry.build_strategy_instance(
        "v5",
        engine=object(),
        vt_symbols=["FLY.NASDAQ"],
        strategy_name_override="TestRunner",
        setting_override={"top_n": 8},
    )

    assert isinstance(strategy, DummyStrategy)
    assert strategy.inited is True
    assert strategy.strategy_name == "TestRunner"
    assert strategy.vt_symbols == ["FLY.NASDAQ"]
    assert strategy.setting == {"top_n": 8}
