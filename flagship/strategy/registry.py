from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Type

from vnpy.alpha.strategy import AlphaStrategy

from flagship.strategy.alpha_momentum_v5 import AlphaMomentumV5
from flagship.strategy.alpha_momentum_v7 import AlphaMomentumV7


@dataclass(frozen=True)
class StrategySpec:
    strategy_class: Type[AlphaStrategy]
    default_name: str
    default_setting: dict[str, Any]


STRATEGY_REGISTRY: dict[str, StrategySpec] = {
    "v5": StrategySpec(
        strategy_class=AlphaMomentumV5,
        default_name="Live_Flagship_V5",
        default_setting={"top_n": 5},
    ),
    "v7": StrategySpec(
        strategy_class=AlphaMomentumV7,
        default_name="Live_Flagship_V7_Aggressive",
        default_setting={"top_n": 8},
    ),
}


def get_strategy_spec(strategy_name: str) -> StrategySpec:
    key = strategy_name.strip().lower()
    if key in STRATEGY_REGISTRY:
        return STRATEGY_REGISTRY[key]
    raise ValueError(f"Unknown strategy version: {strategy_name!r}")


def get_strategy_class(strategy_name: str) -> Type[AlphaStrategy]:
    return get_strategy_spec(strategy_name).strategy_class


def build_strategy_instance(
    strategy_name: str,
    *,
    engine: Any,
    vt_symbols: list[str],
    strategy_name_override: str | None = None,
    setting_override: dict[str, Any] | None = None,
) -> AlphaStrategy:
    spec = get_strategy_spec(strategy_name)
    setting = dict(spec.default_setting)
    if setting_override:
        setting.update(setting_override)
    strategy = spec.strategy_class(
        strategy_engine=engine,  # type: ignore[arg-type]
        strategy_name=strategy_name_override or spec.default_name,
        vt_symbols=vt_symbols,
        setting=setting,
    )
    strategy.on_init()
    return strategy
