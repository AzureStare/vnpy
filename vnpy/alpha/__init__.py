from __future__ import annotations

from typing import Any

from .logger import logger
from .strategy import AlphaStrategy, BacktestingEngine
from .lab import AlphaLab

__all__ = [
    "logger",
    "AlphaDataset",
    "Segment",
    "to_datetime",
    "AlphaModel",
    "AlphaStrategy",
    "BacktestingEngine",
    "AlphaLab",
]


def __getattr__(name: str) -> Any:
    """
    Lazily import optional submodules (dataset/model) so that AlphaLab can be
    used without forcing heavy dependencies such as alphalens unless needed.
    """
    if name in {"AlphaDataset", "Segment", "to_datetime"}:
        from .dataset import AlphaDataset, Segment, to_datetime  # type: ignore

        globals().update(
            {
                "AlphaDataset": AlphaDataset,
                "Segment": Segment,
                "to_datetime": to_datetime,
            }
        )
        return globals()[name]

    if name == "AlphaModel":
        from .model import AlphaModel  # type: ignore

        globals()["AlphaModel"] = AlphaModel
        return AlphaModel

    raise AttributeError(f"module 'vnpy.alpha' has no attribute '{name}'")
