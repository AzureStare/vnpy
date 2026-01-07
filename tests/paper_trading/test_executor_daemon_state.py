from __future__ import annotations

import json
from pathlib import Path

from flagship.trading.execution.executor_daemon import DaemonState, _load_state, _save_state


def test_daemon_state_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "state.json"
    s0 = _load_state(p)
    assert isinstance(s0, DaemonState)
    assert s0.last_rebalance_trade_date is None
    assert s0.last_rebalance_timestamp is None
    assert s0.last_signal_date is None

    s1 = DaemonState(
        last_rebalance_trade_date="2026-01-06",
        last_rebalance_timestamp=123.45,
        last_signal_date="2026-01-05",
    )
    _save_state(p, s1)
    assert p.exists()

    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw["last_rebalance_trade_date"] == "2026-01-06"
    assert raw["last_rebalance_timestamp"] == 123.45
    assert raw["last_signal_date"] == "2026-01-05"

    s2 = _load_state(p)
    assert s2.last_rebalance_trade_date == "2026-01-06"
    assert s2.last_rebalance_timestamp == 123.45
    assert s2.last_signal_date == "2026-01-05"


