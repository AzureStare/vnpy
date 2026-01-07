from __future__ import annotations

import json
from pathlib import Path

from flagship.trading.intraday.intraday_daemon import IntradayDaemonState, _load_state, _save_state


def test_intraday_daemon_state_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "intraday_state.json"
    s0 = _load_state(p)
    assert isinstance(s0, IntradayDaemonState)
    assert s0.last_started_trade_date is None
    assert s0.child_pid is None
    assert s0.last_start_timestamp is None
    assert s0.last_exit_code is None
    assert s0.restarts_today == 0

    s1 = IntradayDaemonState(
        last_started_trade_date="2026-01-06",
        child_pid=1234,
        last_start_timestamp=111.0,
        last_exit_code=0,
        restarts_today=2,
    )
    _save_state(p, s1)
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw["last_started_trade_date"] == "2026-01-06"
    assert raw["child_pid"] == 1234
    assert raw["last_start_timestamp"] == 111.0
    assert raw["last_exit_code"] == 0
    assert raw["restarts_today"] == 2

    s2 = _load_state(p)
    assert s2.last_started_trade_date == "2026-01-06"
    assert s2.child_pid == 1234
    assert s2.last_start_timestamp == 111.0
    assert s2.last_exit_code == 0
    assert s2.restarts_today == 2


