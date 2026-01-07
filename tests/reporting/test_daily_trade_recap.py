from __future__ import annotations

from datetime import date
from pathlib import Path

from flagship.ops.reporting.daily_trade_recap import generate_trade_recap


def test_generate_trade_recap_writes_html_latest_and_index(tmp_path: Path) -> None:
    output_dir = tmp_path / "app"
    log_dir = tmp_path / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    trade_date = date(2026, 1, 5)

    # Minimal snapshots required by generator
    (output_dir / "orders.json").write_text(
        """
{
  "generated_at": "2026-01-05T21:55:03+00:00",
  "orders": [
    {
      "id": "buy-1",
      "symbol": "ASTS",
      "side": "buy",
      "qty": 10,
      "filled_qty": 10,
      "filled_avg_price": 10.0,
      "status": "filled",
      "order_type": "market",
      "submitted_at": "2026-01-05T14:30:00+00:00",
      "filled_at": "2026-01-05T14:30:05+00:00"
    },
    {
      "id": "sell-1",
      "symbol": "ASTS",
      "side": "sell",
      "qty": 10,
      "filled_qty": 10,
      "filled_avg_price": 11.0,
      "status": "filled",
      "order_type": "market",
      "submitted_at": "2026-01-05T16:16:00+00:00",
      "filled_at": "2026-01-05T16:16:05+00:00"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    (output_dir / "portfolio.json").write_text(
        """
{
  "generated_at": "2026-01-05T21:55:03+00:00",
  "account": {"cash": 1000, "equity": 1000, "buying_power": 1000, "status": "ACTIVE"},
  "positions": [
    {"symbol": "KYMR", "qty": 5, "market_value": 500, "avg_entry": 90, "unrealized_pnl": 50}
  ]
}
""".strip(),
        encoding="utf-8",
    )

    (log_dir / "intraday_runner_20260105.log").write_text(
        """
2026-01-05 11:16:01.089 | INFO | Logger | [IntradayRunner] ASTS.NASDAQ 盘中离场: EMA10 趋势止盈, 当前收益: 8.82%
""".strip(),
        encoding="utf-8",
    )

    recap_path = generate_trade_recap(trade_date=trade_date, output_dir=output_dir, log_dir=log_dir, strategy_version="v7")

    assert recap_path.exists()
    assert (output_dir / "trade_recap_latest.html").exists()
    assert (output_dir / "trade_recap_index.json").exists()

    html = recap_path.read_text(encoding="utf-8")
    assert "Flagship V7 交易复盘（2026-01-05）" in html
    assert "ASTS" in html
    assert "EMA10 趋势止盈" in html

    # realized pnl (11 - 10) * 10 = 10
    assert "$10.00" in html


