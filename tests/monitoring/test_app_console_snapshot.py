from datetime import date, datetime

import polars as pl

from flagship.monitoring.app_console_snapshot import build_selection_rows


def test_build_selection_rows_sorts_by_signal_desc_and_limits_top_n():
    # signals with mixed scores (including None)
    df = pl.DataFrame(
        {
            "datetime": [
                datetime(2025, 1, 1),
                datetime(2025, 1, 1),
                datetime(2025, 1, 1),
                datetime(2025, 1, 1),
            ],
            "vt_symbol": ["AAA.NASDAQ", "BBB.NASDAQ", "CCC.NASDAQ", "ZZZ.NASDAQ"],
            "signal": [0.2, None, 0.8, 9.9],
            "as_of_date": [date(2025, 1, 1)] * 4,
        }
    )

    selection_map = {
        "AAA.NASDAQ": {"close_price": 10.0, "adv_usd": 1e6, "med_volume": 1000},
        "BBB.NASDAQ": {"close_price": 20.0, "adv_usd": 2e6, "med_volume": 2000},
        "CCC.NASDAQ": {"close_price": 30.0, "adv_usd": 3e6, "med_volume": 3000},
    }

    rows = build_selection_rows(df, selection_map, top_n=2)

    # Expect sorted by signal desc, None goes last (and trimmed by top_n)
    # NOTE: ZZZ.NASDAQ is not in selection_map so it should be excluded even though it has the highest signal.
    assert [r["vt_symbol"] for r in rows] == ["CCC.NASDAQ", "AAA.NASDAQ"]
    assert rows[0]["signal"] == 0.8
    assert rows[0]["close_price"] == 30.0
    assert rows[1]["signal"] == 0.2
    assert rows[1]["adv_usd"] == 1e6


