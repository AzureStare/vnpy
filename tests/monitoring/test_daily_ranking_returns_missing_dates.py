from datetime import date

from flagship.monitoring.backfill_daily_ranking_returns import _compute_missing_trade_dates


def test_compute_missing_trade_dates_by_coverage() -> None:
    horizons = [1, 3, 5]
    # history TopN rows: day1=50, day2=50
    history_counts = {
        date(2024, 1, 2): 50,
        date(2024, 1, 3): 50,
    }
    # expected = 50 * 3 * 2 = 300 per day
    returns_counts = {
        date(2024, 1, 2): 300,  # complete
        date(2024, 1, 3): 120,  # incomplete
    }

    missing = _compute_missing_trade_dates(
        history_counts=history_counts,
        returns_counts=returns_counts,
        horizons=horizons,
        min_coverage=0.98,
    )
    assert missing == [date(2024, 1, 3)]


def test_compute_missing_trade_dates_ignores_invalid_horizons() -> None:
    missing = _compute_missing_trade_dates(
        history_counts={date(2024, 1, 2): 50},
        returns_counts={},
        horizons=[0, -1],
        min_coverage=0.98,
    )
    assert missing == []

