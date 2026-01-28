from flagship.trading.execution.broker_alpaca import _parse_position_qty_to_int


def test_parse_position_qty_to_int() -> None:
    assert _parse_position_qty_to_int("10") == 10
    assert _parse_position_qty_to_int("0.5") == 0
    assert _parse_position_qty_to_int("1.9") == 1
    assert _parse_position_qty_to_int(2.0) == 2
    assert _parse_position_qty_to_int(-2.2) == -2
    assert _parse_position_qty_to_int(None) == 0
    assert _parse_position_qty_to_int("abc") == 0

