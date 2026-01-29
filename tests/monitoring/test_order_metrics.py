from flagship.monitoring.order_metrics import classify_order_error


def test_classify_order_error_generic_exception() -> None:
    reason, rejected = classify_order_error(RuntimeError("boom"))
    assert reason == "exception"
    assert rejected is False
