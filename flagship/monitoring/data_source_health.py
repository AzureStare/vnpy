"""
数据源健康度探测（Prometheus textfile 指标）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable

from flagship.monitoring.textfile_metrics import Sample, TextfileMetricsWriter
from flagship.config.polygon_config import PolygonConfigError, create_polygon_client


@dataclass(frozen=True)
class ProbeResult:
    source: str
    ok: bool
    latency_seconds: float
    error_type: str | None = None


def _run_probe(source: str, fn: Callable[[], None]) -> ProbeResult:
    start = time.perf_counter()
    try:
        fn()
        latency = max(0.0, time.perf_counter() - start)
        return ProbeResult(source=source, ok=True, latency_seconds=latency, error_type=None)
    except Exception as exc:
        latency = max(0.0, time.perf_counter() - start)
        error_type = type(exc).__name__
        return ProbeResult(source=source, ok=False, latency_seconds=latency, error_type=error_type)


def probe_alpaca() -> ProbeResult:
    from flagship.trading.execution.broker_alpaca import AlpacaAdapter

    adapter = AlpacaAdapter()

    def _probe() -> None:
        adapter.client.get_clock()

    return _run_probe("alpaca", _probe)


def probe_polygon() -> ProbeResult:
    try:
        client = create_polygon_client()
    except PolygonConfigError:
        return ProbeResult(source="polygon", ok=False, latency_seconds=0.0, error_type="missing_api_key")
    except Exception as exc:
        return ProbeResult(source="polygon", ok=False, latency_seconds=0.0, error_type=type(exc).__name__)

    def _probe() -> None:
        # 轻量请求：仅拉 1 条 ticker 作为连通性与延迟探测
        _ = client.list_tickers(market="stocks", type="CS", active=True, limit=1)

    return _run_probe("polygon", _probe)


class DataSourceHealthMonitor:
    def __init__(self, *, min_interval_seconds: int = 300) -> None:
        self.min_interval_seconds = max(30, int(min_interval_seconds))
        self._last_ts: float = 0.0
        self._writer = TextfileMetricsWriter("flagship_data_source_health.prom")

    def maybe_emit(self) -> None:
        now = time.time()
        if now - self._last_ts < self.min_interval_seconds:
            return
        self._last_ts = now
        self.emit()

    def emit(self) -> None:
        results = [probe_alpaca(), probe_polygon()]
        self._emit_results(results)

    def _emit_results(self, results: Iterable[ProbeResult]) -> None:
        now_ts = float(time.time())
        samples: list[Sample] = [
            Sample("flagship_data_source_last_check_timestamp_seconds", now_ts),
        ]
        for r in results:
            samples.append(
                Sample(
                    "flagship_data_source_up",
                    1.0 if r.ok else 0.0,
                    labels={"source": r.source},
                )
            )
            samples.append(
                Sample(
                    "flagship_data_source_latency_seconds",
                    float(r.latency_seconds),
                    labels={"source": r.source},
                )
            )
            if not r.ok:
                samples.append(
                    Sample(
                        "flagship_data_source_last_error_timestamp_seconds",
                        now_ts,
                        labels={"source": r.source},
                    )
                )
                samples.append(
                    Sample(
                        "flagship_data_source_last_error",
                        1.0,
                        labels={"source": r.source, "error": str(r.error_type or "unknown")},
                    )
                )

        self._writer.write(
            samples,
            help_map={
                "flagship_data_source_last_check_timestamp_seconds": "Last data source health check timestamp.",
                "flagship_data_source_up": "Data source connectivity status (1=ok, 0=error).",
                "flagship_data_source_latency_seconds": "Data source probe latency in seconds.",
                "flagship_data_source_last_error_timestamp_seconds": "Timestamp of last data source error.",
                "flagship_data_source_last_error": "Latest error marker for a data source (1=error).",
            },
            type_map={
                "flagship_data_source_last_check_timestamp_seconds": "gauge",
                "flagship_data_source_up": "gauge",
                "flagship_data_source_latency_seconds": "gauge",
                "flagship_data_source_last_error_timestamp_seconds": "gauge",
                "flagship_data_source_last_error": "gauge",
            },
        )

