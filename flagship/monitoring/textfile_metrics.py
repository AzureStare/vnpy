"""
Prometheus textfile collector helper.

We use Prometheus' recommended pattern:
- each component writes its own *.prom file
- node_exporter reads the directory and exposes metrics

This avoids cross-process merge/locking complexity.

Config:
- env FLAGSHIP_TEXTFILE_DIR: directory for *.prom output
  - if not set, default to ./logs/metrics (relative to project root CWD)
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from vnpy.trader.logger import logger


def _escape_label_value(value: str) -> str:
    # Prometheus text format: backslash + quote need escaping.
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace("\"", "\\\"")


def _format_labels(labels: Mapping[str, str] | None) -> str:
    if not labels:
        return ""
    parts = []
    for k in sorted(labels.keys()):
        v = labels[k]
        parts.append(f'{k}="{_escape_label_value(str(v))}"')
    return "{" + ",".join(parts) + "}"


@dataclass(frozen=True)
class Sample:
    name: str
    value: float
    labels: Mapping[str, str] | None = None


def _format_sample(sample: Sample) -> str:
    return f"{sample.name}{_format_labels(sample.labels)} {sample.value}"


def get_textfile_dir() -> Path:
    raw = os.getenv("FLAGSHIP_TEXTFILE_DIR", "").strip()
    if raw:
        return Path(raw)
    return Path("logs") / "metrics"


class TextfileMetricsWriter:
    def __init__(self, filename: str, *, directory: Path | None = None) -> None:
        self.directory = directory or get_textfile_dir()
        self.filename = filename

    def write(
        self,
        samples: Iterable[Sample],
        *,
        help_map: Mapping[str, str] | None = None,
        type_map: Mapping[str, str] | None = None,
    ) -> None:
        """
        Atomically write a .prom file.
        """
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning(f"[metrics] cannot create textfile dir {self.directory}: {exc}")
            return

        lines: list[str] = []

        if help_map:
            for metric_name in sorted(help_map.keys()):
                help_text = help_map[metric_name]
                lines.append(f"# HELP {metric_name} {help_text}")
        if type_map:
            for metric_name in sorted(type_map.keys()):
                metric_type = type_map[metric_name]
                lines.append(f"# TYPE {metric_name} {metric_type}")

        for s in samples:
            lines.append(_format_sample(s))

        content = "\n".join(lines) + "\n"
        target = self.directory / self.filename

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=str(self.directory),
                prefix=f".{self.filename}.",
                suffix=".tmp",
            ) as f:
                tmp_path = Path(f.name)
                f.write(content)
                f.flush()
            tmp_path.replace(target)
            try:
                # IMPORTANT:
                # node_exporter textfile collector may run as non-root, and will skip unreadable files.
                # Ensure metrics are world-readable to avoid "No data" in Grafana.
                target.chmod(0o644)
            except Exception:
                pass
        except Exception as exc:
            logger.warning(f"[metrics] failed to write {target}: {exc}")
            try:
                if "tmp_path" in locals() and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


