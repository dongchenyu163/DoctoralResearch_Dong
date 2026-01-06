"""Timing instrumentation with JSONL output."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from python.utils.config_loader import InstrumentationConfig, InstrumentationSections, TimingOutputConfig


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TimingRecorder:
    """JSONL timing recorder controlled by instrumentation config."""

    def __init__(self, config: InstrumentationConfig):
        self.config = config
        self.output: TimingOutputConfig = config.timing_output
        self.sections: InstrumentationSections = config.sections
        self.enabled = bool(config.enable_timing)
        self.enable_detailed = bool(config.enable_detailed_timing)
        self._output_path = Path(self.output.path)
        if self.enabled:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)

    def _section_allowed(self, section: str) -> bool:
        if not self.enabled:
            return False
        if "/" not in section:
            return False
        prefix, name = section.split("/", maxsplit=1)
        if prefix == "python":
            return bool(self.sections.python.get(name, False))
        if prefix == "cpp":
            return bool(self.sections.cpp.get(name, False))
        return False

    @contextmanager
    def section(self, name: str, metadata: Optional[Dict[str, object]] = None):
        """Context manager to record a timing section."""
        if not self._section_allowed(name):
            yield None
            return
        start_perf = time.perf_counter()
        start_iso = _now_iso()
        try:
            yield None
        finally:
            duration_ms = (time.perf_counter() - start_perf) * 1000.0
            record = {
                "type": "timing",
                "section": name,
                "duration_ms": duration_ms,
                "timestamp": start_iso,
            }
            if self.enable_detailed:
                record["perf_counter_start"] = start_perf
                record["perf_counter_end"] = start_perf + duration_ms / 1000.0
            if metadata:
                record["metadata"] = metadata
            self._write_record(record)

    def emit_event(self, name: str, payload: Optional[Dict[str, object]] = None) -> None:
        """Write a non-timing diagnostic event."""
        if not self.enabled or not self._section_allowed(name):
            return
        record = {"type": "event", "section": name, "timestamp": _now_iso(), "payload": payload or {}}
        self._write_record(record)

    def _write_record(self, record: Dict[str, object]) -> None:
        line = json.dumps(record, ensure_ascii=True)
        with self._output_path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")
