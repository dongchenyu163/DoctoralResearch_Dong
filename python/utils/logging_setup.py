"""Logging configuration helpers for Python and C++ components."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]


def _relative_path(pathname: str) -> str:
    """Return repo-relative path if possible."""
    candidate = Path(pathname).resolve()
    try:
        return str(candidate.relative_to(REPO_ROOT))
    except ValueError:
        return str(candidate)


class _ConsoleFormatter(logging.Formatter):
    """Formatter that uses compact [MMDDhhmmss_mmm] timestamps."""

    def format(self, record: logging.LogRecord) -> str:
        record.custom_time = self.formatTime(record)  # type: ignore[attr-defined]
        return f"[{record.custom_time}] {record.name} {record.levelname}: {record.getMessage()} :: {record.funcName}"

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone()
        return dt.strftime("%m%d%H%M%S") + f"_{int(record.msecs):03d}"


class _FileFormatter(logging.Formatter):
    """ISO-like timestamp formatter for log files."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone().isoformat()
        rel = _relative_path(record.pathname)
        return f"{ts} {record.name} {record.levelname}: {record.getMessage()} [{rel}]:line {record.lineno} :: {record.funcName}"


@dataclass
class ModuleLevel:
    level: str


@dataclass
class PythonLoggingSettings:
    enabled: bool
    console_enabled: bool
    console_level: str
    file_enabled: bool
    file_level: str
    file_path: Path
    module_levels: Dict[str, ModuleLevel] = field(default_factory=dict)


@dataclass
class CppLoggingSettings:
    enabled: bool
    logger_name: str
    level: str
    console: bool
    file_path: Optional[Path]


@dataclass
class LoggingSettings:
    python: PythonLoggingSettings
    cpp: CppLoggingSettings


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_logging_settings(path: Path) -> LoggingSettings:
    data = _load_json(path)
    py_section = data.get("console", {})
    file_section = data.get("file", {})
    modules_cfg = {}
    for name, module_data in data.get("modules", {}).items():
        modules_cfg[name] = ModuleLevel(level=str(module_data.get("level", "INFO")).upper())
    python_settings = PythonLoggingSettings(
        enabled=bool(data.get("enabled", True)),
        console_enabled=bool(py_section.get("enabled", True)),
        console_level=str(py_section.get("level", "INFO")).upper(),
        file_enabled=bool(file_section.get("enabled", True)),
        file_level=str(file_section.get("level", "DEBUG")).upper(),
        file_path=Path(file_section.get("path", "logs/pipeline.log")),
        module_levels=modules_cfg,
    )

    cpp_section = data.get("cpp", {})
    cpp_file = cpp_section.get("file")
    cpp_settings = CppLoggingSettings(
        enabled=bool(cpp_section.get("enabled", True)),
        logger_name=str(cpp_section.get("logger_name", "cpp.score_calculator")),
        level=str(cpp_section.get("level", "info")).lower(),
        console=bool(cpp_section.get("console", True)),
        file_path=Path(cpp_file) if cpp_file else None,
    )
    return LoggingSettings(python=python_settings, cpp=cpp_settings)


def configure_python_logging(settings: PythonLoggingSettings) -> None:
    """Configure Python logging handlers per config."""
    logging.shutdown()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if not settings.enabled:
        logging.getLogger().disabled = True
        return

    logging.getLogger().setLevel(logging.DEBUG)
    handlers = []
    if settings.console_enabled:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, settings.console_level, logging.INFO))
        console_handler.setFormatter(_ConsoleFormatter())
        handlers.append(console_handler)
    if settings.file_enabled:
        settings.file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(settings.file_path, encoding="utf-8")
        file_handler.setLevel(getattr(logging, settings.file_level, logging.DEBUG))
        file_handler.setFormatter(_FileFormatter())
        handlers.append(file_handler)
    for handler in handlers:
        logging.getLogger().addHandler(handler)

    for module_name, module_level in settings.module_levels.items():
        logging.getLogger(module_name).setLevel(getattr(logging, module_level.level, logging.INFO))


__all__ = [
    "LoggingSettings",
    "PythonLoggingSettings",
    "CppLoggingSettings",
    "load_logging_settings",
    "configure_python_logging",
]
