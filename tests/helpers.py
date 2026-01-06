"""Shared helpers for test configuration and recorders."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

BUILD_DIR = Path(__file__).resolve().parents[1] / "build"
if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

from python.instrumentation.timing import TimingRecorder
from python.utils.config_loader import Config, DEFAULT_CONFIG


def _merge_dict(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def make_config(overrides: dict | None = None) -> Config:
    data = _merge_dict(DEFAULT_CONFIG, overrides or {})
    data["instrumentation"]["enable_timing"] = False
    data["instrumentation"]["enable_detailed_timing"] = False
    return Config.from_dict(data)


def make_recorder(config: Config | None = None) -> TimingRecorder:
    cfg = config or make_config()
    return TimingRecorder(cfg.instrumentation)
