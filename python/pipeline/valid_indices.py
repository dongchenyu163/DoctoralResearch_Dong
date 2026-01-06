"""Valid index mask computation for \(\Omega_g\)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from python.instrumentation.timing import TimingRecorder
from python.utils.config_loader import Config


@dataclass
class ValidIndicesResult:
    indices: np.ndarray
    table_threshold: float
    knife_threshold: float
    passed_table: int
    passed_knife: int


def compute_valid_indices(points_low: np.ndarray, config: Config, recorder: TimingRecorder) -> ValidIndicesResult:
    """Compute the valid index set based on table and knife constraints."""

    if points_low.ndim != 2 or points_low.shape[1] != 3:
        raise ValueError("points_low must be shaped (N, 3)")

    table_z = float(config.environment.get("table_z", 0.0))
    table_clearance = float(config.search.get("table_clearance", 0.0))
    knife_height = float(config.knife.get("height", 0.05))
    knife_clearance = float(config.search.get("knife_clearance", 0.0))

    table_threshold = table_z + table_clearance
    knife_threshold = knife_height - knife_clearance

    with recorder.section("python/compute_valid_indices_total"):
        with recorder.section("python/valid_filter_table"):
            table_mask = points_low[:, 2] >= table_threshold
        with recorder.section("python/valid_filter_knife"):
            knife_mask = points_low[:, 2] <= knife_threshold

    valid_mask = table_mask & knife_mask
    indices = np.nonzero(valid_mask)[0].astype(np.int32)
    return ValidIndicesResult(
        indices=indices,
        table_threshold=table_threshold,
        knife_threshold=knife_threshold,
        passed_table=int(table_mask.sum()),
        passed_knife=int(knife_mask.sum()),
    )
