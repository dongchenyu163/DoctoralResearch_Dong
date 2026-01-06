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
    passed_plane: int
    plane_tolerance: float


def compute_valid_indices(
    points_low: np.ndarray,
    config: Config,
    recorder: TimingRecorder,
    knife_point: np.ndarray,
    knife_normal: np.ndarray,
) -> ValidIndicesResult:
    """Compute Ωg mask for Algorithm 1 based on table and knife clearances.

    Args:
        points_low: Shape (M, 3) array for Ω_low.
        config: Supplies thresholds:
            - `environment.table_z` (base table plane height).
            - `search.table_clearance` (increase to demand higher finger positions).
            - `knife.height` and `search.knife_clearance` (decrease to allow closer to blade).
        recorder: Emits instrumentation for table/knife filters.
    """

    if points_low.ndim != 2 or points_low.shape[1] != 3:
        raise ValueError("points_low must be shaped (N, 3)")

    table_z = float(config.environment.get("table_z", 0.0))
    table_clearance = float(config.search.get("table_clearance", 0.0))
    knife_height = float(config.knife.get("height", 0.05))
    knife_clearance = float(config.search.get("knife_clearance", 0.0))

    table_threshold = table_z + table_clearance
    knife_threshold = knife_height - knife_clearance
    plane_tolerance = float(config.search.get("knife_plane_clearance", 0.0))
    normal = np.asarray(knife_normal, dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-9:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        normal /= normal_norm
    plane_point = np.asarray(knife_point, dtype=np.float64)

    with recorder.section("python/compute_valid_indices_total"):
        with recorder.section("python/valid_filter_table"):
            table_mask = points_low[:, 2] >= table_threshold
        with recorder.section("python/valid_filter_knife"):
            knife_mask = points_low[:, 2] <= knife_threshold
        with recorder.section("python/valid_filter_plane"):
            signed_dist = (points_low - plane_point) @ normal
            plane_mask = signed_dist >= -plane_tolerance

    valid_mask = table_mask & knife_mask & plane_mask
    indices = np.nonzero(valid_mask)[0].astype(np.int32)
    return ValidIndicesResult(
        indices=indices,
        table_threshold=table_threshold,
        knife_threshold=knife_threshold,
        passed_table=int(table_mask.sum()),
        passed_knife=int(knife_mask.sum()),
        passed_plane=int(plane_mask.sum()),
        plane_tolerance=plane_tolerance,
    )
