"""Valid index mask computation for \(\Omega_g\)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from python.instrumentation.timing import TimingRecorder
from python.pipeline.knife_model import KnifeInstance, PlaneInstance
from python.utils.config_loader import Config

LOGGER = logging.getLogger("pipeline.valid_indices.core")


@dataclass
class ValidIndicesResult:
    indices: np.ndarray
    table_threshold: float
    knife_threshold: float
    passed_table: int
    passed_knife: int
    passed_center_plane: int
    passed_penetration_plane: int
    plane_tolerance: float


def compute_valid_indices(
    points_low: np.ndarray,
    config: Config,
    recorder: TimingRecorder,
    knife: KnifeInstance,
) -> ValidIndicesResult:
    """Compute Ωg mask using table/knife clearances + knife plane clipping."""

    if points_low.ndim != 2 or points_low.shape[1] != 3:
        raise ValueError("points_low must be shaped (N, 3)")

    table_z = float(config.environment.get("table_z", 0.0))
    table_clearance = float(config.search.get("table_clearance", 0.0))
    knife_height = float(config.knife.get("height", 0.05))
    knife_clearance = float(config.search.get("knife_clearance", 0.0))
    plane_tolerance = float(config.search.get("knife_plane_clearance", 0.0))

    table_threshold = table_z + table_clearance
    knife_threshold = knife_height - knife_clearance
    center_plane = _normalize_plane(knife.center_plane)
    penetration_plane = _normalize_plane(knife.penetration_plane)

    with recorder.section("python/compute_valid_indices_total"):
        with recorder.section("python/valid_filter_table"):
            table_mask = points_low[:, 2] >= table_threshold
        with recorder.section("python/valid_filter_knife"):
            knife_mask = points_low[:, 2] <= knife_threshold
        with recorder.section("python/valid_filter_center_plane"):
            center_mask = _half_space_mask(points_low, center_plane, plane_tolerance)
        with recorder.section("python/valid_filter_slice_plane"):
            penetration_mask = _half_space_mask(points_low, penetration_plane, plane_tolerance)

    valid_mask = table_mask & knife_mask & center_mask & penetration_mask
    indices = np.nonzero(valid_mask)[0].astype(np.int32)
    LOGGER.info(
        "Ωg built: table_pass=%d knife_pass=%d center_pass=%d slice_pass=%d survivors=%d",
        int(table_mask.sum()),
        int(knife_mask.sum()),
        int(center_mask.sum()),
        int(penetration_mask.sum()),
        int(indices.size),
    )
    if indices.size == 0:
        LOGGER.error("Ωg empty after filters (table>=%.4f knife<=%.4f tolerance=%.5f)", table_threshold, knife_threshold, plane_tolerance)
    if center_mask.sum() == 0 or penetration_mask.sum() == 0:
        LOGGER.error("Knife plane clipping removed all points (center=%d slice=%d)", int(center_mask.sum()), int(penetration_mask.sum()))
    return ValidIndicesResult(
        indices=indices,
        table_threshold=table_threshold,
        knife_threshold=knife_threshold,
        passed_table=int(table_mask.sum()),
        passed_knife=int(knife_mask.sum()),
        passed_center_plane=int(center_mask.sum()),
        passed_penetration_plane=int(penetration_mask.sum()),
        plane_tolerance=plane_tolerance,
    )


def _normalize_plane(plane: PlaneInstance) -> PlaneInstance:
    normal = plane.normal
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        raise ValueError("Knife plane normal is degenerate")
    return PlaneInstance(point=np.asarray(plane.point, dtype=np.float64), normal=normal / norm)


def _half_space_mask(points: np.ndarray, plane: PlaneInstance, tolerance: float) -> np.ndarray:
    signed = (points - plane.point) @ plane.normal
    return signed <= tolerance
