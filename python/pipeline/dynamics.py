"""Placeholder dynamics score computation."""

from __future__ import annotations

import numpy as np

from python.pipeline.geo_filter import GeoFilterRunner
from python.utils.config_loader import Config


def compute_dynamics_scores(
    runner: GeoFilterRunner,
    candidate_matrix: np.ndarray,
    wrench: np.ndarray,
    config: Config,
) -> np.ndarray:
    if candidate_matrix.size == 0:
        return np.zeros((0,), dtype=np.float64)
    friction_coef = float(config.physics.get("friction_coef", 0.5))
    friction_angle = float(config.physics.get("friction_cone", {}).get("angle_deg", 40.0))
    scores = runner.calculator.calc_dynamics_scores(candidate_matrix, wrench, friction_coef, friction_angle)
    return np.asarray(scores, dtype=np.float64)


__all__ = ["compute_dynamics_scores"]
