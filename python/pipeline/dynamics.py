"""Placeholder dynamics score computation."""

from __future__ import annotations

import logging
import numpy as np

from python.pipeline.geo_filter import GeoFilterRunner
from python.utils.config_loader import Config

LOGGER = logging.getLogger("pipeline.dynamics")


def compute_dynamics_scores(
    runner: GeoFilterRunner,
    candidate_matrix: np.ndarray,
    wrench: np.ndarray,
    config: Config,
) -> np.ndarray:
    """Forward to Algorithm 4 implementation in C++.

    Args:
        runner: GeoFilterRunner exposing the shared ScoreCalculator.
        candidate_matrix: Shape (K,F) candidates that survived Algorithms 2+3.
        wrench: 6-vector knife wrench (force+torque). Large magnitude generally requires
            more balanced forces, leading to lower scores if infeasible.
        config: Supplies physics knobs:
            - `physics.friction_coef` (μ). Increasing expands admissible tangential force.
            - `physics.friction_cone.angle_deg` (deg). Wider cone also relaxes tangential bounds.
    """
    if candidate_matrix.size == 0:
        return np.zeros((0,), dtype=np.float64)
    friction_coef = float(config.physics.get("friction_coef", 0.5))
    friction_angle = float(config.physics.get("friction_cone", {}).get("angle_deg", 40.0))
    LOGGER.debug(
        "Dynamics scoring rows=%d μ=%.3f cone=%.3f wrench=%s",
        candidate_matrix.shape[0],
        friction_coef,
        friction_angle,
        np.array2string(wrench, precision=4, separator=","),
    )
    scores = runner.calculator.calc_dynamics_scores(candidate_matrix, wrench, friction_coef, friction_angle)
    if scores.size:
        scores = np.clip(scores, 0.0, 1.0)
    LOGGER.debug(
        "Dynamics scores statistics mean=%.4f min=%.4f max=%.4f",
        float(scores.mean()) if scores.size else 0.0,
        float(scores.min()) if scores.size else 0.0,
        float(scores.max()) if scores.size else 0.0,
    )
    return np.asarray(scores, dtype=np.float64)


__all__ = ["compute_dynamics_scores"]
