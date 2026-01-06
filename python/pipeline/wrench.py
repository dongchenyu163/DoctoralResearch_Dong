"""Simplified wrench computation placeholders."""

from __future__ import annotations

import numpy as np

from python.pipeline.contact_surface import ContactSurfaceResult
from python.utils.config_loader import Config


def compute_wrench(surface: ContactSurfaceResult, config: Config) -> np.ndarray:
    planar_only = bool(config.physics.get("planar_constraint", True))
    total_force = np.zeros(3, dtype=np.float64)
    for cluster in surface.faces:
        if cluster.size == 0:
            continue
        centroid = cluster.reshape(-1, 3).mean(axis=0)
        normal = np.array([0.0, 1.0, 0.0])
        total_force += normal * (np.linalg.norm(centroid) + 1.0)
    wrench = np.zeros(6, dtype=np.float64)
    wrench[:3] = total_force
    wrench[3:] = np.cross(np.array([0.0, 0.0, 0.0]), total_force)
    if planar_only:
        wrench[2] = 0.0
        wrench[4:] = 0.0
    return wrench


__all__ = ["compute_wrench"]
