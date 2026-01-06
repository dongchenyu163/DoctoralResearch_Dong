"""Simplified wrench computation placeholders."""

from __future__ import annotations

import numpy as np

from python.pipeline.contact_surface import ContactSurfaceResult
from python.utils.config_loader import Config


def compute_wrench(surface: ContactSurfaceResult, config: Config) -> np.ndarray:
    """Compute simplified fracture + friction wrench (Algorithm 4 input).

    Args:
        surface: ContactSurfaceResult listing faces per connected component.
        config physics knobs:
            - `pressure_distribution` (Pa). Larger values increase normal forces proportionally.
            - `fracture_toughness` (N/m). Larger values yield bigger fracture force contributions.
            - `friction_coef` μ. Controls tangential friction force magnitude.
            - `planar_constraint` bool. When True, zeroes out non-planar wrench components (z force
              + roll/pitch moments) per spec §3.4.
    Returns:
        6-vector `[Fx, Fy, Fz, Tx, Ty, Tz]`.
    """
    pressure = float(config.physics.get("pressure_distribution", 1000.0))
    friction_coef = float(config.physics.get("friction_coef", 0.5))
    fracture_toughness = float(config.physics.get("fracture_toughness", 500.0))
    planar_only = bool(config.physics.get("planar_constraint", True))

    total_force = np.zeros(3, dtype=np.float64)
    total_torque = np.zeros(3, dtype=np.float64)
    for cluster in surface.faces:
        triangles = cluster.reshape((-1, 3, 3))
        for tri in triangles:
            a, b, c = tri
            normal_vec = np.cross(b - a, c - a)
            area = 0.5 * np.linalg.norm(normal_vec)
            if not np.isfinite(area) or area < 1e-9:
                continue
            normal = normal_vec / (np.linalg.norm(normal_vec) + 1e-12)
            centroid = (a + b + c) / 3.0
            fracture_force = fracture_toughness * area * normal
            normal_force = pressure * area * normal
            tangent = np.cross(normal, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(tangent) < 1e-9:
                tangent = np.cross(normal, np.array([0.0, 1.0, 0.0]))
            tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
            friction_force = friction_coef * np.linalg.norm(normal_force) * tangent
            total_force += fracture_force + friction_force
            total_torque += np.cross(centroid, fracture_force) + np.cross(centroid, friction_force)

    wrench = np.zeros(6, dtype=np.float64)
    wrench[:3] = total_force
    wrench[3:] = total_torque
    if planar_only:
        wrench[2] = 0.0
        wrench[3] = 0.0
        wrench[4] = 0.0
    return wrench


__all__ = ["compute_wrench"]
