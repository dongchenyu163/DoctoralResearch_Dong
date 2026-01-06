"""Simplified wrench computation placeholders."""

from __future__ import annotations

import logging
import numpy as np

from python.pipeline.contact_surface import ContactSurfaceResult
from python.utils.config_loader import Config
from python.utils.logging_sections import log_boxed_heading

LOGGER = logging.getLogger("pipeline.wrench")


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
    log_boxed_heading(LOGGER, "4.1", "Algorithm 4 Knife Wrench 估计")
    pressure = float(config.physics.get("pressure_distribution", 1000.0))
    friction_coef = float(config.physics.get("friction_coef", 0.5))
    fracture_toughness = float(config.physics.get("fracture_toughness", 500.0))
    planar_only = bool(config.physics.get("planar_constraint", True))
    LOGGER.debug(
        "Knife wrench params pressure=%.3f fracture=%.3f μ=%.3f planar_only=%s components=%d",
        pressure,
        fracture_toughness,
        friction_coef,
        planar_only,
        len(surface.faces),
    )

    total_force = np.zeros(3, dtype=np.float64)
    total_torque = np.zeros(3, dtype=np.float64)
    face_details = []
    for cluster in surface.faces:
        fracture_sum = np.zeros(3, dtype=np.float64)
        friction_sum = np.zeros(3, dtype=np.float64)
        torque_sum = np.zeros(3, dtype=np.float64)
        total_area = 0.0
        triangles = cluster.reshape((-1, 3, 3))
        for tri in triangles:
            a, b, c = tri
            normal_vec = np.cross(b - a, c - a)
            area = 0.5 * np.linalg.norm(normal_vec)
            if not np.isfinite(area) or area < 1e-9:
                continue
            total_area += area
            normal = normal_vec / (np.linalg.norm(normal_vec) + 1e-12)
            centroid = (a + b + c) / 3.0
            fracture_force = fracture_toughness * area * normal
            normal_force = pressure * area * normal
            tangent = np.cross(normal, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(tangent) < 1e-9:
                tangent = np.cross(normal, np.array([0.0, 1.0, 0.0]))
            tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
            friction_force = friction_coef * np.linalg.norm(normal_force) * tangent
            fracture_sum += fracture_force
            friction_sum += friction_force
            torque_sum += np.cross(centroid, fracture_force) + np.cross(centroid, friction_force)
        face_details.append((fracture_sum.copy(), friction_sum.copy(), torque_sum.copy(), total_area))
        total_force += fracture_sum + friction_sum
        total_torque += torque_sum

    if not face_details:
        LOGGER.warning("未检测到接触面，返回零扭矩")
    wrench = np.zeros(6, dtype=np.float64)
    wrench[:3] = total_force
    wrench[3:] = total_torque
    for idx, (fracture_vec, friction_vec, torque_vec, area_total) in enumerate(face_details):
        LOGGER.info(
            "Face %02d 切割力=%s 摩擦力=%s 合扭矩=%s area=%.6f",
            idx,
            _format_vec(fracture_vec),
            _format_vec(friction_vec),
            _format_vec(torque_vec),
            area_total,
        )
    LOGGER.info("原始合外力=%s 合扭矩=%s", _format_vec(total_force), _format_vec(total_torque))
    if planar_only:
        wrench[2] = 0.0
        wrench[3] = 0.0
        wrench[4] = 0.0
        LOGGER.info("planar_only→处理后的合外力=%s 合扭矩=%s", _format_vec(wrench[:3]), _format_vec(wrench[3:]))
    LOGGER.debug("Final wrench vector=%s", _format_vec(wrench))
    return wrench


def _format_vec(vec: np.ndarray) -> str:
    return np.array2string(np.asarray(vec, dtype=np.float64), precision=4, separator=",")


__all__ = ["compute_wrench"]
