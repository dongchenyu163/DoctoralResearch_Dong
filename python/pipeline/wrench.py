"""Simplified wrench computation placeholders."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None

try:
    from scipy.spatial import Delaunay
except ImportError:  # pragma: no cover
    Delaunay = None

from python.pipeline.contact_surface import ContactSurfaceResult
from python.utils.config_loader import Config
from python.utils.logging_sections import log_boxed_heading

LOGGER = logging.getLogger("pipeline.wrench")


def compute_wrench(surface: ContactSurfaceResult, config: Config, step_idx: int | None = None) -> np.ndarray:
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
    step_label = f"Step {step_idx} " if step_idx is not None else ""
    log_boxed_heading(LOGGER, "4.1", f"{step_label}Algorithm 4 Knife Wrench 估计")
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

    fracture_step = float(config.physics.get("fracture_step", 0.001))
    friction_cfg = config.physics.get("friction_mesh", {})
    friction_sample = int(friction_cfg.get("sample_count", 5000))
    friction_voxel = float(friction_cfg.get("voxel_size", 0.002))
    friction_edge = float(friction_cfg.get("edge_delta", 0.002))

    total_force = np.zeros(3, dtype=np.float64)
    total_torque = np.zeros(3, dtype=np.float64)
    face_details = []

    edge_line = _build_edge_line(surface.faces, step=fracture_step)
    fracture_sum = np.zeros(3, dtype=np.float64)
    fracture_torque = np.zeros(3, dtype=np.float64)
    if edge_line is not None:
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        for point in edge_line:
            force = fracture_toughness * fracture_step * direction
            fracture_sum += force
            fracture_torque += np.cross(point, force)
    else:
        LOGGER.error("Knife edge line unavailable; fracture force defaults to zero")

    for cluster in surface.faces:
        friction_sum = np.zeros(3, dtype=np.float64)
        torque_sum = np.zeros(3, dtype=np.float64)
        total_area = 0.0
        mesh = _build_uniform_contact_mesh(cluster, friction_sample, friction_voxel, friction_edge)
        triangles = mesh.triangles if mesh is not None else cluster.reshape((-1, 3, 3))
        for tri in triangles:
            a, b, c = tri
            normal_vec = np.cross(b - a, c - a)
            area = 0.5 * np.linalg.norm(normal_vec)
            if not np.isfinite(area) or area < 1e-9:
                continue
            total_area += area
            normal = normal_vec / (np.linalg.norm(normal_vec) + 1e-12)
            centroid = (a + b + c) / 3.0
            normal_force = pressure * area * normal
            tangent = np.cross(normal, np.array([0.0, 0.0, 1.0]))
            if np.linalg.norm(tangent) < 1e-9:
                tangent = np.cross(normal, np.array([0.0, 1.0, 0.0]))
            tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
            friction_force = friction_coef * np.linalg.norm(normal_force) * tangent
            friction_sum += friction_force
            torque_sum += np.cross(centroid, friction_force)
        face_details.append((fracture_sum.copy(), friction_sum.copy(), fracture_torque.copy() + torque_sum.copy(), total_area))
        total_force += fracture_sum + friction_sum
        total_torque += fracture_torque + torque_sum

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


def _build_edge_line(face_groups: List[np.ndarray], step: float) -> Optional[np.ndarray]:
    if len(face_groups) < 2:
        return None
    pts1 = np.asarray(face_groups[0], dtype=np.float64).reshape(-1, 3)
    pts2 = np.asarray(face_groups[1], dtype=np.float64).reshape(-1, 3)
    if pts1.size == 0 or pts2.size == 0:
        return None
    n1 = _fit_plane_normal(pts1)
    n2 = _fit_plane_normal(pts2)
    direction = np.cross(n1, n2)
    norm = np.linalg.norm(direction)
    if norm < 1e-9:
        return None
    direction /= norm
    mean = np.vstack([pts1, pts2]).mean(axis=0)
    projections = (np.vstack([pts1, pts2]) - mean) @ direction
    t_min = float(np.min(projections))
    t_max = float(np.max(projections))
    length = max(t_max - t_min, step)
    count = max(int(np.ceil(length / step)) + 1, 2)
    ts = np.linspace(t_min, t_max, count)
    return mean + ts[:, None] * direction


def _fit_plane_normal(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    norm = np.linalg.norm(normal)
    return normal / (norm + 1e-12)


def _build_uniform_contact_mesh(
    face_block: np.ndarray,
    sample_count: int,
    voxel_size: float,
    edge_delta: float,
) -> Optional["trimesh.Trimesh"]:
    if trimesh is None or Delaunay is None:
        LOGGER.error("Uniform contact mesh unavailable (trimesh or scipy missing)")
        return None
    if face_block.size == 0:
        return None
    raw_mesh = _faces_to_trimesh(face_block)
    if raw_mesh is None or raw_mesh.is_empty:
        return None
    if sample_count <= 0:
        sample_count = 5000
    sampled, _ = trimesh.sample.sample_surface(raw_mesh, sample_count)
    sampled = np.asarray(sampled, dtype=np.float64)
    if o3d is not None and voxel_size > 0:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(sampled)
        sampled = np.asarray(cloud.voxel_down_sample(voxel_size).points, dtype=np.float64)
    edge_points = _densify_edges(raw_mesh, edge_delta)
    combined = np.vstack([sampled, edge_points]) if edge_points.size else sampled
    if combined.shape[0] < 3:
        return None
    basis, origin = _pca_basis(combined)
    coords2d = (combined - origin) @ basis.T
    tri = Delaunay(coords2d)
    faces = tri.simplices.astype(np.int64)
    return trimesh.Trimesh(vertices=combined, faces=faces, process=False)


def _densify_edges(mesh: "trimesh.Trimesh", edge_delta: float) -> np.ndarray:
    if edge_delta <= 0:
        return np.empty((0, 3), dtype=np.float64)
    edges = mesh.edges_unique
    pts = []
    for a_idx, b_idx in edges:
        a = mesh.vertices[a_idx]
        b = mesh.vertices[b_idx]
        length = np.linalg.norm(b - a)
        if length < 1e-9:
            continue
        count = max(int(np.ceil(length / edge_delta)) + 1, 2)
        for t in np.linspace(0.0, 1.0, count):
            pts.append(a + t * (b - a))
    if not pts:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def _pca_basis(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2]
    return basis, points.mean(axis=0)


def _faces_to_trimesh(face_block: np.ndarray) -> Optional["trimesh.Trimesh"]:
    if trimesh is None:
        return None
    triangles = np.asarray(face_block, dtype=np.float64).reshape(-1, 3, 3)
    vertices = triangles.reshape(-1, 3)
    faces = np.arange(vertices.shape[0], dtype=np.int64).reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


__all__ = ["compute_wrench"]
