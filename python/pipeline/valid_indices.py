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
    # Fix penetration_plane normal to point +Y direction
    if penetration_plane.normal[1] < 0:
        penetration_plane = PlaneInstance(point=penetration_plane.point, normal=-penetration_plane.normal)
    if bool(config.search.get("debug_valid_indices_viz", False)):
        _visualize_valid_indices(points_low, center_plane, penetration_plane)

    with recorder.section("python/compute_valid_indices_total"):
        with recorder.section("python/valid_filter_table"):
            table_mask = points_low[:, 2] >= table_threshold
        with recorder.section("python/valid_filter_knife"):
            knife_mask = points_low[:, 2] <= knife_threshold
        with recorder.section("python/valid_filter_center_plane"):
            center_mask = _half_space_mask(points_low, center_plane, plane_tolerance)
        with recorder.section("python/valid_filter_slice_plane"):
            penetration_mask = _half_space_mask(points_low, penetration_plane, plane_tolerance)

    valid_mask = table_mask & knife_mask & ~center_mask & ~penetration_mask
    indices = np.nonzero(valid_mask)[0].astype(np.int32)
    LOGGER.info(
        "Ωg built: table_pass=%d knife_pass=%d center_pass=%d slice_pass=%d survivors=%d",
        int(table_mask.sum()),
        int(knife_mask.sum()),
        int(center_mask.sum()),
        int(penetration_mask.sum()),
        int(indices.size),
    )
    LOGGER.debug("Penetration plane normal=%s", np.array2string(penetration_plane.normal, precision=4, separator=","))
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
    return signed >= -tolerance


def _visualize_valid_indices(points_low: np.ndarray, center_plane: PlaneInstance, penetration_plane: PlaneInstance) -> None:
    try:
        import open3d as o3d
    except ImportError:  # pragma: no cover - optional dependency
        LOGGER.warning("open3d unavailable; skip debug visualization for valid indices")
        return
    if points_low.size == 0:
        LOGGER.warning("points_low empty; skip debug visualization for valid indices")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_low.astype(np.float64))
    pcd.paint_uniform_color([0.95, 0.9, 0.1])
    extent = float(np.linalg.norm(points_low.max(axis=0) - points_low.min(axis=0)) * 0.6)
    center_plane_mesh = _make_plane_mesh(center_plane, extent)
    center_arrow_mesh = _make_normal_arrow(center_plane, extent * 0.7)
    penetration_plane_mesh = _make_plane_mesh(penetration_plane, extent)
    penetration_arrow_mesh = _make_normal_arrow(penetration_plane, extent * 0.7)
    o3d.visualization.draw_geometries(
        [pcd, center_plane_mesh, center_arrow_mesh, penetration_plane_mesh, penetration_arrow_mesh],
        window_name="valid_indices: points_low + penetration_plane",
    )


def _make_plane_mesh(plane: PlaneInstance, extent: float):
    import open3d as o3d

    normal = plane.normal
    if abs(normal[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = np.cross(normal, ref)
    u /= max(np.linalg.norm(u), 1e-9)
    v = np.cross(normal, u)
    center = plane.point
    p0 = center - u * extent - v * extent
    p1 = center + u * extent - v * extent
    p2 = center + u * extent + v * extent
    p3 = center - u * extent + v * extent
    vertices = np.vstack([p0, p1, p2, p3])
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(triangles),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.1, 0.8, 0.9])
    return mesh


def _make_normal_arrow(plane: PlaneInstance, length: float):
    import open3d as o3d

    cylinder_height = length * 0.7
    cone_height = length * 0.3
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=length * 0.02,
        cone_radius=length * 0.04,
        cylinder_height=cylinder_height,
        cone_height=cone_height,
    )
    direction = plane.normal / max(np.linalg.norm(plane.normal), 1e-9)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    axis = np.cross(z_axis, direction)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        rotation = np.eye(3)
    else:
        axis /= axis_norm
        angle = float(np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0)))
        rotation = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
    arrow.rotate(rotation, center=(0.0, 0.0, 0.0))
    arrow.translate(plane.point.astype(np.float64))
    arrow.paint_uniform_color([0.95, 0.1, 0.1])
    return arrow
