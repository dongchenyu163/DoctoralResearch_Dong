"""Placeholder dynamics score computation."""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from python.pipeline.geo_filter import GeoFilterRunner
from python.utils.config_loader import Config

LOGGER = logging.getLogger("pipeline.dynamics")


def compute_dynamics_scores(
    runner: GeoFilterRunner,
    candidate_matrix: np.ndarray,
    wrench: np.ndarray,
    center: np.ndarray,
    config: Config,
) -> np.ndarray:
    """Forward to Algorithm 4 implementation in C++.

    Args:
        runner: GeoFilterRunner exposing the shared ScoreCalculator.
        candidate_matrix: Shape (K,F) candidates that survived Algorithms 2+3.
        wrench: 6-vector knife wrench (force+torque). Large magnitude generally requires
            more balanced forces, leading to lower scores if infeasible.
        center: Object center in the same frame as Ω_low points.
        config: Supplies physics knobs:
            - `physics.friction_coef` (μ). Increasing expands admissible tangential force.
            - `physics.friction_cone.angle_deg` (deg). Wider cone also relaxes tangential bounds.
            - `physics.force_generation_attempts` max attempts per candidate.
            - `physics.force_balance_threshold` residual norm threshold.
            - `physics.force_sample_range` controls sampled force magnitudes and cone range.
    """
    if candidate_matrix.size == 0:
        return np.zeros((0,), dtype=np.float64)
    friction_coef = float(config.physics.get("friction_coef", 0.5))
    friction_angle = float(config.physics.get("friction_cone", {}).get("angle_deg", 40.0))
    max_attempts = int(config.physics.get("force_generation_attempts", 600))
    balance_threshold = float(config.physics.get("force_balance_threshold", 1e-4))
    sample_cfg = config.physics.get("force_sample_range", {})
    force_min = float(sample_cfg.get("force_min", 0.1))
    force_max = float(sample_cfg.get("force_max", 1.0))
    cone_angle_max = float(sample_cfg.get("cone_angle_max_deg", friction_angle))
    planar_constraint = bool(config.physics.get("planar_constraint", False))
    LOGGER.debug(
        "Dynamics scoring rows=%d μ=%.3f cone=%.3f attempts=%d balance=%.6f force=[%.3f,%.3f] cone_max=%.3f wrench=%s",
        candidate_matrix.shape[0],
        friction_coef,
        friction_angle,
        max_attempts,
        balance_threshold,
        force_min,
        force_max,
        cone_angle_max,
        np.array2string(wrench, precision=4, separator=","),
    )
    scores = runner.calculator.calc_dynamics_scores(
        candidate_matrix,
        wrench,
        center,
        planar_constraint,
        friction_coef,
        friction_angle,
        max_attempts,
        balance_threshold,
        force_min,
        force_max,
        cone_angle_max,
    )
    LOGGER.debug(
        "Dynamics scores statistics mean=%.4f min=%.4f max=%.4f",
        float(scores.mean()) if scores.size else 0.0,
        float(scores.min()) if scores.size else 0.0,
        float(scores.max()) if scores.size else 0.0,
    )
    return np.asarray(scores, dtype=np.float64)


def debug_visualize_dynamics_forces(
    runner: GeoFilterRunner,
    points_low: np.ndarray,
    normals_low: np.ndarray,
    omega_indices: np.ndarray,
    candidate_matrix: np.ndarray,
    wrench: np.ndarray,
    center: np.ndarray,
    config: Config,
) -> None:
    if not bool(config.search.get("debug_dynamics_force_viz", False)):
        return
    try:
        import open3d as o3d
    except ImportError:  # pragma: no cover
        LOGGER.warning("open3d unavailable; skip dynamics force visualization")
        return
    if candidate_matrix.size == 0:
        LOGGER.warning("No candidates to visualize in dynamics force debug")
        return
    attempts = runner.last_dynamics_attempts()
    if not attempts:
        LOGGER.warning("No dynamics attempts recorded for visualization")
        return

    omega_points = points_low[omega_indices] if omega_indices.size else points_low
    normals = normals_low
    bbox = omega_points if omega_points.size else points_low
    extent = bbox.max(axis=0) - bbox.min(axis=0) if bbox.size else np.array([0.1, 0.1, 0.1])
    aabb_size = float(np.linalg.norm(extent))
    sphere_radius = 0.002
    normal_length = 0.02

    def make_arrow(origin: np.ndarray, direction: np.ndarray, color: Iterable[float]) -> "o3d.geometry.TriangleMesh | None":
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            return None
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.0005,
            cone_radius=0.0015,
            cylinder_height=length * 0.8,
            cone_height=length * 0.2,
        )
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        direction_unit = direction / length
        axis = np.cross(z_axis, direction_unit)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-9:
            angle = float(np.arccos(np.clip(np.dot(z_axis, direction_unit), -1.0, 1.0)))
            arrow.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(axis / axis_norm * angle), center=np.zeros(3))
        arrow.translate(origin)
        arrow.paint_uniform_color(np.asarray(color, dtype=np.float64))
        return arrow

    def make_spheres(points: np.ndarray, color: Iterable[float]) -> list["o3d.geometry.TriangleMesh"]:
        meshes = []
        for pt in points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sphere.translate(pt)
            sphere.paint_uniform_color(np.asarray(color, dtype=np.float64))
            meshes.append(sphere)
        return meshes

    state = {"p_idx": 0, "f_idx": 0}
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Dynamics Force Debug", width=1200, height=900)

    opt = vis.get_render_option()
    opt.point_size = 17.0  # Default is typically 5.0

    def clear_scene():
        vis.clear_geometries()

    def add_geometry(geom):
        if geom is not None:
            vis.add_geometry(geom, reset_bounding_box=False)

    def update_scene() -> None:
        clear_scene()
        p_idx = state["p_idx"] % len(attempts)
        f_list = attempts[p_idx]
        angle_list = []
        if not f_list:
            LOGGER.warning("Candidate %d has no force attempts", p_idx)
            return
        f_idx = state["f_idx"] % len(f_list)
        candidate = candidate_matrix[p_idx]
        points = points_low[candidate]
        normals_local = normals[candidate]
        attempt = f_list[f_idx]
        f_vec = np.asarray(attempt[0], dtype=np.float64).reshape(-1)
        f_init_vec = np.asarray(attempt[1], dtype=np.float64).reshape(-1)
        max_force = 0.0
        for idx in range(points.shape[0]):
            force = f_vec[3 * idx : 3 * idx + 3]
            max_force = max(max_force, float(np.linalg.norm(force)))
            force_init = f_init_vec[3 * idx : 3 * idx + 3]
            max_force = max(max_force, float(np.linalg.norm(force_init)))
        wrench_force = np.asarray(wrench[:3], dtype=np.float64)
        max_force = max(max_force, float(np.linalg.norm(wrench_force)))
        scale = aabb_size / max_force if max_force > 1e-9 else 1.0

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(omega_points.astype(np.float64))
        cloud.paint_uniform_color([0.7, 0.7, 0.7])
        add_geometry(cloud)

        for sphere in make_spheres(points, [0.1, 0.9, 0.1]):
            add_geometry(sphere)

        for pt, n in zip(points, normals_local):
            arrow = make_arrow(pt, n * normal_length, [0.1, 0.3, 0.9])
            add_geometry(arrow)

        for idx in range(points.shape[0]):
            force = f_vec[3 * idx : 3 * idx + 3]
            angle = float(np.arccos(np.clip(np.dot(force, normals_local[idx]) / (np.linalg.norm(force) * np.linalg.norm(normals_local[idx]) + 1e-9), -1.0, 1.0)))
            angle_list.append(np.rad2deg(angle))
            arrow = make_arrow(points[idx], force * scale, [0.9, 0.2, 0.2])
            add_geometry(arrow)
            init_force = f_init_vec[3 * idx : 3 * idx + 3]
            arrow = make_arrow(points[idx], init_force * scale, [0.95, 0.6, 0.6])
            add_geometry(arrow)

        add_geometry(make_arrow(center, wrench_force * scale, [0.9, 0.7, 0.2]))

        planar_constraint = bool(config.physics.get("planar_constraint", False))
        residual = runner.calculator.calc_force_residual(candidate, wrench, center, planar_constraint, f_vec)
        LOGGER.info(
            "Dynamics viz P=%d/%d f=%d/%d residual=%.2f f=%s normal=%s angles=%s",
            p_idx + 1,
            len(attempts),
            f_idx + 1,
            len(f_list),
            residual,
            np.array2string(f_vec, precision=4, separator=","),
            np.array2string(normals_local, precision=2, separator=","),
            np.array2string(np.asarray(angle_list, dtype=np.float64), precision=2, separator=","),
        )
        # vis.reset_view_point(True)

    def find_next_valid(start_idx: int, step: int) -> int:
        p_idx = state["p_idx"] % len(attempts)
        f_list = attempts[p_idx]
        if not f_list:
            return start_idx
        count = len(f_list)
        for offset in range(1, count + 1):
            idx = (start_idx + step * offset) % count
            attempt = f_list[idx]
            if len(attempt) >= 6 and bool(attempt[5]):
                return idx
        return start_idx

    def on_page_up(vis_obj):
        state["p_idx"] = (state["p_idx"] + 1) % len(attempts)
        state["f_idx"] = 0
        update_scene()
        return False

    def on_page_down(vis_obj):
        state["p_idx"] = (state["p_idx"] - 1) % len(attempts)
        state["f_idx"] = 0
        update_scene()
        return False

    def on_up(vis_obj):
        state["f_idx"] += 1
        update_scene()
        return False

    def on_down(vis_obj):
        state["f_idx"] -= 1
        update_scene()
        return False

    def on_home(vis_obj):
        state["f_idx"] = find_next_valid(state["f_idx"], 1)
        update_scene()
        return False

    def on_end(vis_obj):
        state["f_idx"] = find_next_valid(state["f_idx"], -1)
        update_scene()
        return False

    vis.register_key_callback(266, on_page_up)
    vis.register_key_callback(267, on_page_down)
    vis.register_key_callback(265, on_up)
    vis.register_key_callback(264, on_down)
    vis.register_key_callback(268, on_home)
    vis.register_key_callback(269, on_end)

    update_scene()
    vis.reset_view_point(True)
    vis.run()
    vis.destroy_window()


def debug_visualize_dynamics_f_init_forces(
    runner: GeoFilterRunner,
    points_low: np.ndarray,
    normals_low: np.ndarray,
    omega_indices: np.ndarray,
    candidate_matrix: np.ndarray,
    config: Config,
) -> None:
    if not bool(config.search.get("debug_dynamics_force_viz", False)):
        return
    try:
        import open3d as o3d
    except ImportError:  # pragma: no cover
        LOGGER.warning("open3d unavailable; skip dynamics f_init visualization")
        return
    if candidate_matrix.size == 0:
        LOGGER.warning("No candidates to visualize in dynamics f_init debug")
        return
    attempts = runner.last_dynamics_attempts()
    if not attempts:
        LOGGER.warning("No dynamics attempts recorded for f_init visualization")
        return

    omega_points = points_low[omega_indices] if omega_indices.size else points_low
    normals = normals_low
    bbox = omega_points if omega_points.size else points_low
    extent = bbox.max(axis=0) - bbox.min(axis=0) if bbox.size else np.array([0.1, 0.1, 0.1])
    aabb_size = float(np.linalg.norm(extent))
    sphere_radius = 0.002
    normal_length = 0.02

    palette = np.array(
        [
            [0.121, 0.466, 0.705],
            [1.000, 0.498, 0.054],
            [0.172, 0.627, 0.172],
            [0.839, 0.153, 0.157],
            [0.580, 0.404, 0.741],
            [0.549, 0.337, 0.294],
            [0.890, 0.467, 0.761],
            [0.498, 0.498, 0.498],
            [0.737, 0.741, 0.133],
            [0.090, 0.745, 0.811],
            [0.682, 0.780, 0.909],
            [0.992, 0.749, 0.435],
            [0.565, 0.933, 0.565],
            [0.969, 0.506, 0.749],
            [0.796, 0.702, 0.839],
            [0.773, 0.690, 0.670],
            [0.968, 0.714, 0.824],
            [0.780, 0.780, 0.780],
            [0.859, 0.859, 0.553],
            [0.619, 0.854, 0.898],
        ],
        dtype=np.float64,
    )

    def make_arrow(origin: np.ndarray, direction: np.ndarray, color: np.ndarray) -> "o3d.geometry.TriangleMesh | None":
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            return None
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.0005,
            cone_radius=0.0015,
            cylinder_height=length * 0.8,
            cone_height=length * 0.2,
        )
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        direction_unit = direction / length
        axis = np.cross(z_axis, direction_unit)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-9:
            angle = float(np.arccos(np.clip(np.dot(z_axis, direction_unit), -1.0, 1.0)))
            arrow.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(axis / axis_norm * angle), center=np.zeros(3))
        arrow.translate(origin)
        arrow.paint_uniform_color(np.asarray(color, dtype=np.float64))
        return arrow

    def make_spheres(points: np.ndarray, color: np.ndarray) -> list["o3d.geometry.TriangleMesh"]:
        meshes = []
        for pt in points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sphere.translate(pt)
            sphere.paint_uniform_color(np.asarray(color, dtype=np.float64))
            meshes.append(sphere)
        return meshes

    state = {"p_idx": 0}
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Dynamics f_init Debug", width=1200, height=900)

    opt = vis.get_render_option()
    opt.point_size = 17.0  # Default is typically 5.0

    def clear_scene():
        vis.clear_geometries()

    def add_geometry(geom):
        if geom is not None:
            vis.add_geometry(geom, reset_bounding_box=False)

    def update_scene() -> None:
        clear_scene()
        p_idx = state["p_idx"] % len(attempts)
        f_list = attempts[p_idx]
        if not f_list:
            LOGGER.warning("Candidate %d has no force attempts", p_idx)
            return
        candidate = candidate_matrix[p_idx]
        points = points_low[candidate]
        normals_local = normals[candidate]
        k = int(config.search.get("debug_dynamics_f_init_k", 5))
        k = max(1, min(k, len(f_list)))

        max_force = 0.0
        for idx in range(k):
            f_init_vec = np.asarray(f_list[idx][1], dtype=np.float64).reshape(-1)
            for c_idx in range(points.shape[0]):
                force = f_init_vec[3 * c_idx : 3 * c_idx + 3]
                max_force = max(max_force, float(np.linalg.norm(force)))
        scale = aabb_size / max_force if max_force > 1e-9 else 1.0

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(omega_points.astype(np.float64))
        cloud.paint_uniform_color([0.7, 0.7, 0.7])
        add_geometry(cloud)

        for sphere in make_spheres(points, np.array([0.1, 0.9, 0.1], dtype=np.float64)):
            add_geometry(sphere)

        for pt, n in zip(points, normals_local):
            arrow = make_arrow(pt, n * normal_length, np.array([0.1, 0.3, 0.9], dtype=np.float64))
            add_geometry(arrow)

        for idx in range(k):
            f_init_vec = np.asarray(f_list[idx][1], dtype=np.float64).reshape(-1)
            color = palette[idx % palette.shape[0]]
            for c_idx in range(points.shape[0]):
                force = f_init_vec[3 * c_idx : 3 * c_idx + 3]
                arrow = make_arrow(points[c_idx], force * scale, color)
                add_geometry(arrow)

        LOGGER.info("Dynamics f_init viz P=%d/%d k=%d", p_idx + 1, len(attempts), k)

    def on_page_up(vis_obj):
        state["p_idx"] = (state["p_idx"] + 1) % len(attempts)
        update_scene()
        return False

    def on_page_down(vis_obj):
        state["p_idx"] = (state["p_idx"] - 1) % len(attempts)
        update_scene()
        return False

    vis.register_key_callback(266, on_page_up)
    vis.register_key_callback(267, on_page_down)

    update_scene()
    vis.reset_view_point(True)
    vis.run()
    vis.destroy_window()


__all__ = ["compute_dynamics_scores", "debug_visualize_dynamics_forces", "debug_visualize_dynamics_f_init_forces"]
