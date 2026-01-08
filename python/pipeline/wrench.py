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


def compute_wrench(
    surface: ContactSurfaceResult,
    config: Config,
    step_idx: int | None = None,
    velocity: np.ndarray | None = None,
    food_center: np.ndarray | None = None,
) -> np.ndarray:
    """Compute simplified fracture + friction wrench (Algorithm 4 input).

    Args:
        surface: ContactSurfaceResult listing faces per connected component.
        config physics knobs:
            - `pressure_distribution` (Pa). Larger values increase normal forces proportionally.
            - `fracture_toughness` (N/m). Larger values yield bigger fracture force contributions.
            - `friction_coef` μ. Controls tangential friction force magnitude.
            - `planar_constraint` bool. When True, zeroes out non-planar wrench components (z force
              + roll/pitch moments) per spec §3.4.
        velocity: Knife velocity vector; used for fracture/friction directions.
        food_center: Food center of mass g; used for torque arms.
    Returns:
        6-vector `[Fx, Fy, Fz, Tx, Ty, Tz]`.
    """
    step_label = f"Step {step_idx} " if step_idx is not None else ""
    log_boxed_heading(LOGGER, "4.1", f"{step_label}Algorithm 4 Knife Wrench 估计")
    
    # 从配置中提取物理参数
    pressure = float(config.physics.get("pressure_distribution", 1000.0))  # 压强分布 (Pa)
    friction_coef = float(config.physics.get("friction_coef", 0.5))  # 摩擦系数 μ
    fracture_toughness = float(config.physics.get("fracture_toughness", 500.0))  # 断裂韧性 (N/m)
    planar_only = bool(config.physics.get("planar_constraint", True))  # 是否仅保留平面约束分量
    
    LOGGER.debug(
        "Knife wrench params pressure=%.3f fracture=%.3f μ=%.3f planar_only=%s components=%d",
        pressure,
        fracture_toughness,
        friction_coef,
        planar_only,
        len(surface.faces),
    )

    # 提取断裂力和摩擦力网格细化参数
    fracture_step = float(config.physics.get("fracture_step", 0.001))  # 断裂边缘采样步长 (m)
    friction_cfg = config.physics.get("friction_mesh", {})
    friction_sample = int(friction_cfg.get("sample_count", 5000))  # 摩擦面采样点数
    friction_voxel = float(friction_cfg.get("voxel_size", 0.002))  # 体素下采样尺寸 (m)
    friction_edge = float(friction_cfg.get("edge_delta", 0.002))  # 边缘加密步长 (m)

    # 规范化速度向量，作为断裂/摩擦方向
    v_hat, v_norm = _safe_unit(velocity)
    if v_norm < 1e-12:
        LOGGER.warning("Knife velocity missing/zero; wrench forces default to zero")

    # 确定重心 g（用于扭矩臂）
    center = _resolve_food_center(surface, food_center)

    # 初始化累积变量：总力、总扭矩和各面详细信息
    total_force = np.zeros(3, dtype=np.float64)
    total_torque = np.zeros(3, dtype=np.float64)
    face_details = []  # 存储每个面的断裂力、摩擦力、扭矩和面积

    # ===== 1. 计算刀刃断裂力 =====
    # 从两个主接触面的交线构建刀刃线，沿线分布断裂力
    edge_line = _build_edge_line(surface.faces, step=fracture_step)
    fracture_sum = np.zeros(3, dtype=np.float64)
    fracture_torque = np.zeros(3, dtype=np.float64)
    
    if edge_line is not None and v_norm >= 1e-12:
        # blade_normal = _estimate_blade_normal(surface.faces)
        blade_normal = np.array([0, 0, 1])
        blade_total_length = 0
        if blade_normal is None:
            LOGGER.warning("Knife blade normal unavailable; fracture force defaults to zero")
        else:
            for point in edge_line:
                # f_c(u) = -kappa * (v_hat · n_s) * v_hat, integrate with du
                scale = -fracture_toughness * float(np.dot(v_hat, blade_normal))
                force = scale * v_hat * fracture_step
                fracture_sum += force
                fracture_torque += np.cross(point - center, force)  # τ = (r - g) × F
                blade_total_length += fracture_step
            LOGGER.info(
                "刀刃断裂力计算: 刀刃长度=%.4fm 刀刃法线=%s 切割力=%s 切割扭矩=%s",
                blade_total_length,
                _format_vec(blade_normal),
                _format_vec(fracture_sum),
                _format_vec(fracture_torque),
            )
    else:
        LOGGER.error("Knife edge line unavailable; fracture force defaults to zero")

    # ===== 2. 遍历每个接触面簇，计算摩擦力 =====
    for cluster in surface.faces:
        friction_sum = np.zeros(3, dtype=np.float64)
        torque_sum = np.zeros(3, dtype=np.float64)
        total_area = 0.0
        
        # 构建均匀采样的接触面网格（包含表面采样、体素下采样和边缘加密）
        mesh = _build_uniform_contact_mesh(cluster, friction_sample, friction_voxel, friction_edge)
        triangles = mesh.triangles if mesh is not None else cluster.reshape((-1, 3, 3))
        if bool(config.physics.get("debug_friction_contact_viz", False)):
            _show_friction_contact_debug(mesh, surface.mesh)
        
        # 遍历该簇的所有三角形面片
        for tri in triangles:
            a, b, c = tri
            # 计算法向量和面积
            normal_vec = np.cross(b - a, c - a)
            area = 0.5 * np.linalg.norm(normal_vec)
            if not np.isfinite(area) or area < 1e-14:
                continue  # 跳过退化三角形
            total_area += area
            
            # 归一化法向量
            normal = normal_vec / (np.linalg.norm(normal_vec) + 1e-12)
            centroid = (a + b + c) / 3.0  # 三角形质心
            
            if v_norm < 1e-12:
                continue

            # 速度方向投影到接触面：v_cj = v_hat - (v_hat·n) n
            v_proj = v_hat - float(np.dot(v_hat, normal)) * normal
            v_proj_norm = np.linalg.norm(v_proj)
            if v_proj_norm < 1e-12:
                continue
            v_hat_cj = v_proj / v_proj_norm

            # 计算摩擦力：F_f = μ × P × area × v_hat_cj
            friction_force = friction_coef * pressure * area * v_hat_cj
            friction_sum += friction_force
            torque_sum += np.cross(centroid - center, friction_force)  # 累积扭矩

        # 记录该面簇的详细信息（断裂力、摩擦力、总扭矩、总面积）
        face_details.append((fracture_sum.copy(), friction_sum.copy(), fracture_torque.copy() + torque_sum.copy(), total_area))
        
        # 累加到全局总力和总扭矩
        total_force += fracture_sum + friction_sum
        total_torque += fracture_torque + torque_sum

    # ===== 3. 组装最终扭矩向量 =====
    if not face_details:
        LOGGER.warning("未检测到接触面，返回零扭矩")
    
    # 构造6维扭矩向量 [Fx, Fy, Fz, Tx, Ty, Tz]
    wrench = np.zeros(6, dtype=np.float64)
    wrench[:3] = total_force  # 前3维：合力
    wrench[3:] = total_torque  # 后3维：合扭矩
    
    # 输出各面簇的详细信息
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
    
    # ===== 4. 应用平面约束（可选） =====
    # 根据规格书§3.4，平面约束会将Z向力和Roll/Pitch扭矩置零
    if planar_only:
        wrench[2] = 0.0   # Fz = 0
        wrench[3] = 0.0   # Tx (Roll) = 0
        wrench[4] = 0.0   # Ty (Pitch) = 0
        LOGGER.info("planar_only→处理后的合外力=%s 合扭矩=%s", _format_vec(wrench[:3]), _format_vec(wrench[3:]))
    
    LOGGER.debug("Final wrench vector=%s", _format_vec(wrench))
    return wrench


def _format_vec(vec: np.ndarray) -> str:
    return np.array2string(np.asarray(vec, dtype=np.float64), precision=5, separator=",", suppress_small=True)


def _safe_unit(vec: np.ndarray | None) -> Tuple[np.ndarray, float]:
    if vec is None:
        return np.zeros(3, dtype=np.float64), 0.0
    arr = np.asarray(vec, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64), norm
    return arr / norm, norm


def _resolve_food_center(surface: ContactSurfaceResult, food_center: np.ndarray | None) -> np.ndarray:
    if food_center is not None:
        return np.asarray(food_center, dtype=np.float64).reshape(3)
    if surface.mesh is not None and not surface.mesh.is_empty:
        if surface.mesh.is_volume:
            return np.asarray(surface.mesh.center_mass, dtype=np.float64)
        return np.asarray(surface.mesh.centroid, dtype=np.float64)
    face_points = _flatten_faces(surface.faces)
    if face_points.size:
        return face_points.mean(axis=0)
    return np.zeros(3, dtype=np.float64)


def _estimate_blade_normal(face_groups: List[np.ndarray]) -> Optional[np.ndarray]:
    normals = []
    for group in face_groups:
        pts = np.asarray(group, dtype=np.float64).reshape(-1, 3)
        if pts.size:
            normals.append(_fit_plane_normal(pts))
    if not normals:
        return None
    summed = np.sum(normals, axis=0)
    norm = np.linalg.norm(summed)
    if norm < 1e-9:
        return normals[0]
    return summed / norm


def _flatten_faces(face_groups: List[np.ndarray]) -> np.ndarray:
    if not face_groups:
        return np.empty((0, 3), dtype=np.float64)
    points = [np.asarray(group, dtype=np.float64).reshape(-1, 3) for group in face_groups if group.size]
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(points)


def _show_friction_contact_debug(mesh: Optional["trimesh.Trimesh"], surface_mesh: Optional["trimesh.Trimesh"]) -> None:
    if trimesh is None:
        LOGGER.warning("trimesh unavailable; skip friction contact visualization")
        return
    if mesh is None and surface_mesh is None:
        LOGGER.warning("No meshes for friction contact visualization")
        return
    scene = trimesh.Scene()
    if surface_mesh is not None:
        surface_copy = surface_mesh.copy()
        surface_copy.apply_translation([0.0, 0.02, 0.0])
        surface_edges = surface_copy.edges_unique
        surface_entities = [trimesh.path.entities.Line(edge) for edge in surface_edges]
        surface_wireframe = trimesh.path.Path3D(entities=surface_entities, vertices=surface_copy.vertices)
        scene.add_geometry(surface_wireframe)
        scene.add_geometry(surface_copy)
    if mesh is not None:
        scene.add_geometry(mesh)
        edges = mesh.edges_unique
        entities = [trimesh.path.entities.Line(edge) for edge in edges]
        wireframe = trimesh.path.Path3D(entities=entities, vertices=mesh.vertices)
        scene.add_geometry(wireframe)
    scene.show()


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
