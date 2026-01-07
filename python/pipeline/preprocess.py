"""Point cloud loading, downsampling, normal estimation, and mesh building."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError:  # pragma: no cover - optional dependency
    o3d = None

try:
    import trimesh
except ImportError:  # pragma: no cover - optional dependency
    trimesh = None

from python.instrumentation.timing import TimingRecorder
from python.utils.config_loader import Config
from python.utils import py_gpt

LOGGER = logging.getLogger("pipeline.preprocess")


@dataclass
class RawPointCloud:
    """Raw point cloud loaded from disk or synthesized."""

    source_path: Optional[Path]
    points: np.ndarray


@dataclass
class PreprocessResult:
    """Downsampled point cloud and estimated normals."""

    source_path: Optional[Path]
    original_point_count: int
    downsampled_point_count: int
    points_low: np.ndarray
    normals_low: np.ndarray
    points_high: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float64))
    food_mesh: Optional["trimesh.Trimesh"] = None


def load_point_cloud(config: Config, recorder: TimingRecorder) -> RawPointCloud:
    """Load or synthesize Ω_high according to the config.

    Args:
        config.preprocess.point_cloud_path: File path. If absent, a synthetic cube of
            `synthetic_point_count` points is generated.
        recorder: emits IO instrumentation.
    """

    preprocess_cfg = config.preprocess
    path_value = preprocess_cfg.get("point_cloud_path")
    synthetic_count = int(preprocess_cfg.get("synthetic_point_count", 512))

    with recorder.section("python/io"):
        if path_value:
            source_path = Path(path_value)
        else:
            source_path = None

        points: np.ndarray
        if source_path and source_path.exists():
            try:
                points = _load_points_from_file(source_path)
            except Exception as exc:  # pragma: no cover - exercised via fallback branch
                recorder.emit_event(
                    "python/io",
                    {
                        "level": "warning",
                        "message": f"Failed to load {source_path}: {exc}",
                    },
                )
                points = _generate_synthetic_point_cloud(synthetic_count)
                source_path = None
        else:
            points = _generate_synthetic_point_cloud(synthetic_count)
            source_path = None

    return RawPointCloud(source_path=source_path, points=points)


def preprocess_point_cloud(
    raw_cloud: RawPointCloud, config: Config, recorder: TimingRecorder
) -> PreprocessResult:
    """Downsample (Ω_low) and compute normals.

    Args:
        raw_cloud: Output of `load_point_cloud`.
        config.preprocess:
            - `downsample_num`: Target M. Larger values give more candidates but
              increase Algorithm 2–4 cost roughly linearly (and combinatorially for C_M^N).
            - `normal_estimation_radius`: Open3D ball radius for PCA normals. Increase for
              smoother normals, decrease for sharp details.
        recorder: instrumentation hooks.
    """

    preprocess_cfg = config.preprocess
    downsample_target = int(preprocess_cfg.get("downsample_num", raw_cloud.points.shape[0]))
    normal_radius = float(preprocess_cfg.get("normal_estimation_radius", 0.01))

    with recorder.section("python/preprocess_total"):
        with recorder.section("python/high_res_downsample"):
            high_res = _downsample_points(raw_cloud.points, target=0, voxel_size=float(preprocess_cfg.get("high_res_voxel", 0.001)))
        with recorder.section("python/downsample"):
            downsampled = _downsample_points(high_res, downsample_target)
        with recorder.section("python/estimate_normals"):
            normals = _estimate_normals(downsampled, search_radius=normal_radius)
        food_mesh = _build_food_mesh(high_res, config)

    return PreprocessResult(
        source_path=raw_cloud.source_path,
        original_point_count=int(raw_cloud.points.shape[0]),
        downsampled_point_count=int(downsampled.shape[0]),
        points_low=downsampled,
        normals_low=normals,
        points_high=high_res,
        food_mesh=food_mesh,
    )


def _load_points_from_file(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".ply", ".pcd"}:
        if o3d is None:
            raise RuntimeError("open3d is required to load .ply/.pcd point clouds")
        cloud = o3d.io.read_point_cloud(str(path))  # pragma: no cover - depends on optional data
        if not cloud.has_points():
            raise ValueError(f"Point cloud {path} is empty")
        return np.asarray(cloud.points, dtype=np.float64)
    if suffix == ".npy":
        return np.load(path).astype(np.float64)
    if suffix == ".npz":
        data = np.load(path)
        for key in ("points", "pts", "point_cloud"):
            if key in data:
                return np.asarray(data[key], dtype=np.float64)
        raise KeyError(f"{path} does not contain a 'points' array")
    # Fallback: assume whitespace-separated XYZ
    return np.loadtxt(path, dtype=np.float64, ndmin=2)


def _generate_synthetic_point_cloud(count: int) -> np.ndarray:
    grid = int(round(count ** (1.0 / 3))) or 1
    lin = np.linspace(-0.05, 0.05, grid)
    mesh = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), axis=-1).reshape(-1, 3)
    if mesh.shape[0] >= count:
        return mesh[:count]
    repeat = int(np.ceil(count / mesh.shape[0]))
    tiled = np.tile(mesh, (repeat, 1))
    return tiled[:count]


def _to_point_cloud(points: np.ndarray) -> "o3d.geometry.PointCloud":
    if o3d is None:
        raise RuntimeError("open3d must be installed for point cloud processing")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points, dtype=np.float64))
    return cloud


_VOXEL_CACHE: Dict[Tuple[str, int], float] = {}


def _points_hash(points: np.ndarray) -> str:
    view = np.ascontiguousarray(points, dtype=np.float32)
    return hashlib.sha1(view.tobytes()).hexdigest()


def _downsample_points(points: np.ndarray, target: int, voxel_size: Optional[float] = None) -> np.ndarray:
    """Binary-search voxel size so |points| ≈ target (Ω_low size).

    Args:
        points: Ω_high array (N×3).
        target: Desired downsample_num. Smaller => faster rest of pipeline; larger => better accuracy.
    """
    if target <= 0 and voxel_size is not None:
        return _voxel_downsample(points, voxel_size)
    if target <= 0 or target >= points.shape[0]:
        return np.ascontiguousarray(points, dtype=np.float64)

    cache_key = (_points_hash(points), target)
    cached_voxel = _VOXEL_CACHE.get(cache_key)

    cloud = _to_point_cloud(points)
    bbox = cloud.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    diag = float(np.linalg.norm(extent)) or 0.1
    low = max(diag * 1e-4, 1e-4)
    high = max(diag, 0.2)

    best_points = None
    best_voxel = cached_voxel if cached_voxel is not None else high

    def sample(voxel: float) -> np.ndarray:
        sampled = cloud.voxel_down_sample(voxel)
        return np.asarray(sampled.points, dtype=np.float64)

    if cached_voxel is not None:
        best_points = sample(cached_voxel)
    else:
        for _ in range(20):
            voxel = (low + high) / 2.0
            sampled_points = sample(voxel)
            count = sampled_points.shape[0]
            if best_points is None or abs(count - target) < abs(best_points.shape[0] - target):
                best_points = sampled_points
                best_voxel = voxel
            if count > target:
                low = voxel
            elif count < target:
                high = voxel
            else:
                break

    if best_points is None or best_points.size == 0:
        best_points = np.asarray(points[:target], dtype=np.float64)
    _VOXEL_CACHE[cache_key] = best_voxel

    if best_points.shape[0] > target:
        best_points = best_points[:target]
    elif best_points.shape[0] < target:
        deficit = target - best_points.shape[0]
        fallback = np.ascontiguousarray(points[:deficit], dtype=np.float64)
        best_points = np.concatenate([best_points, fallback], axis=0)
    return np.ascontiguousarray(best_points, dtype=np.float64)


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if o3d is None:
        raise RuntimeError("open3d required for voxel downsample")
    cloud = _to_point_cloud(points)
    sampled = cloud.voxel_down_sample(max(voxel_size, 1e-5))
    return np.asarray(sampled.points, dtype=np.float64)


def _estimate_normals(points: np.ndarray, search_radius: float, max_nn: int = 30) -> np.ndarray:
    """Estimate normals via Open3D ball neighborhoods."""
    if points.size == 0:
        return np.zeros_like(points)
    cloud = _to_point_cloud(points)
    radius = max(search_radius, 1e-4)
    cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    normals = np.asarray(cloud.normals, dtype=np.float64)
    if normals.shape[0] != points.shape[0]:
        raise RuntimeError("open3d failed to estimate normals")
    return normals


def _build_food_mesh(points: np.ndarray, config: Config) -> Optional["trimesh.Trimesh"]:
    """Generate dense food mesh using configured reconstruction method."""
    mesh_cfg = config.preprocess.get("mesh", {})
    if not bool(mesh_cfg.get("enabled", True)):
        return None
    method = mesh_cfg.get("method", "greedy").lower()
    if method == "greedy":
        return _build_mesh_greedy(points, mesh_cfg.get("greedy", {}), config)
    LOGGER.error("Unknown mesh reconstruction method '%s'", method)
    return None


def _build_mesh_greedy(points: np.ndarray, method_cfg: Dict[str, object], config: Config) -> Optional["trimesh.Trimesh"]:
    if points.size < 3:
        LOGGER.error("Ω_high cloud too small for mesh reconstruction")
        return None
    if o3d is None or trimesh is None:
        LOGGER.error("Cannot build food mesh (open3d or trimesh missing)")
        return None
    mls_cfg = method_cfg.get("mls", {})
    if bool(mls_cfg.get("enabled", False)):
        radius = float(mls_cfg.get("radius", 0.01))
        points = _apply_mls(points, radius)
    cloud = _to_point_cloud(points)
    normal_radius = float(method_cfg.get("normal_radius", config.preprocess.get("normal_estimation_radius", 0.01)))
    normal_max_nn = int(method_cfg.get("normal_max_nn", 60))
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(normal_radius, 1e-4), max_nn=max(3, normal_max_nn))
    )
    cloud.orient_normals_consistent_tangent_plane(30)
    normals = np.asarray(cloud.normals, dtype=np.float64)
    if normals.shape[0] != points.shape[0]:
        LOGGER.error("Failed to estimate normals for Ω_high; mesh reconstruction aborted")
        return None
    pts = np.asarray(cloud.points, dtype=np.float64)
    pts_normals = np.hstack([pts, normals])
    params = py_gpt.GPTParams()
    overrides = method_cfg.get("gpt_params", {})
    for key, value in (overrides or {}).items():
        if hasattr(params, key):
            setattr(params, key, value)
    try:
        faces = py_gpt.compute_mesh(pts_normals, params)
    except Exception as exc:  # pragma: no cover - depends on native module
        LOGGER.error("py_gpt mesh reconstruction raised %s", exc)
        return None
    if faces.size == 0:
        LOGGER.error("py_gpt returned empty face set; skip food mesh build")
        return None
    mesh = trimesh.Trimesh(vertices=pts, faces=np.asarray(faces, dtype=np.int64), process=False)
    mesh.fix_normals()
    fill_cfg = method_cfg.get("fill_holes", {})
    if bool(fill_cfg.get("enabled", True)):
        mesh = _fill_mesh_holes(
            mesh,
            skip_longest=bool(fill_cfg.get("skip_longest", True)),
            close_largest=bool(fill_cfg.get("close_largest", True)),
            close_offset_z=float(fill_cfg.get("close_offset_z", 0.01)),
        )
    mesh.fix_normals()
    LOGGER.info(
        "Food mesh (greedy) built via py_gpt: vertices=%d faces=%d radius=%.4f max_nn=%d",
        mesh.vertices.shape[0],
        mesh.faces.shape[0],
        normal_radius,
        normal_max_nn,
    )
    return mesh


def _fill_mesh_holes(mesh: "trimesh.Trimesh", skip_longest: bool, close_largest: bool = True, close_offset_z: float = 0.01) -> "trimesh.Trimesh":
    if mesh is None or mesh.is_empty:
        return mesh
    boundary_edges, edge_to_face = _collect_boundary_edges(mesh)
    loops = _extract_boundary_loops(boundary_edges)
    if not loops:
        return mesh
    skip_index = None
    longest_index = None
    if skip_longest:
        lengths = [_loop_length(mesh.vertices, loop) for loop in loops]
        longest_index = int(np.argmax(lengths))
        skip_index = longest_index
    new_vertices = mesh.vertices.tolist()
    new_faces = mesh.faces.tolist()
    added_faces = 0
    for idx, loop in enumerate(loops):
        if skip_index is not None and idx == skip_index:
            continue
        if len(loop) < 3:
            continue
        desired_normal = _loop_desired_normal(mesh, loop, edge_to_face)
        loop = _orient_loop(mesh.vertices, loop, desired_normal)
        centroid = np.mean(mesh.vertices[loop], axis=0)
        centroid_index = len(new_vertices)
        new_vertices.append(centroid.tolist())
        added_faces += _triangulate_loop(loop, centroid_index, new_faces, mesh.vertices, desired_normal)
    if added_faces == 0:
        return mesh
    if close_largest and longest_index is not None and 0 <= longest_index < len(loops):
        longest_loop = loops[longest_index]
        if len(longest_loop) >= 3:
            _cap_largest_loop(
                longest_loop,
                mesh,
                new_vertices,
                new_faces,
                edge_to_face,
                close_offset_z,
            )
    filled = trimesh.Trimesh(vertices=np.asarray(new_vertices), faces=np.asarray(new_faces, dtype=np.int64), process=False)
    LOGGER.info("Filled mesh holes: loops=%d added_faces=%d skip_longest=%s", len(loops), added_faces, skip_longest)
    return filled


def _cap_largest_loop(
    loop: List[int],
    mesh: "trimesh.Trimesh",
    vertices_out: List[List[float]],
    faces_out: List[List[int]],
    edge_to_face: Dict[Tuple[int, int], int],
    offset_z: float,
) -> None:
    bounds = mesh.bounds
    z_target = float(bounds[0][2]) - float(offset_z)
    desired_normal = _loop_desired_normal(mesh, loop, edge_to_face)
    loop = _orient_loop(mesh.vertices, loop, desired_normal)
    copy_indices: List[int] = []
    for idx in loop:
        v = mesh.vertices[int(idx)].copy()
        v[2] = z_target
        copy_indices.append(len(vertices_out))
        vertices_out.append(v.tolist())
    mesh_center = mesh.vertices.mean(axis=0)
    vertices_all = np.asarray(vertices_out, dtype=np.float64)
    n = len(loop)
    for i in range(n):
        a = int(loop[i])
        b = int(loop[(i + 1) % n])
        a2 = copy_indices[i]
        b2 = copy_indices[(i + 1) % n]
        tri1 = [a, b, b2]
        tri2 = [a, b2, a2]
        if _tri_normal_points_in(vertices_all, tri1, mesh_center):
            faces_out.append(tri1)
        else:
            faces_out.append([b, a, b2])
        if _tri_normal_points_in(vertices_all, tri2, mesh_center):
            faces_out.append(tri2)
        else:
            faces_out.append([b2, a, a2])
    cap_loop = copy_indices
    cap_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    cap_loop = _orient_loop(np.asarray(vertices_out, dtype=np.float64), cap_loop, cap_normal)
    centroid = np.mean(np.asarray(vertices_out, dtype=np.float64)[cap_loop], axis=0)
    centroid_index = len(vertices_out)
    vertices_out.append(centroid.tolist())
    _triangulate_loop(cap_loop, centroid_index, faces_out, np.asarray(vertices_out, dtype=np.float64), cap_normal)


def _tri_normal_points_in(vertices: np.ndarray, tri: List[int], center: np.ndarray) -> bool:
    a, b, c = (vertices[int(idx)] for idx in tri)
    normal = np.cross(b - a, c - a)
    return np.dot(normal, center - ((a + b + c) / 3.0)) < 0.0


def _collect_boundary_edges(mesh: "trimesh.Trimesh") -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    edge_to_face: Dict[Tuple[int, int], int] = {}
    if hasattr(mesh, "edges_unique_faces"):
        edges = mesh.edges_unique
        faces = mesh.edges_unique_faces
        boundary_mask = np.any(faces == -1, axis=1)
        boundary_edges = edges[boundary_mask]
        boundary_faces = faces[boundary_mask]
        for edge, face_pair in zip(boundary_edges, boundary_faces):
            face_idx = int(face_pair[0] if face_pair[0] != -1 else face_pair[1])
            edge_to_face[_edge_key(int(edge[0]), int(edge[1]))] = face_idx
        return boundary_edges, edge_to_face
    face_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    for face_idx, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = _edge_key(int(a), int(b))
            face_counts[key] += 1
            edge_to_face.setdefault(key, int(face_idx))
    boundary_edges = np.array([list(edge) for edge, count in face_counts.items() if count == 1], dtype=np.int64)
    return boundary_edges, edge_to_face


def _extract_boundary_loops(boundary_edges: np.ndarray) -> List[List[int]]:
    adj: Dict[int, List[int]] = defaultdict(list)
    for edge in boundary_edges:
        a = int(edge[0])
        b = int(edge[1])
        adj[a].append(b)
        adj[b].append(a)
    visited_edges: Set[Tuple[int, int]] = set()
    loops: List[List[int]] = []
    for edge in boundary_edges:
        start = int(edge[0])
        nxt = int(edge[1])
        key = _edge_key(start, nxt)
        if key in visited_edges:
            continue
        visited_edges.add(key)
        loop = [start, nxt]
        prev = start
        current = nxt
        while True:
            neighbors = adj.get(current, [])
            next_vertex = None
            for candidate in neighbors:
                if candidate == prev:
                    continue
                candidate_key = _edge_key(current, candidate)
                if candidate_key in visited_edges:
                    continue
                next_vertex = candidate
                break
            if next_vertex is None:
                break
            if next_vertex == loop[0]:
                visited_edges.add(_edge_key(current, next_vertex))
                loops.append(loop)
                break
            loop.append(next_vertex)
            visited_edges.add(_edge_key(current, next_vertex))
            prev, current = current, next_vertex
    return loops


def _loop_length(vertices: np.ndarray, loop: List[int]) -> float:
    points = vertices[loop]
    shifted = np.roll(points, -1, axis=0)
    return float(np.sum(np.linalg.norm(points - shifted, axis=1)))


def _loop_desired_normal(
    mesh: "trimesh.Trimesh", loop: List[int], edge_to_face: Dict[Tuple[int, int], int]
) -> Optional[np.ndarray]:
    normals = []
    for i in range(len(loop)):
        a = int(loop[i])
        b = int(loop[(i + 1) % len(loop)])
        face_idx = edge_to_face.get(_edge_key(a, b))
        if face_idx is not None and 0 <= face_idx < len(mesh.face_normals):
            normals.append(mesh.face_normals[face_idx])
    if normals:
        normal = np.mean(normals, axis=0)
        norm = np.linalg.norm(normal)
        if norm > 1e-8:
            return normal / norm
    return None


def _orient_loop(vertices: np.ndarray, loop: List[int], desired_normal: Optional[np.ndarray]) -> List[int]:
    if desired_normal is None:
        return loop
    loop_normal = _polygon_normal(vertices, loop)
    if np.dot(loop_normal, desired_normal) < 0.0:
        return list(reversed(loop))
    return loop


def _polygon_normal(vertices: np.ndarray, loop: List[int]) -> np.ndarray:
    points = vertices[loop]
    shifted = np.roll(points, -1, axis=0)
    normal = np.sum(np.cross(points, shifted), axis=0)
    norm = np.linalg.norm(normal)
    if norm > 1e-8:
        return normal / norm
    return normal


def _triangulate_loop(
    loop: List[int],
    centroid_index: int,
    faces_out: List[List[int]],
    vertices: np.ndarray,
    desired_normal: Optional[np.ndarray],
) -> int:
    added = 0
    n = len(loop)
    centroid = vertices[loop].mean(axis=0)
    for i in range(n):
        a = int(loop[i])
        b = int(loop[(i + 1) % n])
        tri = [a, b, centroid_index]
        if desired_normal is not None:
            normal = np.cross(vertices[b] - vertices[a], centroid - vertices[a])
            if np.dot(normal, desired_normal) < 0.0:
                tri = [b, a, centroid_index]
        faces_out.append(tri)
        added += 1
    return added


def _edge_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _apply_mls(points: np.ndarray, radius: float) -> np.ndarray:
    try:
        mls_fn = _load_mls_function()
    except RuntimeError as exc:  # pragma: no cover - depends on env
        LOGGER.error("MLS smoothing unavailable: %s", exc)
        return points
    try:
        cloud = _to_point_cloud(points)
        smoothed = mls_fn(cloud, radius=radius)
        LOGGER.info("Applied MLS smoothing (radius=%.4f)", radius)
        return np.asarray(smoothed.points, dtype=np.float64)
    except Exception as exc:  # pragma: no cover
        LOGGER.error("MLS smoothing failed (%s); fallback to raw points", exc)
        return points


def _load_mls_function():
    module_name = "algorithms.points_algo.mls_surface_smooth_numba"
    try:
        module = importlib.import_module(module_name)
        return module.mls_smoothing_numba
    except Exception as first_exc:  # pragma: no cover
        os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        if module_name in sys.modules:
            del sys.modules[module_name]
        try:
            module = importlib.import_module(module_name)
            LOGGER.warning("Imported mls_smoothing_numba with NUMBA_DISABLE_JIT=1 due to %s", first_exc)
            return module.mls_smoothing_numba
        except Exception as final_exc:
            raise RuntimeError(final_exc) from final_exc
