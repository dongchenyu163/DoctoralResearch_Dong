"""Point cloud loading, downsampling, and normal estimation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError:  # pragma: no cover - optional dependency
    o3d = None

from python.instrumentation.timing import TimingRecorder
from python.utils.config_loader import Config


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
        with recorder.section("python/downsample"):
            downsampled = _downsample_points(raw_cloud.points, downsample_target)
        with recorder.section("python/estimate_normals"):
            normals = _estimate_normals(downsampled, search_radius=normal_radius)

    return PreprocessResult(
        source_path=raw_cloud.source_path,
        original_point_count=int(raw_cloud.points.shape[0]),
        downsampled_point_count=int(downsampled.shape[0]),
        points_low=downsampled,
        normals_low=normals,
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


def _downsample_points(points: np.ndarray, target: int) -> np.ndarray:
    """Binary-search voxel size so |points| ≈ target (Ω_low size).

    Args:
        points: Ω_high array (N×3).
        target: Desired downsample_num. Smaller => faster rest of pipeline; larger => better accuracy.
    """
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
