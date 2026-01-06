"""Point cloud loading, downsampling, and normal estimation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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
    """Load or synthesize a point cloud according to the config."""

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
    """Downsample and compute normals for the low-density cloud."""

    preprocess_cfg = config.preprocess
    downsample_target = int(preprocess_cfg.get("downsample_num", raw_cloud.points.shape[0]))
    normal_radius = float(preprocess_cfg.get("normal_estimation_radius", 0.01))

    with recorder.section("python/preprocess_total"):
        with recorder.section("python/downsample"):
            downsampled = _downsample_points(raw_cloud.points, downsample_target, seed=config.seed)
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


def _downsample_points(points: np.ndarray, target: int, seed: Optional[int]) -> np.ndarray:
    if target <= 0 or target >= points.shape[0]:
        return np.ascontiguousarray(points, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=target, replace=False)
    indices.sort()
    return np.ascontiguousarray(points[indices], dtype=np.float64)


def _estimate_normals(points: np.ndarray, search_radius: float, min_neighbors: int = 6) -> np.ndarray:
    if points.size == 0:
        return np.zeros_like(points)
    normals = np.zeros_like(points)
    radius_sq = max(search_radius, 1e-6) ** 2
    for idx, point in enumerate(points):
        offsets = points - point
        dists = np.einsum("ij,ij->i", offsets, offsets)
        neighbor_mask = (dists > 0.0) & (dists <= radius_sq)
        neighbor_indices = np.nonzero(neighbor_mask)[0]
        if neighbor_indices.size < min_neighbors:
            neighbor_indices = np.argsort(dists)[1 : min_neighbors + 1]
        neighborhood = offsets[neighbor_indices]
        cov = neighborhood.T @ neighborhood
        try:
            _, _, vh = np.linalg.svd(cov, hermitian=True)
            normal = vh[-1]
        except np.linalg.LinAlgError:
            normal = np.array([0.0, 0.0, 1.0])
        norm = float(np.linalg.norm(normal))
        if norm < 1e-12:
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal = normal / norm
        normals[idx] = normal
    return normals
