"""Utilities for saving debug point clouds / meshes."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    import open3d as o3d
except ImportError:  # pragma: no cover
    o3d = None

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None

LOGGER = logging.getLogger("pipeline.debug_pc")


@dataclass
class PointCloudLoggingSettings:
    enabled: bool
    output_dir: Path
    stages: Dict[str, bool]


def load_pointcloud_logging_config(path: Path) -> PointCloudLoggingSettings:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    output_dir = Path(data.get("output_dir", "debug_pc_calc_data"))
    stages = {name: bool(value) for name, value in data.get("stages", {}).items()}
    return PointCloudLoggingSettings(enabled=bool(data.get("enabled", True)), output_dir=output_dir, stages=stages)


class PointCloudDebugSaver:
    """Handles optional saving of intermediate clouds/meshes."""

    def __init__(self, settings: PointCloudLoggingSettings):
        self.settings = settings
        if self.settings.enabled:
            self.settings.output_dir.mkdir(parents=True, exist_ok=True)

    def enabled_for(self, stage: str) -> bool:
        if not self.settings.enabled:
            return False
        return self.settings.stages.get(stage, False)

    def save_point_cloud(self, stage: str, timestep: int, points: np.ndarray, colors: Optional[np.ndarray] = None) -> Optional[Path]:
        if not self.enabled_for(stage):
            return None
        if points.size == 0:
            LOGGER.warning("Skip saving stage %s at step %d because point cloud empty", stage, timestep)
            return None
        if colors is None:
            colors = np.ones_like(points, dtype=np.float64)
        if colors.shape != points.shape:
            raise ValueError("colors shape must match points shape")
        path = self.settings.output_dir / f"step_{timestep:03d}_{stage}.ply"
        if o3d is None:
            np.save(path.with_suffix(".npy"), np.hstack([points, colors]))
        else:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
            o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)
        return path

    def save_mesh(self, stage: str, timestep: int, mesh: "trimesh.Trimesh") -> Optional[Path]:
        if not self.enabled_for(stage):
            return None
        if mesh is None or mesh.is_empty:
            LOGGER.warning("Skip saving mesh stage %s at step %d because mesh empty", stage, timestep)
            return None
        path = self.settings.output_dir / f"step_{timestep:03d}_{stage}.obj"
        if trimesh is None:
            LOGGER.warning("trimesh not available; skip saving %s", path)
            return None
        mesh.export(path)
        return path


__all__ = ["PointCloudLoggingSettings", "PointCloudDebugSaver", "load_pointcloud_logging_config"]
