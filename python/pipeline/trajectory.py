"""Trajectory helpers mirroring the C++ test generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from python.instrumentation.timing import TimingRecorder
from python.pipeline.preprocess import PreprocessResult
from python.utils.config_loader import Config


@dataclass
class TrajectoryNode:
    pose: np.ndarray  # Shape (4, 4)
    velocity: np.ndarray  # Shape (3,)


def build_test_trajectory(
    preprocess: PreprocessResult, config: Config, recorder: TimingRecorder
) -> List[TrajectoryNode]:
    points = preprocess.points_low
    if points.size == 0:
        center = np.zeros(3, dtype=np.float64)
    else:
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = (mins + maxs) / 2.0

    offsets = np.asarray(config.trajectory.get("offsets", []), dtype=np.float64)
    if offsets.size == 0:
        offsets = np.zeros((1, 3), dtype=np.float64)

    scalar_velocities = list(config.trajectory.get("scalar_velocities", []))
    if len(scalar_velocities) < offsets.shape[0]:
        scalar_velocities.extend([scalar_velocities[-1] if scalar_velocities else 0.0] * (offsets.shape[0] - len(scalar_velocities)))

    nodes: List[TrajectoryNode] = []
    with recorder.section("python/trajectory_build"):
        for offset in offsets:
            pose = np.eye(4, dtype=np.float64)
            pose[:3, 3] = center + offset
            nodes.append(TrajectoryNode(pose=pose, velocity=np.zeros(3, dtype=np.float64)))

    _calculate_velocity(nodes, scalar_velocities)
    return nodes


def _calculate_velocity(nodes: List[TrajectoryNode], scalars: List[float]) -> None:
    if len(nodes) < 2:
        return
    for idx in range(len(nodes) - 1):
        current = nodes[idx].pose[:3, 3]
        next_pt = nodes[idx + 1].pose[:3, 3]
        direction = next_pt - current
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            nodes[idx].velocity = np.zeros(3, dtype=np.float64)
        else:
            scalar = scalars[min(idx, len(scalars) - 1)]
            nodes[idx].velocity = (direction / norm) * scalar
    nodes[-1].velocity = nodes[-2].velocity.copy()


__all__ = ["TrajectoryNode", "build_test_trajectory"]
