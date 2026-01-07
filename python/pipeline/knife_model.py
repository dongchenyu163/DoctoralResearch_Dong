"""Knife geometry helpers: canonical model + per-pose plane transforms."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Tuple

import numpy as np

try:
    import trimesh
except ImportError:  # pragma: no cover - trimesh optional in some environments
    trimesh = None

from python.utils.config_loader import Config

LOGGER = logging.getLogger("pipeline.knife")


@dataclass
class PlaneInstance:
    """Represents a plane via a point and outward normal."""

    point: np.ndarray
    normal: np.ndarray


@dataclass
class KnifeInstance:
    """Knife geometry at a specific pose."""

    mesh: "trimesh.Trimesh"
    center_plane: PlaneInstance
    positive_plane: PlaneInstance
    negative_plane: PlaneInstance

    @property
    def penetration_plane(self) -> PlaneInstance:
        """Plane located on the -Y side (inside the food)."""
        center = self.center_plane
        pos_side = float(np.dot(self.positive_plane.point - center.point, center.normal))
        neg_side = float(np.dot(self.negative_plane.point - center.point, center.normal))
        plane = self.positive_plane if pos_side < neg_side else self.negative_plane
        normal = center.normal.copy()
        if float(np.dot(normal, center.normal)) < 0.0:
            normal = -normal
        return PlaneInstance(point=plane.point, normal=normal)


class KnifeModel:
    """Canonical knife geometry (isosceles cross section extruded along X)."""

    def __init__(self, mesh: "trimesh.Trimesh", center_plane: PlaneInstance, positive_plane: PlaneInstance, negative_plane: PlaneInstance):
        self._mesh = mesh
        self._center = center_plane
        self._positive = positive_plane
        self._negative = negative_plane

    def instantiate(self, pose: np.ndarray) -> KnifeInstance:
        """Return mesh + plane data transformed by pose (4x4)."""
        if trimesh is None:
            raise RuntimeError("trimesh is required to instantiate KnifeModel")
        if pose.shape != (4, 4):
            raise ValueError("pose must be 4x4 homogeneous matrix")
        mesh = self._mesh.copy()
        mesh.apply_transform(pose)
        center = _transform_plane(self._center, pose)
        positive = _transform_plane(self._positive, pose)
        negative = _transform_plane(self._negative, pose)
        return KnifeInstance(mesh=mesh, center_plane=center, positive_plane=positive, negative_plane=negative)


def build_knife_model(config: Config, reference_points: np.ndarray) -> KnifeModel:
    """Construct canonical knife geometry using config + Ω_low bounds."""
    if trimesh is None:
        raise RuntimeError("trimesh must be installed for knife geometry")
    knife_cfg = config.knife
    height = float(knife_cfg.get("height", 0.05))
    edge_angle = float(knife_cfg.get("edge_angle_deg", 5.0))
    base_length = knife_cfg.get("length")
    if base_length is None:
        base_length = _estimate_length(reference_points, float(knife_cfg.get("length_margin", 0.05)))
    cross_length = float(knife_cfg.get("cross_section_length", base_length))
    mesh, planes = _create_wedge_mesh(float(cross_length), height, edge_angle)
    mesh.fix_normals()
    LOGGER.debug("Knife mesh built length=%.4f height=%.4f edge_angle=%.2f", float(base_length), height, edge_angle)
    return KnifeModel(mesh=mesh, center_plane=planes[0], positive_plane=planes[1], negative_plane=planes[2])


def _estimate_length(points: np.ndarray, margin: float) -> float:
    if points.size == 0:
        return 0.2
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extent = float(maxs[0] - mins[0])
    return max(extent + margin, 0.05)


def _create_wedge_mesh(length: float, height: float, edge_angle_deg: float) -> Tuple["trimesh.Trimesh", Tuple[PlaneInstance, PlaneInstance, PlaneInstance]]:
    half_width = height * np.tan(np.deg2rad(edge_angle_deg) / 2.0)
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, half_width, height],
            [0.0, -half_width, height],
            [length, 0.0, 0.0],
            [length, half_width, height],
            [length, -half_width, height],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [3, 5, 4],
            [0, 3, 4],
            [0, 4, 1],
            [0, 5, 3],
            [0, 2, 5],
            [1, 4, 5],
            [1, 5, 2],
        ],
        dtype=np.int64,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    center_plane = PlaneInstance(point=np.array([0.0, 0.0, height * 0.5], dtype=np.float64), normal=np.array([0.0, 1.0, 0.0], dtype=np.float64))
    positive_plane = _plane_from_points(vertices[0], vertices[1], vertices[4])
    negative_plane = _plane_from_points(vertices[0], vertices[2], vertices[5])
    _ensure_normal_direction(positive_plane, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    _ensure_normal_direction(negative_plane, np.array([0.0, -1.0, 0.0], dtype=np.float64))
    return mesh, (center_plane, positive_plane, negative_plane)


def _plane_from_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> PlaneInstance:
    normal = np.cross(b - a, c - a)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        raise ValueError("Degenerate plane definition for knife mesh")
    normal /= norm
    point = (a + b + c) / 3.0
    return PlaneInstance(point=point, normal=normal)


def _ensure_normal_direction(plane: PlaneInstance, reference: np.ndarray) -> None:
    if float(np.dot(plane.normal, reference)) < 0.0:
        plane.normal *= -1.0


def _transform_plane(plane: PlaneInstance, pose: np.ndarray) -> PlaneInstance:
    point_h = np.array([plane.point[0], plane.point[1], plane.point[2], 1.0], dtype=np.float64)
    point = (pose @ point_h)[:3]
    normal = pose[:3, :3] @ plane.normal
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        normal = plane.normal.copy()
    else:
        normal = normal / norm
    return PlaneInstance(point=point, normal=normal)


__all__ = ["KnifeInstance", "KnifeModel", "PlaneInstance", "build_knife_model"]
