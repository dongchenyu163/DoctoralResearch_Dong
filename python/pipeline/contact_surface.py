"""Contact surface extraction using MLS-derived food mesh and knife planes."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import trimesh

from python.instrumentation.timing import TimingRecorder
from python.pipeline.knife_model import KnifeInstance, PlaneInstance
from python.pipeline.preprocess import PreprocessResult

LOGGER = logging.getLogger("pipeline.contact_surface")
_BLENDER_PREFIX = "/home/cookteam/Documents/blender-4.5.3-linux-x64"
if _BLENDER_PREFIX not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"{_BLENDER_PREFIX}:{os.environ.get('PATH', '')}"


@dataclass
class ContactSurfaceResult:
    faces: List[np.ndarray]
    metadata: Dict[str, float]
    mesh: Optional[trimesh.Trimesh]


def extract_contact_surface(
    preprocess: PreprocessResult,
    recorder: TimingRecorder,
    knife_instance: KnifeInstance,
    normal_threshold: float = 0.9,
) -> ContactSurfaceResult:
    """Compute contact faces for Algorithm 4 based on current knife pose."""
    base_mesh = preprocess.food_mesh or _build_dense_mesh(preprocess.points_low)
    if base_mesh is None:
        LOGGER.error("Base food mesh missing; cannot extract contact surfaces")
        return ContactSurfaceResult(faces=[], metadata={"total_faces": 0.0, "components": 0.0}, mesh=None)
    with recorder.section("python/contact_surface_total"):
        with recorder.section("python/mesh_boolean"):
            try:
                intersection = base_mesh.intersection(knife_instance.mesh, engine='blender', check_volume=False, use_exact=True)
                if isinstance(intersection, (list, tuple)):
                    intersection = trimesh.util.concatenate(intersection)
            except Exception as exc:  # pragma: no cover - backend specific
                LOGGER.error("Mesh boolean failed (%s); returning empty contact surface", exc)
                LOGGER.error("Mesh boolean failed (%s); export PATH=${PATH}:/home/cookteam/Documents/blender-4.5.3-linux-x64", exc)
                intersection = None
        faces = _filter_faces_with_planes(
            intersection,
            knife_instance.center_plane,
            [knife_instance.positive_plane, knife_instance.negative_plane],
            normal_threshold,
        )
    total_faces = sum(face_group.shape[0] for face_group in faces)
    metadata = {
        "total_faces": float(total_faces),
        "components": float(len([arr for arr in faces if arr.size])),
        "side_counts": [float(group.shape[0]) for group in faces],
    }
    LOGGER.debug(
        "Contact surface faces=%d components=%d bounds=%s",
        int(metadata["total_faces"]),
        int(metadata["components"]),
        np.vstack([base_mesh.bounds[0], base_mesh.bounds[1]]).tolist(),
    )
    for idx, group in enumerate(faces):
        if group.size == 0:
            LOGGER.error("Contact surface side %d empty after filtering", idx)
        else:
            LOGGER.info("Contact surface side %d triangles=%d", idx, group.shape[0])
    return ContactSurfaceResult(faces=faces, metadata=metadata, mesh=intersection)


def _build_dense_mesh(points: np.ndarray) -> Optional[trimesh.Trimesh]:
    if points.size == 0:
        return None
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return trimesh.creation.box(bounds=np.vstack([mins, maxs]))


def _filter_faces_with_planes(
    mesh: Optional[trimesh.Trimesh],
    center_plane: PlaneInstance,
    side_planes: List[PlaneInstance],
    normal_threshold: float,
) -> List[np.ndarray]:
    if mesh is None or mesh.is_empty:
        LOGGER.error("Intersection mesh empty; no contact faces found")
        return [np.empty((0, 3, 3), dtype=np.float64), np.empty((0, 3, 3), dtype=np.float64)]
    triangles = mesh.triangles
    normals = mesh.face_normals
    centers = mesh.triangles_center
    grouped: List[List[np.ndarray]] = [[], []]
    for tri, normal, centroid in zip(triangles, normals, centers):
        matches = [idx for idx, plane in enumerate(side_planes) if float(np.dot(normal, plane.normal)) >= normal_threshold]
        if not matches:
            continue
        side_value = float(np.dot(centroid - center_plane.point, center_plane.normal))
        bucket = 0 if side_value >= 0 else 1
        grouped[bucket].append(tri)
    return [np.asarray(group, dtype=np.float64) if group else np.empty((0, 3, 3), dtype=np.float64) for group in grouped]


__all__ = ["ContactSurfaceResult", "extract_contact_surface"]
