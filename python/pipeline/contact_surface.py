"""Contact surface extraction using trimesh booleans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import trimesh

from python.instrumentation.timing import TimingRecorder
from python.pipeline.preprocess import PreprocessResult


@dataclass
class ContactSurfaceResult:
    faces: List[np.ndarray]
    metadata: Dict[str, float]


def extract_contact_surface(
    preprocess: PreprocessResult,
    recorder: TimingRecorder,
    knife_pose: np.ndarray,
) -> ContactSurfaceResult:
    """Compute contact faces used by fracture/friction integrals.

    Steps (per spec §3.4):
        1. Build dense proxy mesh (currently bbox placeholder) from Ω_low.
        2. Intersect with knife mesh transformed by current pose.
        3. Filter faces by alignment with knife side plane to isolate contact surface.
        4. Split connected components to recover Ω_c1, Ω_c2.
    """
    with recorder.section("python/contact_surface_total"):
        dense_mesh = _build_dense_mesh(preprocess)
        knife_mesh = _build_knife_mesh(knife_pose)
        with recorder.section("python/mesh_boolean"):
            try:
                intersection = trimesh.boolean.intersection([dense_mesh, knife_mesh])
            except Exception:
                intersection = dense_mesh.copy()
        purified_faces = _filter_faces(intersection)
        with recorder.section("python/contact_surface_purify"):
            clusters = _split_connected_components(purified_faces)
    metadata = {
        "total_faces": float(len(purified_faces)),
        "components": float(len(clusters)),
    }
    return ContactSurfaceResult(faces=clusters, metadata=metadata)


def _build_dense_mesh(preprocess: PreprocessResult) -> trimesh.Trimesh:
    # Placeholder: use bounding box as proxy
    points = preprocess.points_low
    if points.size == 0:
        points = np.array([[-0.05, -0.05, -0.05], [0.05, 0.05, 0.05]], dtype=np.float64)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    box = trimesh.creation.box(bounds=np.vstack([mins, maxs]))
    return box


def _build_knife_mesh(knife_pose: np.ndarray) -> trimesh.Trimesh:
    verts = np.array(
        [
            [0.0, -0.01, -0.1],
            [0.0, 0.01, -0.1],
            [0.2, 0.0, 0.1],
        ]
    )
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    if knife_pose.shape == (4, 4):
        mesh.apply_transform(knife_pose)
    return mesh


def _filter_faces(mesh: trimesh.Trimesh) -> List[np.ndarray]:
    if mesh.is_empty:
        return []
    faces = []
    normals = mesh.face_normals
    for face, normal in zip(mesh.faces, normals):
        if abs(normal[1]) >= 0.99:  # keep faces parallel to knife side plane
            faces.append(mesh.vertices[face])
    return faces


def _split_connected_components(faces: List[np.ndarray]) -> List[np.ndarray]:
    if not faces:
        return []
    components: List[np.ndarray] = []
    current = []
    last_centroid = None
    for face in faces:
        centroid = face.mean(axis=0)
        if last_centroid is not None and np.linalg.norm(centroid - last_centroid) > 0.05:
            components.append(np.array(current))
            current = []
        current.append(face)
        last_centroid = centroid
    if current:
        components.append(np.array(current))
    return components


__all__ = ["ContactSurfaceResult", "extract_contact_surface"]
