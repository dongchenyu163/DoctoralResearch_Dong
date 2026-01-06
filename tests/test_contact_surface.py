"""Tests for the contact surface extraction pipeline."""

from __future__ import annotations

import unittest

import numpy as np
import trimesh

from python.pipeline.contact_surface import ContactSurfaceResult, extract_contact_surface
from python.pipeline.knife_model import KnifeInstance, PlaneInstance
from python.pipeline.preprocess import PreprocessResult
from tests.helpers import make_config, make_recorder


def _make_dummy_knife_instance() -> KnifeInstance:
    mesh = trimesh.creation.box(extents=(0.1, 0.02, 0.02))
    center = PlaneInstance(point=np.zeros(3, dtype=np.float64), normal=np.array([0.0, 1.0, 0.0], dtype=np.float64))
    positive = PlaneInstance(point=np.array([0.0, 0.01, 0.0], dtype=np.float64), normal=np.array([0.0, 1.0, 0.0], dtype=np.float64))
    negative = PlaneInstance(point=np.array([0.0, -0.01, 0.0], dtype=np.float64), normal=np.array([0.0, -1.0, 0.0], dtype=np.float64))
    return KnifeInstance(mesh=mesh, center_plane=center, positive_plane=positive, negative_plane=negative)


class ContactSurfaceTests(unittest.TestCase):
    def test_extract_contact_surface_returns_metadata(self) -> None:
        config = make_config()
        recorder = make_recorder(config)
        food_mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
        preprocess = PreprocessResult(
            source_path=None,
            original_point_count=2,
            downsampled_point_count=2,
            points_low=np.array([[-0.05, 0.0, -0.05], [0.05, 0.0, 0.05]], dtype=np.float64),
            normals_low=np.zeros((2, 3), dtype=np.float64),
            food_mesh=food_mesh,
        )
        result = extract_contact_surface(preprocess, recorder, knife_instance=_make_dummy_knife_instance())
        self.assertIsInstance(result, ContactSurfaceResult)
        self.assertIn("total_faces", result.metadata)
        self.assertEqual(len(result.faces), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
