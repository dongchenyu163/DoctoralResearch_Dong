"""Tests for the contact surface extraction pipeline."""

from __future__ import annotations

import unittest

import numpy as np

from python.pipeline.contact_surface import ContactSurfaceResult, extract_contact_surface
from python.pipeline.preprocess import PreprocessResult
from tests.helpers import make_config, make_recorder


class ContactSurfaceTests(unittest.TestCase):
    def test_extract_contact_surface_returns_metadata(self) -> None:
        config = make_config()
        recorder = make_recorder(config)
        preprocess = PreprocessResult(
            source_path=None,
            original_point_count=2,
            downsampled_point_count=2,
            points_low=np.array([[-0.05, 0.0, -0.05], [0.05, 0.0, 0.05]], dtype=np.float64),
            normals_low=np.zeros((2, 3), dtype=np.float64),
        )
        result = extract_contact_surface(preprocess, recorder, knife_pose=np.eye(4, dtype=np.float64))
        self.assertIsInstance(result, ContactSurfaceResult)
        self.assertIn("total_faces", result.metadata)
        self.assertGreaterEqual(len(result.faces), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
