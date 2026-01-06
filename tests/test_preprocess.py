"""Tests for the preprocessing stage."""

from __future__ import annotations

import unittest

import numpy as np

from tests.helpers import make_config, make_recorder
from python.pipeline.preprocess import PreprocessResult, load_point_cloud, preprocess_point_cloud


class PreprocessTests(unittest.TestCase):
    def test_load_point_cloud_generates_synthetic_when_missing(self) -> None:
        config = make_config({"preprocess": {"point_cloud_path": None, "synthetic_point_count": 16}})
        recorder = make_recorder(config)
        raw_cloud = load_point_cloud(config, recorder)
        self.assertIsNone(raw_cloud.source_path)
        self.assertEqual(raw_cloud.points.shape, (16, 3))

    def test_preprocess_point_cloud_produces_normals(self) -> None:
        config = make_config(
            {
                "preprocess": {
                    "point_cloud_path": None,
                    "synthetic_point_count": 64,
                    "downsample_num": 20,
                    "normal_estimation_radius": 0.02,
                }
            }
        )
        recorder = make_recorder(config)
        raw_cloud = load_point_cloud(config, recorder)
        result: PreprocessResult = preprocess_point_cloud(raw_cloud, config, recorder)
        self.assertEqual(result.points_low.shape, (20, 3))
        self.assertEqual(result.normals_low.shape, (20, 3))
        norms = np.linalg.norm(result.normals_low, axis=1)
        self.assertTrue(np.all(np.isfinite(norms)))
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-6))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
