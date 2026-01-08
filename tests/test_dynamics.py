"""Tests for dynamics scoring placeholder."""

from __future__ import annotations

import unittest

import numpy as np

from python.pipeline.dynamics import compute_dynamics_scores
from python.pipeline.geo_filter import GeoFilterRunner
from python.pipeline.preprocess import PreprocessResult
from tests.helpers import make_config, make_recorder


class DynamicsTests(unittest.TestCase):
    def test_compute_dynamics_scores_returns_values(self) -> None:
        config = make_config()
        recorder = make_recorder(config)
        runner = GeoFilterRunner(config)
        preprocess = PreprocessResult(
            source_path=None,
            original_point_count=3,
            downsampled_point_count=3,
            points_low=np.zeros((3, 3), dtype=np.float64),
            normals_low=np.zeros((3, 3), dtype=np.float64),
        )
        runner.set_point_cloud(preprocess)
        candidates = np.array([[0, 1], [1, 2]], dtype=np.int32)
        wrench = np.ones(6, dtype=np.float64)
        scores = compute_dynamics_scores(runner, candidates, wrench, np.zeros(3, dtype=np.float64), config)
        self.assertEqual(scores.shape[0], candidates.shape[0])
        self.assertFalse(np.any(np.isnan(scores)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
