"""Tests for trajectory generation and velocity computation."""

from __future__ import annotations

import unittest

import numpy as np

from python.pipeline.preprocess import PreprocessResult
from python.pipeline.trajectory import TrajectoryNode, build_test_trajectory
from tests.helpers import make_config, make_recorder


class TrajectoryTests(unittest.TestCase):
    def test_offsets_are_applied_relative_to_center(self) -> None:
        config = make_config()
        recorder = make_recorder(config)
        preprocess = PreprocessResult(
            source_path=None,
            original_point_count=8,
            downsampled_point_count=8,
            points_low=np.array(
                [
                    [-0.05, 0.0, -0.05],
                    [0.05, 0.0, 0.05],
                ],
                dtype=np.float64,
            ),
            normals_low=np.zeros((2, 3), dtype=np.float64),
        )
        nodes = build_test_trajectory(preprocess, config, recorder)
        self.assertEqual(len(nodes), 3)
        positions = [node.pose[:3, 3] for node in nodes]
        expected_center = np.array([0.0, 0.0, 0.0])
        offsets = np.asarray(config.trajectory["offsets"], dtype=np.float64)
        for pos, offset in zip(positions, offsets):
            np.testing.assert_allclose(pos, expected_center + offset)

    def test_velocity_matches_direction_and_scalar(self) -> None:
        config = make_config({"trajectory": {"scalar_velocities": [0.02, 0.03, 0.04]}})
        recorder = make_recorder(config)
        preprocess = PreprocessResult(
            source_path=None,
            original_point_count=3,
            downsampled_point_count=3,
            points_low=np.zeros((3, 3), dtype=np.float64),
            normals_low=np.zeros((3, 3), dtype=np.float64),
        )
        nodes = build_test_trajectory(preprocess, config, recorder)
        self.assertGreater(len(nodes), 1)
        first_dir = nodes[0].pose[:3, 3] - nodes[1].pose[:3, 3]
        self.assertTrue(np.linalg.norm(nodes[0].velocity) > 0)
        self.assertAlmostEqual(np.linalg.norm(nodes[0].velocity), 0.02, places=6)
        self.assertAlmostEqual(np.linalg.norm(nodes[1].velocity), 0.03, places=6)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
