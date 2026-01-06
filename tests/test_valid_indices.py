"""Tests for valid index computation."""

from __future__ import annotations

import unittest

import numpy as np
import trimesh

from python.pipeline.knife_model import KnifeInstance, PlaneInstance
from python.pipeline.valid_indices import compute_valid_indices
from tests.helpers import make_config, make_recorder


def _dummy_knife() -> KnifeInstance:
    mesh = trimesh.creation.box(extents=(0.1, 0.02, 0.02))
    center = PlaneInstance(point=np.zeros(3, dtype=np.float64), normal=np.array([0.0, 1.0, 0.0], dtype=np.float64))
    positive = PlaneInstance(point=np.array([0.0, 0.01, 0.0], dtype=np.float64), normal=np.array([0.0, 1.0, 0.0], dtype=np.float64))
    negative = PlaneInstance(point=np.array([0.0, -0.01, 0.0], dtype=np.float64), normal=np.array([0.0, -1.0, 0.0], dtype=np.float64))
    return KnifeInstance(mesh=mesh, center_plane=center, positive_plane=positive, negative_plane=negative)


class ValidIndicesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.points = np.array(
            [
                [0.0, 0.0, -0.01],
                [0.0, 0.0, 0.005],
                [0.0, 0.0, 0.02],
                [0.0, 0.0, 0.06],
            ],
            dtype=np.float64,
        )
        self.knife = _dummy_knife()

    def test_threshold_filters_match_expected_indices(self) -> None:
        config = make_config(
            {
                "environment": {"table_z": 0.0},
                "search": {"table_clearance": 0.01, "knife_clearance": 0.005},
                "knife": {"height": 0.05},
            }
        )
        recorder = make_recorder(config)
        result = compute_valid_indices(self.points, config, recorder, self.knife)
        self.assertEqual(result.indices.tolist(), [2])
        self.assertEqual(result.passed_table, 2)
        self.assertEqual(result.passed_knife, 3)
        self.assertEqual(result.passed_center_plane, 4)
        self.assertEqual(result.passed_penetration_plane, 4)

    def test_extreme_thresholds_return_empty(self) -> None:
        config = make_config({"search": {"table_clearance": 0.2}})
        recorder = make_recorder(config)
        result = compute_valid_indices(self.points, config, recorder, self.knife)
        self.assertEqual(result.indices.size, 0)

    def test_extreme_thresholds_return_full_set(self) -> None:
        config = make_config({"search": {"table_clearance": -1.0, "knife_clearance": -1.0, "knife_plane_clearance": 1.0}})
        recorder = make_recorder(config)
        result = compute_valid_indices(self.points, config, recorder, self.knife)
        self.assertEqual(result.indices.size, self.points.shape[0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
