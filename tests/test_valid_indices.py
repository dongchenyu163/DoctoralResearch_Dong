"""Tests for valid index computation."""

from __future__ import annotations

import unittest

import numpy as np

from python.pipeline.valid_indices import compute_valid_indices
from tests.helpers import make_config, make_recorder


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

    def test_threshold_filters_match_expected_indices(self) -> None:
        config = make_config(
            {
                "environment": {"table_z": 0.0},
                "search": {"table_clearance": 0.01, "knife_clearance": 0.005},
                "knife": {"height": 0.05},
            }
        )
        recorder = make_recorder(config)
        result = compute_valid_indices(self.points, config, recorder)
        self.assertEqual(result.indices.tolist(), [2])
        self.assertEqual(result.passed_table, 2)
        self.assertEqual(result.passed_knife, 3)

    def test_extreme_thresholds_return_empty(self) -> None:
        config = make_config({"search": {"table_clearance": 0.2}})
        recorder = make_recorder(config)
        result = compute_valid_indices(self.points, config, recorder)
        self.assertEqual(result.indices.size, 0)

    def test_extreme_thresholds_return_full_set(self) -> None:
        config = make_config({"search": {"table_clearance": -1.0, "knife_clearance": -1.0}})
        recorder = make_recorder(config)
        result = compute_valid_indices(self.points, config, recorder)
        self.assertEqual(result.indices.size, self.points.shape[0])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
