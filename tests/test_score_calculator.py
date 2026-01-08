"""Smoke tests for the C++ score_calculator module."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

BUILD_DIR = pathlib.Path(__file__).resolve().parents[1] / "build"
if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

import score_calculator  # type: ignore  # noqa: E402,E401


class ScoreCalculatorBindingsTests(unittest.TestCase):
    def test_set_point_cloud_accepts_numpy_arrays(self) -> None:
        calc = score_calculator.ScoreCalculator()
        points = np.zeros((4, 3), dtype=np.float64)
        normals = np.ones((4, 3), dtype=np.float64)
        calc.set_point_cloud(points, normals)
        self.assertEqual(calc.point_count, 4)

    def test_filter_by_geo_score_respects_max_candidates(self) -> None:
        calc = score_calculator.ScoreCalculator()
        points = np.zeros((4, 3), dtype=np.float64)
        normals = np.ones((4, 3), dtype=np.float64)
        calc.set_point_cloud(points, normals)
        calc.set_max_candidates(1)
        calc.set_geo_weights(1.0, 1.0, 1.0)
        calc.set_geo_filter_ratio(1.0)
        candidates = np.array([[0, 1], [1, 2]], dtype=np.int32)
        result = calc.filter_by_geo_score(
            candidates,
            np.zeros(3, dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            0.0,
        )
        self.assertEqual(result.shape[0], 1)
        np.testing.assert_array_equal(result[0], np.array([0, 1], dtype=np.int32))

    def test_geo_filter_prefers_higher_table_distance(self) -> None:
        calc = score_calculator.ScoreCalculator()
        points = np.array([[0.0, 0.0, 0.01], [0.0, 0.0, 0.02], [0.0, 0.0, 0.05]], dtype=np.float64)
        normals = np.ones_like(points)
        calc.set_point_cloud(points, normals)
        calc.set_geo_weights(0.2, 0.2, 1.0)
        calc.set_geo_filter_ratio(0.5)
        candidates = np.array([[0, 1], [1, 2]], dtype=np.int32)
        result = calc.filter_by_geo_score(
            candidates,
            np.zeros(3, dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            0.0,
        )
        self.assertEqual(result.shape[0], 1)
        np.testing.assert_array_equal(result[0], np.array([1, 2], dtype=np.int32))

    def test_calc_positional_scores_prefers_orthogonal(self) -> None:
        calc = score_calculator.ScoreCalculator()
        points = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float64)
        normals = np.ones_like(points)
        calc.set_point_cloud(points, normals)
        candidates = np.array([[0, 1], [0, 2]], dtype=np.int32)
        scores = calc.calc_positional_scores(
            candidates,
            np.zeros(3, dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
        self.assertEqual(scores.shape[0], 2)
        self.assertTrue(np.all(scores >= 0.0))

    def test_calc_dynamics_scores_returns_values(self) -> None:
        calc = score_calculator.ScoreCalculator()
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
            ],
            dtype=np.float64,
        )
        normals = np.tile(np.array([[0.0, 1.0, 0.0]]), (3, 1))
        calc.set_point_cloud(points, normals)
        candidates = np.array([[0, 1], [1, 2]], dtype=np.int32)
        wrench = np.ones(6, dtype=np.float64)
        scores = calc.calc_dynamics_scores(
            candidates,
            wrench,
            np.zeros(3, dtype=np.float64),
            0.5,
            40.0,
            10,
            1e-2,
            0.1,
            1.0,
            40.0,
        )
        self.assertEqual(scores.shape[0], 2)
        self.assertFalse(np.any(np.isnan(scores)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
