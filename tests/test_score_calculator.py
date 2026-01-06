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
        np.testing.assert_array_equal(np.array(calc.points), points)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
