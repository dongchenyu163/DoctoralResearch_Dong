"""Tests for candidate combination generation."""

from __future__ import annotations

import unittest

import numpy as np

from python.pipeline.accumulate import build_all_combinations
from tests.helpers import make_recorder


class CombinationTests(unittest.TestCase):
    def test_build_all_combinations_returns_expected_shape(self) -> None:
        recorder = make_recorder()
        combos = build_all_combinations(4, 2, recorder)
        self.assertEqual(combos.shape, (6, 2))
        self.assertTrue(np.array_equal(combos[0], np.array([0, 1], dtype=np.int32)))

    def test_cached_result_is_not_mutated(self) -> None:
        recorder = make_recorder()
        combos_first = build_all_combinations(5, 2, recorder)
        combos_first[0, 0] = 99
        combos_second = build_all_combinations(5, 2, recorder)
        self.assertEqual(combos_second[0, 0], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
