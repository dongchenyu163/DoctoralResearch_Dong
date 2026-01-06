"""Tests for candidate combination generation."""

from __future__ import annotations

import unittest

import numpy as np

from python.pipeline.accumulate import ScoreAccumulator, build_all_combinations
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


class ScoreAccumulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.combinations = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int32)
        self.acc = ScoreAccumulator(self.combinations)

    def test_accumulate_and_best_candidate(self) -> None:
        self.acc.mark_eliminated([1], "geo_filter", 0)
        self.acc.accumulate([0, 2], np.array([0.5, 0.2]), np.array([0.25, 0.3]))
        best = self.acc.best_candidate_index()
        self.assertIsNotNone(best)
        self.assertEqual(best, 0)
        top = self.acc.top_candidates(2)
        self.assertEqual(top, [0, 2])

    def test_elimination_summary_tracks_reasons(self) -> None:
        self.acc.mark_eliminated([0], "invalid_indices", 0)
        self.acc.mark_eliminated([1], "geo_filter", 0)
        summary = self.acc.elimination_summary()
        self.assertEqual(summary["invalid_indices"], 1)
        self.assertEqual(summary["geo_filter"], 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
