"""Tests for the GeoFilterRunner wrapper."""

from __future__ import annotations

import unittest

import numpy as np

from python.pipeline.geo_filter import GeoFilterRunner
from python.pipeline.preprocess import PreprocessResult
from python.pipeline.valid_indices import ValidIndicesResult
from tests.helpers import make_config, make_recorder


class GeoFilterRunnerTests(unittest.TestCase):
    def test_geo_filter_respects_valid_indices_and_limits(self) -> None:
        config = make_config({"search": {"max_geo_candidates": 1}})
        runner = GeoFilterRunner(config)
        recorder = make_recorder(config)

        preprocess = PreprocessResult(
            source_path=None,
            original_point_count=3,
            downsampled_point_count=3,
            points_low=np.zeros((3, 3), dtype=np.float64),
            normals_low=np.ones((3, 3), dtype=np.float64),
        )
        runner.set_point_cloud(preprocess)

        valid = ValidIndicesResult(
            indices=np.array([0, 1], dtype=np.int32),
            table_threshold=0.0,
            knife_threshold=1.0,
            passed_table=2,
            passed_knife=2,
        )
        candidate_matrix = np.array([[0, 1], [1, 2], [0, 2]], dtype=np.int32)
        result = runner.run(valid, candidate_matrix, recorder)
        self.assertEqual(result.shape[0], 1)
        np.testing.assert_array_equal(result[0], np.array([0, 1], dtype=np.int32))

    def test_positional_scores_return_values(self) -> None:
        config = make_config()
        runner = GeoFilterRunner(config)
        recorder = make_recorder(config)
        preprocess = PreprocessResult(
            source_path=None,
            original_point_count=3,
            downsampled_point_count=3,
            points_low=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.01, 0.0, 0.0],
                    [0.0, 0.01, 0.0],
                ],
                dtype=np.float64,
            ),
            normals_low=np.zeros((3, 3), dtype=np.float64),
        )
        runner.set_point_cloud(preprocess)
        valid = ValidIndicesResult(
            indices=np.array([0, 1, 2], dtype=np.int32),
            table_threshold=0.0,
            knife_threshold=1.0,
            passed_table=3,
            passed_knife=3,
        )
        candidate_matrix = np.array([[0, 1], [1, 2]], dtype=np.int32)
        filtered = runner.run(valid, candidate_matrix, recorder)
        scores = runner.calc_positional_scores(filtered, np.zeros(3, dtype=np.float64), np.array([0.0, 0.0, 1.0]))
        self.assertEqual(scores.shape[0], filtered.shape[0])
        self.assertTrue(np.all(scores >= 0.0))

    def test_positional_distances_normalize(self) -> None:
        config = make_config()
        runner = GeoFilterRunner(config)
        recorder = make_recorder(config)
        preprocess = PreprocessResult(
            source_path=None,
            original_point_count=3,
            downsampled_point_count=3,
            points_low=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.1],
                    [0.0, 0.0, 0.2],
                ],
                dtype=np.float64,
            ),
            normals_low=np.zeros((3, 3), dtype=np.float64),
        )
        runner.set_point_cloud(preprocess)
        valid = ValidIndicesResult(
            indices=np.array([0, 1, 2], dtype=np.int32),
            table_threshold=0.0,
            knife_threshold=1.0,
            passed_table=3,
            passed_knife=3,
        )
        candidate_matrix = np.array([[0, 1], [1, 2]], dtype=np.int32)
        filtered = runner.run(valid, candidate_matrix, recorder)
        distances = runner.calc_positional_distances(filtered, np.zeros(3, dtype=np.float64), np.array([0.0, 0.0, 1.0]))
        self.assertEqual(distances.shape[0], filtered.shape[0])
        self.assertTrue(np.all(distances >= 0.0))
        self.assertTrue(np.all(distances <= 1.0))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
