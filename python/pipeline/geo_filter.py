"""Python wrapper around the C++ ScoreCalculator GeoFilter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import score_calculator  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    build_dir = Path(__file__).resolve().parents[2] / "build"
    sys.path.insert(0, str(build_dir))
    import score_calculator  # type: ignore

from python.instrumentation.timing import TimingRecorder
from python.pipeline.preprocess import PreprocessResult
from python.pipeline.valid_indices import ValidIndicesResult
from python.utils.config_loader import Config


class GeoFilterRunner:
    """High-level helper that coordinates ScoreCalculator usage."""

    def __init__(self, config: Config):
        self.config = config
        self._calculator = score_calculator.ScoreCalculator()
        max_candidates = int(config.search.get("max_geo_candidates", 0))
        if max_candidates > 0:
            self._calculator.set_max_candidates(max_candidates)
        geo_weights = config.weights.get("geo_score", {})
        self._calculator.set_geo_weights(
            float(geo_weights.get("w_fin", 1.0)),
            float(geo_weights.get("w_knf", 1.0)),
            float(geo_weights.get("w_tbl", 1.0)),
        )
        geo_ratio = float(config.search.get("geo_filter_ratio", 1.0))
        self._calculator.set_geo_filter_ratio(geo_ratio)

    def set_point_cloud(self, preprocess_result: PreprocessResult) -> None:
        self._calculator.set_point_cloud(preprocess_result.points_low, preprocess_result.normals_low)

    def run(
        self,
        valid_indices: ValidIndicesResult,
        candidate_matrix: np.ndarray,
        recorder: TimingRecorder,
    ) -> np.ndarray:
        filtered_candidates = _mask_candidates(candidate_matrix, valid_indices.indices)
        knife_position = np.zeros(3, dtype=np.float64)
        knife_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        table_z = float(self.config.environment.get("table_z", 0.0))

        with recorder.section("python/geo_filter_call_cpp"):
            result = self._calculator.filter_by_geo_score(filtered_candidates, knife_position, knife_normal, table_z)
        return np.asarray(result, dtype=np.int32)

    def calc_positional_scores(
        self,
        candidate_matrix: np.ndarray,
        knife_position: np.ndarray,
        knife_normal: np.ndarray,
    ) -> np.ndarray:
        if candidate_matrix.size == 0:
            return np.zeros((0,), dtype=np.float64)
        scores = self._calculator.calc_positional_scores(candidate_matrix, knife_position, knife_normal)
        return np.asarray(scores, dtype=np.float64)

    def calc_positional_distances(
        self,
        candidate_matrix: np.ndarray,
        knife_position: np.ndarray,
        knife_normal: np.ndarray,
    ) -> np.ndarray:
        if candidate_matrix.size == 0:
            return np.zeros((0,), dtype=np.float64)
        scores = self._calculator.calc_positional_distances(candidate_matrix, knife_position, knife_normal)
        return np.asarray(scores, dtype=np.float64)

    @property
    def calculator(self):
        return self._calculator


def _mask_candidates(candidate_matrix: np.ndarray, valid_indices: np.ndarray) -> np.ndarray:
    if candidate_matrix.ndim != 2:
        return np.empty((0, 0), dtype=np.int32)
    if candidate_matrix.size == 0 or valid_indices.size == 0:
        return np.empty((0, candidate_matrix.shape[1]), dtype=np.int32)
    valid_mask = np.isin(candidate_matrix, valid_indices)
    row_mask = np.all(valid_mask, axis=1)
    return candidate_matrix[row_mask]


__all__ = ["GeoFilterRunner"]
