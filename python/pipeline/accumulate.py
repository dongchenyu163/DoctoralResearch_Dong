"""Candidate combination helpers."""

from __future__ import annotations

import itertools
from typing import Dict, Tuple

import numpy as np

from python.instrumentation.timing import TimingRecorder

_COMBINATION_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


def build_all_combinations(point_count: int, finger_count: int, recorder: TimingRecorder) -> np.ndarray:
    """Build (and cache) the \(C_M^N\) index matrix."""

    if finger_count <= 0:
        raise ValueError("finger_count must be positive")
    if point_count <= 0:
        return np.empty((0, finger_count), dtype=np.int32)

    cache_key = (point_count, finger_count)
    cached = _COMBINATION_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    with recorder.section("python/build_P_all"):
        combos = list(itertools.combinations(range(point_count), finger_count))
        matrix = np.array(combos, dtype=np.int32) if combos else np.empty((0, finger_count), dtype=np.int32)
    _COMBINATION_CACHE[cache_key] = matrix
    return matrix.copy()
