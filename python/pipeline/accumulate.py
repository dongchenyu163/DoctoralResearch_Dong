"""Candidate combination helpers and score accumulation utilities."""

from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

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


@dataclass
class ScoreAccumulator:
    """Track per-candidate cumulative scores and elimination state."""

    combination_matrix: np.ndarray
    total_scores: np.ndarray = field(init=False)
    positional_scores: np.ndarray = field(init=False)
    dynamic_scores: np.ndarray = field(init=False)
    hit_counts: np.ndarray = field(init=False)
    active_mask: np.ndarray = field(init=False)
    eliminated_step: np.ndarray = field(init=False)
    eliminated_reason: List[str] = field(init=False)
    _lookup: Dict[Tuple[int, ...], int] = field(init=False)

    def __post_init__(self) -> None:
        combos = np.asarray(self.combination_matrix, dtype=np.int32)
        self.combination_matrix = combos
        candidate_count = combos.shape[0]
        self.total_scores = np.zeros(candidate_count, dtype=np.float64)
        self.positional_scores = np.zeros(candidate_count, dtype=np.float64)
        self.dynamic_scores = np.zeros(candidate_count, dtype=np.float64)
        self.hit_counts = np.zeros(candidate_count, dtype=np.int32)
        self.active_mask = np.ones(candidate_count, dtype=bool)
        self.eliminated_step = np.full(candidate_count, -1, dtype=np.int32)
        self.eliminated_reason = ["" for _ in range(candidate_count)]
        self._lookup = {
            tuple(int(v) for v in combos[i].tolist()): int(i) for i in range(candidate_count)
        }

    def has_active_candidates(self) -> bool:
        return bool(self.active_mask.size and np.any(self.active_mask))

    def active_ids(self) -> np.ndarray:
        if not self.active_mask.size:
            return np.empty((0,), dtype=np.int64)
        return np.flatnonzero(self.active_mask)

    def lookup_rows(self, rows: np.ndarray) -> np.ndarray:
        if rows.size == 0:
            return np.empty((0,), dtype=np.int64)
        ids = np.empty(rows.shape[0], dtype=np.int64)
        for idx, row in enumerate(rows):
            key = tuple(int(v) for v in row.tolist())
            ids[idx] = self._lookup.get(key, -1)
        if np.any(ids < 0):
            raise ValueError("row not found in combination matrix lookup")
        return ids

    def mark_eliminated(self, candidate_ids: Iterable[int], reason: str, timestep: int) -> None:
        ids = self._normalize_ids(candidate_ids)
        if ids.size == 0:
            return
        ids = ids[(ids >= 0) & (ids < self.active_mask.size)]
        if ids.size == 0:
            return
        self.active_mask[ids] = False
        self.eliminated_step[ids] = int(timestep)
        for cid in ids:
            self.eliminated_reason[int(cid)] = reason

    def accumulate(self, candidate_ids: Iterable[int], positional: np.ndarray, dynamic: np.ndarray) -> None:
        ids = self._normalize_ids(candidate_ids)
        if ids.size == 0:
            return
        if positional.shape[0] != ids.size or dynamic.shape[0] != ids.size:
            raise ValueError("score arrays must align with candidate_ids")
        self.total_scores[ids] += positional + dynamic
        self.positional_scores[ids] += positional
        self.dynamic_scores[ids] += dynamic
        self.hit_counts[ids] += 1

    def best_candidate_index(self) -> int | None:
        if not self.total_scores.size:
            return None
        survivors = np.flatnonzero(self.active_mask & (self.hit_counts > 0))
        if survivors.size == 0:
            survivors = np.flatnonzero(self.hit_counts > 0)
        if survivors.size == 0:
            return None
        best_offset = np.argmax(self.total_scores[survivors])
        return int(survivors[best_offset])

    def top_candidates(self, k: int) -> List[int]:
        if k <= 0 or not self.total_scores.size:
            return []
        survivors = np.flatnonzero(self.active_mask & (self.hit_counts > 0))
        if survivors.size == 0:
            return []
        order = np.argsort(self.total_scores[survivors])[::-1]
        top_ids = survivors[order[: min(k, survivors.size)]]
        return [int(idx) for idx in top_ids]

    def elimination_summary(self) -> Dict[str, int]:
        counter: Counter[str] = Counter(reason for reason in self.eliminated_reason if reason)
        return dict(counter)

    def _normalize_ids(self, candidate_ids: Iterable[int]) -> np.ndarray:
        if isinstance(candidate_ids, np.ndarray):
            return candidate_ids.astype(np.int64, copy=False)
        return np.fromiter((int(cid) for cid in candidate_ids), dtype=np.int64)
