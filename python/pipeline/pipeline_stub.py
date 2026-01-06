"""Pipeline implementation that wires together preprocessing, scoring, and accumulation."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from python.instrumentation.timing import TimingRecorder
from python.pipeline.accumulate import ScoreAccumulator, build_all_combinations
from python.pipeline.contact_surface import ContactSurfaceResult, extract_contact_surface
from python.pipeline.dynamics import compute_dynamics_scores
from python.pipeline.geo_filter import GeoFilterRunner
from python.pipeline.preprocess import PreprocessResult, RawPointCloud, load_point_cloud, preprocess_point_cloud
from python.pipeline.trajectory import TrajectoryNode, build_test_trajectory
from python.pipeline.valid_indices import compute_valid_indices
from python.pipeline.wrench import compute_wrench
from python.utils.config_loader import Config


def run_pipeline(config: Config, recorder: TimingRecorder) -> Dict[str, object]:
    """Execute the integrated holding-point search pipeline."""
    raw_cloud: RawPointCloud = load_point_cloud(config, recorder)
    preprocess_result: PreprocessResult = preprocess_point_cloud(raw_cloud, config, recorder)

    dataset_info = {
        "source": str(raw_cloud.source_path) if raw_cloud.source_path else "synthetic",
        "original_point_count": preprocess_result.original_point_count,
    }
    preprocess_summary = {
        "downsample_num": config.preprocess.get("downsample_num"),
        "downsampled_point_count": preprocess_result.downsampled_point_count,
        "normal_estimation_radius": config.preprocess.get("normal_estimation_radius"),
    }

    geo_filter = GeoFilterRunner(config)
    geo_filter.set_point_cloud(preprocess_result)

    finger_count = int(config.search.get("finger_count", 2))
    combination_matrix = build_all_combinations(preprocess_result.points_low.shape[0], finger_count, recorder)
    accumulator = ScoreAccumulator(combination_matrix)
    combinations_summary = {
        "finger_count": finger_count,
        "total_combinations": int(combination_matrix.shape[0]),
    }

    contact_surface: ContactSurfaceResult = extract_contact_surface(preprocess_result, recorder)
    with recorder.section("python/wrench_compute"):
        wrench = compute_wrench(contact_surface, config)
    mesh_boolean_summary = dict(contact_surface.metadata)

    trajectory_nodes: List[TrajectoryNode] = build_test_trajectory(preprocess_result, config, recorder)
    timestep_reports: List[Dict[str, object]] = []
    instrumentation = config.instrumentation
    valid_summary: Dict[str, object] | None = None

    pos_weights = config.weights.get("pos_score", {})
    w_pdir = float(pos_weights.get("w_pdir", 1.0))
    w_pdis = float(pos_weights.get("w_pdis", 1.0))

    with recorder.section("python/trajectory_loop"):
        for step_idx, node in enumerate(trajectory_nodes):
            valid_result = compute_valid_indices(preprocess_result.points_low, config, recorder)
            if valid_summary is None:
                valid_summary = {
                    "count": int(valid_result.indices.size),
                    "table_threshold": valid_result.table_threshold,
                    "knife_threshold": valid_result.knife_threshold,
                    "passed_table": valid_result.passed_table,
                    "passed_knife": valid_result.passed_knife,
                }

            active_ids = accumulator.active_ids()
            if active_ids.size == 0:
                break

            active_candidates = combination_matrix[active_ids]
            valid_mask = _valid_row_mask(active_candidates, valid_result.indices)
            valid_ids = active_ids[valid_mask]
            valid_candidates = active_candidates[valid_mask]
            invalid_ids = active_ids[~valid_mask]
            if invalid_ids.size:
                accumulator.mark_eliminated(invalid_ids, "invalid_indices", step_idx)

            knife_position = node.pose[:3, 3]
            knife_normal = node.pose[:3, 2]
            step_report = {
                "timestep": step_idx,
                "knife_position": knife_position.tolist(),
                "knife_normal": knife_normal.tolist(),
                "velocity": node.velocity.tolist(),
                "candidate_counts": {
                    "active": int(active_ids.size),
                    "valid": int(valid_ids.size),
                },
            }

            if valid_candidates.size == 0:
                if instrumentation.emit_per_timestep_report:
                    timestep_reports.append(step_report)
                continue

            filtered_candidates = geo_filter.run(valid_result, valid_candidates, recorder)
            candidate_lookup = _build_row_lookup(valid_candidates, valid_ids)
            filtered_ids = _rows_to_ids(filtered_candidates, candidate_lookup)

            eliminated_geo = np.setdiff1d(valid_ids, filtered_ids, assume_unique=True)
            if eliminated_geo.size:
                accumulator.mark_eliminated(eliminated_geo, "geo_filter", step_idx)

            step_report["candidate_counts"]["geo_filtered"] = int(filtered_ids.size)

            if filtered_candidates.size == 0:
                if instrumentation.emit_per_timestep_report:
                    timestep_reports.append(step_report)
                continue

            positional_scores = geo_filter.calc_positional_scores(filtered_candidates, knife_position, knife_normal)
            positional_distances = geo_filter.calc_positional_distances(filtered_candidates, knife_position, knife_normal)
            pos_combined = w_pdir * positional_scores + w_pdis * positional_distances
            dynamics_scores = compute_dynamics_scores(geo_filter, filtered_candidates, wrench, config)

            with recorder.section("python/accumulate_scores"):
                accumulator.accumulate(filtered_ids, pos_combined, dynamics_scores)

            step_report["positional_score_mean"] = float(pos_combined.mean()) if pos_combined.size else 0.0
            step_report["dynamic_score_mean"] = float(dynamics_scores.mean()) if dynamics_scores.size else 0.0
            step_report["candidate_counts"]["alive"] = int(accumulator.active_mask.sum())

            if instrumentation.emit_per_timestep_report:
                timestep_reports.append(step_report)

    if valid_summary is None:
        valid_result = compute_valid_indices(preprocess_result.points_low, config, recorder)
        valid_summary = {
            "count": int(valid_result.indices.size),
            "table_threshold": valid_result.table_threshold,
            "knife_threshold": valid_result.knife_threshold,
            "passed_table": valid_result.passed_table,
            "passed_knife": valid_result.passed_knife,
        }

    best_idx = accumulator.best_candidate_index()
    best_summary = (
        _build_candidate_summary(best_idx, accumulator, preprocess_result) if best_idx is not None else None
    )

    top_ids = accumulator.top_candidates(5)
    top_candidates = [_build_candidate_summary(idx, accumulator, preprocess_result) for idx in top_ids]

    candidate_debug: List[Dict[str, object]] = []
    if instrumentation.emit_per_candidate_debug:
        debug_ids = accumulator.top_candidates(10)
        candidate_debug = [_build_candidate_summary(idx, accumulator, preprocess_result) for idx in debug_ids]

    score_section = {
        "best_candidate": best_summary,
        "top_candidates": top_candidates,
        "survivor_count": int(accumulator.active_mask.sum()),
        "eliminated_counts": accumulator.elimination_summary(),
    }
    if candidate_debug:
        score_section["candidate_debug"] = candidate_debug

    result_summary = {
        "status": "ok" if best_summary else "no_candidates",
        "message": "Pipeline executed with integrated accumulation.",
        "dataset": dataset_info,
        "preprocess": preprocess_summary,
        "valid_indices": valid_summary,
        "combinations": combinations_summary,
        "contact_surface": mesh_boolean_summary,
        "wrench": wrench.tolist(),
        "trajectory": {"nodes": len(trajectory_nodes)},
        "scores": score_section,
        "timesteps": timestep_reports if instrumentation.emit_per_timestep_report else [],
    }
    return result_summary


def _valid_row_mask(candidate_matrix: np.ndarray, valid_indices: np.ndarray) -> np.ndarray:
    if candidate_matrix.size == 0 or candidate_matrix.shape[0] == 0:
        return np.zeros(candidate_matrix.shape[0], dtype=bool)
    if valid_indices.size == 0:
        return np.zeros(candidate_matrix.shape[0], dtype=bool)
    membership = np.isin(candidate_matrix, valid_indices)
    return np.all(membership, axis=1)


def _build_row_lookup(rows: np.ndarray, ids: np.ndarray) -> Dict[Tuple[int, ...], int]:
    if rows.size == 0 or ids.size == 0:
        return {}
    lookup: Dict[Tuple[int, ...], int] = {}
    for row, cid in zip(rows, ids):
        lookup[tuple(int(v) for v in row.tolist())] = int(cid)
    return lookup


def _rows_to_ids(rows: np.ndarray, lookup: Dict[Tuple[int, ...], int]) -> np.ndarray:
    if rows.size == 0:
        return np.empty((0,), dtype=np.int64)
    ids = np.empty(rows.shape[0], dtype=np.int64)
    for idx, row in enumerate(rows):
        ids[idx] = lookup.get(tuple(int(v) for v in row.tolist()), -1)
    if np.any(ids < 0):
        raise ValueError("Encountered candidate row without lookup entry.")
    return ids


def _build_candidate_summary(index: int, accumulator: ScoreAccumulator, preprocess: PreprocessResult) -> Dict[str, object]:
    combo = accumulator.combination_matrix[index]
    points = preprocess.points_low[combo] if preprocess.points_low.size else np.empty((0, 3))
    return {
        "combination_index": int(index),
        "point_indices": combo.tolist(),
        "points": points.tolist(),
        "score_total": float(accumulator.total_scores[index]),
        "score_positional": float(accumulator.positional_scores[index]),
        "score_dynamic": float(accumulator.dynamic_scores[index]),
        "hit_count": int(accumulator.hit_counts[index]),
    }
