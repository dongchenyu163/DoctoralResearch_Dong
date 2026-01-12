"""Full Algorithm 1 integration loop.

- PREPARE_DATA / lines 1-4: `load_point_cloud` + `preprocess_point_cloud`
- FILTER_BY_GEO_SCORE / Algorithm 2: `GeoFilterRunner.run`
- CAL_POSITIONAL_SCORE / Algorithm 3: `calc_positional_*`
- CAL_KNIFE_FORCE + CAL_DYNAMICS_SCORE / Algorithm 4: `compute_wrench` + `calc_dynamics_scores`
- Score accumulation + elimination mirrors Algorithm 1 lines 10-18.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from python.instrumentation.timing import TimingRecorder
from python.pipeline.accumulate import ScoreAccumulator, build_all_combinations
from python.pipeline.contact_surface import ContactSurfaceResult, extract_contact_surface
from python.pipeline.dynamics import (
    compute_dynamics_scores,
    debug_visualize_dynamics_f_init_forces,
    debug_visualize_dynamics_forces,
)
from python.pipeline.geo_filter import GeoFilterRunner
from python.pipeline.knife_model import KnifeModel, build_knife_model
from python.pipeline.preprocess import PreprocessResult, RawPointCloud, load_point_cloud, preprocess_point_cloud
from python.pipeline.trajectory import TrajectoryNode, build_test_trajectory
from python.pipeline.valid_indices import compute_valid_indices
from python.pipeline.wrench import compute_wrench
from python.utils.config_loader import Config
from python.utils.logging_sections import log_boxed_heading
from python.utils.logging_setup import CppLoggingSettings
from python.utils.pointcloud_logging import PointCloudDebugSaver

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None

LOGGER = logging.getLogger("pipeline")
PREPROCESS_LOGGER = logging.getLogger("pipeline.preprocess")
VALID_LOGGER = logging.getLogger("pipeline.valid_indices")
CONTACT_LOGGER = logging.getLogger("pipeline.contact_surface")
SCORES_LOGGER = logging.getLogger("pipeline.scores")


def run_pipeline(
    config: Config,
    recorder: TimingRecorder,
    *,
    pc_logger: Optional[PointCloudDebugSaver] = None,
    cpp_logging: Optional[CppLoggingSettings] = None,
) -> Dict[str, object]:
    """Execute Algorithm 1 end-to-end and return summarized diagnostics.

    Args:
        config: Parsed JSON config controlling thresholds/weights.
        recorder: Timing recorder that emits per-section instrumentation.
    """
    log_boxed_heading(LOGGER, "1", "PREPARE DATA / Ω_low 构建")
    log_boxed_heading(PREPROCESS_LOGGER, "1.1", "Load Raw Point Cloud")
    raw_cloud: RawPointCloud = load_point_cloud(config, recorder)
    LOGGER.info("Loaded point cloud source=%s count=%d", raw_cloud.source_path, raw_cloud.points.shape[0])
    log_boxed_heading(PREPROCESS_LOGGER, "1.2", "Preprocess → Downsample + Normals")
    preprocess_result: PreprocessResult = preprocess_point_cloud(raw_cloud, config, recorder)
    PREPROCESS_LOGGER.info(
        "Downsampled to %d points (target=%s)",
        preprocess_result.downsampled_point_count,
        config.preprocess.get("downsample_num"),
    )
    knife_model = build_knife_model(config, preprocess_result.points_low)

    dataset_info = {
        "source": str(raw_cloud.source_path) if raw_cloud.source_path else "synthetic",
        "original_point_count": preprocess_result.original_point_count,
    }
    preprocess_summary = {
        "downsample_num": config.preprocess.get("downsample_num"),
        "downsampled_point_count": preprocess_result.downsampled_point_count,
        "normal_estimation_radius": config.preprocess.get("normal_estimation_radius"),
    }

    geo_filter = GeoFilterRunner(config, cpp_logging=cpp_logging)
    geo_filter.set_point_cloud(preprocess_result)
    if bool(config.search.get("debug_points_normals_viz", False)):
        _show_points_normals(preprocess_result.points_low, preprocess_result.normals_low)

    log_boxed_heading(LOGGER, "2", "COMBINATIONS + TRAJECTORY INIT")
    finger_count = int(config.search.get("finger_count", 2))
    combination_matrix = build_all_combinations(preprocess_result.points_low.shape[0], finger_count, recorder)
    accumulator = ScoreAccumulator(combination_matrix)
    combinations_summary = {
        "finger_count": finger_count,
        "total_combinations": int(combination_matrix.shape[0]),
    }

    # Knife wrench (Algorithm 4 pre-step) computed per timestep.
    trajectory_nodes: List[TrajectoryNode] = build_test_trajectory(preprocess_result, config, recorder)
    step_count = len(trajectory_nodes)
    candidate_count = int(combination_matrix.shape[0])
    valid_mask_table = np.zeros((candidate_count, step_count), dtype=np.int8)
    global_valid_mask = np.ones(candidate_count, dtype=bool)
    geo_score_table = np.full((candidate_count, step_count), np.nan, dtype=np.float64)
    positional_score_table = np.full((candidate_count, step_count), np.nan, dtype=np.float64)
    dynamic_score_table = np.full((candidate_count, step_count), np.nan, dtype=np.float64)
    timestep_reports: List[Dict[str, object]] = []
    instrumentation = config.instrumentation
    valid_summary: Dict[str, object] | None = None
    last_contact_metadata: Dict[str, float] = {}
    last_wrench: np.ndarray | None = None
    last_valid_indices: np.ndarray | None = None
    manual_selection: Dict[str, object] | None = None

    pos_weights = config.weights.get("pos_score", {})
    w_pdir = float(pos_weights.get("w_pdir", 1.0))
    w_pdis = float(pos_weights.get("w_pdis", 1.0))

    if pc_logger and pc_logger.enabled_for("omega_high") and preprocess_result.points_high.size:
        pc_logger.save_point_cloud("omega_high", -1, preprocess_result.points_high)
    if pc_logger and pc_logger.enabled_for("omega_low") and preprocess_result.points_low.size:
        pc_logger.save_point_cloud("omega_low", -1, preprocess_result.points_low)
    log_boxed_heading(LOGGER, "3", "TRAJECTORY LOOP / 多阶段计算")
    with recorder.section("python/trajectory_loop"):
        for step_idx, node in enumerate(trajectory_nodes):
            log_boxed_heading(LOGGER, f"3.{step_idx + 1}", f"Step {step_idx} Pose + Ωg 计算")
            knife_position = node.pose[:3, 3]
            knife_normal = node.pose[:3, 2]
            knife_instance = knife_model.instantiate(node.pose)
            if pc_logger and pc_logger.enabled_for("knife_mesh"):
                pc_logger.save_mesh("knife_mesh", step_idx, knife_instance.mesh)
            if pc_logger and pc_logger.enabled_for("food_mesh"):
                if preprocess_result.food_mesh is None:
                    logging.getLogger("pipeline.debug_pc").error("food_mesh missing at step %d", step_idx)
                else:
                    pc_logger.save_mesh("food_mesh", step_idx, preprocess_result.food_mesh)
            # PREPARE_DATA line 4: recompute Ωg for each timestep using knife plane.
            valid_result = compute_valid_indices(preprocess_result.points_low, config, recorder, knife_instance)
            last_valid_indices = valid_result.indices
            valid_mask_all = _valid_row_mask(combination_matrix, valid_result.indices)
            if candidate_count:
                global_valid_mask &= valid_mask_all
                valid_mask_table[:, step_idx] = global_valid_mask.astype(np.int8)
            VALID_LOGGER.info(
                "Step %d Ωg count=%d thresholds(table>=%.4f knife<=%.4f)",
                step_idx,
                valid_result.indices.size,
                valid_result.table_threshold,
                valid_result.knife_threshold,
            )
            VALID_LOGGER.debug(
                "Step %d table_pass=%d knife_pass=%d center_pass=%d slice_pass=%d plane_tol=%.5f",
                step_idx,
                valid_result.passed_table,
                valid_result.passed_knife,
                valid_result.passed_center_plane,
                valid_result.passed_penetration_plane,
                valid_result.plane_tolerance,
            )
            if pc_logger and pc_logger.enabled_for("omega_g") and valid_result.indices.size:
                pc_logger.save_point_cloud("omega_g", step_idx, preprocess_result.points_low[valid_result.indices])
            if valid_result.indices.size == 0:
                VALID_LOGGER.error(
                    "Step %d Ωg empty after filters (table=%d knife=%d center=%d slice=%d)",
                    step_idx,
                    valid_result.passed_table,
                    valid_result.passed_knife,
                    valid_result.passed_center_plane,
                    valid_result.passed_penetration_plane,
                )
            if valid_summary is None:
                valid_summary = {
                    "count": int(valid_result.indices.size),
                    "table_threshold": valid_result.table_threshold,
                    "knife_threshold": valid_result.knife_threshold,
                    "passed_table": valid_result.passed_table,
                    "passed_knife": valid_result.passed_knife,
                    "passed_center_plane": valid_result.passed_center_plane,
                    "passed_penetration_plane": valid_result.passed_penetration_plane,
                    "plane_tolerance": valid_result.plane_tolerance,
                }

            active_ids = accumulator.active_ids()
            if active_ids.size == 0:
                if candidate_count:
                    valid_mask_table[:, step_idx] = global_valid_mask.astype(np.int8)
                break

            # Mask out candidates that violate current Ωg.
            active_candidates = combination_matrix[active_ids]
            valid_mask = _valid_row_mask(active_candidates, valid_result.indices)
            valid_ids = active_ids[valid_mask]
            valid_candidates = active_candidates[valid_mask]
            invalid_ids = active_ids[~valid_mask]
            if invalid_ids.size:
                accumulator.mark_eliminated(invalid_ids, "invalid_indices", step_idx)
                SCORES_LOGGER.debug("Step %d eliminated %d candidates by Ωg", step_idx, invalid_ids.size)
                if candidate_count:
                    global_valid_mask[invalid_ids] = False

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
                if candidate_count:
                    valid_mask_table[:, step_idx] = global_valid_mask.astype(np.int8)
                if instrumentation.emit_per_timestep_report:
                    timestep_reports.append(step_report)
                continue

            log_boxed_heading(CONTACT_LOGGER, f"3.{step_idx + 1}.1", f"Step {step_idx} Contact Surface + Wrench")
            contact_surface: ContactSurfaceResult = extract_contact_surface(preprocess_result, recorder, knife_instance)
            if pc_logger:
                if pc_logger.enabled_for("boolean_base_mesh") and preprocess_result.food_mesh is not None:
                    pc_logger.save_mesh("boolean_base_mesh", step_idx, preprocess_result.food_mesh)
                if pc_logger.enabled_for("boolean_knife_mesh"):
                    pc_logger.save_mesh("boolean_knife_mesh", step_idx, knife_instance.mesh)
            if pc_logger:
                if pc_logger.enabled_for("contact_faces"):
                    contact_points = _flatten_contact_faces(contact_surface.faces)
                    omega_points = preprocess_result.points_low[valid_result.indices] if valid_result.indices.size else np.empty((0, 3))
                    combined_points, combined_colors = _combine_points_for_debug(omega_points, contact_points)
                    pc_logger.save_point_cloud("contact_faces", step_idx, combined_points, combined_colors)
                if pc_logger.enabled_for("knife_food_intersection") and contact_surface.mesh is not None:
                    pc_logger.save_mesh("knife_food_intersection", step_idx, contact_surface.mesh)
                for side_idx, face_block in enumerate(contact_surface.faces):
                    stage_name = f"contact_mesh_side{side_idx}"
                    if pc_logger.enabled_for(stage_name):
                        mesh = _faces_to_trimesh(face_block)
                        if mesh is not None:
                            pc_logger.save_mesh(stage_name, step_idx, mesh)
            food_center = _food_center(preprocess_result)
            with recorder.section("python/wrench_compute"):
                wrench = compute_wrench(
                    contact_surface,
                    config,
                    step_idx=step_idx,
                    velocity=node.velocity,
                    food_center=food_center,
                )
            last_contact_metadata = dict(contact_surface.metadata)
            last_wrench = wrench
            CONTACT_LOGGER.info(
                "Step %d contact faces=%s components=%s",
                step_idx,
                contact_surface.metadata.get("total_faces"),
                contact_surface.metadata.get("components"),
            )
            CONTACT_LOGGER.debug("Step %d contact metadata=%s", step_idx, contact_surface.metadata)
            for side_idx, count in enumerate(contact_surface.metadata.get("side_counts", [])):
                if count <= 0:
                    CONTACT_LOGGER.error("Step %d contact side %d empty", step_idx, side_idx)
                else:
                    CONTACT_LOGGER.info("Step %d contact side %d triangles=%d", step_idx, side_idx, int(count))

            # Algorithm 2: Geometry filter (Table 1 section Ωg).
            log_boxed_heading(SCORES_LOGGER, f"3.{step_idx + 1}.2", f"Step {step_idx} GeoFilter + Scores")
            filtered_candidates = geo_filter.run(valid_result, valid_candidates, recorder)
            if bool(config.search.get("debug_geo_filter_viz", False)):
                masked_candidates = geo_filter.mask_candidates(valid_candidates, valid_result.indices)
                order = geo_filter.last_geo_order()
                _show_geo_filter_debug(
                    preprocess_result.points_low,
                    valid_result.indices,
                    masked_candidates,
                    order,
                    int(config.search.get("debug_geo_filter_k", 3)),
                    float(config.search.get("geo_filter_ratio", 1.0)),
                    int(config.search.get("debug_geo_filter_seed", config.seed)),
                    high_scores=True,
                    window_name=f"GeoFilter High Scores step {step_idx}",
                )
                _show_geo_filter_debug(
                    preprocess_result.points_low,
                    valid_result.indices,
                    masked_candidates,
                    order,
                    int(config.search.get("debug_geo_filter_k", 3)),
                    float(config.search.get("geo_filter_ratio", 1.0)),
                    int(config.search.get("debug_geo_filter_seed", config.seed)),
                    high_scores=False,
                    window_name=f"GeoFilter Low Scores step {step_idx}",
                )
            candidate_lookup = _build_row_lookup(valid_candidates, valid_ids)
            filtered_ids = _rows_to_ids(filtered_candidates, candidate_lookup)
            if filtered_candidates.size:
                geo_scores = geo_filter.calc_geo_scores(
                    valid_candidates,
                    knife_position,
                    knife_normal,
                    float(config.environment.get("table_z", 0.0)),
                )
                if candidate_count:
                    geo_scores_full = np.full(candidate_count, np.nan, dtype=np.float64)
                    geo_scores_full[valid_ids] = geo_scores
                    geo_score_table[filtered_ids, step_idx] = geo_scores_full[filtered_ids]

            eliminated_geo = np.setdiff1d(valid_ids, filtered_ids, assume_unique=True)
            if eliminated_geo.size:
                accumulator.mark_eliminated(eliminated_geo, "geo_filter", step_idx)
                if candidate_count:
                    global_valid_mask[eliminated_geo] = False

            step_report["candidate_counts"]["geo_filtered"] = int(filtered_ids.size)

            if filtered_candidates.size == 0:
                if candidate_count:
                    valid_mask_table[:, step_idx] = global_valid_mask.astype(np.int8)
                if instrumentation.emit_per_timestep_report:
                    timestep_reports.append(step_report)
                continue

            # Algorithm 3: positional scores (direction + lever arm terms).
            positional_scores = geo_filter.calc_positional_scores(filtered_candidates, knife_position, knife_normal)
            positional_distances = geo_filter.calc_positional_distances(filtered_candidates, knife_position, knife_normal)
            pos_combined = w_pdir * positional_scores + w_pdis * positional_distances
            # Algorithm 4: dynamics score based on wrench equilibrium.
            dynamics_scores = compute_dynamics_scores(geo_filter, filtered_candidates, wrench, food_center, config)
            debug_visualize_dynamics_f_init_forces(
                geo_filter,
                preprocess_result.points_low,
                preprocess_result.normals_low,
                valid_result.indices,
                filtered_candidates,
                config,
            )
            debug_visualize_dynamics_forces(
                geo_filter,
                preprocess_result.points_low,
                preprocess_result.normals_low,
                valid_result.indices,
                filtered_candidates,
                wrench,
                food_center,
                config,
            )
            invalid_mask = ~np.isfinite(dynamics_scores)
            if np.any(invalid_mask):
                invalid_ids = filtered_ids[invalid_mask]
                accumulator.mark_eliminated(invalid_ids, "dynamic_infeasible", step_idx)
                SCORES_LOGGER.debug("Step %d eliminated %d candidates by dynamics infeasibility", step_idx, invalid_ids.size)
                if candidate_count:
                    global_valid_mask[invalid_ids] = False
            valid_mask = ~invalid_mask
            filtered_ids = filtered_ids[valid_mask]
            pos_combined = pos_combined[valid_mask]
            dynamics_scores = dynamics_scores[valid_mask]
            SCORES_LOGGER.debug(
                "Step %d pos(mean=%.4f,min=%.4f,max=%.4f) dyn(mean=%.4f,min=%.4f,max=%.4f)",
                step_idx,
                float(pos_combined.mean()) if pos_combined.size else 0.0,
                float(pos_combined.min()) if pos_combined.size else 0.0,
                float(pos_combined.max()) if pos_combined.size else 0.0,
                float(dynamics_scores.mean()) if dynamics_scores.size else 0.0,
                float(dynamics_scores.min()) if dynamics_scores.size else 0.0,
                float(dynamics_scores.max()) if dynamics_scores.size else 0.0,
            )

            if filtered_ids.size:
                with recorder.section("python/accumulate_scores"):
                    accumulator.accumulate(filtered_ids, pos_combined, dynamics_scores)
                positional_score_table[filtered_ids, step_idx] = pos_combined
                dynamic_score_table[filtered_ids, step_idx] = dynamics_scores

            step_report["positional_score_mean"] = float(pos_combined.mean()) if pos_combined.size else 0.0
            step_report["dynamic_score_mean"] = float(dynamics_scores.mean()) if dynamics_scores.size else 0.0
            step_report["candidate_counts"]["alive"] = int(accumulator.active_mask.sum())
            SCORES_LOGGER.info(
                "Step %d alive=%d geo_filtered=%d pos_mean=%.3f dyn_mean=%.3f",
                step_idx,
                step_report["candidate_counts"]["alive"],
                step_report["candidate_counts"]["geo_filtered"],
                step_report["positional_score_mean"],
                step_report["dynamic_score_mean"],
            )

            if candidate_count:
                valid_mask_table[:, step_idx] = global_valid_mask.astype(np.int8)

            if instrumentation.emit_per_timestep_report:
                timestep_reports.append(step_report)

    if valid_summary is None:
        # If no steps executed, fall back to static Ωg summary.
        identity_pose = np.eye(4, dtype=np.float64)
        knife_instance = knife_model.instantiate(identity_pose)
        valid_result = compute_valid_indices(preprocess_result.points_low, config, recorder, knife_instance)
        last_valid_indices = valid_result.indices
        valid_summary = {
            "count": int(valid_result.indices.size),
            "table_threshold": valid_result.table_threshold,
            "knife_threshold": valid_result.knife_threshold,
            "passed_table": valid_result.passed_table,
            "passed_knife": valid_result.passed_knife,
            "passed_center_plane": valid_result.passed_center_plane,
            "passed_penetration_plane": valid_result.passed_penetration_plane,
            "plane_tolerance": valid_result.plane_tolerance,
        }

    # best_idx = accumulator.best_candidate_index()
    best_idx = accumulator.top_candidates(100)[50]
    best_summary = (
        _build_candidate_summary(best_idx, accumulator, preprocess_result) if best_idx is not None else None
    )
    if config.manual_select_result:
        if trajectory_nodes:
            manual_pose = trajectory_nodes[-1].pose
        else:
            manual_pose = np.eye(4, dtype=np.float64)
        manual_instance = knife_model.instantiate(manual_pose)
        manual_valid = compute_valid_indices(preprocess_result.points_low, config, recorder, manual_instance)
        manual_selection = _manual_select_grasp(
            preprocess_result.points_low,
            preprocess_result.normals_low,
            manual_valid.indices,
        )
    if bool(config.search.get("debug_best_candidate_viz", False)):
        debug_best_k = int(config.search.get("debug_best_candidate_k", 5))
        top_ids_for_viz = accumulator.top_candidates(max(debug_best_k, 1))
        _show_best_candidate_grasp(
            preprocess_result.points_low,
            preprocess_result.normals_low,
            last_valid_indices if last_valid_indices is not None else np.empty((0,), dtype=np.int32),
            accumulator.combination_matrix,
            np.array(top_ids_for_viz),
        )
    if best_summary:
        SCORES_LOGGER.info(
            "Best candidate idx=%d total=%.3f hit_count=%d",
            best_summary["combination_index"],
            best_summary["score_total"],
            best_summary["hit_count"],
        )
    elif manual_selection is None:
        SCORES_LOGGER.warning("No valid candidates remained after processing.")

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

    final_points = []
    final_normals = []
    if manual_selection is not None:
        final_points = manual_selection.get("points", [])
        final_normals = manual_selection.get("normals", [])
    elif best_summary:
        final_points = best_summary.get("points", [])
        final_normals = best_summary.get("normals", [])

    status_ok = bool(best_summary) or manual_selection is not None
    result_summary = {
        "manual_select_result": bool(config.manual_select_result),
        "status": "ok" if status_ok else "no_candidates",
        "message": "Pipeline executed with integrated accumulation.",
        "dataset": dataset_info,
        "preprocess": preprocess_summary,
        "valid_indices": valid_summary,
        "combinations": combinations_summary,
        "contact_surface": last_contact_metadata,
        "wrench": last_wrench.tolist() if last_wrench is not None else [],
        "trajectory": {"nodes": len(trajectory_nodes)},
        "scores": score_section,
        "selected_points": final_points,
        "selected_normals": final_normals,
        "manual_selection": manual_selection,
        "timesteps": timestep_reports if instrumentation.emit_per_timestep_report else [],
    }
    _write_candidate_metric_csv(Path("output/valid_mask.csv"), valid_mask_table, float_format=False)
    _write_candidate_metric_csv(Path("output/geo_score.csv"), geo_score_table, float_format=True)
    _write_candidate_metric_csv(Path("output/positional_score.csv"), positional_score_table, float_format=True)
    _write_candidate_metric_csv(Path("output/dynamic_score.csv"), dynamic_score_table, float_format=True)
    return result_summary


def _valid_row_mask(candidate_matrix: np.ndarray, valid_indices: np.ndarray) -> np.ndarray:
    """Return boolean mask for rows whose every finger index ∈ Ωg.

    Args:
        candidate_matrix: Shape (K, F) matrix of finger index combinations.
        valid_indices: 1-D array of Ωg indices (monotonic but not necessarily contiguous).
    """
    if candidate_matrix.size == 0 or candidate_matrix.shape[0] == 0:
        return np.zeros(candidate_matrix.shape[0], dtype=bool)
    if valid_indices.size == 0:
        return np.zeros(candidate_matrix.shape[0], dtype=bool)
    membership = np.isin(candidate_matrix, valid_indices)
    return np.all(membership, axis=1)


def _build_row_lookup(rows: np.ndarray, ids: np.ndarray) -> Dict[Tuple[int, ...], int]:
    """Map each finger combination (tuple) back to its global candidate id."""
    if rows.size == 0 or ids.size == 0:
        return {}
    lookup: Dict[Tuple[int, ...], int] = {}
    for row, cid in zip(rows, ids):
        lookup[tuple(int(v) for v in row.tolist())] = int(cid)
    return lookup


def _rows_to_ids(rows: np.ndarray, lookup: Dict[Tuple[int, ...], int]) -> np.ndarray:
    """Resolve locally filtered rows back into absolute candidate ids."""
    if rows.size == 0:
        return np.empty((0,), dtype=np.int64)
    ids = np.empty(rows.shape[0], dtype=np.int64)
    for idx, row in enumerate(rows):
        ids[idx] = lookup.get(tuple(int(v) for v in row.tolist()), -1)
    if np.any(ids < 0):
        raise ValueError("Encountered candidate row without lookup entry.")
    return ids


def _build_candidate_summary(index: int, accumulator: ScoreAccumulator, preprocess: PreprocessResult) -> Dict[str, object]:
    """Assemble reporting payload for a winning candidate."""
    combo = accumulator.combination_matrix[index]
    points = preprocess.points_low[combo] if preprocess.points_low.size else np.empty((0, 3))
    normals = (
        preprocess.normals_low[combo]
        if preprocess.normals_low.shape == preprocess.points_low.shape
        else np.zeros_like(points)
    )
    return {
        "combination_index": int(index),
        "point_indices": combo.tolist(),
        "points": points.tolist(),
        "normals": normals.tolist(),
        "score_total": float(accumulator.total_scores[index]),
        "score_positional": float(accumulator.positional_scores[index]),
        "score_dynamic": float(accumulator.dynamic_scores[index]),
        "hit_count": int(accumulator.hit_counts[index]),
    }


def _write_candidate_metric_csv(path: Path, data: np.ndarray, *, float_format: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    step_count = data.shape[1] if data.ndim == 2 else 0
    header = ["P"] + [f"step_{idx}" for idx in range(step_count)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row_idx in range(data.shape[0]):
            row = [row_idx]
            if step_count:
                if float_format:
                    row.extend(_format_csv_float(value) for value in data[row_idx])
                else:
                    row.extend(int(value) for value in data[row_idx])
            writer.writerow(row)


def _format_csv_float(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{float(value):.6f}"


def _flatten_contact_faces(faces: List[np.ndarray]) -> np.ndarray:
    flattened = [np.asarray(face).reshape(-1, 3) for face in faces if face.size]
    if not flattened:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(flattened)


def _food_center(preprocess: PreprocessResult) -> np.ndarray:
    mesh = preprocess.food_mesh
    if mesh is not None and not mesh.is_empty:
        if mesh.is_volume:
            return np.asarray(mesh.center_mass, dtype=np.float64)
        return np.asarray(mesh.centroid, dtype=np.float64)
    if preprocess.points_high.size:
        return preprocess.points_high.mean(axis=0)
    if preprocess.points_low.size:
        return preprocess.points_low.mean(axis=0)
    return np.zeros(3, dtype=np.float64)


def _combine_points_for_debug(omega_points: np.ndarray, contact_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    point_blocks = []
    color_blocks = []
    if omega_points.size:
        point_blocks.append(omega_points)
        color_blocks.append(np.tile(np.array([[0.0, 1.0, 0.0]]), (omega_points.shape[0], 1)))
    if contact_points.size:
        point_blocks.append(contact_points)
        color_blocks.append(np.tile(np.array([[1.0, 0.0, 0.0]]), (contact_points.shape[0], 1)))
    if not point_blocks:
        return np.empty((0, 3)), np.empty((0, 3))
    return np.vstack(point_blocks), np.vstack(color_blocks)


def _show_points_normals(points: np.ndarray, normals: np.ndarray) -> None:
    try:
        import open3d as o3d
    except ImportError:  # pragma: no cover
        LOGGER.warning("open3d unavailable; skip points+normals visualization")
        return
    if points.size == 0:
        LOGGER.warning("points_low empty; skip points+normals visualization")
        return
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if normals.size == points.size:
        cloud.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    o3d.visualization.draw_geometries([cloud], point_show_normal=True)


def _show_best_candidate_grasp(
    points_low: np.ndarray,
    normals_low: np.ndarray,
    omega_indices: np.ndarray,
    candidate_matrix: np.ndarray,
    candidate_ids: np.ndarray,
) -> None:
    try:
        import open3d as o3d
    except ImportError:  # pragma: no cover
        SCORES_LOGGER.warning("open3d unavailable; skip best candidate visualization")
        return
    if candidate_ids.size == 0:
        SCORES_LOGGER.warning("No candidate ids available for visualization")
        return
    if candidate_matrix.size == 0:
        SCORES_LOGGER.warning("Candidate matrix empty; skip visualization")
        return

    omega_points = points_low[omega_indices] if omega_indices.size else points_low
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    max_index = int(candidate_matrix.shape[0] - 1)
    candidate_ids = candidate_ids[(candidate_ids >= 0) & (candidate_ids <= max_index)]
    if candidate_ids.size == 0:
        SCORES_LOGGER.warning("Candidate ids out of range; skip visualization")
        return

    sphere_radius = 0.002
    normal_length = 0.02

    def make_arrow(origin: np.ndarray, direction: np.ndarray, color: np.ndarray) -> "o3d.geometry.TriangleMesh | None":
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            return None
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.0005,
            cone_radius=0.0015,
            cylinder_height=length * 0.8,
            cone_height=length * 0.2,
        )
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        direction_unit = direction / length
        axis = np.cross(z_axis, direction_unit)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-9:
            angle = float(np.arccos(np.clip(np.dot(z_axis, direction_unit), -1.0, 1.0)))
            arrow.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(axis / axis_norm * angle), center=np.zeros(3))
        arrow.translate(origin)
        arrow.paint_uniform_color(np.asarray(color, dtype=np.float64))
        return arrow

    def build_geometries(candidate_id: int) -> list:
        candidate = candidate_matrix[candidate_id]
        grasp_points = points_low[candidate]
        normals_local = normals_low[candidate] if normals_low.shape == points_low.shape else np.zeros_like(grasp_points)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(omega_points.astype(np.float64))
        cloud.paint_uniform_color([0.7, 0.7, 0.7])
        geoms = [cloud]
        for pt in grasp_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sphere.translate(pt)
            sphere.paint_uniform_color(np.asarray([0.1, 0.9, 0.1], dtype=np.float64))
            geoms.append(sphere)
        for pt, n in zip(grasp_points, normals_local):
            arrow = make_arrow(pt, n * normal_length, np.asarray([0.1, 0.3, 0.9], dtype=np.float64))
            if arrow is not None:
                geoms.append(arrow)
        return geoms

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Top-K Grasp Candidates", width=1200, height=900)
    state = {"index": 0}

    def redraw() -> None:
        view = vis.get_view_control().convert_to_pinhole_camera_parameters()
        vis.clear_geometries()
        geoms = build_geometries(int(candidate_ids[state["index"]]))
        for geom in geoms:
            vis.add_geometry(geom)
        vis.get_view_control().convert_from_pinhole_camera_parameters(view, allow_arbitrary=True)
        SCORES_LOGGER.info(
            "Top-K grasp view %d/%d candidate=%d",
            state["index"] + 1,
            int(candidate_ids.size),
            int(candidate_ids[state["index"]]),
        )

    def on_up(_vis) -> bool:
        state["index"] = (state["index"] - 1) % int(candidate_ids.size)
        redraw()
        return False

    def on_down(_vis) -> bool:
        state["index"] = (state["index"] + 1) % int(candidate_ids.size)
        redraw()
        return False

    vis.register_key_callback(265, on_up)
    vis.register_key_callback(264, on_down)
    redraw()
    vis.run()
    vis.destroy_window()


def _manual_select_grasp(
    points_low: np.ndarray,
    normals_low: np.ndarray,
    omega_indices: np.ndarray,
) -> Dict[str, object] | None:
    try:
        import open3d as o3d
    except ImportError:  # pragma: no cover
        SCORES_LOGGER.warning("open3d unavailable; skip manual selection")
        return None

    if omega_indices.size == 0:
        SCORES_LOGGER.warning("OmegaG empty; skip manual selection")
        return None

    omega_points = points_low[omega_indices]
    omega_normals = (
        normals_low[omega_indices] if normals_low.shape == points_low.shape else np.zeros_like(omega_points)
    )

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(omega_points.astype(np.float64))
    cloud.paint_uniform_color([0.7, 0.7, 0.7])
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Manual Select Result", width=1200, height=900)
    vis.add_geometry(cloud)
    vis.run()
    vis.destroy_window()

    picked = vis.get_picked_points()
    if len(picked) != 2:
        SCORES_LOGGER.warning("Manual selection incomplete; selected=%d", len(picked))
        return None

    selected_points = omega_points[picked]
    selected_normals = omega_normals[picked]
    selected_global = omega_indices[picked].astype(int)
    return {
        "point_indices": selected_global.tolist(),
        "points": selected_points.tolist(),
        "normals": selected_normals.tolist(),
    }


def _show_geo_filter_debug(
    points_low: np.ndarray,
    omega_indices: np.ndarray,
    candidate_matrix: np.ndarray,
    order: np.ndarray,
    k: int,
    geo_ratio: float,
    seed: int,
    high_scores: bool,
    window_name: str,
) -> None:
    try:
        import open3d as o3d
    except ImportError:  # pragma: no cover
        SCORES_LOGGER.warning("open3d unavailable; skip geo filter debug visualization")
        return
    
    def make_R_align_z_to_v(v, up=np.array([0.0, 1.0, 0.0])):
        v = np.asarray(v, dtype=float)
        z = v / np.linalg.norm(v)

        up = np.asarray(up, dtype=float)
        # 若 up 与 z 太接近平行，换一个 up
        if abs(np.dot(up, z)) > 0.99:
            up = np.array([1.0, 0.0, 0.0])

        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)

        R = np.column_stack([x, y, z])  # ✅ R[:,2] == z == normalize(v)
        return R
    
    def line_to_cylinder(p0, p1, radius=0.0002, resolution=8):
        length = np.linalg.norm(p1 - p0)
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius,
            height=length,
            resolution=resolution
        )
        cylinder.compute_vertex_normals()

        # 对齐方向
        z = np.array([0, 0, 1])
        v = p1 - p0
        v /= np.linalg.norm(v)
        axis = np.cross(z, -v)
        # angle = np.arccos(np.dot(z, -v))
        if np.linalg.norm(axis) > 1e-6:
            R = make_R_align_z_to_v(v, up=z)
            cylinder.rotate(R, center=np.zeros(3))

        cylinder.translate((p0 + p1) / 2)
        return cylinder

    if k <= 0:
        return
    if points_low.size == 0 or omega_indices.size == 0 or candidate_matrix.size == 0:
        SCORES_LOGGER.warning("Geo filter debug visualization skipped (empty data)")
        return
    if order.size != candidate_matrix.shape[0]:
        SCORES_LOGGER.warning("Geo filter debug order mismatch (order=%d rows=%d)", order.size, candidate_matrix.shape[0])
        return
    if high_scores:
        # Random sampling from top geo_ratio fraction:
        ratio = float(np.clip(geo_ratio, 0.0, 1.0))
        top_count = max(1, int(np.round(ratio * order.size)))
        top_slice = order[:top_count]
        pick = min(k, top_slice.size)
        if pick <= 0:
            return
        rng = np.random.default_rng(seed)
        chosen = rng.choice(top_slice, size=pick, replace=False)

        # Top k directly
        # chosen = order[:min(k, order.size)]

    else:
        # Random sampling from bottom geo_ratio fraction:
        ratio = float(np.clip(1 - geo_ratio, 0.0, 1.0))
        back_count = max(1, int(np.round(ratio * order.size)))
        back_count = min(back_count, k * 10)  # limit to avoid too large random pool
        back_slice = order[-back_count:]
        pick = min(k, back_slice.size)
        if pick <= 0:
            return
        rng = np.random.default_rng(seed)
        chosen = rng.choice(back_slice, size=pick, replace=False)

        # Bottom k directly
        # chosen = order[-min(k, order.size):]
        # chosen = chosen[::-1]

    omega_points = points_low[omega_indices]
    omega_pcd = o3d.geometry.PointCloud()
    omega_pcd.points = o3d.utility.Vector3dVector(omega_points.astype(np.float64))
    omega_pcd.paint_uniform_color([0.15, 0.15, 0.15])
    geometries = [omega_pcd]
    palette = [
        [0.9, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.1, 0.2, 0.9],
        [0.9, 0.6, 0.1],
        [0.6, 0.2, 0.7],
        [0.1, 0.8, 0.8],
        [0.8, 0.8, 0.1],
        [0.5, 0.1, 0.6],
        [0.2, 0.6, 0.9],
        [0.9, 0.3, 0.5],
        [0.3, 0.9, 0.4],
        [0.4, 0.3, 0.9],
        [0.9, 0.9, 0.3],
        [0.2, 0.9, 0.7],
        [0.7, 0.2, 0.9],
        [0.3, 0.5, 0.9],
        [0.9, 0.5, 0.3],
        [0.2, 0.8, 0.5],
        [0.5, 0.8, 0.2],
        [0.8, 0.2, 0.4],
    ]
    for i, row_idx in enumerate(chosen):
        candidate = candidate_matrix[row_idx]
        candidate_points = points_low[candidate]
        color = palette[i % len(palette)]
        cand_pcd = o3d.geometry.PointCloud()
        cand_pcd.points = o3d.utility.Vector3dVector(candidate_points.astype(np.float64))
        cand_pcd.paint_uniform_color(color)
        for a in range(candidate.shape[0]):
            for b in range(a + 1, candidate.shape[0]):
                cylinder = line_to_cylinder(candidate_points[a], candidate_points[b])
                cylinder.paint_uniform_color(color)
                geometries.append(cylinder)
        geometries.append(cand_pcd)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name)
    for geometry in geometries:
        vis.add_geometry(geometry)

    # Access and set line width
    opt = vis.get_render_option()
    opt.point_size = 17.0  # Default is typically 5.0

    vis.run()
    vis.destroy_window()
    # o3d.visualization.draw_geometries(geometries, window_name=window_name)



def _faces_to_trimesh(face_block: np.ndarray):
    if face_block.size == 0 or trimesh is None:
        return None
    triangles = np.asarray(face_block, dtype=np.float64).reshape(-1, 3, 3)
    vertices = triangles.reshape(-1, 3)
    faces = np.arange(vertices.shape[0], dtype=np.int64).reshape(-1, 3)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
