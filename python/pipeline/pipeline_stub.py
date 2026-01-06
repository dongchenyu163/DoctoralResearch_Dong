"""Minimal pipeline skeleton that wires together early phases."""

from __future__ import annotations

from typing import Dict, List

from python.instrumentation.timing import TimingRecorder
from python.pipeline.accumulate import build_all_combinations
from python.pipeline.contact_surface import ContactSurfaceResult, extract_contact_surface
from python.pipeline.dynamics import compute_dynamics_scores
from python.pipeline.geo_filter import GeoFilterRunner
from python.pipeline.preprocess import PreprocessResult, RawPointCloud, load_point_cloud, preprocess_point_cloud
from python.pipeline.trajectory import TrajectoryNode, build_test_trajectory
from python.pipeline.wrench import compute_wrench
from python.pipeline.valid_indices import ValidIndicesResult, compute_valid_indices
from python.utils.config_loader import Config


def run_pipeline(config: Config, recorder: TimingRecorder) -> Dict[str, object]:
    """Execute a placeholder pipeline that mirrors the expected structure."""
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

    valid_result: ValidIndicesResult = compute_valid_indices(preprocess_result.points_low, config, recorder)
    geo_filter = GeoFilterRunner(config)
    geo_filter.set_point_cloud(preprocess_result)

    finger_count = int(config.search.get("finger_count", 2))
    combination_matrix = build_all_combinations(preprocess_result.points_low.shape[0], finger_count, recorder)
    combinations_summary = {
        "finger_count": finger_count,
        "total_combinations": int(combination_matrix.shape[0]),
    }

    valid_summary = {
        "count": int(valid_result.indices.size),
        "table_threshold": valid_result.table_threshold,
        "knife_threshold": valid_result.knife_threshold,
        "passed_table": valid_result.passed_table,
        "passed_knife": valid_result.passed_knife,
    }

    filtered_candidates = geo_filter.run(valid_result, combination_matrix, recorder)

    contact_surface: ContactSurfaceResult = extract_contact_surface(preprocess_result, recorder)
    with recorder.section("python/wrench_compute"):
        wrench = compute_wrench(contact_surface, config)
    mesh_boolean_summary = {
        "contact_faces": contact_surface.metadata.get("total_faces", 0.0),
        "purified_faces": contact_surface.metadata.get("total_faces", 0.0),
    }

    timestep_reports: List[Dict[str, object]] = []
    trajectory_nodes: List[TrajectoryNode] = build_test_trajectory(preprocess_result, config, recorder)

    with recorder.section("python/trajectory_loop"):
        for step_idx, node in enumerate(trajectory_nodes):
            valid_indices_count = int(valid_result.indices.size)
            knife_position = node.pose[:3, 3]
            knife_normal = node.pose[:3, 2]
            positional_scores = geo_filter.calc_positional_scores(filtered_candidates, knife_position, knife_normal)
            positional_distances = geo_filter.calc_positional_distances(filtered_candidates, knife_position, knife_normal)
            dynamics_scores = compute_dynamics_scores(geo_filter, filtered_candidates, wrench, config)
            pos_mean = float(positional_scores.mean()) if positional_scores.size else 0.0
            pdis_mean = float(positional_distances.mean()) if positional_distances.size else 0.0
            dyn_mean = float(dynamics_scores.mean()) if dynamics_scores.size else 0.0
            with recorder.section("python/accumulate_scores"):
                timestep_reports.append(
                    {
                        "timestep": step_idx,
                        "valid_indices": valid_indices_count,
                        "geo_filter_ratio": config.search.get("geo_filter_ratio"),
                        "pose": node.pose[:3, 3].tolist(),
                        "velocity": node.velocity.tolist(),
                        "positional_score_mean": pos_mean,
                        "positional_distance_mean": pdis_mean,
                        "dynamics_score_mean": dyn_mean,
                    }
                )

    result_summary = {
        "status": "stub",
        "message": "Pipeline skeleton executed; no geometry computed.",
        "dataset": dataset_info,
        "preprocess": preprocess_summary,
        "valid_indices": valid_summary,
        "combinations": combinations_summary,
        "contact_surface": mesh_boolean_summary,
        "wrench": wrench.tolist(),
        "geo_filter": {"candidates_after_cpp": int(filtered_candidates.shape[0])},
        "trajectory": {"nodes": len(trajectory_nodes)},
        "timesteps": timestep_reports,
    }
    return result_summary
