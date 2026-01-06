"""Minimal pipeline skeleton that wires together early phases."""

from __future__ import annotations

from typing import Dict, List

from python.instrumentation.timing import TimingRecorder
from python.pipeline.accumulate import build_all_combinations
from python.pipeline.preprocess import PreprocessResult, RawPointCloud, load_point_cloud, preprocess_point_cloud
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

    with recorder.section("python/mesh_boolean"):
        mesh_boolean_summary = {"contact_faces": 0, "purified_faces": 0}

    with recorder.section("python/contact_surface_purify"):
        mesh_boolean_summary["purified_faces"] = mesh_boolean_summary["contact_faces"]

    timestep_reports: List[Dict[str, object]] = []
    with recorder.section("python/trajectory_loop"):
        for step_idx in range(1):
            valid_indices_count = int(valid_result.indices.size)
            with recorder.section("python/accumulate_scores"):
                timestep_reports.append(
                    {
                        "timestep": step_idx,
                        "valid_indices": valid_indices_count,
                        "geo_filter_ratio": config.search.get("geo_filter_ratio"),
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
        "timesteps": timestep_reports,
    }
    return result_summary
