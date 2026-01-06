"""Minimal stub pipeline to exercise instrumentation and IO."""

from __future__ import annotations

from typing import Dict, List

from python.instrumentation.timing import TimingRecorder
from python.utils.config_loader import Config


def run_pipeline(config: Config, recorder: TimingRecorder) -> Dict[str, object]:
    """Execute a placeholder pipeline that mirrors the expected structure."""
    with recorder.section("python/io"):
        dataset_info = {"points_loaded": 0, "normals_loaded": 0}

    with recorder.section("python/preprocess"):
        preprocess_summary = {
            "downsample_num": config.preprocess.get("downsample_num"),
            "normal_estimation_radius": config.preprocess.get("normal_estimation_radius"),
        }

    with recorder.section("python/mesh_boolean"):
        mesh_boolean_summary = {"contact_faces": 0, "purified_faces": 0}

    with recorder.section("python/contact_surface_purify"):
        mesh_boolean_summary["purified_faces"] = mesh_boolean_summary["contact_faces"]

    timestep_reports: List[Dict[str, object]] = []
    with recorder.section("python/trajectory_loop"):
        for step_idx in range(1):
            with recorder.section("python/valid_indices"):
                valid_indices_count = 0
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
        "contact_surface": mesh_boolean_summary,
        "timesteps": timestep_reports,
    }
    return result_summary
