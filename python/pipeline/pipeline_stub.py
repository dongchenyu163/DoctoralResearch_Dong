"""Minimal stub pipeline to exercise IO."""

from __future__ import annotations

from typing import Dict, List

from python.utils.config_loader import Config


def run_pipeline(config: Config) -> Dict[str, object]:
    """Execute a placeholder pipeline that mirrors the expected structure."""
    dataset_info = {"points_loaded": 0, "normals_loaded": 0}
    preprocess_summary = {
        "downsample_num": config.preprocess.get("downsample_num"),
        "normal_estimation_radius": config.preprocess.get("normal_estimation_radius"),
    }
    mesh_boolean_summary = {"contact_faces": 0, "purified_faces": 0}

    timestep_reports: List[Dict[str, object]] = []
    for step_idx in range(1):
        valid_indices_count = 0
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
