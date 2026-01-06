"""Configuration loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


def _deep_merge(user_data: Dict[str, Any], default_data: Dict[str, Any]) -> Dict[str, Any]:
    """Merge user_data over default_data recursively."""
    merged = dict(default_data)
    for key, value in user_data.items():
        if isinstance(value, dict) and isinstance(default_data.get(key), dict):
            merged[key] = _deep_merge(value, default_data[key])
        else:
            merged[key] = value
    return merged


@dataclass
class TimingOutputConfig:
    format: str
    path: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimingOutputConfig":
        return cls(format=data["format"], path=data["path"])


@dataclass
class InstrumentationSections:
    python: Dict[str, bool]
    cpp: Dict[str, bool]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstrumentationSections":
        return cls(python=dict(data.get("python", {})), cpp=dict(data.get("cpp", {})))


@dataclass
class InstrumentationConfig:
    enable_timing: bool
    enable_detailed_timing: bool
    emit_per_timestep_report: bool
    emit_per_candidate_debug: bool
    timing_output: TimingOutputConfig
    sections: InstrumentationSections

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InstrumentationConfig":
        return cls(
            enable_timing=bool(data.get("enable_timing", True)),
            enable_detailed_timing=bool(data.get("enable_detailed_timing", True)),
            emit_per_timestep_report=bool(data.get("emit_per_timestep_report", False)),
            emit_per_candidate_debug=bool(data.get("emit_per_candidate_debug", False)),
            timing_output=TimingOutputConfig.from_dict(data["timing_output"]),
            sections=InstrumentationSections.from_dict(data.get("sections", {})),
        )


@dataclass
class Config:
    preprocess: Dict[str, Any]
    weights: Dict[str, Any]
    knife: Dict[str, Any]
    physics: Dict[str, Any]
    environment: Dict[str, Any]
    search: Dict[str, Any]
    instrumentation: InstrumentationConfig
    seed: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        return cls(
            preprocess=dict(data["preprocess"]),
            weights=dict(data["weights"]),
            knife=dict(data["knife"]),
            physics=dict(data["physics"]),
            environment=dict(data.get("environment", {})),
            search=dict(data["search"]),
            instrumentation=InstrumentationConfig.from_dict(data["instrumentation"]),
            seed=int(data.get("seed", 42)),
        )


DEFAULT_CONFIG: Dict[str, Any] = {
    "preprocess": {
        "point_cloud_path": None,
        "synthetic_point_count": 512,
        "downsample_num": 100,
        "normal_estimation_radius": 0.01,
    },
    "weights": {
        "geo_score": {"w_fin": 1.0, "w_knf": 4.4, "w_tbl": 6.0},
        "pos_score": {"w_pdir": 5.0, "w_pdis": 4.0},
        "force_score": {"w_mag": 2.0, "w_dir": 2.0, "w_var": 1.0},
    },
    "knife": {"edge_angle_deg": 30.0, "height": 0.05},
    "physics": {
        "friction_coef": 0.5,
        "fracture_toughness": 400.0,
        "pressure_distribution": 3000.0,
        "planar_constraint": True,
        "friction_cone": {"angle_deg": 40.0, "num_sides": 8},
        "force_balance_threshold": 1e-4,
    },
    "environment": {"table_z": 0.0},
    "search": {
        "geo_filter_ratio": 0.6,
        "force_sample_count": 1000,
        "finger_count": 2,
        "table_clearance": 0.002,
        "knife_clearance": 0.002,
        "max_geo_candidates": 2048,
    },
    "instrumentation": {
        "enable_timing": True,
        "enable_detailed_timing": True,
        "emit_per_timestep_report": False,
        "emit_per_candidate_debug": False,
        "timing_output": {"format": "jsonl", "path": "logs/timing.jsonl"},
        "sections": {
            "python": {
                "io": True,
                "preprocess": True,
                "preprocess_total": True,
                "downsample": True,
                "estimate_normals": True,
                "trajectory_loop": True,
                "valid_indices": True,
                "compute_valid_indices_total": True,
                "valid_filter_table": True,
                "valid_filter_knife": True,
                "mesh_boolean": True,
                "contact_surface_purify": True,
                "accumulate_scores": True,
                "build_P_all": True,
            },
            "cpp": {
                "kdtree": True,
                "geo_filter": True,
                "pos_score": True,
                "knife_wrench": True,
                "dyn_score_total": True,
                "force_generate": True,
                "force_check": True,
                "grasp_matrix": True,
                "pinv": True,
                "normalize": True,
            },
        },
    },
    "seed": 42,
}


def load_config(config_path: str | Path) -> Config:
    """Load JSON config and fill missing fields with defaults."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        user_config = json.load(f)
    merged_config = _deep_merge(user_config, DEFAULT_CONFIG)
    return Config.from_dict(merged_config)
