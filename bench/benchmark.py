"""Benchmark runner for sweeping point cloud and force parameters."""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List

from python.instrumentation.timing import TimingRecorder
from python.pipeline.pipeline_stub import run_pipeline
from python.utils.config_loader import Config, load_config
from python.utils.logging_setup import CppLoggingSettings, configure_python_logging, load_logging_settings
from python.utils.pointcloud_logging import PointCloudDebugSaver, load_pointcloud_logging_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark holding-point pipeline under varied settings.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.json",
        help="Base config file to load before applying overrides.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        nargs="+",
        default=[64, 128, 256],
        help="List of preprocess.downsample_num values to evaluate.",
    )
    parser.add_argument(
        "--force-samples",
        type=int,
        nargs="+",
        default=[200, 500, 1000],
        help="List of search.force_sample_count values to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/benchmark.jsonl",
        help="Where to write benchmark results (JSONL).",
    )
    return parser.parse_args()


def _clone_config(base: Config) -> Config:
    return deepcopy(base)


def _run_trial(config: Config, pc_logger: PointCloudDebugSaver, cpp_logging: CppLoggingSettings) -> Dict[str, object]:
    recorder = TimingRecorder(config.instrumentation)
    start = time.perf_counter()
    result = run_pipeline(config, recorder, pc_logger=pc_logger, cpp_logging=cpp_logging)
    duration_ms = (time.perf_counter() - start) * 1000.0
    best = result.get("scores", {}).get("best_candidate")
    metrics = {
        "status": result.get("status"),
        "duration_ms": duration_ms,
        "dataset": result.get("dataset", {}),
        "downsample_num": config.preprocess.get("downsample_num"),
        "force_sample_count": config.search.get("force_sample_count"),
        "total_combinations": result.get("combinations", {}).get("total_combinations"),
        "survivor_count": result.get("scores", {}).get("survivor_count"),
        "best_score": best.get("score_total") if best else None,
        "best_hit_count": best.get("hit_count") if best else None,
    }
    return metrics


def main() -> None:
    args = parse_args()
    logging_settings = load_logging_settings(Path("logging_config.json"))
    configure_python_logging(logging_settings.python)
    pc_settings = load_pointcloud_logging_config(Path("logging_pointcloud_config.json"))
    pc_logger = PointCloudDebugSaver(pc_settings)
    base_config = load_config(args.config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    for downsample in args.downsample:
        for force_samples in args.force_samples:
            cfg = _clone_config(base_config)
            cfg.preprocess["downsample_num"] = int(downsample)
            cfg.search["force_sample_count"] = int(force_samples)
            metrics = _run_trial(cfg, pc_logger, logging_settings.cpp)
            metrics["downsample_num"] = downsample
            metrics["force_sample_count"] = force_samples
            records.append(metrics)
            with output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False))
                f.write("\n")
            print(
                f"[benchmark] downsample={downsample} force_samples={force_samples} "
                f"duration_ms={metrics['duration_ms']:.2f} status={metrics['status']}"
            )

    summary = {
        "trials": len(records),
        "downsample_values": args.downsample,
        "force_sample_values": args.force_samples,
        "output": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
