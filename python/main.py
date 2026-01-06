"""CLI entry for the holding point search pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from python.instrumentation.timing import TimingRecorder
from python.pipeline.pipeline_stub import run_pipeline
from python.utils.config_loader import Config, load_config
from python.utils.logging_setup import configure_python_logging, load_logging_settings
from python.utils.pointcloud_logging import PointCloudDebugSaver, load_pointcloud_logging_config
from python.utils.random_seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Holding point search pipeline (skeleton).")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/result.json",
        help="Path to write pipeline results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging_settings = load_logging_settings(Path("logging_config.json"))
    configure_python_logging(logging_settings.python)
    pc_logging_settings = load_pointcloud_logging_config(Path("logging_pointcloud_config.json"))
    pc_logger = PointCloudDebugSaver(pc_logging_settings)

    config: Config = load_config(args.config)
    set_global_seed(config.seed)

    recorder = TimingRecorder(config.instrumentation)

    logging.getLogger("pipeline").info("Starting pipeline with config %s", args.config)
    result = run_pipeline(config, recorder, pc_logger=pc_logger, cpp_logging=logging_settings.cpp)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Result written to {output_path}")


if __name__ == "__main__":
    main()
