"""CLI entry for the holding point search pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from python.instrumentation.timing import TimingRecorder
from python.pipeline.pipeline_stub import run_pipeline
from python.utils.config_loader import Config, load_config


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
    config: Config = load_config(args.config)

    recorder = TimingRecorder(config.instrumentation)

    result = run_pipeline(config, recorder)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Result written to {output_path}")


if __name__ == "__main__":
    main()
