"""Command-line entry point for the isolated EDoF reproduction."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from .config import load_config
from .runner import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DeepLens EDoF reproduction")
    parser.add_argument("--config", required=True, help="YAML configuration path")
    parser.add_argument("--output", help="explicit run directory")
    parser.add_argument("--resume", help="checkpoint to resume")
    parser.add_argument("--force-cache", action="store_true", help="recompute cached ray fields")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.resume:
        config = replace(config, training=replace(config.training, resume=arguments.resume))
    result = run_training(
        config,
        output_override=arguments.output,
        force_cache=arguments.force_cache,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
