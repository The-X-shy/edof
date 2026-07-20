"""Command-line entry point for the isolated EDoF reproduction."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from .config import load_config
from .runner import run_checkpoint_evaluation, run_memory_smoke, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the DeepLens EDoF reproduction")
    parser.add_argument("--config", required=True, help="YAML configuration path")
    parser.add_argument("--output", help="explicit run directory")
    parser.add_argument("--resume", help="checkpoint to resume")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="evaluate --resume on the configured validation set without training",
    )
    parser.add_argument(
        "--memory-smoke",
        action="store_true",
        help="build the optical cache and run one forward/backward batch",
    )
    parser.add_argument("--force-cache", action="store_true", help="recompute cached ray fields")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.resume:
        config = replace(config, training=replace(config.training, resume=arguments.resume))
    if arguments.evaluate_only and arguments.memory_smoke:
        parser.error("--evaluate-only and --memory-smoke are mutually exclusive")
    if arguments.evaluate_only:
        if not arguments.resume:
            parser.error("--evaluate-only requires --resume")
        result = run_checkpoint_evaluation(
            config,
            arguments.resume,
            output_override=arguments.output,
        )
    elif arguments.memory_smoke:
        result = run_memory_smoke(
            config,
            output_override=arguments.output,
            force_cache=arguments.force_cache,
        )
    else:
        result = run_training(
            config,
            output_override=arguments.output,
            force_cache=arguments.force_cache,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
