"""Run the corrected strict fine-tune with spatially varying training PSFs."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edof_reproduction.config import load_config
from edof_reproduction.convergence import (
    build_strict_finetune_config,
    select_optical_settings,
)
from edof_reproduction.runner import run_training


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = Path("configs/edof_reproduction/windows_strict_finetune.yaml")
TRAINING_CROP_SIZE = 125
TRAINING_FIELD_GRID = 5


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def run(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    workspace = project_root / "workspace" / "edof_reproduction"
    convergence_path = workspace / "windows_strict_optics_convergence" / "summary.json"
    output = workspace / "windows_strict_spatial_finetune"
    state_path = output / "strict_state.json"
    output.mkdir(parents=True, exist_ok=True)
    if not convergence_path.exists():
        raise FileNotFoundError(f"optical convergence summary is missing: {convergence_path}")
    convergence = read_json(convergence_path)
    if convergence.get("status") != "completed":
        raise RuntimeError("optical convergence did not complete")
    decision = select_optical_settings(convergence["cases"])
    grid = int(decision["selected_simulation_grid"])
    psf_size = int(decision["selected_psf_size"])
    if not decision["compact_psf_truncated"]:
        raise RuntimeError("measured compact PSF does not exceed the truncation threshold")

    checkpoint = workspace / "windows_optimized" / "checkpoints" / "epoch_050.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"joint epoch-50 checkpoint is missing: {checkpoint}")

    training_summary_path = output / "summary.json"
    result_path = output / "strict_result.json"
    if training_summary_path.exists() and result_path.exists():
        result = read_json(result_path)
        if result.get("status") == "completed":
            return result

    cache_file = (
        f"../windows_strict_optics_convergence/caches/"
        f"wavefront_cache_3x3_{grid}.pt"
    )
    fixed_psf_cache = f"fixed_psf_map_40x40_{grid}_k{psf_size}.pt"
    base = load_config(project_root / BASE_CONFIG)
    config = build_strict_finetune_config(
        base,
        decision,
        cache_file=cache_file,
        fixed_psf_cache_file=fixed_psf_cache,
        initialize_from=str(checkpoint),
        training_crop_size=TRAINING_CROP_SIZE,
        spatial_field_crop_grid=TRAINING_FIELD_GRID,
    )
    latest = output / "checkpoints" / "latest.pt"
    if latest.exists() and not training_summary_path.exists():
        config = replace(
            config,
            training=replace(
                config.training,
                resume=str(latest),
                initialize_from=None,
            ),
        )

    state = {
        "status": "running",
        "started_at": now(),
        "updated_at": now(),
        "selected_simulation_grid": grid,
        "selected_psf_size": psf_size,
        "compact_psf_edge_energy": decision["compact_psf_edge_energy"],
        "cache_file": cache_file,
        "fixed_psf_cache_file": fixed_psf_cache,
        "training_crop_size": TRAINING_CROP_SIZE,
        "training_field_grid": TRAINING_FIELD_GRID,
        "field_positions_per_axis": 40 - TRAINING_FIELD_GRID + 1,
        "planned_finetune_epochs": 50,
    }
    write_json(state_path, state)
    try:
        training = run_training(config, output_override=output)
        validation = training["validation_mean"]
        reconstruction_gain = (
            float(validation["psnr"]) - float(validation["raw"]["psnr"])
        )
        result = {
            "status": "completed",
            "completed_at": now(),
            "selected_simulation_grid": grid,
            "selected_psf_size": psf_size,
            "compact_psf_edge_energy": decision["compact_psf_edge_energy"],
            "training_summary": str(training_summary_path),
            "selected_checkpoint": training["selected_checkpoint"],
            "validation_mean": validation,
            "reconstruction_gain_over_raw_db": reconstruction_gain,
            "target_gain_db": 4.0,
            "target_passed": reconstruction_gain >= 4.0,
            "spatial_training": {
                "full_field_grid": 40,
                "crop_field_grid": TRAINING_FIELD_GRID,
                "crop_pixels": TRAINING_CROP_SIZE,
                "field_cell_pixels": TRAINING_CROP_SIZE // TRAINING_FIELD_GRID,
            },
            "paper_loss": {
                "pixel_loss_type": "rmse",
                "quality_weight": 0.3,
                "cross_depth_weight": 1.0,
                "perceptual_weight": 0.0,
            },
        }
        write_json(result_path, result)
        state.update(
            {
                "status": "completed",
                "updated_at": now(),
                "completed_at": now(),
                "result": result,
            }
        )
        write_json(state_path, state)
        print(
            json.dumps({"strict_spatial_finetune_result": result}, ensure_ascii=False),
            flush=True,
        )
        return result
    except BaseException as error:
        state.update(
            {
                "status": "failed",
                "updated_at": now(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        write_json(state_path, state)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    run(args.project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
