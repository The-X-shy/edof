"""Run the A/B/C short loss branches on the accepted smooth PSF model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from edof_reproduction.config import EDOFConfig, load_config, validate_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = Path("configs/edof_reproduction/windows_strict_finetune.yaml")
BRANCHES = {
    "a_quality_only": {
        "pixel_loss_weight": 1.0,
        "cross_depth_loss_weight": 0.0,
        "perceptual_weight": 0.0,
        "claim": "strict_loss_ablation",
    },
    "b_low_cross_depth": {
        "pixel_loss_weight": 1.0,
        "cross_depth_loss_weight": 0.1,
        "perceptual_weight": 0.0,
        "claim": "strict_loss_ablation",
    },
    "c_low_perceptual": {
        "pixel_loss_weight": 1.0,
        "cross_depth_loss_weight": 0.1,
        "perceptual_weight": 0.02,
        "claim": "practical_loss_ablation",
    },
}
MAX_EPOCHS = 20
VALIDATE_EVERY = 5
GATE_EPOCH = 10
GATE_PSNR = 16.4
RAW_TARGET_PSNR = 17.34


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


def branch_config(
    project_root: Path,
    branch_name: str,
    settings: dict[str, Any],
) -> EDOFConfig:
    workspace = project_root / "workspace" / "edof_reproduction"
    base = load_config(project_root / BASE_CONFIG)
    convergence_cache = (
        workspace
        / "windows_strict_optics_convergence"
        / "caches"
        / "wavefront_cache_3x3_1024.pt"
    )
    fixed_psf_cache = (
        workspace
        / "windows_strict_spatial_finetune"
        / "fixed_psf_map_40x40_1024_k127.pt"
    )
    initialize_from = (
        workspace / "windows_optimized" / "checkpoints" / "epoch_050.pt"
    )
    for path in (convergence_cache, fixed_psf_cache, initialize_from):
        if not path.exists():
            raise FileNotFoundError(f"loss-ablation input is missing: {path}")
    config = replace(
        base,
        optics=replace(
            base.optics,
            cache_file=str(convergence_cache),
            field_grid=3,
            simulation_grid=1024,
            psf_size=127,
            finetune_field_grid=40,
            finetune_psf_mode="exact",
            finetune_psf_cache_file=str(fixed_psf_cache),
            propagation_batch_size=1,
            spatial_psf_refine_factor=5,
        ),
        dataset=replace(
            base.dataset,
            crop_size=125,
        ),
        evaluation=replace(
            base.evaluation,
            crop_size=1000,
            max_images=100,
            every_n_epochs=VALIDATE_EVERY,
            early_stopping_patience=3,
            early_stopping_min_delta=0.02,
            field_grid=40,
            local_field_patches=False,
            minimum_psnr_epoch=GATE_EPOCH,
            minimum_psnr=GATE_PSNR,
        ),
        training=replace(
            base.training,
            joint_epochs=0,
            finetune_epochs=MAX_EPOCHS,
            finetune_lr=0.00005,
            warmup_epochs=0,
            pixel_loss_weight=float(settings["pixel_loss_weight"]),
            perceptual_weight=float(settings["perceptual_weight"]),
            pixel_loss_type="rmse",
            cross_depth_loss_weight=float(settings["cross_depth_loss_weight"]),
            local_field_patches=True,
            spatial_field_crop_grid=5,
            checkpoint_every=VALIDATE_EVERY,
            initialize_from=str(initialize_from),
            resume=None,
        ),
        output=replace(
            base.output,
            run_name=f"windows_smooth_{branch_name}",
        ),
    )
    validate_config(config)
    return config


def run(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    workspace = project_root / "workspace" / "edof_reproduction"
    smooth_gate_path = workspace / "windows_smooth_psf_gate" / "summary.json"
    if not smooth_gate_path.exists():
        raise FileNotFoundError(f"smooth PSF gate is missing: {smooth_gate_path}")
    smooth_gate = read_json(smooth_gate_path)
    if not smooth_gate.get("passed"):
        raise RuntimeError("smooth PSF gate did not pass; loss ablation was not started")

    sequence_output = workspace / "windows_smooth_loss_ablation"
    state_path = sequence_output / "state.json"
    summary_path = sequence_output / "summary.json"
    sequence_output.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        previous = read_json(summary_path)
        if previous.get("status") == "completed":
            return previous
    state = {
        "status": "running",
        "started_at": now(),
        "updated_at": now(),
        "current_step": "initializing",
        "branches": {},
    }
    write_json(state_path, state)

    results: dict[str, Any] = {}
    for branch_name, settings in BRANCHES.items():
        output = workspace / f"windows_smooth_{branch_name}"
        branch_summary_path = output / "summary.json"
        state["current_step"] = branch_name
        state["updated_at"] = now()
        state["branches"][branch_name] = {
            "status": "running",
            "updated_at": now(),
            "output": str(output),
        }
        write_json(state_path, state)
        if not branch_summary_path.exists():
            config = branch_config(project_root, branch_name, settings)
            config_path = sequence_output / f"{branch_name}.json"
            write_json(config_path, config.as_dict())
            arguments = [
                sys.executable,
                "-u",
                "-m",
                "edof_reproduction",
                "--config",
                str(config_path),
                "--output",
                str(output),
            ]
            process = subprocess.run(arguments, cwd=project_root, check=False)
            if process.returncode != 0:
                state["status"] = "failed"
                state["branches"][branch_name].update(
                    {
                        "status": "failed",
                        "exit_code": process.returncode,
                        "updated_at": now(),
                    }
                )
                write_json(state_path, state)
                raise RuntimeError(
                    f"{branch_name} failed with exit code {process.returncode}"
                )
        if not branch_summary_path.exists():
            raise RuntimeError(f"{branch_name} did not write a training summary")
        training = read_json(branch_summary_path)
        mean = training["validation_mean"]
        gain = float(mean["psnr"]) - float(mean["raw"]["psnr"])
        branch_result = {
            "status": "completed",
            "claim": settings["claim"],
            "output": str(output),
            "epochs_completed": training["epochs_completed"],
            "stopped_early": training["stopped_early"],
            "stop_reason": training.get("stop_reason"),
            "best_epoch": training["best_epoch"],
            "best_validation_psnr": training["best_validation_psnr"],
            "validation_mean": mean,
            "gain_over_raw_db": gain,
            "exceeded_raw": float(mean["psnr"]) >= RAW_TARGET_PSNR,
        }
        results[branch_name] = branch_result
        state["branches"][branch_name] = {
            **branch_result,
            "updated_at": now(),
        }
        write_json(state_path, state)
        print(
            json.dumps(
                {"smooth_loss_ablation": {branch_name: branch_result}},
                ensure_ascii=False,
            ),
            flush=True,
        )

    winner_name = max(
        results,
        key=lambda name: float(results[name]["validation_mean"]["psnr"]),
    )
    any_exceeded_raw = any(row["exceeded_raw"] for row in results.values())
    summary = {
        "status": "completed",
        "completed_at": now(),
        "protocol": {
            "simulation_grid": 1024,
            "psf_size": 127,
            "base_field_grid": 40,
            "spatial_refine_factor": 5,
            "training_crop": 125,
            "training_field_grid": 5,
            "max_epochs": MAX_EPOCHS,
            "validate_every": VALIDATE_EVERY,
            "gate_epoch": GATE_EPOCH,
            "gate_psnr": GATE_PSNR,
            "raw_target_psnr": RAW_TARGET_PSNR,
        },
        "branches": results,
        "winner": {
            "name": winner_name,
            **results[winner_name],
        },
        "any_exceeded_raw": any_exceeded_raw,
        "doe_optimization_required": not any_exceeded_raw,
    }
    write_json(summary_path, summary)
    state.update(
        {
            "status": "completed",
            "current_step": (
                "completed"
                if any_exceeded_raw
                else "completed_network_ablation_doe_required"
            ),
            "updated_at": now(),
            "completed_at": now(),
            "winner": summary["winner"],
            "doe_optimization_required": not any_exceeded_raw,
        }
    )
    write_json(state_path, state)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    run(args.project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
