"""Conditionally re-optimize the DOE and require a one-decibel raw-image gain."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from edof_reproduction.config import load_config, validate_config
from edof_reproduction.dataset import build_loader, build_validation_loader
from edof_reproduction.evaluation import evaluate_reconstruction
from edof_reproduction.imaging import spatial_convolution, wavelength_choice
from edof_reproduction.optics import (
    CachedRayWaveOptics,
    load_or_build_cache,
    load_or_build_fixed_psf_map,
)
from edof_reproduction.runner import _pairwise_rmse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = Path("configs/edof_reproduction/windows_strict_finetune.yaml")
DOE_STEPS = 200
DOE_LEARNING_RATE = 0.01
RAW_GAIN_TARGET_DB = 1.0


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


def build_optics_config(project_root: Path, output: Path):
    workspace = project_root / "workspace" / "edof_reproduction"
    base = load_config(project_root / BASE_CONFIG)
    config = replace(
        base,
        optics=replace(
            base.optics,
            cache_file=str(
                workspace
                / "windows_strict_optics_convergence"
                / "caches"
                / "wavefront_cache_3x3_1024.pt"
            ),
            field_grid=3,
            simulation_grid=1024,
            psf_size=127,
            finetune_field_grid=40,
            finetune_psf_mode="exact",
            finetune_psf_cache_file=str(
                output / "fixed_psf_map_40x40_1024_k127_doe_optimized.pt"
            ),
            propagation_batch_size=1,
            spatial_psf_refine_factor=5,
        ),
        dataset=replace(
            base.dataset,
            crop_size=125,
            batch_size=1,
            workers=2,
        ),
        evaluation=replace(
            base.evaluation,
            crop_size=1000,
            max_images=100,
            every_n_epochs=5,
            field_grid=40,
            local_field_patches=False,
            minimum_psnr_epoch=None,
            minimum_psnr=None,
        ),
        training=replace(
            base.training,
            joint_epochs=0,
            finetune_epochs=20,
            local_field_patches=True,
            spatial_field_crop_grid=5,
        ),
    )
    validate_config(config)
    return config


def optimize_doe(
    config,
    optics: CachedRayWaveOptics,
    loader,
    output: Path,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(optics.doe.parameters(), lr=DOE_LEARNING_RATE)
    history: list[dict[str, float]] = []
    iterator = iter(loader)
    log_path = output / "doe_optimization.jsonl"
    for step in range(DOE_STEPS):
        try:
            clean = next(iterator)
        except StopIteration:
            if hasattr(loader.dataset, "set_epoch"):
                loader.dataset.set_epoch(step // max(len(loader), 1) + 1)
            iterator = iter(loader)
            clean = next(iterator)
        clean = clean.to(optics.device_for_compute, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        field_index = step % optics.field_count
        psfs = optics.psfs(
            wavelength_choice(step, averaged=False),
            field_indices=(field_index,),
        )
        sensors = [
            spatial_convolution(clean, psfs[depth_index])
            for depth_index in range(psfs.shape[0])
        ]
        quality = torch.stack(
            [
                torch.sqrt(torch.nn.functional.mse_loss(sensor, clean) + 1e-12)
                for sensor in sensors
            ]
        ).mean()
        depth_similarity = _pairwise_rmse(sensors)
        loss = quality + 0.1 * depth_similarity
        loss.backward()
        torch.nn.utils.clip_grad_norm_(optics.doe.parameters(), 1.0)
        optimizer.step()
        row = {
            "step": step + 1,
            "field_index": field_index,
            "loss": float(loss.detach()),
            "raw_rmse": float(quality.detach()),
            "depth_similarity": float(depth_similarity.detach()),
        }
        history.append(row)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if (step + 1) % 10 == 0 or step == 0:
            print(json.dumps({"doe_optimization": row}, ensure_ascii=False), flush=True)
    return history


def run(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    workspace = project_root / "workspace" / "edof_reproduction"
    ablation_path = workspace / "windows_smooth_loss_ablation" / "summary.json"
    smooth_gate_path = workspace / "windows_smooth_psf_gate" / "summary.json"
    if not ablation_path.exists() or not smooth_gate_path.exists():
        raise FileNotFoundError("DOE prerequisites are missing")
    ablation = read_json(ablation_path)
    smooth_gate = read_json(smooth_gate_path)
    if not ablation.get("doe_optimization_required"):
        return {
            "status": "skipped",
            "reason": "a network branch already exceeded raw imaging",
        }

    output = workspace / "windows_doe_raw_optimization"
    state_path = output / "state.json"
    summary_path = output / "summary.json"
    output.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        return read_json(summary_path)
    state = {
        "status": "running",
        "started_at": now(),
        "updated_at": now(),
        "current_step": "optimizing_doe",
    }
    write_json(state_path, state)
    if not torch.cuda.is_available():
        raise RuntimeError("DOE optimization requires CUDA")

    config = build_optics_config(project_root, output)
    device = torch.device("cuda")
    cache, _ = load_or_build_cache(config.optics, output)
    optics = CachedRayWaveOptics(config.optics, cache, device).to(device)
    source_checkpoint_path = (
        workspace / "windows_optimized" / "checkpoints" / "epoch_050.pt"
    )
    source_checkpoint = torch.load(
        source_checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    optics.doe.load_exported_state(source_checkpoint["doe"])
    loader = build_loader(config.dataset, config.training.seed)
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(0)
    history = optimize_doe(config, optics, loader, output)
    optimized_checkpoint = dict(source_checkpoint)
    optimized_checkpoint["doe"] = optics.doe.export_state()
    optimized_checkpoint["doe_optimization"] = {
        "steps": DOE_STEPS,
        "learning_rate": DOE_LEARNING_RATE,
        "first": history[0],
        "last": history[-1],
    }
    checkpoint_path = output / "doe_optimized_initialization.pt"
    torch.save(optimized_checkpoint, checkpoint_path)

    state["current_step"] = "building_exact_psf_map"
    state["updated_at"] = now()
    write_json(state_path, state)
    fixed_psfs, fixed_path = load_or_build_fixed_psf_map(
        config.optics,
        optics,
        output,
    )
    state["current_step"] = "evaluating_raw_gain"
    state["updated_at"] = now()
    write_json(state_path, state)
    validation_loader = build_validation_loader(
        config.dataset,
        config.evaluation,
        config.training.seed,
    )
    raw_metrics, _ = evaluate_reconstruction(
        optics,
        torch.nn.Identity().to(device),
        validation_loader,
        device=device,
        depths_mm=config.optics.depths_mm,
        noise_std=config.evaluation.noise_std,
        use_lpips=False,
        seed=config.training.seed,
        field_grid=40,
        local_field_patches=False,
        fixed_psfs=fixed_psfs,
        field_refine_factor=5,
    )
    baseline_raw_psnr = float(
        smooth_gate["measurements"]["smooth_raw_psnr"]
    )
    optimized_raw_psnr = float(raw_metrics["mean"]["raw"]["psnr"])
    raw_gain = optimized_raw_psnr - baseline_raw_psnr
    passed = raw_gain >= RAW_GAIN_TARGET_DB

    network_training = None
    if passed:
        state["current_step"] = "training_network_after_doe"
        state["updated_at"] = now()
        write_json(state_path, state)
        winner = ablation["winner"]
        winner_name = winner["name"]
        winner_settings = {
            "a_quality_only": (1.0, 0.0, 0.0),
            "b_low_cross_depth": (1.0, 0.1, 0.0),
            "c_low_perceptual": (1.0, 0.1, 0.02),
        }[winner_name]
        training_config = replace(
            config,
            training=replace(
                config.training,
                joint_epochs=0,
                finetune_epochs=50,
                finetune_lr=0.00005,
                warmup_epochs=0,
                pixel_loss_weight=winner_settings[0],
                cross_depth_loss_weight=winner_settings[1],
                perceptual_weight=winner_settings[2],
                pixel_loss_type="rmse",
                checkpoint_every=5,
                initialize_from=str(checkpoint_path),
                resume=None,
            ),
            evaluation=replace(
                config.evaluation,
                every_n_epochs=5,
                early_stopping_patience=3,
                minimum_psnr_epoch=None,
                minimum_psnr=None,
            ),
        )
        validate_config(training_config)
        training_output = workspace / "windows_doe_optimized_network"
        config_path = output / "network_config.json"
        write_json(config_path, training_config.as_dict())
        process = subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "edof_reproduction",
                "--config",
                str(config_path),
                "--output",
                str(training_output),
            ],
            cwd=project_root,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"post-DOE network training failed with exit code {process.returncode}"
            )
        network_training = read_json(training_output / "summary.json")

    summary = {
        "status": "completed",
        "completed_at": now(),
        "doe_steps": DOE_STEPS,
        "doe_learning_rate": DOE_LEARNING_RATE,
        "source_checkpoint": str(source_checkpoint_path),
        "optimized_checkpoint": str(checkpoint_path),
        "fixed_psf_cache": str(fixed_path),
        "raw_evaluation": raw_metrics,
        "baseline_raw_psnr": baseline_raw_psnr,
        "optimized_raw_psnr": optimized_raw_psnr,
        "raw_gain_db": raw_gain,
        "raw_gain_target_db": RAW_GAIN_TARGET_DB,
        "raw_gate_passed": passed,
        "network_training": network_training,
    }
    write_json(summary_path, summary)
    state.update(
        {
            "status": "completed",
            "current_step": (
                "completed_network_training"
                if passed
                else "completed_raw_gate_failed"
            ),
            "updated_at": now(),
            "completed_at": now(),
            "raw_gain_db": raw_gain,
            "raw_gate_passed": passed,
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
