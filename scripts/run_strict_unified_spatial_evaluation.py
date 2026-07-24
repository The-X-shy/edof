"""Evaluate the shared 127-pixel PSF protocol with repeat and spatial metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from edof_reproduction.config import load_config, validate_config
from edof_reproduction.dataset import build_validation_loader
from edof_reproduction.evaluation import evaluate_reconstruction
from edof_reproduction.metrics import LPIPSMetric
from edof_reproduction.nafnet import NAFNet
from edof_reproduction.optics import (
    CachedRayWaveOptics,
    load_or_build_cache,
    load_or_build_fixed_psf_map,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = Path("configs/edof_reproduction/windows_strict_finetune.yaml")
REPEATABILITY_THRESHOLD = 0.02


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


def numeric_deltas(left: Any, right: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(left, dict) and isinstance(right, dict):
        result: dict[str, float] = {}
        for key in sorted(left.keys() & right.keys()):
            child = f"{prefix}.{key}" if prefix else key
            result.update(numeric_deltas(left[key], right[key], child))
        return result
    if (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
    ):
        return {prefix: abs(float(left) - float(right))}
    return {}


def run(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    workspace = project_root / "workspace" / "edof_reproduction"
    strict_output = workspace / "windows_strict_spatial_finetune"
    output = workspace / "windows_strict_unified_evaluation"
    state_path = output / "state.json"
    summary_path = output / "summary.json"
    output.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        previous = read_json(summary_path)
        if previous.get("status") == "completed":
            return previous

    checkpoints = {
        "joint_epoch50": workspace
        / "windows_optimized"
        / "checkpoints"
        / "epoch_050.pt",
        "spatial_epoch20": strict_output / "checkpoints" / "epoch_020.pt",
        "spatial_epoch50": strict_output / "checkpoints" / "epoch_050.pt",
    }
    convergence_cache = (
        workspace
        / "windows_strict_optics_convergence"
        / "caches"
        / "wavefront_cache_3x3_1024.pt"
    )
    fixed_psf_cache = strict_output / "fixed_psf_map_40x40_1024_k127.pt"
    required = [*checkpoints.values(), convergence_cache, fixed_psf_cache]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required unified-evaluation inputs are missing: {missing}")

    state = {
        "status": "running",
        "started_at": now(),
        "updated_at": now(),
        "current_step": "loading_shared_optics",
        "checkpoints": {key: str(value) for key, value in checkpoints.items()},
    }
    write_json(state_path, state)

    base = load_config(project_root / BASE_CONFIG)
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
            spatial_psf_refine_factor=1,
        ),
        evaluation=replace(
            base.evaluation,
            crop_size=1000,
            max_images=100,
            field_grid=40,
            local_field_patches=False,
            minimum_psnr_epoch=None,
            minimum_psnr=None,
        ),
    )
    validate_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_checkpoint = torch.load(
        checkpoints["joint_epoch50"],
        map_location=device,
        weights_only=False,
    )
    cache, _ = load_or_build_cache(config.optics, strict_output)
    optics = CachedRayWaveOptics(config.optics, cache, device).to(device)
    optics.doe.load_exported_state(reference_checkpoint["doe"])
    fixed_psfs, _ = load_or_build_fixed_psf_map(config.optics, optics, strict_output)
    reference_coefficients = optics.doe.coefficients.detach().cpu().clone()
    network = NAFNet(
        in_chan=3,
        out_chan=3,
        width=config.network.width,
        middle_blk_num=config.network.middle_blk_num,
        enc_blk_nums=config.network.enc_blk_nums,
        dec_blk_nums=config.network.dec_blk_nums,
    ).to(device)
    loader = build_validation_loader(
        config.dataset,
        config.evaluation,
        config.training.seed,
    )
    lpips_metric = LPIPSMetric(device)

    evaluations: dict[str, Any] = {}
    repeatability: dict[str, Any] = {}
    for name, checkpoint_path in checkpoints.items():
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        optics.doe.load_exported_state(checkpoint["doe"])
        if not torch.allclose(
            optics.doe.coefficients.detach().cpu(),
            reference_coefficients,
            atol=0.0,
            rtol=0.0,
        ):
            raise RuntimeError(f"{name} does not share the fixed PSF DOE state")
        network.load_state_dict(checkpoint["network"])
        repeats = []
        for repeat in (1, 2):
            state["current_step"] = f"{name}_repeat_{repeat}"
            state["updated_at"] = now()
            write_json(state_path, state)
            metrics, _ = evaluate_reconstruction(
                optics,
                network,
                loader,
                device=device,
                depths_mm=config.optics.depths_mm,
                noise_std=config.evaluation.noise_std,
                use_lpips=True,
                seed=config.training.seed,
                lpips_metric=lpips_metric,
                field_grid=40,
                local_field_patches=False,
                fixed_psfs=fixed_psfs,
                field_refine_factor=1,
            )
            repeats.append(metrics)
            print(
                json.dumps(
                    {
                        "unified_evaluation": {
                            "checkpoint": name,
                            "repeat": repeat,
                            "mean": metrics["mean"],
                        }
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        deltas = numeric_deltas(repeats[0], repeats[1])
        maximum_delta = max(deltas.values(), default=0.0)
        repeatability[name] = {
            "maximum_absolute_delta": maximum_delta,
            "threshold": REPEATABILITY_THRESHOLD,
            "passed": maximum_delta <= REPEATABILITY_THRESHOLD,
            "deltas": deltas,
        }
        evaluations[name] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
            "validation": repeats[0],
        }

    passed = all(row["passed"] for row in repeatability.values())
    summary = {
        "status": "completed" if passed else "failed_repeatability",
        "completed_at": now(),
        "protocol": {
            "simulation_grid": 1024,
            "psf_size": 127,
            "field_grid": 40,
            "sensor_crop": [1000, 1000],
            "validation_images": 100,
            "noise_seed": config.training.seed,
            "repeats": 2,
            "repeatability_threshold": REPEATABILITY_THRESHOLD,
        },
        "evaluations": evaluations,
        "repeatability": repeatability,
    }
    write_json(summary_path, summary)
    state.update(
        {
            "status": summary["status"],
            "current_step": "completed" if passed else "repeatability_failed",
            "updated_at": now(),
            "completed_at": now(),
        }
    )
    write_json(state_path, state)
    if not passed:
        raise RuntimeError("unified evaluation did not satisfy repeatability threshold")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    run(args.project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
