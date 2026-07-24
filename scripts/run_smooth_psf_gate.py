"""Compare hard and smoothly interpolated PSF maps before retraining."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from edof_reproduction.config import load_config, validate_config
from edof_reproduction.dataset import build_validation_loader
from edof_reproduction.evaluation import evaluate_reconstruction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = Path("configs/edof_reproduction/windows_strict_finetune.yaml")
REFINE_FACTOR = 5
MINIMUM_SEAM_REDUCTION = 0.50
MAXIMUM_RAW_PSNR_DROP_DB = 0.10


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


def save_image(tensor: torch.Tensor, path: Path) -> None:
    array = (
        tensor[0]
        .detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(array).save(path)


def run(project_root: Path, *, force: bool = False) -> dict[str, Any]:
    project_root = project_root.resolve()
    workspace = project_root / "workspace" / "edof_reproduction"
    strict_output = workspace / "windows_strict_spatial_finetune"
    output = workspace / "windows_smooth_psf_gate"
    state_path = output / "state.json"
    summary_path = output / "summary.json"
    output.mkdir(parents=True, exist_ok=True)
    if summary_path.exists() and not force:
        return read_json(summary_path)

    fixed_psf_cache = strict_output / "fixed_psf_map_40x40_1024_k127.pt"
    if not fixed_psf_cache.exists():
        raise FileNotFoundError(f"fixed PSF cache is missing: {fixed_psf_cache}")
    payload = torch.load(fixed_psf_cache, map_location="cpu", weights_only=False)
    fixed_psfs = payload["psfs"]

    base = load_config(project_root / BASE_CONFIG)
    config = replace(
        base,
        optics=replace(
            base.optics,
            field_grid=3,
            simulation_grid=1024,
            psf_size=127,
            finetune_field_grid=40,
            finetune_psf_mode="exact",
            finetune_psf_cache_file=str(fixed_psf_cache),
            spatial_psf_refine_factor=REFINE_FACTOR,
        ),
        dataset=replace(base.dataset, crop_size=125),
        evaluation=replace(
            base.evaluation,
            crop_size=1000,
            max_images=100,
            field_grid=40,
            local_field_patches=False,
            use_lpips=False,
            minimum_psnr_epoch=None,
            minimum_psnr=None,
        ),
        training=replace(
            base.training,
            spatial_field_crop_grid=5,
        ),
    )
    validate_config(config)
    state = {
        "status": "running",
        "started_at": now(),
        "updated_at": now(),
        "current_step": "hard_psf",
    }
    write_json(state_path, state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_validation_loader(
        config.dataset,
        config.evaluation,
        config.training.seed,
    )
    identity = torch.nn.Identity().to(device)
    evaluations: dict[str, Any] = {}
    samples: dict[str, dict[str, torch.Tensor]] = {}
    for name, factor in (("hard", 1), ("smooth", REFINE_FACTOR)):
        state["current_step"] = f"{name}_psf"
        state["updated_at"] = now()
        write_json(state_path, state)
        metrics, sample = evaluate_reconstruction(
            None,
            identity,
            loader,
            device=device,
            depths_mm=config.optics.depths_mm,
            noise_std=config.evaluation.noise_std,
            use_lpips=False,
            seed=config.training.seed,
            field_grid=40,
            local_field_patches=False,
            fixed_psfs=fixed_psfs,
            field_refine_factor=factor,
        )
        evaluations[name] = metrics
        samples[name] = sample
        save_image(sample["sensor"], output / f"{name}_sensor.png")
        print(
            json.dumps(
                {"smooth_psf_gate": {"mode": name, "mean": metrics["mean"]}},
                ensure_ascii=False,
            ),
            flush=True,
        )

    hard_mean = evaluations["hard"]["mean"]
    smooth_mean = evaluations["smooth"]["mean"]
    hard_seam = float(hard_mean["spatial"]["raw_boundary_excess"])
    smooth_seam = float(smooth_mean["spatial"]["raw_boundary_excess"])
    seam_reduction = 1.0 - smooth_seam / max(hard_seam, 1e-12)
    raw_psnr_drop = float(hard_mean["raw"]["psnr"]) - float(
        smooth_mean["raw"]["psnr"]
    )
    seam_passed = seam_reduction >= MINIMUM_SEAM_REDUCTION
    psnr_passed = raw_psnr_drop <= MAXIMUM_RAW_PSNR_DROP_DB
    passed = seam_passed and psnr_passed
    summary = {
        "status": "completed" if passed else "failed_gate",
        "completed_at": now(),
        "passed": passed,
        "protocol": {
            "simulation_grid": 1024,
            "psf_size": 127,
            "base_field_grid": 40,
            "refine_factor": REFINE_FACTOR,
            "seam_metric": "raw_boundary_excess",
            "seam_neighborhood_pixels": 3,
            "validation_images": 100,
            "noise_seed": config.training.seed,
        },
        "thresholds": {
            "minimum_seam_reduction": MINIMUM_SEAM_REDUCTION,
            "maximum_raw_psnr_drop_db": MAXIMUM_RAW_PSNR_DROP_DB,
        },
        "measurements": {
            "hard_seam": hard_seam,
            "smooth_seam": smooth_seam,
            "seam_reduction": seam_reduction,
            "hard_absolute_boundary": hard_mean["spatial"][
                "raw_boundary_discontinuity"
            ],
            "smooth_absolute_boundary": smooth_mean["spatial"][
                "raw_boundary_discontinuity"
            ],
            "hard_raw_psnr": hard_mean["raw"]["psnr"],
            "smooth_raw_psnr": smooth_mean["raw"]["psnr"],
            "raw_psnr_drop_db": raw_psnr_drop,
            "seam_passed": seam_passed,
            "psnr_passed": psnr_passed,
        },
        "evaluations": evaluations,
    }
    write_json(summary_path, summary)
    state.update(
        {
            "status": summary["status"],
            "current_step": "completed" if passed else "gate_failed",
            "updated_at": now(),
            "completed_at": now(),
            "measurements": summary["measurements"],
        }
    )
    write_json(state_path, state)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute the gate even when a previous summary exists",
    )
    args = parser.parse_args()
    run(args.project_root, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
