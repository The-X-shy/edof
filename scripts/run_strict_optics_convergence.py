"""Measure representative-field convergence across optical grids and PSF extents."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from edof_reproduction.config import load_config
from edof_reproduction.convergence import (
    crop_normalized_psfs,
    mean_edge_energy,
    select_optical_settings,
)
from edof_reproduction.dataset import build_validation_loader
from edof_reproduction.imaging import spatial_convolution, wavelength_choice
from edof_reproduction.metrics import batch_psnr, batch_ssim
from edof_reproduction.optics import (
    CachedRayWaveOptics,
    cache_description,
    load_or_build_cache,
)
from edof_reproduction.runner import _seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = Path("configs/edof_reproduction/windows_strict_full_fov_eval.yaml")
SIMULATION_GRIDS = (512, 768, 1024)
PSF_SIZES = (63, 127)
REPRESENTATIVE_FIELD_GRID = 3


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


@torch.no_grad()
def evaluate_raw_psfs(
    psfs: torch.Tensor,
    images: list[torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, Any]:
    totals = [
        {"psnr": 0.0, "ssim": 0.0, "samples": 0}
        for _ in range(psfs.shape[0])
    ]
    for clean_cpu in images:
        clean = clean_cpu.to(device, non_blocking=True)
        for depth_index in range(psfs.shape[0]):
            sensor = spatial_convolution(
                clean,
                psfs[depth_index].to(device, non_blocking=True),
                field_chunk_size=32,
            ).clamp(0.0, 1.0)
            psnr = batch_psnr(sensor, clean)
            ssim = batch_ssim(sensor, clean)
            totals[depth_index]["psnr"] += float(psnr.sum())
            totals[depth_index]["ssim"] += float(ssim.sum())
            totals[depth_index]["samples"] += clean.shape[0]
    depths = []
    for total in totals:
        count = int(total["samples"])
        if count == 0:
            raise RuntimeError("the convergence validation set is empty")
        depths.append(
            {
                "samples": count,
                "raw_psnr": total["psnr"] / count,
                "raw_ssim": total["ssim"] / count,
            }
        )
    return {
        "depth_metrics": depths,
        "mean_raw_psnr": sum(row["raw_psnr"] for row in depths) / len(depths),
        "mean_raw_ssim": sum(row["raw_ssim"] for row in depths) / len(depths),
    }


class OpticsConvergenceRunner:
    def __init__(self, project_root: Path, max_images: int) -> None:
        self.project_root = project_root.resolve()
        self.output = (
            self.project_root
            / "workspace"
            / "edof_reproduction"
            / "windows_strict_optics_convergence"
        )
        self.cache_output = self.output / "caches"
        self.output.mkdir(parents=True, exist_ok=True)
        self.cache_output.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output / "state.json"
        self.max_images = max_images
        self.state: dict[str, Any] = {
            "status": "running",
            "started_at": now(),
            "updated_at": now(),
            "current_grid": None,
            "grids": {},
        }
        if self.state_path.exists():
            previous = json.loads(self.state_path.read_text(encoding="utf-8-sig"))
            self.state["started_at"] = previous.get("started_at", self.state["started_at"])
            self.state["grids"] = previous.get("grids", {})
        self._save_state()

    def _save_state(self) -> None:
        self.state["updated_at"] = now()
        write_json(self.state_path, self.state)

    def _set_grid(self, grid: int, status: str, **details: Any) -> None:
        key = str(grid)
        self.state["current_grid"] = grid
        row = self.state["grids"].setdefault(key, {})
        row.update({"status": status, "updated_at": now(), **details})
        self._save_state()
        print(
            json.dumps(
                {"strict_optics_convergence": {"grid": grid, "status": status, **details}},
                ensure_ascii=False,
            ),
            flush=True,
        )

    def run(self) -> dict[str, Any]:
        base = load_config(self.project_root / BASE_CONFIG)
        checkpoint_path = (
            self.project_root
            / "workspace"
            / "edof_reproduction"
            / "windows_optimized"
            / "checkpoints"
            / "epoch_050.pt"
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"joint epoch-50 checkpoint is missing: {checkpoint_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("strict optical convergence requires CUDA")
        device = torch.device("cuda")
        _seed_everything(base.training.seed)
        evaluation = replace(
            base.evaluation,
            crop_size=1000,
            max_images=self.max_images,
            use_lpips=False,
            noise_std=0.0,
            field_grid=REPRESENTATIVE_FIELD_GRID,
            local_field_patches=False,
        )
        loader = build_validation_loader(base.dataset, evaluation, base.training.seed)
        images = [batch.cpu() for batch in loader]
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        cases: list[dict[str, Any]] = []
        for grid in SIMULATION_GRIDS:
            grid_summary_path = self.output / f"grid_{grid}" / "summary.json"
            if grid_summary_path.exists():
                cached_summary = json.loads(
                    grid_summary_path.read_text(encoding="utf-8-sig")
                )
                if cached_summary.get("status") == "completed":
                    cases.extend(cached_summary["cases"])
                    self._set_grid(grid, "completed", skipped=True)
                    continue

            self._set_grid(grid, "building_cache")
            started = time.monotonic()
            optics_config = replace(
                base.optics,
                cache_file=f"wavefront_cache_3x3_{grid}.pt",
                field_grid=REPRESENTATIVE_FIELD_GRID,
                simulation_grid=grid,
                psf_size=max(PSF_SIZES),
                finetune_field_grid=REPRESENTATIVE_FIELD_GRID,
                finetune_psf_mode="interpolate",
                propagation_batch_size=1,
            )
            _seed_everything(base.training.seed)
            cache, cache_path = load_or_build_cache(optics_config, self.cache_output)
            optics = CachedRayWaveOptics(optics_config, cache, device).to(device)
            optics.doe.load_exported_state(checkpoint["doe"])
            self._set_grid(grid, "computing_psfs", cache=str(cache_path))
            largest_psfs = optics.psfs(wavelength_choice(0, averaged=True)).detach()
            grid_cases = []
            for psf_size in PSF_SIZES:
                case_psfs = crop_normalized_psfs(largest_psfs, psf_size)
                metrics = evaluate_raw_psfs(case_psfs, images, device=device)
                case = {
                    "status": "completed",
                    "simulation_grid": grid,
                    "psf_size": psf_size,
                    "representative_field_grid": REPRESENTATIVE_FIELD_GRID,
                    "validation_images": len(images),
                    "noise_std": 0.0,
                    "mean_edge_energy": mean_edge_energy(case_psfs),
                    **metrics,
                }
                grid_cases.append(case)
                cases.append(case)
                print(json.dumps({"strict_optics_case": case}, ensure_ascii=False), flush=True)
            grid_summary = {
                "status": "completed",
                "completed_at": now(),
                "simulation_grid": grid,
                "cache": cache_description(cache),
                "cache_path": str(cache_path),
                "runtime_seconds": time.monotonic() - started,
                "cases": grid_cases,
            }
            write_json(grid_summary_path, grid_summary)
            self._set_grid(
                grid,
                "completed",
                runtime_seconds=grid_summary["runtime_seconds"],
            )
            del largest_psfs, optics, cache
            torch.cuda.empty_cache()

        decision = select_optical_settings(cases)
        summary = {
            "status": "completed",
            "completed_at": now(),
            "protocol": {
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
                "simulation_grids": list(SIMULATION_GRIDS),
                "psf_sizes": list(PSF_SIZES),
                "representative_field_grid": REPRESENTATIVE_FIELD_GRID,
                "representative_fields": REPRESENTATIVE_FIELD_GRID**2,
                "sensor_crop": [1000, 1000],
                "validation_images": len(images),
                "noise_std": 0.0,
            },
            "cases": sorted(
                cases,
                key=lambda row: (row["simulation_grid"], row["psf_size"]),
            ),
            "decision": decision,
        }
        write_json(self.output / "summary.json", summary)
        self.state["status"] = "completed"
        self.state["current_grid"] = None
        self.state["completed_at"] = now()
        self.state["decision"] = decision
        self._save_state()
        print(json.dumps({"strict_optics_summary": summary}, ensure_ascii=False), flush=True)
        return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--max-images", type=int, default=10)
    args = parser.parse_args()
    if args.max_images < 1:
        parser.error("--max-images must be positive")
    OpticsConvergenceRunner(args.project_root, args.max_images).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
