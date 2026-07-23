"""Selection rules and PSF helpers for strict optical convergence checks."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

import torch
from torch import Tensor

from .config import EDOFConfig, validate_config


def crop_normalized_psfs(psfs: Tensor, target_size: int) -> Tensor:
    """Take a centred odd-sized crop and renormalize every PSF."""

    source_size = int(psfs.shape[-1])
    if psfs.shape[-2] != source_size:
        raise ValueError("PSFs must be square")
    if target_size < 3 or target_size % 2 == 0 or target_size > source_size:
        raise ValueError("target_size must be an odd integer within the source PSF")
    offset = (source_size - target_size) // 2
    cropped = psfs[..., offset : offset + target_size, offset : offset + target_size]
    return cropped / cropped.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)


def mean_edge_energy(psfs: Tensor, border: int = 3) -> float:
    """Return the mean normalized energy in the outer PSF border."""

    size = int(psfs.shape[-1])
    if psfs.shape[-2] != size:
        raise ValueError("PSFs must be square")
    if border < 1 or border * 2 >= size:
        raise ValueError("border must leave a non-empty PSF interior")
    normalized = psfs / psfs.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    interior = normalized[..., border:-border, border:-border].sum(dim=(-2, -1))
    return float((1.0 - interior).mean().detach().cpu())


def select_optical_settings(
    cases: Iterable[dict[str, Any]],
    *,
    convergence_threshold_db: float = 0.1,
    grid_limited_threshold_db: float = 0.3,
    psf_threshold_db: float = 0.1,
) -> dict[str, Any]:
    """Apply the predeclared grid and PSF-size gates to measured raw PSNR."""

    rows = list(cases)
    if not rows:
        raise ValueError("at least one optical convergence case is required")
    lookup = {
        (int(row["simulation_grid"]), int(row["psf_size"])): float(
            row["mean_raw_psnr"]
        )
        for row in rows
    }
    grids = sorted({key[0] for key in lookup})
    psf_sizes = sorted({key[1] for key in lookup})
    if len(grids) < 2 or len(psf_sizes) < 2:
        raise ValueError("convergence selection requires multiple grids and PSF sizes")
    missing = [
        (grid, psf_size)
        for grid in grids
        for psf_size in psf_sizes
        if (grid, psf_size) not in lookup
    ]
    if missing:
        raise ValueError(f"missing convergence cases: {missing}")

    largest_grid = grids[-1]
    reference_psf = psf_sizes[-1]
    high_resolution_psnr = lookup[(largest_grid, reference_psf)]
    grid_deltas = {
        str(grid): high_resolution_psnr - lookup[(grid, reference_psf)]
        for grid in grids
    }
    selected_grid = next(
        grid
        for grid in grids
        if grid_deltas[str(grid)] <= convergence_threshold_db
    )
    grid_gain = high_resolution_psnr - lookup[(grids[0], reference_psf)]

    compact_psf, largest_psf = psf_sizes[0], psf_sizes[-1]
    psf_gain = (
        lookup[(selected_grid, largest_psf)]
        - lookup[(selected_grid, compact_psf)]
    )
    selected_psf = largest_psf if psf_gain > psf_threshold_db else compact_psf

    if grid_gain > grid_limited_threshold_db:
        bottleneck = "simulation_grid"
    elif grid_gain < convergence_threshold_db:
        bottleneck = "proxy_lens_parameters"
    else:
        bottleneck = "mixed"

    return {
        "selected_simulation_grid": selected_grid,
        "selected_psf_size": selected_psf,
        "largest_grid_reference_psnr": high_resolution_psnr,
        "grid_deltas_to_largest_db": grid_deltas,
        "grid_gain_512_to_largest_db": grid_gain,
        "psf_gain_at_selected_grid_db": psf_gain,
        "grid_limited": grid_gain > grid_limited_threshold_db,
        "primary_bottleneck": bottleneck,
        "thresholds_db": {
            "grid_converged": convergence_threshold_db,
            "grid_limited": grid_limited_threshold_db,
            "psf_range": psf_threshold_db,
        },
    }


def build_strict_finetune_config(
    base: EDOFConfig,
    decision: dict[str, Any],
    *,
    cache_file: str,
    fixed_psf_cache_file: str,
    initialize_from: str,
) -> EDOFConfig:
    """Apply a convergence decision without changing the disclosed paper loss."""

    simulation_grid = int(decision["selected_simulation_grid"])
    psf_size = int(decision["selected_psf_size"])
    config = replace(
        base,
        optics=replace(
            base.optics,
            cache_file=cache_file,
            field_grid=3,
            simulation_grid=simulation_grid,
            psf_size=psf_size,
            finetune_field_grid=40,
            finetune_psf_mode="exact",
            finetune_psf_cache_file=fixed_psf_cache_file,
            propagation_batch_size=1,
        ),
        dataset=replace(base.dataset, crop_size=128),
        evaluation=replace(
            base.evaluation,
            crop_size=1000,
            max_images=100,
            field_grid=40,
            local_field_patches=False,
        ),
        training=replace(
            base.training,
            joint_epochs=0,
            finetune_epochs=50,
            warmup_epochs=0,
            pixel_loss_weight=0.3,
            perceptual_weight=0.0,
            pixel_loss_type="rmse",
            cross_depth_loss_weight=1.0,
            local_field_patches=True,
            resume=None,
            initialize_from=initialize_from,
        ),
    )
    validate_config(config)
    return config
