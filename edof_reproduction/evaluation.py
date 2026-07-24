"""Independent DIV2K validation over all configured depths."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .imaging import interpolate_psf_grid, spatial_convolution, wavelength_choice
from .metrics import LPIPSMetric, batch_psnr, batch_ssim
from .optics import CachedRayWaveOptics


def _regional_psnr(prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
    height, width = target.shape[-2:]
    y = (torch.arange(height, device=target.device, dtype=target.dtype) + 0.5) / height
    x = (torch.arange(width, device=target.device, dtype=target.dtype) + 0.5) / width
    radius = torch.maximum(
        (y[:, None] - 0.5).abs() * 2.0,
        (x[None, :] - 0.5).abs() * 2.0,
    )
    masks = {
        "center": radius < (1.0 / 3.0),
        "middle": (radius >= (1.0 / 3.0)) & (radius < (2.0 / 3.0)),
        "edge": radius >= (2.0 / 3.0),
    }
    squared_error = (prediction - target).square()
    result = {}
    for name, mask in masks.items():
        weights = mask.to(dtype=target.dtype)[None, None]
        denominator = weights.sum() * target.shape[1]
        mse = (squared_error * weights).sum(dim=(1, 2, 3)) / denominator
        result[name] = -10.0 * torch.log10(mse.clamp_min(1e-12))
    return result


def _boundary_discontinuity(image: Tensor, field_side: int) -> Tensor:
    if field_side <= 1:
        return image.new_zeros(image.shape[0])
    height, width = image.shape[-2:]
    horizontal_positions = sorted(
        {(index * width) // field_side for index in range(1, field_side)}
    )
    vertical_positions = sorted(
        {(index * height) // field_side for index in range(1, field_side)}
    )
    vertical = torch.stack(
        [
            (image[..., position] - image[..., position - 1])
            .abs()
            .mean(dim=(1, 2))
            for position in horizontal_positions
            if 0 < position < width
        ]
    ).mean(dim=0)
    horizontal = torch.stack(
        [
            (image[..., position, :] - image[..., position - 1, :])
            .abs()
            .mean(dim=(1, 2))
            for position in vertical_positions
            if 0 < position < height
        ]
    ).mean(dim=0)
    return (vertical + horizontal) * 0.5


@torch.no_grad()
def evaluate_reconstruction(
    optics: CachedRayWaveOptics,
    network: nn.Module,
    loader,
    *,
    device: torch.device,
    depths_mm: tuple[float, ...],
    noise_std: float,
    use_lpips: bool,
    seed: int,
    lpips_metric: LPIPSMetric | None = None,
    field_grid: int | None = None,
    local_field_patches: bool = False,
    fixed_psfs: Tensor | None = None,
    field_refine_factor: int = 1,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    network.eval()
    if fixed_psfs is None:
        psfs = optics.psfs(wavelength_choice(0, averaged=True)).detach()
        psfs = interpolate_psf_grid(psfs, field_grid).detach()
    else:
        psfs = fixed_psfs.detach()
        expected_prefix = (len(depths_mm), 3)
        if psfs.ndim != 5 or (psfs.shape[0], psfs.shape[2]) != expected_prefix:
            raise ValueError("fixed PSFs must have shape [depth, field, RGB, kernel, kernel]")
    metric = lpips_metric
    if use_lpips and metric is None:
        metric = LPIPSMetric(device)

    totals = [
        dict(
            psnr=0.0,
            ssim=0.0,
            lpips=0.0,
            raw_psnr=0.0,
            raw_ssim=0.0,
            raw_lpips=0.0,
            center_psnr=0.0,
            middle_psnr=0.0,
            edge_psnr=0.0,
            raw_center_psnr=0.0,
            raw_middle_psnr=0.0,
            raw_edge_psnr=0.0,
            boundary_discontinuity=0.0,
            raw_boundary_discontinuity=0.0,
            samples=0,
        )
        for _ in depths_mm
    ]
    sample: dict[str, Tensor] = {}
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch_index, clean in enumerate(loader):
        clean = clean.to(device, non_blocking=True)
        for depth_index, depth in enumerate(depths_mm):
            depth_psfs = psfs[depth_index]
            if local_field_patches:
                field_index = batch_index % depth_psfs.shape[0]
                depth_psfs = depth_psfs[field_index : field_index + 1]
            depth_psfs = depth_psfs.to(device, non_blocking=True)
            sensor = spatial_convolution(
                clean,
                depth_psfs,
                field_refine_factor=field_refine_factor,
            )
            if noise_std > 0.0:
                noise = torch.randn(
                    sensor.shape,
                    generator=generator,
                    device=device,
                    dtype=sensor.dtype,
                )
                sensor = sensor + noise * noise_std
            reconstruction = network(sensor).clamp(0.0, 1.0)
            sensor_clamped = sensor.clamp(0.0, 1.0)
            psnr = batch_psnr(reconstruction, clean)
            ssim = batch_ssim(reconstruction, clean)
            lpips_values = metric(reconstruction, clean) if metric is not None else None
            raw_psnr = batch_psnr(sensor_clamped, clean)
            raw_ssim = batch_ssim(sensor_clamped, clean)
            raw_lpips = metric(sensor_clamped, clean) if metric is not None else None
            regional = _regional_psnr(reconstruction, clean)
            raw_regional = _regional_psnr(sensor_clamped, clean)
            field_side = round(depth_psfs.shape[0] ** 0.5)
            boundary = _boundary_discontinuity(reconstruction, field_side)
            raw_boundary = _boundary_discontinuity(sensor_clamped, field_side)
            count = clean.shape[0]
            totals[depth_index]["psnr"] += float(psnr.sum())
            totals[depth_index]["ssim"] += float(ssim.sum())
            if lpips_values is not None:
                totals[depth_index]["lpips"] += float(lpips_values.sum())
                totals[depth_index]["raw_lpips"] += float(raw_lpips.sum())
            totals[depth_index]["raw_psnr"] += float(raw_psnr.sum())
            totals[depth_index]["raw_ssim"] += float(raw_ssim.sum())
            for region in ("center", "middle", "edge"):
                totals[depth_index][f"{region}_psnr"] += float(
                    regional[region].sum()
                )
                totals[depth_index][f"raw_{region}_psnr"] += float(
                    raw_regional[region].sum()
                )
            totals[depth_index]["boundary_discontinuity"] += float(boundary.sum())
            totals[depth_index]["raw_boundary_discontinuity"] += float(
                raw_boundary.sum()
            )
            totals[depth_index]["samples"] += count
            if batch_index == 0 and depth_index == 1:
                sample = {
                    "clean": clean[:1].detach(),
                    "sensor": sensor[:1].detach(),
                    "reconstruction": reconstruction[:1].detach(),
                    "psfs": psfs.detach(),
                }

    depth_metrics = []
    for depth, total in zip(depths_mm, totals):
        count = total["samples"]
        if count == 0:
            raise RuntimeError("validation loader produced no samples")
        lpips_value = total["lpips"] / count if metric is not None else None
        depth_metrics.append(
            {
                "depth_mm": depth,
                "samples": count,
                "psnr": total["psnr"] / count,
                "ssim": total["ssim"] / count,
                "lpips": lpips_value,
                "one_minus_lpips": 1.0 - lpips_value if lpips_value is not None else None,
                "raw": {
                    "psnr": total["raw_psnr"] / count,
                    "ssim": total["raw_ssim"] / count,
                    "lpips": total["raw_lpips"] / count if metric is not None else None,
                    "one_minus_lpips": (
                        1.0 - total["raw_lpips"] / count if metric is not None else None
                    ),
                },
                "spatial": {
                    "center_psnr": total["center_psnr"] / count,
                    "middle_psnr": total["middle_psnr"] / count,
                    "edge_psnr": total["edge_psnr"] / count,
                    "boundary_discontinuity": (
                        total["boundary_discontinuity"] / count
                    ),
                    "raw_center_psnr": total["raw_center_psnr"] / count,
                    "raw_middle_psnr": total["raw_middle_psnr"] / count,
                    "raw_edge_psnr": total["raw_edge_psnr"] / count,
                    "raw_boundary_discontinuity": (
                        total["raw_boundary_discontinuity"] / count
                    ),
                },
            }
        )
    mean_metrics = {
        key: sum(item[key] for item in depth_metrics) / len(depth_metrics)
        for key in ("psnr", "ssim")
    }
    lpips_items = [item["lpips"] for item in depth_metrics if item["lpips"] is not None]
    mean_metrics["lpips"] = sum(lpips_items) / len(lpips_items) if lpips_items else None
    mean_metrics["one_minus_lpips"] = (
        1.0 - mean_metrics["lpips"] if mean_metrics["lpips"] is not None else None
    )
    mean_metrics["raw"] = {
        key: sum(item["raw"][key] for item in depth_metrics) / len(depth_metrics)
        for key in ("psnr", "ssim")
    }
    raw_lpips_items = [item["raw"]["lpips"] for item in depth_metrics if item["raw"]["lpips"] is not None]
    mean_metrics["raw"]["lpips"] = (
        sum(raw_lpips_items) / len(raw_lpips_items) if raw_lpips_items else None
    )
    mean_metrics["raw"]["one_minus_lpips"] = (
        1.0 - mean_metrics["raw"]["lpips"]
        if mean_metrics["raw"]["lpips"] is not None
        else None
    )
    mean_metrics["spatial"] = {
        key: sum(item["spatial"][key] for item in depth_metrics)
        / len(depth_metrics)
        for key in (
            "center_psnr",
            "middle_psnr",
            "edge_psnr",
            "boundary_discontinuity",
            "raw_center_psnr",
            "raw_middle_psnr",
            "raw_edge_psnr",
            "raw_boundary_discontinuity",
        )
    }
    return {"depth_metrics": depth_metrics, "mean": mean_metrics}, sample
