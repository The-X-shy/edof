"""Independent DIV2K validation over all configured depths."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .imaging import spatial_convolution, wavelength_choice
from .metrics import LPIPSMetric, batch_psnr, batch_ssim
from .optics import CachedRayWaveOptics


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
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    network.eval()
    psfs = optics.psfs(wavelength_choice(0, averaged=True)).detach()
    metric = lpips_metric
    if use_lpips and metric is None:
        metric = LPIPSMetric(device)

    totals = [dict(psnr=0.0, ssim=0.0, lpips=0.0, samples=0) for _ in depths_mm]
    sample: dict[str, Tensor] = {}
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch_index, clean in enumerate(loader):
        clean = clean.to(device, non_blocking=True)
        for depth_index, depth in enumerate(depths_mm):
            sensor = spatial_convolution(clean, psfs[depth_index])
            if noise_std > 0.0:
                noise = torch.randn(
                    sensor.shape,
                    generator=generator,
                    device=device,
                    dtype=sensor.dtype,
                )
                sensor = sensor + noise * noise_std
            reconstruction = network(sensor).clamp(0.0, 1.0)
            psnr = batch_psnr(reconstruction, clean)
            ssim = batch_ssim(reconstruction, clean)
            lpips_values = metric(reconstruction, clean) if metric is not None else None
            count = clean.shape[0]
            totals[depth_index]["psnr"] += float(psnr.sum())
            totals[depth_index]["ssim"] += float(ssim.sum())
            if lpips_values is not None:
                totals[depth_index]["lpips"] += float(lpips_values.sum())
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
    return {"depth_metrics": depth_metrics, "mean": mean_metrics}, sample
