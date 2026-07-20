"""End-to-end DOE/NAFNet training with checkpoints, traces, and artifacts."""

from __future__ import annotations

import json
import math
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from .provenance import (
    FileArtifactStore,
    MetaTrace,
    MetaTraceWriter,
    SQLiteStore,
    compute_file_sha256,
    make_deterministic_id,
)

from .config import EDOFConfig
from .dataset import build_loader, build_validation_loader
from .evaluation import evaluate_reconstruction
from .imaging import spatial_convolution as _spatial_convolution
from .imaging import wavelength_choice as _wavelength_choice
from .metrics import LPIPSMetric, VGG16PerceptualLoss, batch_psnr, batch_ssim
from .nafnet import NAFNet
from .optics import CachedRayWaveOptics, cache_description, load_or_build_cache


PAPER_SOURCES = {
    "paper": "https://arxiv.org/abs/2406.00834",
    "project": "https://vccimaging.org/Publications/Yang2024HybridLens/",
    "supplement": "https://vccimaging.org/Publications/Yang2024HybridLens/Yang2024HybridLens_supp.pdf",
    "deeplens": "https://github.com/singer-yang/DeepLens/tree/7df9613ca06be4093d094ad3095bd8712641a77d",
    "historical_poly1d": "https://github.com/singer-yang/DeepLens/blob/e354456/deeplens/optics/surfaces_diffractive.py",
    "nafnet_example": "https://github.com/singer-yang/End2endImaging/tree/0d4661eba50c97359f8e72d71913517b3a005bd4",
}

CLAIM_BOUNDARY = (
    "The publication does not release the exact Optolife prescription, trained DOE coefficients, "
    "DOE-to-sensor spacing, sensor response, DIV2K crop schedule, noise model, or exact NAFNet config. "
    "This run reproduces the disclosed training protocol with the public DeepLens A489 refractive "
    "proxy and the historical public Poly1D DOE; it is not a numerical identity claim."
)


def _device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def _run_directory(config: EDOFConfig, output_override: str | Path | None) -> Path:
    if output_override is not None:
        output = Path(output_override)
    elif config.training.resume:
        output = Path(config.training.resume).resolve().parent.parent
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path(config.output.root) / config.output.run_name / stamp
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    return output


def _pairwise_rmse(values: list[Tensor]) -> Tensor:
    losses = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            losses.append(torch.sqrt(F.mse_loss(values[left], values[right]) + 1e-12))
    return torch.stack(losses).mean()


def _loss_from_psfs(
    clean: Tensor,
    psfs: Tensor,
    network: NAFNet,
    *,
    pixel_loss_weight: float,
    perceptual_weight: float,
    perceptual_loss: VGG16PerceptualLoss | None,
    noise_std: float,
) -> tuple[Tensor, dict[str, float], list[Tensor]]:
    reconstructions: list[Tensor] = []
    for depth_index in range(psfs.shape[0]):
        sensor = _spatial_convolution(clean, psfs[depth_index])
        if noise_std > 0:
            sensor = sensor + torch.randn_like(sensor) * noise_std
        reconstructions.append(network(sensor))
    similarity = _pairwise_rmse(reconstructions)
    pixel_mse = torch.stack([F.mse_loss(item, clean) for item in reconstructions]).mean()
    truth = torch.sqrt(pixel_mse + 1e-12)
    if perceptual_loss is not None and perceptual_weight > 0.0:
        perceptual = torch.stack(
            [perceptual_loss(item.clamp(0.0, 1.0), clean) for item in reconstructions]
        ).mean()
    else:
        perceptual = pixel_mse.new_zeros(())
    loss = pixel_loss_weight * pixel_mse + perceptual_weight * perceptual
    metrics = {
        "loss": float(loss.detach()),
        "pixel_mse": float(pixel_mse.detach()),
        "perceptual_loss": float(perceptual.detach()),
        "cross_depth_rmse": float(similarity.detach()),
        "truth_rmse": float(truth.detach()),
    }
    return loss, metrics, reconstructions


def _loss_for_batch(
    clean: Tensor,
    optics: CachedRayWaveOptics,
    network: NAFNet,
    *,
    step: int,
    pixel_loss_weight: float,
    perceptual_weight: float,
    perceptual_loss: VGG16PerceptualLoss | None,
    averaged_wavelengths: bool,
    noise_std: float,
) -> tuple[Tensor, dict[str, float], list[Tensor], Tensor]:
    psfs = optics.psfs(_wavelength_choice(step, averaged=averaged_wavelengths))
    loss, metrics, reconstructions = _loss_from_psfs(
        clean,
        psfs,
        network,
        pixel_loss_weight=pixel_loss_weight,
        perceptual_weight=perceptual_weight,
        perceptual_loss=perceptual_loss,
        noise_std=noise_std,
    )
    return loss, metrics, reconstructions, psfs


def _save_tensor_image(tensor: Tensor, path: Path) -> None:
    data = tensor.detach().float().cpu()
    if data.ndim == 2:
        minimum, maximum = data.min(), data.max()
        data = (data - minimum) / (maximum - minimum).clamp_min(1e-12)
        array = (data.numpy() * 255).round().astype(np.uint8)
        Image.fromarray(array, mode="L").save(path)
    else:
        if data.ndim == 4:
            data = data[0]
        array = (data.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(array, mode="RGB").save(path)


def _save_psf_grid(psfs: Tensor, path: Path) -> None:
    # Center field, RGB-averaged, three depths placed side by side.
    panels = []
    for depth_index in range(psfs.shape[0]):
        panel = psfs[depth_index, psfs.shape[1] // 2].mean(dim=0)
        panel = torch.log1p(panel / panel.max().clamp_min(1e-12) * 100.0)
        panel = panel / panel.max().clamp_min(1e-12)
        panels.append(panel)
    _save_tensor_image(torch.cat(panels, dim=1), path)


def _checkpoint_payload(
    epoch: int,
    global_step: int,
    stage: str,
    optics: CachedRayWaveOptics,
    network: NAFNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: EDOFConfig,
    training_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "format": "edof_reproduction",
        "version": 2,
        "epoch": epoch,
        "global_step": global_step,
        "stage": stage,
        "doe": optics.doe.export_state(),
        "network": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config.as_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    payload.update(training_state or {})
    return payload


def _write_epoch_trace(
    writer: MetaTraceWriter,
    *,
    config: EDOFConfig,
    run_id: str,
    epoch: int,
    stage: str,
    metrics: dict[str, Any],
    outputs: list[Path],
    cache_path: Path,
) -> MetaTrace:
    trace = MetaTrace(
        trace_id=make_deterministic_id("trace", run_id, epoch, stage),
        workspace_id=config.output.workspace_id,
        run_id=run_id,
        branch_id=None,
        step_id=f"epoch_{epoch:03d}",
        actor="SimulationExperimentalist",
        phase="Execute",
        task=f"Train paper EDoF reproduction epoch {epoch} ({stage})",
        skill_id="deeplens_edof_reproduction",
        skill_version="1.0.0",
        tool="python -m edof_reproduction",
        input_refs=[str(config.source_config), str(cache_path)],
        output_refs=[str(path) for path in outputs],
        findings=[f"{key}={value}" for key, value in metrics.items()],
        limitations=[CLAIM_BOUNDARY],
        next_action="continue_training",
        status="succeeded",
        timestamp_start=None,
        timestamp_end=datetime.now(timezone.utc),
        parents=[],
        content_hash=None,
        metadata={"epoch": epoch, "stage": stage, **metrics},
    )
    return writer.write_trace(trace)


def _pretrain_psf(optics: CachedRayWaveOptics, steps: int, learning_rate: float) -> list[float]:
    if steps <= 0:
        return []
    optimizer = torch.optim.Adam(optics.doe.parameters(), lr=learning_rate)
    losses = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        psfs = optics.psfs(_wavelength_choice(step, averaged=False))
        center = psfs[:, psfs.shape[1] // 2]
        loss = _pairwise_rmse([center[index] for index in range(center.shape[0])])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def run_training(
    config: EDOFConfig,
    *,
    output_override: str | Path | None = None,
    force_cache: bool = False,
) -> dict[str, Any]:
    _seed_everything(config.training.seed)
    output = _run_directory(config, output_override)
    device = _device(config.training.device)
    resolved_config_path = output / "resolved_config.json"
    resolved_config_path.write_text(json.dumps(config.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    sources_path = output / "sources_and_boundary.json"
    sources_path.write_text(
        json.dumps({"sources": PAPER_SOURCES, "claim_boundary": CLAIM_BOUNDARY}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cache, cache_path = load_or_build_cache(config.optics, output, force=force_cache)
    optics = CachedRayWaveOptics(config.optics, cache, device).to(device)
    network = NAFNet(
        in_chan=3,
        out_chan=3,
        width=config.network.width,
        middle_blk_num=config.network.middle_blk_num,
        enc_blk_nums=config.network.enc_blk_nums,
        dec_blk_nums=config.network.dec_blk_nums,
    ).to(device)
    loader = build_loader(config.dataset, config.training.seed)
    validation_loader = (
        build_validation_loader(config.dataset, config.evaluation, config.training.seed)
        if config.evaluation.enabled
        else None
    )
    perceptual_loss = (
        VGG16PerceptualLoss(device)
        if config.training.perceptual_weight > 0.0
        else None
    )
    lpips_metric = (
        LPIPSMetric(device)
        if validation_loader is not None and config.evaluation.use_lpips
        else None
    )
    optimizer = torch.optim.Adam(
        [
            {"params": optics.doe.parameters(), "lr": config.training.doe_lr},
            {"params": network.parameters(), "lr": config.training.network_lr},
        ]
    )
    total_epochs = config.training.joint_epochs + config.training.finetune_epochs

    def lr_multiplier(epoch: int) -> float:
        if epoch < config.training.warmup_epochs:
            return max((epoch + 1) / max(config.training.warmup_epochs, 1), 1e-3)
        progress = (epoch - config.training.warmup_epochs) / max(
            total_epochs - config.training.warmup_epochs - 1, 1
        )
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    start_epoch, global_step = 0, 0
    best_validation_psnr = float("-inf")
    best_epoch: int | None = None
    evaluations_without_improvement = 0
    resumed_stage = None
    if config.training.resume:
        checkpoint = torch.load(config.training.resume, map_location=device, weights_only=False)
        optics.doe.load_exported_state(checkpoint["doe"])
        network.load_state_dict(checkpoint["network"])
        resumed_stage = checkpoint["stage"]
        if resumed_stage == "finetune":
            for parameter in optics.doe.parameters():
                parameter.requires_grad_(False)
            optimizer = torch.optim.Adam(network.parameters(), lr=config.training.finetune_lr)
            remaining = max(total_epochs - checkpoint["epoch"] - 1, 1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, global_step = checkpoint["epoch"] + 1, checkpoint["global_step"]
        best_validation_psnr = checkpoint.get("best_validation_psnr", float("-inf"))
        best_epoch = checkpoint.get("best_epoch")
        evaluations_without_improvement = checkpoint.get("evaluations_without_improvement", 0)
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    pretrain_losses = _pretrain_psf(
        optics, config.training.psf_pretrain_steps if start_epoch == 0 else 0, config.training.doe_lr
    )
    trace_store = SQLiteStore(output / "trace.sqlite")
    trace_writer = MetaTraceWriter(trace_store)
    run_id = make_deterministic_id("run", str(output.resolve()), config.training.seed)
    log_path = output / "training_log.jsonl"
    validation_log_path = output / "validation_log.jsonl"
    last_clean: Tensor | None = None
    last_reconstructions: list[Tensor] = []
    last_psfs: Tensor | None = None
    epoch_history: list[dict[str, Any]] = []
    fixed_finetune_psfs: Tensor | None = None
    stopped_early = False
    epochs_completed = start_epoch

    for epoch in range(start_epoch, total_epochs):
        stage = "joint" if epoch < config.training.joint_epochs else "finetune"
        if stage == "finetune" and fixed_finetune_psfs is None:
            if any(parameter.requires_grad for parameter in optics.doe.parameters()):
                for parameter in optics.doe.parameters():
                    parameter.requires_grad_(False)
                optimizer = torch.optim.Adam(network.parameters(), lr=config.training.finetune_lr)
                remaining = max(total_epochs - epoch, 1)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)
            fixed_finetune_psfs = optics.psfs(_wavelength_choice(global_step, averaged=True)).detach()
            evaluations_without_improvement = 0
        if hasattr(loader.dataset, "set_epoch"):
            loader.dataset.set_epoch(epoch)
        network.train()
        batch_metrics: list[dict[str, float]] = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index, clean in enumerate(loader):
            if config.training.max_batches_per_epoch is not None and batch_index >= config.training.max_batches_per_epoch:
                break
            clean = clean.to(device)
            if fixed_finetune_psfs is None:
                loss, metrics, reconstructions, psfs = _loss_for_batch(
                    clean,
                    optics,
                    network,
                    step=global_step,
                    pixel_loss_weight=config.training.pixel_loss_weight,
                    perceptual_weight=config.training.perceptual_weight,
                    perceptual_loss=perceptual_loss,
                    averaged_wavelengths=False,
                    noise_std=0.0,
                )
            else:
                loss, metrics, reconstructions = _loss_from_psfs(
                    clean,
                    fixed_finetune_psfs,
                    network,
                    pixel_loss_weight=config.training.pixel_loss_weight,
                    perceptual_weight=config.training.perceptual_weight,
                    perceptual_loss=perceptual_loss,
                    noise_std=config.training.sensor_noise_std,
                )
                psfs = fixed_finetune_psfs
            (loss / config.training.accumulation_steps).backward()
            if (batch_index + 1) % config.training.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(network.parameters(), config.training.gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            batch_metrics.append(metrics)
            global_step += 1
            last_clean, last_reconstructions, last_psfs = clean.detach(), [item.detach() for item in reconstructions], psfs.detach()
        if batch_metrics and len(batch_metrics) % config.training.accumulation_steps:
            torch.nn.utils.clip_grad_norm_(network.parameters(), config.training.gradient_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        averaged = {
            key: sum(item[key] for item in batch_metrics) / len(batch_metrics) for key in batch_metrics[0]
        }
        averaged.update(
            {
                "epoch": epoch + 1,
                "stage": stage,
                "lr": optimizer.param_groups[-1]["lr"],
                "doe_coefficients": [float(value) for value in optics.doe.coefficients.detach().cpu()],
            }
        )
        epoch_history.append(averaged)
        epochs_completed = epoch + 1
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(averaged, ensure_ascii=False) + "\n")
        print(json.dumps(averaged, ensure_ascii=False), flush=True)

        validation_result = None
        should_validate = validation_loader is not None and (
            (epoch + 1) % config.evaluation.every_n_epochs == 0 or epoch + 1 == total_epochs
        )
        if should_validate:
            validation_result, _ = evaluate_reconstruction(
                optics,
                network,
                validation_loader,
                device=device,
                depths_mm=config.optics.depths_mm,
                noise_std=config.evaluation.noise_std,
                use_lpips=config.evaluation.use_lpips,
                seed=config.training.seed + epoch + 1,
                lpips_metric=lpips_metric,
            )
            validation_row = {"epoch": epoch + 1, "stage": stage, **validation_result}
            with validation_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(validation_row, ensure_ascii=False) + "\n")
            print(json.dumps({"validation": validation_row}, ensure_ascii=False), flush=True)

            validation_psnr = validation_result["mean"]["psnr"]
            if validation_psnr > best_validation_psnr + config.evaluation.early_stopping_min_delta:
                best_validation_psnr = validation_psnr
                best_epoch = epoch + 1
                evaluations_without_improvement = 0
            elif stage == "finetune":
                evaluations_without_improvement += 1

        training_state = {
            "best_validation_psnr": best_validation_psnr,
            "best_epoch": best_epoch,
            "evaluations_without_improvement": evaluations_without_improvement,
        }
        checkpoint_path = output / "checkpoints" / f"epoch_{epoch + 1:03d}.pt"
        if (epoch + 1) % config.training.checkpoint_every == 0 or epoch + 1 == total_epochs:
            torch.save(
                _checkpoint_payload(
                    epoch, global_step, stage, optics, network, optimizer, scheduler, config, training_state
                ),
                checkpoint_path,
            )
            latest_path = output / "checkpoints" / "latest.pt"
            torch.save(
                _checkpoint_payload(
                    epoch, global_step, stage, optics, network, optimizer, scheduler, config, training_state
                ),
                latest_path,
            )
            outputs = [log_path, checkpoint_path, latest_path]
        else:
            outputs = [log_path]
        if validation_result is not None and best_epoch == epoch + 1:
            best_path = output / "checkpoints" / "best.pt"
            torch.save(
                _checkpoint_payload(
                    epoch, global_step, stage, optics, network, optimizer, scheduler, config, training_state
                ),
                best_path,
            )
            outputs.extend([validation_log_path, best_path])
        _write_epoch_trace(
            trace_writer,
            config=config,
            run_id=run_id,
            epoch=epoch + 1,
            stage=stage,
            metrics=averaged,
            outputs=outputs,
            cache_path=cache_path,
        )

        if (
            stage == "finetune"
            and validation_loader is not None
            and evaluations_without_improvement >= config.evaluation.early_stopping_patience
        ):
            stopped_early = True
            print(
                json.dumps(
                    {
                        "early_stopping": True,
                        "epoch": epoch + 1,
                        "best_epoch": best_epoch,
                        "best_validation_psnr": best_validation_psnr,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            break

    if last_clean is None or last_psfs is None:
        raise RuntimeError("training produced no batches")
    best_path = output / "checkpoints" / "best.pt"
    selected_checkpoint = output / "checkpoints" / "latest.pt"
    if validation_loader is not None and best_path.exists():
        best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        optics.doe.load_exported_state(best_checkpoint["doe"])
        network.load_state_dict(best_checkpoint["network"])
        selected_checkpoint = best_path

    if validation_loader is not None:
        final_evaluation, final_sample = evaluate_reconstruction(
            optics,
            network,
            validation_loader,
            device=device,
            depths_mm=config.optics.depths_mm,
            noise_std=config.evaluation.noise_std,
            use_lpips=config.evaluation.use_lpips,
            seed=config.training.seed,
            lpips_metric=lpips_metric,
        )
        depth_metrics = final_evaluation["depth_metrics"]
        final_psfs = final_sample["psfs"]
        final_clean = final_sample["clean"]
        final_sensor = final_sample["sensor"]
        final_reconstruction = final_sample["reconstruction"]
    else:
        depth_metrics = []
        for depth, reconstruction in zip(config.optics.depths_mm, last_reconstructions):
            depth_metrics.append(
                {
                    "depth_mm": depth,
                    "samples": last_clean.shape[0],
                    "psnr": float(batch_psnr(reconstruction, last_clean).mean()),
                    "ssim": float(batch_ssim(reconstruction, last_clean).mean()),
                    "lpips": None,
                    "one_minus_lpips": None,
                }
            )
        final_evaluation = {
            "depth_metrics": depth_metrics,
            "mean": {
                "psnr": sum(item["psnr"] for item in depth_metrics) / len(depth_metrics),
                "ssim": sum(item["ssim"] for item in depth_metrics) / len(depth_metrics),
                "lpips": None,
                "one_minus_lpips": None,
            },
        }
        final_psfs = last_psfs
        final_clean = last_clean
        final_sensor = _spatial_convolution(last_clean, last_psfs[1])
        final_reconstruction = last_reconstructions[1]
    phase_path = output / "doe_phase.png"
    phase = optics.doe.quantize_phase(optics.doe.wrap_phase(optics.doe.raw_phase(optics.grid_x, optics.grid_y)), straight_through=False)
    _save_tensor_image(phase, phase_path)
    psf_path = output / "psfs.png"
    _save_psf_grid(final_psfs, psf_path)
    clean_path = output / "validation_clean.png"
    _save_tensor_image(final_clean, clean_path)
    sensor_path = output / "validation_sensor.png"
    _save_tensor_image(final_sensor, sensor_path)
    reconstruction_path = output / "reconstruction.png"
    _save_tensor_image(final_reconstruction, reconstruction_path)
    comparison_path = output / "validation_comparison.png"
    _save_tensor_image(
        torch.cat((final_clean, final_sensor.clamp(0.0, 1.0), final_reconstruction), dim=-1),
        comparison_path,
    )
    summary = {
        "status": "completed",
        "run_id": run_id,
        "output": str(output),
        "device": str(device),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cache": cache_description(cache),
        "epochs_planned": total_epochs,
        "epochs_completed": epochs_completed,
        "joint_epochs": config.training.joint_epochs,
        "finetune_epochs": config.training.finetune_epochs,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_validation_psnr": best_validation_psnr if best_epoch is not None else None,
        "selected_checkpoint": str(selected_checkpoint),
        "pretrain_steps": len(pretrain_losses),
        "pretrain_first_loss": pretrain_losses[0] if pretrain_losses else None,
        "pretrain_last_loss": pretrain_losses[-1] if pretrain_losses else None,
        "final_depth_metrics": depth_metrics,
        "validation_mean": final_evaluation["mean"],
        "doe_coefficients": [float(value) for value in optics.doe.coefficients.detach().cpu()],
        "sources": PAPER_SOURCES,
        "claim_boundary": CLAIM_BOUNDARY,
        "paper_table4_targets": [
            {"depth_mm": -200.0, "psnr": 27.5, "ssim": 0.821, "one_minus_lpips": 0.782},
            {"depth_mm": -300.0, "psnr": 28.9, "ssim": 0.869, "one_minus_lpips": 0.842},
            {"depth_mm": -10000.0, "psnr": 27.4, "ssim": 0.818, "one_minus_lpips": 0.787},
        ],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    final_trace = MetaTrace(
        trace_id=make_deterministic_id("trace", run_id, "final"),
        workspace_id=config.output.workspace_id,
        run_id=run_id,
        branch_id=None,
        step_id="final",
        actor="CriticalReviewer",
        phase="Review",
        task="Review and register the EDoF reproduction run",
        skill_id="deeplens_edof_reproduction",
        skill_version="1.0.0",
        tool="run_training",
        input_refs=[str(resolved_config_path), str(cache_path), str(log_path)],
        output_refs=[
            str(summary_path),
            str(phase_path),
            str(psf_path),
            str(comparison_path),
            str(reconstruction_path),
        ],
        findings=[f"epochs_completed={epochs_completed}", f"metrics={depth_metrics}"],
        limitations=[CLAIM_BOUNDARY],
        next_action="compare_against_paper_targets",
        status="succeeded",
        timestamp_start=None,
        timestamp_end=datetime.now(timezone.utc),
        parents=[
            make_deterministic_id(
                "trace",
                run_id,
                epochs_completed,
                "finetune" if epochs_completed > config.training.joint_epochs else "joint",
            )
        ],
        content_hash=None,
        metadata=summary,
    )
    written_trace = trace_writer.write_trace(final_trace)
    trace_path = output / "meta_trace.json"
    trace_path.write_text(json.dumps(written_trace.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    artifact_store = FileArtifactStore(output / "registered_artifacts", trace_store)
    registered = []
    artifacts = [
        (summary_path, "summary", {"epochs_completed": epochs_completed}),
        (output / "checkpoints" / "latest.pt", "checkpoint", {}),
        (log_path, "training_log", {}),
        (phase_path, "doe_phase", {}),
        (psf_path, "psf_visualization", {}),
        (clean_path, "validation_clean", {}),
        (sensor_path, "validation_sensor", {}),
        (reconstruction_path, "reconstruction", {}),
        (comparison_path, "validation_comparison", {}),
        (resolved_config_path, "resolved_config", {}),
        (sources_path, "evidence", {}),
        (trace_path, "meta_trace", {}),
    ]
    if validation_log_path.exists():
        artifacts.append((validation_log_path, "validation_log", {}))
    if best_path.exists():
        artifacts.append((best_path, "best_checkpoint", {"best_epoch": best_epoch}))
    for path, artifact_type, metrics in artifacts:
        registered.append(
            artifact_store.register_file(
                path,
                workspace_id=config.output.workspace_id,
                run_id=run_id,
                trace_id=written_trace.trace_id,
                producer="DeepLensEDOFReproduction",
                metadata={"artifact_type": artifact_type, "validation_completed": True},
                metrics=metrics,
            )
        )
    manifest = {
        "run_id": run_id,
        "trace_id": written_trace.trace_id,
        "files": {
            path.name: {"path": str(path), "sha256": compute_file_sha256(path)}
            for path in (
                summary_path,
                log_path,
                phase_path,
                psf_path,
                clean_path,
                sensor_path,
                reconstruction_path,
                comparison_path,
                trace_path,
            )
        },
        "registered_artifacts": [reference.model_dump(mode="json") for reference in registered],
    }
    manifest_path = output / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_ref = artifact_store.register_file(
        manifest_path,
        workspace_id=config.output.workspace_id,
        run_id=run_id,
        trace_id=written_trace.trace_id,
        producer="DeepLensEDOFReproduction",
        metadata={"artifact_type": "manifest", "validation_completed": True},
        metrics={},
    )
    summary["manifest"] = str(manifest_path)
    summary["artifact_ids"] = [item.artifact_id for item in registered] + [manifest_ref.artifact_id]
    return summary


def run_checkpoint_evaluation(
    config: EDOFConfig,
    checkpoint_path: str | Path,
    *,
    output_override: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint on the independent validation set without training."""

    if not config.evaluation.enabled:
        raise ValueError("evaluation.enabled must be true for checkpoint evaluation")
    _seed_everything(config.training.seed)
    checkpoint_path = Path(checkpoint_path)
    training_output = checkpoint_path.resolve().parent.parent
    output = Path(output_override) if output_override else training_output / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    device = _device(config.training.device)
    cache, cache_path = load_or_build_cache(config.optics, training_output)
    optics = CachedRayWaveOptics(config.optics, cache, device).to(device)
    network = NAFNet(
        in_chan=3,
        out_chan=3,
        width=config.network.width,
        middle_blk_num=config.network.middle_blk_num,
        enc_blk_nums=config.network.enc_blk_nums,
        dec_blk_nums=config.network.dec_blk_nums,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    optics.doe.load_exported_state(checkpoint["doe"])
    network.load_state_dict(checkpoint["network"])
    loader = build_validation_loader(config.dataset, config.evaluation, config.training.seed)
    lpips_metric = LPIPSMetric(device) if config.evaluation.use_lpips else None
    metrics, sample = evaluate_reconstruction(
        optics,
        network,
        loader,
        device=device,
        depths_mm=config.optics.depths_mm,
        noise_std=config.evaluation.noise_std,
        use_lpips=config.evaluation.use_lpips,
        seed=config.training.seed,
        lpips_metric=lpips_metric,
    )

    clean_path = output / "validation_clean.png"
    sensor_path = output / "validation_sensor.png"
    reconstruction_path = output / "validation_reconstruction.png"
    comparison_path = output / "validation_comparison.png"
    psf_path = output / "psfs.png"
    phase_path = output / "doe_phase.png"
    _save_tensor_image(sample["clean"], clean_path)
    _save_tensor_image(sample["sensor"], sensor_path)
    _save_tensor_image(sample["reconstruction"], reconstruction_path)
    _save_tensor_image(
        torch.cat(
            (sample["clean"], sample["sensor"].clamp(0.0, 1.0), sample["reconstruction"]),
            dim=-1,
        ),
        comparison_path,
    )
    _save_psf_grid(sample["psfs"], psf_path)
    phase = optics.doe.quantize_phase(
        optics.doe.wrap_phase(optics.doe.raw_phase(optics.grid_x, optics.grid_y)),
        straight_through=False,
    )
    _save_tensor_image(phase, phase_path)

    summary = {
        "status": "completed",
        "mode": "checkpoint_evaluation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]) + 1,
        "output": str(output),
        "device": str(device),
        "cache": cache_description(cache),
        "validation": metrics,
        "sources": PAPER_SOURCES,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    summary_path = output / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_memory_smoke(
    config: EDOFConfig,
    *,
    output_override: str | Path | None = None,
    force_cache: bool = False,
) -> dict[str, Any]:
    """Build the configured cache and run one real forward/backward batch."""

    _seed_everything(config.training.seed)
    output = _run_directory(config, output_override)
    device = _device(config.training.device)
    cache, cache_path = load_or_build_cache(config.optics, output, force=force_cache)
    optics = CachedRayWaveOptics(config.optics, cache, device).to(device)
    network = NAFNet(
        in_chan=3,
        out_chan=3,
        width=config.network.width,
        middle_blk_num=config.network.middle_blk_num,
        enc_blk_nums=config.network.enc_blk_nums,
        dec_blk_nums=config.network.dec_blk_nums,
    ).to(device)
    perceptual_loss = (
        VGG16PerceptualLoss(device)
        if config.training.perceptual_weight > 0.0
        else None
    )
    loader = build_loader(config.dataset, config.training.seed)
    if hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(0)
    clean = next(iter(loader)).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss, metrics, _, psfs = _loss_for_batch(
        clean,
        optics,
        network,
        step=0,
        pixel_loss_weight=config.training.pixel_loss_weight,
        perceptual_weight=config.training.perceptual_weight,
        perceptual_loss=perceptual_loss,
        averaged_wavelengths=False,
        noise_std=0.0,
    )
    loss.backward()
    result = {
        "status": "completed",
        "mode": "memory_smoke",
        "output": str(output),
        "cache_path": str(cache_path),
        "cache": cache_description(cache),
        "psf_shape": list(psfs.shape),
        "metrics": metrics,
        "cuda_max_memory_allocated": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "cuda_max_memory_reserved": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
    }
    result_path = output / "memory_smoke.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
