from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from edof_reproduction.config import (
    DatasetConfig,
    EDOFConfig,
    EvaluationConfig,
    NetworkConfig,
    OpticsConfig,
    OutputConfig,
    TrainingConfig,
    load_config,
)
from edof_reproduction.dataset import DIV2KDataset
from edof_reproduction.imaging import spatial_convolution
from edof_reproduction.optics import CachedRayWaveOptics, _call_doe_field, load_or_build_cache
from edof_reproduction.runner import _loss_from_psfs, run_memory_smoke, run_training


def tiny_config(tmp_path: Path, epochs: int = 3) -> EDOFConfig:
    return EDOFConfig(
        optics=OpticsConfig(
            backend="analytic",
            simulation_grid=16,
            field_grid=1,
            psf_size=5,
            cache_complex_dtype="complex128",
        ),
        network=NetworkConfig(
            width=2,
            middle_blk_num=1,
            enc_blk_nums=(1,),
            dec_blk_nums=(1,),
        ),
        dataset=DatasetConfig(
            mode="synthetic", crop_size=16, synthetic_images=1, batch_size=1, workers=0
        ),
        training=TrainingConfig(
            seed=123,
            device="cpu",
            joint_epochs=epochs,
            finetune_epochs=0,
            max_batches_per_epoch=1,
            warmup_epochs=1,
            checkpoint_every=1,
        ),
        output=OutputConfig(root=str(tmp_path), run_name="test", workspace_id="test"),
        source_config="test-config",
    )


def test_mac_config_encodes_three_epoch_smoke() -> None:
    config = load_config("configs/edof_reproduction/mac_smoke.yaml")
    assert config.training.joint_epochs == 3
    assert config.training.finetune_epochs == 0
    assert config.optics.backend == "deeplens"
    assert config.optics.depths_mm == (-200.0, -300.0, -10000.0)
    assert sum(len(group) for group in config.optics.wavelengths_rgb_um) == 9


def test_analytic_optics_is_differentiable(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    output = tmp_path / "cache"
    output.mkdir()
    cache, _ = load_or_build_cache(config.optics, output)
    optics = CachedRayWaveOptics(config.optics, cache, torch.device("cpu"))
    optics.doe.set_coefficients((0.2, 0.1, -0.1, 0.03, 0.02, -0.01))
    psfs = optics.psfs(((0,), (3,), (6,)))
    assert psfs.shape == (3, 1, 3, 5, 5)
    assert torch.allclose(psfs.sum(dim=(-2, -1)), torch.ones(3, 1, 3), atol=1e-5)
    psfs[0, 0, 0, 0, 0].backward()
    assert optics.doe.a2.grad is not None
    assert torch.isfinite(optics.doe.a2.grad)


def test_deeplens_25_disables_automatic_4000_grid_upsampling() -> None:
    class FakeLens:
        def __init__(self) -> None:
            self.kwargs = None

        def doe_field(self, point, wvln=None, spp=None, upsample_factor=None):
            self.kwargs = {
                "point": point,
                "wvln": wvln,
                "spp": spp,
                "upsample_factor": upsample_factor,
            }
            return torch.ones(8, 8, dtype=torch.complex128), [0.0, 0.0]

    lens = FakeLens()
    field, _ = _call_doe_field(
        lens,
        point=torch.tensor([0.0, 0.0, -300.0]),
        wavelength=0.55,
        coherent_rays=1_000_000,
    )
    assert field.shape == (8, 8)
    assert lens.kwargs["upsample_factor"] == 1


def test_three_epoch_run_writes_trace_checkpoint_and_artifacts(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, epochs=3)
    output = tmp_path / "run"
    result = run_training(config, output_override=output)
    assert result["status"] == "completed"
    assert result["epochs_completed"] == 3
    rows = (output / "training_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(row)["epoch"] for row in rows] == [1, 2, 3]
    assert (output / "checkpoints" / "latest.pt").exists()
    assert (output / "summary.json").exists()
    assert (output / "trace.sqlite").exists()
    assert (output / "artifact_manifest.json").exists()
    assert (output / "validation_comparison.png").exists()
    assert len(result["artifact_ids"]) == 13


def test_div2k_training_crop_changes_by_epoch_but_is_reproducible(tmp_path: Path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    yy, xx = np.mgrid[:64, :64]
    image = np.stack((xx * 4, yy * 4, (xx + yy) * 2), axis=-1).clip(0, 255).astype(np.uint8)
    Image.fromarray(image, mode="RGB").save(root / "sample.png")
    dataset = DIV2KDataset(
        root,
        24,
        7,
        training=True,
        random_resize_min_scale=0.5,
        color_jitter=0.2,
        horizontal_flip=True,
    )
    dataset.set_epoch(0)
    first = dataset[0]
    dataset.set_epoch(1)
    second = dataset[0]
    dataset.set_epoch(0)
    repeated = dataset[0]
    assert torch.equal(first, repeated)
    assert not torch.equal(first, second)


def test_div2k_validation_uses_stable_center_crop(tmp_path: Path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    image = np.arange(48 * 64 * 3, dtype=np.uint32).reshape(48, 64, 3).astype(np.uint8)
    Image.fromarray(image, mode="RGB").save(root / "sample.png")
    dataset = DIV2KDataset(root, 24, 9, training=False)
    dataset.set_epoch(0)
    first = dataset[0]
    dataset.set_epoch(20)
    assert torch.equal(first, dataset[0])


def test_spatial_convolution_matches_full_image_reference() -> None:
    torch.manual_seed(4)
    image = torch.rand(1, 3, 17, 19)
    psfs = torch.rand(4, 3, 5, 5)
    psfs = psfs / psfs.sum(dim=(-2, -1), keepdim=True)
    actual = spatial_convolution(image, psfs)
    expected = torch.zeros_like(image)
    for index in range(4):
        row, column = divmod(index, 2)
        top, bottom = round(row * 17 / 2), round((row + 1) * 17 / 2)
        left, right = round(column * 19 / 2), round((column + 1) * 19 / 2)
        blurred = F.conv2d(image, psfs[index, :, None], padding=2, groups=3)
        expected[..., top:bottom, left:right] = blurred[..., top:bottom, left:right]
    assert torch.allclose(actual, expected, atol=1e-6)


def test_paper_loss_uses_pixel_and_perceptual_terms() -> None:
    clean = torch.rand(1, 3, 8, 8)
    psfs = torch.zeros(3, 1, 3, 3, 3)
    psfs[..., 1, 1] = 1.0

    class Scale(torch.nn.Module):
        def forward(self, value):
            return value * 0.5

    class FakePerceptual(torch.nn.Module):
        def forward(self, prediction, target):
            return F.mse_loss(prediction, target) * 2.0

    loss, metrics, _ = _loss_from_psfs(
        clean,
        psfs,
        Scale(),
        pixel_loss_weight=1.0,
        perceptual_weight=0.1,
        perceptual_loss=FakePerceptual(),
        noise_std=0.0,
    )
    assert torch.allclose(loss, torch.tensor(metrics["pixel_mse"] * 1.2), atol=1e-6)


def test_validation_averages_all_samples_and_saves_best_checkpoint(tmp_path: Path) -> None:
    config = replace(
        tiny_config(tmp_path, epochs=2),
        evaluation=EvaluationConfig(
            enabled=True,
            crop_size=16,
            batch_size=1,
            workers=0,
            every_n_epochs=1,
            use_lpips=False,
            noise_std=0.0,
            early_stopping_patience=2,
            early_stopping_min_delta=0.0,
        ),
    )
    output = tmp_path / "validated"
    result = run_training(config, output_override=output)
    assert result["epochs_completed"] == 2
    assert result["best_epoch"] in {1, 2}
    assert len(result["final_depth_metrics"]) == 3
    assert all(item["samples"] == 1 for item in result["final_depth_metrics"])
    assert (output / "checkpoints" / "best.pt").exists()
    assert len((output / "validation_log.jsonl").read_text().strip().splitlines()) == 2


def test_memory_smoke_runs_real_backward_and_reports_cache(tmp_path: Path) -> None:
    output = tmp_path / "memory"
    result = run_memory_smoke(tiny_config(tmp_path, epochs=1), output_override=output)
    assert result["status"] == "completed"
    assert result["psf_shape"] == [3, 1, 3, 5, 5]
    assert result["metrics"]["pixel_mse"] > 0.0
    assert (output / "memory_smoke.json").exists()
