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
    validate_config,
)
from edof_reproduction.dataset import DIV2KDataset
from edof_reproduction.convergence import (
    build_strict_finetune_config,
    crop_normalized_psfs,
    mean_edge_energy,
    select_optical_settings,
)
from edof_reproduction.imaging import (
    _spatial_convolution_loop,
    interpolate_psf_grid,
    spatial_convolution,
    wavelength_choice,
)
from edof_reproduction.optics import (
    CachedRayWaveOptics,
    _call_doe_field,
    _field_coordinates,
    load_or_build_cache,
    load_or_build_fixed_psf_map,
)
from edof_reproduction.runner import (
    _loss_from_psfs,
    _psf_pretrain_loss,
    run_memory_smoke,
    run_training,
)


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
            checkpoint_every=5,
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


def test_optimized_config_matches_disclosed_edof_schedule() -> None:
    config = load_config("configs/edof_reproduction/windows_optimized.yaml")
    assert (config.training.joint_epochs, config.training.finetune_epochs) == (50, 50)
    assert config.optics.field_grid == 5
    assert config.optics.finetune_field_grid == 40
    assert config.optics.propagation_precision == "float64"
    assert not config.optics.quantize_during_training
    assert config.training.pixel_loss_type == "rmse"
    assert config.training.pixel_loss_weight == 0.3
    assert config.training.cross_depth_loss_weight == 1.0
    assert config.training.local_field_patches


def test_practical_config_uses_exact_psfs_and_perceptual_finetune() -> None:
    config = load_config("configs/edof_reproduction/windows_practical_finetune.yaml")
    assert (config.training.joint_epochs, config.training.finetune_epochs) == (0, 50)
    assert config.optics.simulation_grid == 512
    assert config.optics.finetune_field_grid == 40
    assert config.optics.finetune_psf_mode == "exact"
    assert config.training.initialize_from.endswith("windows_optimized/checkpoints/best.pt")
    assert config.training.pixel_loss_type == "mse"
    assert config.training.perceptual_weight == 0.1
    assert config.training.cross_depth_loss_weight == 0.1


def test_recommended_sequence_configs_share_exact_cache_and_initialization() -> None:
    baseline = load_config("configs/edof_reproduction/windows_exact_baseline_eval.yaml")
    assert baseline.optics.finetune_psf_mode == "exact"
    assert baseline.optics.finetune_psf_cache_file.startswith(
        "../windows_practical_finetune/"
    )
    expected = {
        "windows_practical_p002_short.yaml": (15, 0.02),
        "windows_practical_p005_short.yaml": (15, 0.05),
        "windows_practical_p002_full.yaml": (50, 0.02),
        "windows_practical_p005_full.yaml": (50, 0.05),
    }
    for name, (epochs, weight) in expected.items():
        config = load_config(f"configs/edof_reproduction/{name}")
        assert (config.training.joint_epochs, config.training.finetune_epochs) == (0, epochs)
        assert config.training.perceptual_weight == weight
        assert config.training.pixel_loss_weight == 1.0
        assert config.training.cross_depth_loss_weight == 0.1
        assert config.training.seed == 240608
        assert config.training.initialize_from.endswith(
            "windows_optimized/checkpoints/best.pt"
        )
        assert config.optics.cache_file.startswith("../windows_practical_finetune/")
        assert config.optics.finetune_psf_cache_file.startswith(
            "../windows_practical_finetune/"
        )


def test_strict_configs_use_full_field_evaluation_and_paper_loss() -> None:
    evaluation = load_config(
        "configs/edof_reproduction/windows_strict_full_fov_eval.yaml"
    )
    assert evaluation.evaluation.crop_size == 1000
    assert evaluation.evaluation.field_grid == 40
    assert not evaluation.evaluation.local_field_patches
    assert evaluation.optics.finetune_psf_mode == "exact"

    finetune = load_config("configs/edof_reproduction/windows_strict_finetune.yaml")
    assert (finetune.training.joint_epochs, finetune.training.finetune_epochs) == (
        0,
        50,
    )
    assert finetune.training.pixel_loss_type == "rmse"
    assert finetune.training.pixel_loss_weight == 0.3
    assert finetune.training.perceptual_weight == 0.0
    assert finetune.training.cross_depth_loss_weight == 1.0
    assert finetune.training.initialize_from.endswith(
        "windows_optimized/checkpoints/epoch_050.pt"
    )
    assert finetune.evaluation.crop_size == 1000
    assert not finetune.evaluation.local_field_patches


def test_resume_and_initialize_from_are_mutually_exclusive(tmp_path: Path) -> None:
    config = tiny_config(tmp_path, epochs=1)
    config = replace(
        config,
        training=replace(config.training, resume="latest.pt", initialize_from="best.pt"),
    )
    try:
        validate_config(config)
    except ValueError as error:
        assert "mutually exclusive" in str(error)
    else:
        raise AssertionError("conflicting checkpoint modes were accepted")


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


def test_square_doe_uses_corner_radius_and_pixel_centres(tmp_path: Path) -> None:
    config = tiny_config(tmp_path)
    output = tmp_path / "normalization"
    output.mkdir()
    cache, _ = load_or_build_cache(config.optics, output)
    optics = CachedRayWaveOptics(config.optics, cache, torch.device("cpu"))
    assert torch.allclose(
        optics.doe.doe_radius,
        torch.tensor(3.0 / np.sqrt(2.0), dtype=optics.doe.doe_radius.dtype),
    )
    assert torch.allclose(optics.grid_x[0, 0], torch.tensor(-1.40625, dtype=optics.grid_x.dtype))
    assert torch.allclose(optics.grid_x[0, -1], torch.tensor(1.40625, dtype=optics.grid_x.dtype))


def test_field_grid_matches_deeplens_patch_centres() -> None:
    fields = _field_coordinates(5)
    assert fields[0] == (-0.875, 0.875)
    assert fields[4] == (0.875, 0.875)
    assert fields[-1] == (0.875, -0.875)


def test_vectorized_full_field_convolution_matches_reference() -> None:
    generator = torch.Generator().manual_seed(240608)
    image = torch.rand(2, 3, 8, 10, generator=generator)
    psfs = torch.rand(4, 3, 3, 3, generator=generator)
    psfs = psfs / psfs.sum(dim=(-2, -1), keepdim=True)
    expected = _spatial_convolution_loop(image, psfs)
    actual = spatial_convolution(image, psfs, field_chunk_size=5)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_full_field_convolution_rejects_invalid_chunk_size() -> None:
    image = torch.ones(1, 3, 8, 8)
    psfs = torch.ones(4, 3, 3, 3) / 9.0
    try:
        spatial_convolution(image, psfs, field_chunk_size=0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("invalid field chunk size was accepted")


def test_convergence_psf_crop_is_centered_and_normalized() -> None:
    psfs = torch.zeros(3, 9, 3, 7, 7)
    psfs[..., 2:5, 2:5] = 1.0
    cropped = crop_normalized_psfs(psfs, 5)
    assert cropped.shape == (3, 9, 3, 5, 5)
    assert torch.allclose(
        cropped.sum(dim=(-2, -1)),
        torch.ones(3, 9, 3),
    )
    assert mean_edge_energy(cropped, border=1) == 0.0


def test_convergence_selection_applies_declared_psnr_gates() -> None:
    cases = [
        {"simulation_grid": 512, "psf_size": 63, "mean_raw_psnr": 16.00},
        {"simulation_grid": 512, "psf_size": 127, "mean_raw_psnr": 16.10},
        {"simulation_grid": 768, "psf_size": 63, "mean_raw_psnr": 16.30},
        {"simulation_grid": 768, "psf_size": 127, "mean_raw_psnr": 16.45},
        {"simulation_grid": 1024, "psf_size": 63, "mean_raw_psnr": 16.50},
        {"simulation_grid": 1024, "psf_size": 127, "mean_raw_psnr": 16.52},
    ]
    decision = select_optical_settings(cases)
    assert decision["selected_simulation_grid"] == 768
    assert decision["selected_psf_size"] == 127
    assert decision["grid_limited"]
    assert decision["primary_bottleneck"] == "simulation_grid"


def test_convergence_selection_keeps_compact_settings_when_converged() -> None:
    cases = [
        {"simulation_grid": 512, "psf_size": 63, "mean_raw_psnr": 16.00},
        {"simulation_grid": 512, "psf_size": 127, "mean_raw_psnr": 16.03},
        {"simulation_grid": 768, "psf_size": 63, "mean_raw_psnr": 16.02},
        {"simulation_grid": 768, "psf_size": 127, "mean_raw_psnr": 16.05},
        {"simulation_grid": 1024, "psf_size": 63, "mean_raw_psnr": 16.04},
        {"simulation_grid": 1024, "psf_size": 127, "mean_raw_psnr": 16.07},
    ]
    decision = select_optical_settings(cases)
    assert decision["selected_simulation_grid"] == 512
    assert decision["selected_psf_size"] == 63
    assert not decision["grid_limited"]
    assert decision["primary_bottleneck"] == "proxy_lens_parameters"


def test_convergence_decision_preserves_strict_paper_finetune() -> None:
    base = load_config("configs/edof_reproduction/windows_strict_finetune.yaml")
    config = build_strict_finetune_config(
        base,
        {
            "selected_simulation_grid": 768,
            "selected_psf_size": 127,
        },
        cache_file="../convergence/cache_768.pt",
        fixed_psf_cache_file="fixed_40x40_768_k127.pt",
        initialize_from="joint_epoch_050.pt",
    )
    assert config.optics.field_grid == 3
    assert config.optics.simulation_grid == 768
    assert config.optics.psf_size == 127
    assert config.optics.finetune_field_grid == 40
    assert config.optics.finetune_psf_mode == "exact"
    assert config.training.initialize_from == "joint_epoch_050.pt"
    assert (config.training.joint_epochs, config.training.finetune_epochs) == (0, 50)
    assert config.training.pixel_loss_type == "rmse"
    assert config.training.pixel_loss_weight == 0.3
    assert config.training.cross_depth_loss_weight == 1.0
    assert config.training.perceptual_weight == 0.0
    assert config.dataset.crop_size == 128
    assert config.evaluation.crop_size == 1000
    assert config.evaluation.field_grid == 40
    assert not config.evaluation.local_field_patches


def test_strict_background_workers_expose_project_to_python() -> None:
    for name in (
        "windows_strict_optics_worker.ps1",
        "windows_strict_finetune_worker.ps1",
    ):
        source = Path("scripts", name).read_text(encoding="utf-8")
        assert "$env:PYTHONPATH = $ProjectRoot" in source


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
    assert not (output / "checkpoints" / "epoch_001.pt").exists()
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
    padded = F.pad(image, (2, 2, 2, 2), mode="reflect")
    for index in range(4):
        row, column = divmod(index, 2)
        top, bottom = (row * 17) // 2, ((row + 1) * 17) // 2
        left, right = (column * 19) // 2, ((column + 1) * 19) // 2
        patch = padded[..., top : bottom + 4, left : right + 4]
        expected[..., top:bottom, left:right] = F.conv2d(
            patch,
            torch.flip(psfs[index], dims=(-2, -1))[:, None],
            groups=3,
        )
    assert torch.allclose(actual, expected, atol=1e-6)


def test_psf_map_interpolation_is_normalized() -> None:
    psfs = torch.rand(3, 25, 3, 5, 5)
    psfs = psfs / psfs.sum(dim=(-2, -1), keepdim=True)
    result = interpolate_psf_grid(psfs, 10)
    assert result.shape == (3, 100, 3, 5, 5)
    assert torch.allclose(result.sum(dim=(-2, -1)), torch.ones(3, 100, 3), atol=1e-6)


def test_exact_fixed_psf_map_is_field_sampled_normalized_and_reusable(tmp_path: Path) -> None:
    base = tiny_config(tmp_path, epochs=1)
    optics_config = replace(
        base.optics,
        finetune_field_grid=2,
        finetune_psf_mode="exact",
        finetune_psf_cache_file="exact_psfs.pt",
        finetune_psf_save_every_fields=1,
    )
    cache_output = tmp_path / "exact-cache"
    cache_output.mkdir()
    cache, _ = load_or_build_cache(optics_config, cache_output)
    optics = CachedRayWaveOptics(optics_config, cache, torch.device("cpu"))
    optics.doe.set_coefficients((0.2, 0.1, -0.1, 0.03, 0.02, -0.01))
    psfs, path = load_or_build_fixed_psf_map(optics_config, optics, cache_output)
    assert psfs.shape == (3, 4, 3, 5, 5)
    assert torch.allclose(psfs.sum(dim=(-2, -1)), torch.ones(3, 4, 3), atol=1e-5)
    repeated, repeated_path = load_or_build_fixed_psf_map(optics_config, optics, cache_output)
    assert repeated_path == path
    assert torch.equal(repeated, psfs)
    metadata = json.loads(path.with_suffix(".pt.metadata.json").read_text(encoding="utf-8"))
    assert metadata["complete"] is True
    assert metadata["fields_completed"] == 4


def test_training_wavelengths_are_independent_and_replayable() -> None:
    choices = [wavelength_choice(step, averaged=False) for step in range(40)]
    assert choices == [wavelength_choice(step, averaged=False) for step in range(40)]
    assert len(set(choices)) > 3
    assert all(
        0 <= value[0][0] <= 2 and 3 <= value[1][0] <= 5 and 6 <= value[2][0] <= 8
        for value in choices
    )


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


def test_edof_loss_uses_rmse_quality_and_cross_depth_similarity() -> None:
    clean = torch.zeros(1, 3, 9, 9)
    clean[..., 4, 4] = 1.0
    psfs = torch.zeros(3, 1, 3, 3, 3)
    psfs[0, ..., 1, 1] = 1.0
    psfs[1, ..., 1, 2] = 1.0
    psfs[2, ..., 2, 1] = 1.0
    network = torch.nn.Identity()
    reconstructions = [spatial_convolution(clean, psfs[index]) for index in range(3)]
    quality = torch.stack(
        [torch.sqrt(F.mse_loss(reconstruction, clean) + 1e-12) for reconstruction in reconstructions]
    ).mean()
    pairs = torch.stack(
        [
            torch.sqrt(F.mse_loss(reconstructions[left], reconstructions[right]) + 1e-12)
            for left, right in ((0, 1), (0, 2), (1, 2))
        ]
    ).mean()
    loss, metrics, _ = _loss_from_psfs(
        clean,
        psfs,
        network,
        pixel_loss_weight=0.3,
        perceptual_weight=0.0,
        perceptual_loss=None,
        noise_std=0.0,
        pixel_loss_type="rmse",
        cross_depth_loss_weight=1.0,
    )
    assert torch.allclose(loss, 0.3 * quality + pairs, atol=1e-6)
    assert np.isclose(metrics["quality_loss"], float(quality), atol=1e-6)
    assert np.isclose(metrics["cross_depth_rmse"], float(pairs), atol=1e-6)


def test_psf_pretraining_combines_size_and_depth_similarity() -> None:
    psfs = torch.rand(3, 1, 3, 5, 5)
    psfs = psfs / psfs.sum(dim=(-2, -1), keepdim=True)
    loss, metrics = _psf_pretrain_loss(psfs, size_weight=0.2, similarity_weight=1.0)
    assert torch.allclose(
        loss,
        torch.tensor(0.2 * metrics["size"] + metrics["similarity"]),
        atol=1e-6,
    )


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


def test_local_field_fixed_optics_finetune_runs(tmp_path: Path) -> None:
    base = tiny_config(tmp_path, epochs=1)
    config = replace(
        base,
        optics=replace(base.optics, finetune_field_grid=2),
        training=replace(
            base.training,
            joint_epochs=1,
            finetune_epochs=1,
            local_field_patches=True,
            pixel_loss_type="rmse",
            pixel_loss_weight=0.3,
            cross_depth_loss_weight=1.0,
        ),
    )
    result = run_training(config, output_override=tmp_path / "finetune")
    assert result["epochs_completed"] == 2
    assert result["joint_epochs"] == 1
    assert result["finetune_epochs"] == 1


def test_initialize_from_starts_independent_exact_finetune(tmp_path: Path) -> None:
    base = tiny_config(tmp_path, epochs=1)
    source_output = tmp_path / "source"
    run_training(base, output_override=source_output)
    source_checkpoint = source_output / "checkpoints" / "latest.pt"
    config = replace(
        base,
        optics=replace(
            base.optics,
            finetune_field_grid=2,
            finetune_psf_mode="exact",
            finetune_psf_cache_file="exact_psfs.pt",
        ),
        training=replace(
            base.training,
            joint_epochs=0,
            finetune_epochs=1,
            initialize_from=str(source_checkpoint),
            psf_pretrain_steps=3,
            local_field_patches=True,
        ),
    )
    result = run_training(config, output_override=tmp_path / "independent-finetune")
    assert result["epochs_completed"] == 1
    assert result["initialized_from"] == str(source_checkpoint)
    assert result["pretrain_steps"] == 0
    assert result["finetune_psf_mode"] == "exact"


def test_memory_smoke_runs_real_backward_and_reports_cache(tmp_path: Path) -> None:
    output = tmp_path / "memory"
    result = run_memory_smoke(tiny_config(tmp_path, epochs=1), output_override=output)
    assert result["status"] == "completed"
    assert result["psf_shape"] == [3, 1, 3, 5, 5]
    assert result["metrics"]["pixel_mse"] > 0.0
    assert (output / "memory_smoke.json").exists()
