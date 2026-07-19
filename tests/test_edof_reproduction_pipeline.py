from __future__ import annotations

import json
from pathlib import Path

import torch

from edof_reproduction.config import (
    DatasetConfig,
    EDOFConfig,
    NetworkConfig,
    OpticsConfig,
    OutputConfig,
    TrainingConfig,
    load_config,
)
from edof_reproduction.optics import CachedRayWaveOptics, load_or_build_cache
from edof_reproduction.runner import run_training


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
    assert len(result["artifact_ids"]) == 10
