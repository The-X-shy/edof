"""Configuration loading and validation for the EDoF reproduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OpticsConfig:
    backend: str = "deeplens"
    cache_device: str = "auto"
    lens_file: str = "configs/edof_reproduction/a489_poly1d_base.json"
    cache_file: str = "wavefront_cache.pt"
    focus_depth_mm: float = -300.0
    depths_mm: tuple[float, ...] = (-200.0, -300.0, -10000.0)
    wavelengths_rgb_um: tuple[tuple[float, ...], ...] = (
        (0.62, 0.66, 0.70),
        (0.50, 0.53, 0.56),
        (0.45, 0.47, 0.49),
    )
    field_grid: int = 1
    simulation_grid: int = 64
    psf_size: int = 31
    coherent_rays: int = 1_000_000
    doe_size_mm: float = 3.0
    doe_sensor_distance_mm: float = 2.9472
    design_wavelength_um: float = 0.55
    design_refractive_index: float = 1.4599
    quantization_levels: int = 16
    quantize_during_training: bool = True
    cache_complex_dtype: str = "complex128"


@dataclass(frozen=True)
class NetworkConfig:
    width: int = 16
    middle_blk_num: int = 1
    enc_blk_nums: tuple[int, ...] = (1, 1, 1, 18)
    dec_blk_nums: tuple[int, ...] = (1, 1, 1, 1)


@dataclass(frozen=True)
class DatasetConfig:
    mode: str = "synthetic"
    root: str = "datasets/DIV2K"
    crop_size: int = 64
    synthetic_images: int = 2
    batch_size: int = 1
    workers: int = 0


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 240608
    device: str = "auto"
    joint_epochs: int = 3
    finetune_epochs: int = 0
    max_batches_per_epoch: int | None = 1
    doe_lr: float = 0.1
    network_lr: float = 0.0001
    finetune_lr: float = 0.0001
    warmup_epochs: int = 1
    accumulation_steps: int = 1
    quality_weight: float = 0.3
    gradient_clip: float = 1.0
    sensor_noise_std: float = 0.01
    psf_pretrain_steps: int = 0
    checkpoint_every: int = 1
    resume: str | None = None


@dataclass(frozen=True)
class OutputConfig:
    root: str = "workspace/edof_reproduction"
    run_name: str = "mac_smoke"
    workspace_id: str = "default"


@dataclass(frozen=True)
class EDOFConfig:
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    source_config: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tuple_nested(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_nested(item) for item in value)
    return value


def _section(cls: type, payload: dict[str, Any], name: str):
    raw = {key: _tuple_nested(value) for key, value in payload.get(name, {}).items()}
    return cls(**raw)


def load_config(path: str | Path) -> EDOFConfig:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    config = EDOFConfig(
        optics=_section(OpticsConfig, payload, "optics"),
        network=_section(NetworkConfig, payload, "network"),
        dataset=_section(DatasetConfig, payload, "dataset"),
        training=_section(TrainingConfig, payload, "training"),
        output=_section(OutputConfig, payload, "output"),
        source_config=str(source),
    )
    validate_config(config)
    return config


def validate_config(config: EDOFConfig) -> None:
    optics, training, dataset = config.optics, config.training, config.dataset
    if optics.backend not in {"deeplens", "analytic"}:
        raise ValueError("optics.backend must be deeplens or analytic")
    if optics.cache_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("optics.cache_device must be auto, cpu, or cuda")
    if len(optics.depths_mm) != 3:
        raise ValueError("the paper reproduction requires exactly three depths")
    if tuple(len(group) for group in optics.wavelengths_rgb_um) != (3, 3, 3):
        raise ValueError("the paper reproduction requires three wavelengths per RGB channel")
    if optics.field_grid < 1 or optics.simulation_grid < 8:
        raise ValueError("field_grid and simulation_grid are too small")
    if optics.psf_size < 3 or optics.psf_size % 2 == 0:
        raise ValueError("psf_size must be an odd integer of at least three")
    if optics.coherent_rays < 1_000_000 and optics.backend == "deeplens":
        raise ValueError("DeepLens coherent ray tracing requires at least 1,000,000 rays")
    if training.joint_epochs < 0 or training.finetune_epochs < 0:
        raise ValueError("epoch counts must be non-negative")
    if training.joint_epochs + training.finetune_epochs == 0:
        raise ValueError("at least one training epoch is required")
    if training.accumulation_steps < 1 or dataset.batch_size < 1:
        raise ValueError("batch and accumulation sizes must be positive")
    if dataset.mode not in {"synthetic", "div2k"}:
        raise ValueError("dataset.mode must be synthetic or div2k")
