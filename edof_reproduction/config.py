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
    f_number: float = 2.2
    sensor_resolution: tuple[int, int] = (1000, 1000)
    doe_normalization_radius_mm: float | None = None
    finetune_field_grid: int | None = None
    finetune_psf_mode: str = "interpolate"
    finetune_psf_cache_file: str = "fixed_psf_map.pt"
    finetune_psf_save_every_fields: int = 100
    design_wavelength_um: float = 0.55
    design_refractive_index: float = 1.4599
    quantization_levels: int = 16
    quantize_during_training: bool = True
    cache_complex_dtype: str = "complex128"
    propagation_precision: str = "float32"
    propagation_batch_size: int | None = None
    spatial_psf_refine_factor: int = 1


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
    random_crop: bool = True
    random_resize_min_scale: float = 0.8
    color_jitter: float = 0.2
    horizontal_flip: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    enabled: bool = False
    root: str | None = None
    crop_size: int = 128
    batch_size: int = 1
    workers: int = 0
    max_images: int | None = None
    every_n_epochs: int = 5
    use_lpips: bool = True
    noise_std: float = 0.01
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 0.02
    field_grid: int | None = None
    local_field_patches: bool = False
    minimum_psnr_epoch: int | None = None
    minimum_psnr: float | None = None


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
    pixel_loss_weight: float = 1.0
    perceptual_weight: float = 0.0
    pixel_loss_type: str = "mse"
    cross_depth_loss_weight: float = 0.0
    depth_loss_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    local_field_patches: bool = False
    spatial_field_crop_grid: int | None = None
    gradient_clip: float = 1.0
    sensor_noise_std: float = 0.01
    psf_pretrain_steps: int = 0
    psf_pretrain_size_weight: float = 0.0
    psf_pretrain_similarity_weight: float = 1.0
    checkpoint_every: int = 1
    log_every_batches: int = 100
    resume: str | None = None
    initialize_from: str | None = None


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
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
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
        evaluation=_section(EvaluationConfig, payload, "evaluation"),
        training=_section(TrainingConfig, payload, "training"),
        output=_section(OutputConfig, payload, "output"),
        source_config=str(source),
    )
    validate_config(config)
    return config


def validate_config(config: EDOFConfig) -> None:
    optics, training, dataset, evaluation = (
        config.optics,
        config.training,
        config.dataset,
        config.evaluation,
    )
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
    if len(optics.sensor_resolution) != 2 or min(optics.sensor_resolution) < optics.psf_size:
        raise ValueError("sensor_resolution must contain two values at least as large as psf_size")
    if optics.doe_normalization_radius_mm is not None and optics.doe_normalization_radius_mm <= 0:
        raise ValueError("doe_normalization_radius_mm must be positive when provided")
    if optics.f_number <= 0:
        raise ValueError("f_number must be positive")
    if optics.finetune_field_grid is not None and optics.finetune_field_grid < optics.field_grid:
        raise ValueError("finetune_field_grid must be at least field_grid")
    if optics.finetune_psf_mode not in {"interpolate", "exact"}:
        raise ValueError("finetune_psf_mode must be interpolate or exact")
    if not optics.finetune_psf_cache_file:
        raise ValueError("finetune_psf_cache_file must not be empty")
    if optics.finetune_psf_save_every_fields < 1:
        raise ValueError("finetune_psf_save_every_fields must be positive")
    if optics.propagation_precision not in {"float32", "float64"}:
        raise ValueError("propagation_precision must be float32 or float64")
    if optics.propagation_batch_size is not None and optics.propagation_batch_size < 1:
        raise ValueError("propagation_batch_size must be positive when provided")
    if optics.spatial_psf_refine_factor < 1:
        raise ValueError("spatial_psf_refine_factor must be positive")
    if optics.coherent_rays < 1_000_000 and optics.backend == "deeplens":
        raise ValueError("DeepLens coherent ray tracing requires at least 1,000,000 rays")
    if training.joint_epochs < 0 or training.finetune_epochs < 0:
        raise ValueError("epoch counts must be non-negative")
    if training.joint_epochs + training.finetune_epochs == 0:
        raise ValueError("at least one training epoch is required")
    if training.resume and training.initialize_from:
        raise ValueError("training.resume and training.initialize_from are mutually exclusive")
    if optics.finetune_psf_mode == "exact" and training.finetune_epochs == 0:
        raise ValueError("exact finetune PSFs require at least one finetune epoch")
    if training.accumulation_steps < 1 or dataset.batch_size < 1:
        raise ValueError("batch and accumulation sizes must be positive")
    if training.checkpoint_every < 1 or training.log_every_batches < 1:
        raise ValueError("checkpoint and batch-log intervals must be positive")
    if dataset.mode not in {"synthetic", "div2k"}:
        raise ValueError("dataset.mode must be synthetic or div2k")
    if not 0.0 < dataset.random_resize_min_scale <= 1.0:
        raise ValueError("dataset.random_resize_min_scale must be in (0, 1]")
    if dataset.color_jitter < 0.0:
        raise ValueError("dataset.color_jitter must be non-negative")
    if training.pixel_loss_weight <= 0.0 or training.perceptual_weight < 0.0:
        raise ValueError("training loss weights must be non-negative with a positive pixel weight")
    if training.pixel_loss_type not in {"mse", "rmse"}:
        raise ValueError("training.pixel_loss_type must be mse or rmse")
    if training.cross_depth_loss_weight < 0.0:
        raise ValueError("training.cross_depth_loss_weight must be non-negative")
    if len(training.depth_loss_weights) != len(optics.depths_mm) or min(training.depth_loss_weights) <= 0:
        raise ValueError("training.depth_loss_weights must provide one positive weight per depth")
    if training.spatial_field_crop_grid is not None:
        patch_grid = training.spatial_field_crop_grid
        field_grid = optics.finetune_field_grid
        sensor_height, sensor_width = optics.sensor_resolution
        if patch_grid < 2:
            raise ValueError("training.spatial_field_crop_grid must be at least two")
        if not training.local_field_patches:
            raise ValueError("spatial field crops require training.local_field_patches")
        if field_grid is None or patch_grid > field_grid:
            raise ValueError(
                "spatial field crops require a compatible optics.finetune_field_grid"
            )
        if sensor_height % field_grid or sensor_width % field_grid:
            raise ValueError("sensor resolution must be divisible by finetune_field_grid")
        if dataset.crop_size % patch_grid:
            raise ValueError("dataset crop_size must be divisible by spatial_field_crop_grid")
        training_tile = dataset.crop_size // patch_grid
        if (
            training_tile != sensor_height // field_grid
            or training_tile != sensor_width // field_grid
        ):
            raise ValueError(
                "training crop and PSF subgrid must preserve the full-sensor field-cell size"
            )
        refined_training_grid = patch_grid * optics.spatial_psf_refine_factor
        if dataset.crop_size % refined_training_grid:
            raise ValueError(
                "dataset crop_size must be divisible by the refined training field grid"
            )
    if training.psf_pretrain_size_weight < 0.0 or training.psf_pretrain_similarity_weight < 0.0:
        raise ValueError("PSF pretraining weights must be non-negative")
    if training.psf_pretrain_steps > 0 and (
        training.psf_pretrain_size_weight + training.psf_pretrain_similarity_weight <= 0.0
    ):
        raise ValueError("PSF pretraining needs at least one positive loss weight")
    if evaluation.enabled and dataset.mode == "div2k" and not evaluation.root:
        raise ValueError("evaluation.root is required for DIV2K validation")
    if evaluation.crop_size < 8 or evaluation.batch_size < 1 or evaluation.workers < 0:
        raise ValueError("evaluation crop, batch, and worker settings are invalid")
    if evaluation.every_n_epochs < 1 or evaluation.early_stopping_patience < 1:
        raise ValueError("evaluation interval and early-stopping patience must be positive")
    if evaluation.field_grid is not None and evaluation.field_grid < optics.field_grid:
        raise ValueError("evaluation.field_grid must be at least optics.field_grid")
    if (
        evaluation.enabled
        and evaluation.field_grid is not None
        and optics.spatial_psf_refine_factor > 1
        and evaluation.crop_size
        % (evaluation.field_grid * optics.spatial_psf_refine_factor)
    ):
        raise ValueError(
            "evaluation crop_size must be divisible by the refined evaluation field grid"
        )
    if evaluation.early_stopping_min_delta < 0.0 or evaluation.noise_std < 0.0:
        raise ValueError("evaluation deltas and noise must be non-negative")
    gate_values = (evaluation.minimum_psnr_epoch, evaluation.minimum_psnr)
    if (gate_values[0] is None) != (gate_values[1] is None):
        raise ValueError("minimum PSNR gate requires both epoch and threshold")
    if evaluation.minimum_psnr_epoch is not None:
        if evaluation.minimum_psnr_epoch < 1 or evaluation.minimum_psnr <= 0.0:
            raise ValueError("minimum PSNR gate values must be positive")
        total_epochs = training.joint_epochs + training.finetune_epochs
        if evaluation.minimum_psnr_epoch > total_epochs:
            raise ValueError("minimum PSNR gate epoch exceeds the training schedule")
        if evaluation.minimum_psnr_epoch % evaluation.every_n_epochs:
            raise ValueError("minimum PSNR gate epoch must be a validation epoch")
