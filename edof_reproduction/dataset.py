"""Deterministic synthetic data plus augmented DIV2K train/validation loading."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

from .config import DatasetConfig, EvaluationConfig


class SyntheticEDOFDataset(Dataset):
    def __init__(self, count: int, size: int, seed: int) -> None:
        self.count, self.size, self.seed = count, size, seed

    def __len__(self) -> int:
        return self.count

    def set_epoch(self, epoch: int) -> None:
        del epoch

    def __getitem__(self, index: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.seed + index)
        axis = torch.linspace(0.0, 1.0, self.size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        frequencies = torch.randint(2, 10, (3,), generator=generator).float()
        phases = torch.rand(3, generator=generator) * torch.pi
        channels = []
        for channel in range(3):
            pattern = 0.45 + 0.20 * torch.sin(
                frequencies[channel] * torch.pi * xx + phases[channel]
            )
            pattern += 0.20 * torch.cos(
                (frequencies[channel] + 1) * torch.pi * yy - phases[channel]
            )
            checker = ((xx * 12).floor() + (yy * 12).floor()).remainder(2)
            channels.append((pattern + checker * 0.15).clamp(0.0, 1.0))
        return torch.stack(channels)


class DIV2KDataset(Dataset):
    """DIV2K images with epoch-varying train crops and fixed validation crops."""

    def __init__(
        self,
        root: str | Path,
        crop_size: int,
        seed: int,
        *,
        training: bool,
        random_resize_min_scale: float = 0.8,
        color_jitter: float = 0.2,
        horizontal_flip: bool = True,
        max_images: int | None = None,
    ) -> None:
        root_path = Path(root)
        candidates: list[Path] = []
        for suffix in ("*.png", "*.jpg", "*.jpeg"):
            candidates.extend(root_path.rglob(suffix))
        self.paths = sorted(candidates)
        if max_images is not None:
            self.paths = self.paths[:max_images]
        if not self.paths:
            raise FileNotFoundError(f"no DIV2K images found below {root_path}")
        self.crop_size = crop_size
        self.seed = seed
        self.training = training
        self.random_resize_min_scale = random_resize_min_scale
        self.color_jitter = color_jitter
        self.horizontal_flip = horizontal_flip
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.paths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _generator(self, index: int) -> torch.Generator:
        return torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003 + index)

    def _ensure_size(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if min(width, height) >= self.crop_size:
            return image
        scale = self.crop_size / min(width, height)
        return image.resize(
            (math.ceil(width * scale), math.ceil(height * scale)),
            Image.Resampling.BICUBIC,
        )

    def _training_transform(self, image: Image.Image, generator: torch.Generator) -> Image.Image:
        image = self._ensure_size(image)
        width, height = image.size
        scale = float(
            torch.empty(1).uniform_(self.random_resize_min_scale, 1.0, generator=generator)
        )
        # Interpret scale as retained area, matching RandomResizedCrop while
        # keeping 128px training patches local instead of shrinking a full 2K image.
        side = max(self.crop_size, round(self.crop_size / math.sqrt(scale)))
        side = min(side, width, height)
        top = int(torch.randint(0, height - side + 1, (1,), generator=generator))
        left = int(torch.randint(0, width - side + 1, (1,), generator=generator))
        image = image.crop((left, top, left + side, top + side))
        if side != self.crop_size:
            image = image.resize((self.crop_size, self.crop_size), Image.Resampling.BICUBIC)

        if self.color_jitter > 0.0:
            factors = 1.0 + (
                torch.rand(3, generator=generator) * 2.0 - 1.0
            ) * self.color_jitter
            enhancers = (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color)
            for index in torch.randperm(3, generator=generator).tolist():
                image = enhancers[index](image).enhance(float(factors[index]))
        if self.horizontal_flip and bool(torch.rand(1, generator=generator) < 0.5):
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image

    def _validation_transform(self, image: Image.Image) -> Image.Image:
        image = self._ensure_size(image)
        width, height = image.size
        top = (height - self.crop_size) // 2
        left = (width - self.crop_size) // 2
        return image.crop((left, top, left + self.crop_size, top + self.crop_size))

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.paths[index]).convert("RGB")
        if self.training:
            image = self._training_transform(image, self._generator(index))
        else:
            image = self._validation_transform(image)
        array = np.asarray(image, dtype=np.float32).copy()
        return torch.from_numpy(array).permute(2, 0, 1) / 255.0


def _loader(dataset: Dataset, *, batch_size: int, workers: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        generator=generator,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


def build_loader(config: DatasetConfig, seed: int) -> DataLoader:
    if config.mode == "synthetic":
        dataset: Dataset = SyntheticEDOFDataset(config.synthetic_images, config.crop_size, seed)
    else:
        dataset = DIV2KDataset(
            config.root,
            config.crop_size,
            seed,
            training=config.random_crop,
            random_resize_min_scale=config.random_resize_min_scale,
            color_jitter=config.color_jitter,
            horizontal_flip=config.horizontal_flip,
        )
    return _loader(
        dataset,
        batch_size=config.batch_size,
        workers=config.workers,
        shuffle=True,
        seed=seed,
    )


def build_validation_loader(
    dataset_config: DatasetConfig,
    evaluation_config: EvaluationConfig,
    seed: int,
) -> DataLoader:
    if dataset_config.mode == "synthetic":
        dataset: Dataset = SyntheticEDOFDataset(
            dataset_config.synthetic_images,
            evaluation_config.crop_size,
            seed + 10_000,
        )
    else:
        dataset = DIV2KDataset(
            evaluation_config.root or "",
            evaluation_config.crop_size,
            seed + 10_000,
            training=False,
            max_images=evaluation_config.max_images,
        )
    return _loader(
        dataset,
        batch_size=evaluation_config.batch_size,
        workers=evaluation_config.workers,
        shuffle=False,
        seed=seed + 10_000,
    )
