"""Deterministic synthetic smoke data and DIV2K image loading."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import DatasetConfig


class SyntheticEDOFDataset(Dataset):
    def __init__(self, count: int, size: int, seed: int) -> None:
        self.count, self.size, self.seed = count, size, seed

    def __len__(self) -> int:
        return self.count

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
    def __init__(self, root: str | Path, crop_size: int, seed: int) -> None:
        root_path = Path(root)
        candidates = []
        for suffix in ("*.png", "*.jpg", "*.jpeg"):
            candidates.extend(root_path.rglob(suffix))
        self.paths = sorted(candidates)
        if not self.paths:
            raise FileNotFoundError(f"no DIV2K images found below {root_path}")
        self.crop_size, self.seed = crop_size, seed

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.paths[index]).convert("RGB")
        tensor = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
        height, width = tensor.shape[-2:]
        if height < self.crop_size or width < self.crop_size:
            scale = max(self.crop_size / height, self.crop_size / width)
            image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
            tensor = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
            height, width = tensor.shape[-2:]
        generator = torch.Generator().manual_seed(self.seed + index)
        top = int(torch.randint(0, height - self.crop_size + 1, (1,), generator=generator))
        left = int(torch.randint(0, width - self.crop_size + 1, (1,), generator=generator))
        return tensor[:, top : top + self.crop_size, left : left + self.crop_size]


def build_loader(config: DatasetConfig, seed: int) -> DataLoader:
    if config.mode == "synthetic":
        dataset: Dataset = SyntheticEDOFDataset(config.synthetic_images, config.crop_size, seed)
    else:
        dataset = DIV2KDataset(config.root, config.crop_size, seed)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        generator=generator,
        drop_last=False,
    )
