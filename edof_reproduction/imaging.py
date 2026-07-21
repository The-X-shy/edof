"""Spatially varying image formation shared by training and validation."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def spatial_convolution(image: Tensor, psfs: Tensor) -> Tensor:
    """Apply one RGB PSF per field cell without convolving unused image regions."""

    field_count, channels, kernel, _ = psfs.shape
    radius = kernel // 2
    padding_mode = "reflect" if min(image.shape[-2:]) > radius else "replicate"
    padded = F.pad(image, (radius, radius, radius, radius), mode=padding_mode)
    # F.conv2d computes cross-correlation.  Image formation is convolution,
    # matching DeepLens conv_psf / conv_psf_map.
    flipped = torch.flip(psfs, dims=(-2, -1))
    if field_count == 1:
        return F.conv2d(padded, flipped[0, :, None], groups=channels)
    side = round(math.sqrt(field_count))
    if side * side != field_count:
        raise ValueError("field count must be a square grid")
    height, width = image.shape[-2:]
    output = torch.zeros_like(image)
    for index in range(field_count):
        row, column = divmod(index, side)
        top, bottom = (row * height) // side, ((row + 1) * height) // side
        left, right = (column * width) // side, ((column + 1) * width) // side
        patch = padded[..., top : bottom + 2 * radius, left : right + 2 * radius]
        output[..., top:bottom, left:right] = F.conv2d(
            patch,
            flipped[index, :, None],
            groups=channels,
        )
    return output


def interpolate_psf_grid(psfs: Tensor, target_side: int | None) -> Tensor:
    """Bilinearly densify ``[depth, field, RGB, k, k]`` PSF maps.

    The optical joint stage can use the paper's affordable 5x5 map while the
    fixed-optics stage uses a denser spatial map without retaining additional
    coherent wave fields.  Each interpolated kernel is renormalized.
    """

    if target_side is None:
        return psfs
    if psfs.ndim != 5:
        raise ValueError("psfs must have shape [depth, field, channels, kernel, kernel]")
    depth_count, field_count, channels, kernel, _ = psfs.shape
    source_side = round(math.sqrt(field_count))
    if source_side * source_side != field_count:
        raise ValueError("field count must be a square grid")
    if target_side < source_side:
        raise ValueError("target PSF grid cannot be smaller than the source grid")
    if target_side == source_side:
        return psfs
    maps = psfs.permute(0, 2, 3, 4, 1).reshape(
        depth_count * channels * kernel * kernel, 1, source_side, source_side
    )
    maps = F.interpolate(
        maps,
        size=(target_side, target_side),
        mode="bilinear",
        align_corners=True,
    )
    result = maps.reshape(
        depth_count, channels, kernel, kernel, target_side * target_side
    ).permute(0, 4, 1, 2, 3)
    return result / result.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)


def wavelength_choice(step: int, *, averaged: bool) -> tuple[tuple[int, ...], ...]:
    if averaged:
        return ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    # The supplement samples each colour independently. Use a local seeded
    # generator so all 27 RGB combinations are reachable while checkpoint
    # replay remains deterministic.
    generator = torch.Generator(device="cpu").manual_seed(int(step))
    choices = torch.randint(0, 3, (3,), generator=generator).tolist()
    return tuple((base + int(choice),) for base, choice in zip((0, 3, 6), choices))
