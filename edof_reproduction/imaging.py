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
    if field_count == 1:
        return F.conv2d(image, psfs[0, :, None], padding=radius, groups=channels)
    side = round(math.sqrt(field_count))
    if side * side != field_count:
        raise ValueError("field count must be a square grid")
    height, width = image.shape[-2:]
    padded = F.pad(image, (radius, radius, radius, radius))
    output = torch.zeros_like(image)
    for index in range(field_count):
        row, column = divmod(index, side)
        top, bottom = round(row * height / side), round((row + 1) * height / side)
        left, right = round(column * width / side), round((column + 1) * width / side)
        patch = padded[..., top : bottom + 2 * radius, left : right + 2 * radius]
        output[..., top:bottom, left:right] = F.conv2d(
            patch,
            psfs[index, :, None],
            groups=channels,
        )
    return output


def wavelength_choice(step: int, *, averaged: bool) -> tuple[tuple[int, ...], ...]:
    if averaged:
        return ((0, 1, 2), (3, 4, 5), (6, 7, 8))
    return tuple((base + ((step + channel) % 3),) for channel, base in enumerate((0, 3, 6)))
