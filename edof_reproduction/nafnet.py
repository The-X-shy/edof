"""Standalone RGB NAFNet used by the EDOF reproduction pipeline.

The module preserves the architecture and state-dict layout of the public
End2endImaging implementation at commit
``0d4661eba50c97359f8e72d71913517b3a005bd4``.  It intentionally depends only
on PyTorch so the reconstruction network can be trained without installing
End2endImaging or BasicSR.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


__all__ = ["LayerNorm2d", "NAFBlock", "NAFNet", "SimpleGate"]


class LayerNormFunction(torch.autograd.Function):
    """Channel-wise layer normalization for ``NCHW`` image tensors."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        channels = x.shape[1]
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + eps)
        ctx.save_for_backward(normalized, variance, weight)
        return (
            weight.view(1, channels, 1, 1) * normalized
            + bias.view(1, channels, 1, 1)
        )

    @staticmethod
    def backward(ctx, grad_output):
        normalized, variance, weight = ctx.saved_tensors
        channels = grad_output.shape[1]
        scaled_grad = grad_output * weight.view(1, channels, 1, 1)
        mean_grad = scaled_grad.mean(dim=1, keepdim=True)
        mean_grad_normalized = (scaled_grad * normalized).mean(
            dim=1, keepdim=True
        )
        grad_input = (
            scaled_grad - mean_grad - normalized * mean_grad_normalized
        ) / torch.sqrt(variance + ctx.eps)
        grad_weight = (grad_output * normalized).sum(dim=(0, 2, 3))
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_input, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    """Learnable channel-wise layer normalization for image tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """Split the channels in half and multiply the two halves."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first, second = x.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    """Nonlinear-activation-free restoration block from NAFNet."""

    def __init__(
        self,
        c: int,
        DW_Expand: int = 2,
        FFN_Expand: int = 2,
        drop_out_rate: float = 0.0,
    ) -> None:
        super().__init__()
        channels = c
        depthwise_channels = channels * DW_Expand
        ffn_channels = channels * FFN_Expand
        if depthwise_channels % 2 or ffn_channels % 2:
            raise ValueError("expanded NAFBlock channel counts must be even")

        self.conv1 = nn.Conv2d(channels, depthwise_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(
            depthwise_channels,
            depthwise_channels,
            kernel_size=3,
            padding=1,
            groups=depthwise_channels,
        )
        self.conv3 = nn.Conv2d(depthwise_channels // 2, channels, kernel_size=1)

        # Simplified Channel Attention (SCA).
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                depthwise_channels // 2,
                depthwise_channels // 2,
                kernel_size=1,
            ),
        )
        self.sg = SimpleGate()

        self.conv4 = nn.Conv2d(channels, ffn_channels, kernel_size=1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1)

        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.dropout1 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )
        self.dropout2 = (
            nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        )

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.norm1(inp))
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.dropout1(self.conv3(x))
        residual = inp + x * self.beta

        x = self.conv4(self.norm2(residual))
        x = self.sg(x)
        x = self.dropout2(self.conv5(x))
        return residual + x * self.gamma


class NAFNet(nn.Module):
    """RGB NAFNet with automatic spatial padding and output cropping.

    Args:
        in_chan: Number of input channels. The EDOF model uses RGB (3).
        out_chan: Number of output channels. The EDOF model uses RGB (3).
        width: Base feature width.
        middle_blk_num: Number of NAFBlocks at the bottleneck.
        enc_blk_nums: Number of NAFBlocks in each encoder stage.
        dec_blk_nums: Number of NAFBlocks in each decoder stage.
    """

    def __init__(
        self,
        in_chan: int = 3,
        out_chan: int = 3,
        width: int = 16,
        middle_blk_num: int = 1,
        enc_blk_nums: Sequence[int] = (1, 1, 1, 18),
        dec_blk_nums: Sequence[int] = (1, 1, 1, 1),
    ) -> None:
        super().__init__()
        encoder_counts = tuple(enc_blk_nums)
        decoder_counts = tuple(dec_blk_nums)
        self._validate_configuration(
            in_chan,
            out_chan,
            width,
            middle_blk_num,
            encoder_counts,
            decoder_counts,
        )

        self.intro = nn.Conv2d(in_chan, width, kernel_size=3, padding=1)
        self.ending = nn.Conv2d(width, out_chan, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        channels = width
        for block_count in encoder_counts:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(channels) for _ in range(block_count)])
            )
            self.downs.append(
                nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2)
            )
            channels *= 2

        self.middle_blks = nn.Sequential(
            *[NAFBlock(channels) for _ in range(middle_blk_num)]
        )

        for block_count in decoder_counts:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(channels) for _ in range(block_count)])
            )

        self.padder_size = 2 ** len(self.encoders)
        self.initialize_weights()

    @staticmethod
    def _validate_configuration(
        in_chan: int,
        out_chan: int,
        width: int,
        middle_blk_num: int,
        encoder_counts: tuple[int, ...],
        decoder_counts: tuple[int, ...],
    ) -> None:
        if in_chan <= 0 or out_chan <= 0 or width <= 0:
            raise ValueError("channel counts and width must be positive")
        if in_chan < out_chan:
            raise ValueError("in_chan must be at least out_chan for the residual path")
        if len(encoder_counts) != len(decoder_counts):
            raise ValueError("encoder and decoder stage counts must match")
        if middle_blk_num < 0 or any(
            count < 0 for count in encoder_counts + decoder_counts
        ):
            raise ValueError("NAFBlock counts must be non-negative")

    def initialize_weights(self) -> None:
        """Use the initialization from the pinned End2endImaging version."""

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.ending.weight)
        if self.ending.bias is not None:
            nn.init.zeros_(self.ending.bias)

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        """Pad height and width to the encoder downsampling factor."""

        height, width = x.shape[-2:]
        pad_height = (
            self.padder_size - height % self.padder_size
        ) % self.padder_size
        pad_width = (self.padder_size - width % self.padder_size) % self.padder_size
        return F.pad(x, (0, pad_width, 0, pad_height))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if inp.ndim != 4:
            raise ValueError("NAFNet input must have shape (N, C, H, W)")
        height, width = inp.shape[-2:]
        padded_input = self.check_image_size(inp)

        x = self.intro(padded_input)
        skip_connections = []
        for encoder, downsample in zip(self.encoders, self.downs):
            x = encoder(x)
            skip_connections.append(x)
            x = downsample(x)

        x = self.middle_blks(x)
        for decoder, upsample, skip in zip(
            self.decoders, self.ups, reversed(skip_connections)
        ):
            x = upsample(x)
            x = decoder(x + skip)

        x = self.ending(x)
        x = x + padded_input[:, : x.shape[1], :, :]
        return x[:, :, :height, :width]
