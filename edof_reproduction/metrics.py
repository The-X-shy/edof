"""Paper-aligned reconstruction loss and image-quality metrics."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VGG16PerceptualLoss(nn.Module):
    """VGG16 feature MSE used by the authors' public End2endImaging code."""

    feature_layers = {3, 8, 15, 22, 29}

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        self.features = vgg16(weights=VGG16_Weights.DEFAULT).features.to(device).eval()
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)

    def _activations(self, image: Tensor) -> list[Tensor]:
        activations = []
        for index, layer in enumerate(self.features):
            image = layer(image)
            if index in self.feature_layers:
                activations.append(image)
            if index >= max(self.feature_layers):
                break
        return activations

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction_features = self._activations(prediction)
        with torch.no_grad():
            target_features = self._activations(target)
        return torch.stack(
            [F.mse_loss(left, right) for left, right in zip(prediction_features, target_features)]
        ).sum()


class LPIPSMetric(nn.Module):
    """AlexNet LPIPS metric used by the authors' public evaluation code."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        import lpips

        self.metric = lpips.LPIPS(net="alex").to(device).eval()
        for parameter in self.metric.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction = prediction.clamp(0.0, 1.0) * 2.0 - 1.0
        target = target.clamp(0.0, 1.0) * 2.0 - 1.0
        return self.metric(prediction, target).flatten(1).mean(dim=1)


def batch_psnr(prediction: Tensor, target: Tensor) -> Tensor:
    error = (prediction.clamp(0.0, 1.0) - target).square().flatten(1).mean(dim=1)
    return -10.0 * torch.log10(error.clamp_min(1e-12))


def batch_ssim(prediction: Tensor, target: Tensor) -> Tensor:
    prediction = prediction.clamp(0.0, 1.0)
    mu_x = F.avg_pool2d(prediction, 7, 1, 3)
    mu_y = F.avg_pool2d(target, 7, 1, 3)
    sigma_x = F.avg_pool2d(prediction.square(), 7, 1, 3) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), 7, 1, 3) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, 7, 1, 3) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + 0.01**2) * (2 * sigma_xy + 0.03**2)) / (
        (mu_x.square() + mu_y.square() + 0.01**2) * (sigma_x + sigma_y + 0.03**2)
    )
    return score.flatten(1).mean(dim=1)
