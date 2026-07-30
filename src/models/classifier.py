"""
Binary wildlife classifier — architecture-agnostic wrapper around a
torchvision backbone, single logit output (mammal=1, bird=0).

Supports EfficientNet-B3 (the primary model) and ResNet-50 (the
literature-comparison baseline) behind the same interface, so
train.py/evaluate.py don't need to know which one they're using --
they just read `architecture` out of whichever configs/model/*.yaml
was pointed at.

Use BCEWithLogitsLoss for training, torch.sigmoid(logit) for a
probability at inference. Never apply sigmoid twice.

MC Dropout: predict_proba() with mc_dropout_passes > 1 keeps dropout
active across N forward passes and returns (mean probability, std
across passes) as an epistemic uncertainty score. With
mc_dropout_passes=1 it's a normal deterministic eval-mode forward pass
and uncertainty is exactly 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B3_Weights,
    ResNet50_Weights,
    efficientnet_b3,
    resnet50,
)

SUPPORTED_ARCHITECTURES = ("efficientnet_b3", "resnet50")


def _build_backbone(architecture: str, pretrained: bool, dropout_rate: float) -> nn.Module:
    if architecture == "efficientnet_b3":
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b3(weights=weights)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 1),
        )
        return backbone

    if architecture == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 1),
        )
        return backbone

    raise ValueError(
        f"Unknown architecture {architecture!r} -- expected one of {SUPPORTED_ARCHITECTURES}"
    )


class WildlifeClassifier(nn.Module):
    def __init__(
        self,
        architecture: str = "efficientnet_b3",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.backbone = _build_backbone(architecture, pretrained, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(-1)  # (batch,) raw logits

    def predict_proba(
        self, x: torch.Tensor, mc_dropout_passes: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.train(mc_dropout_passes > 1)  # keep dropout active only for MC passes

        with torch.no_grad():
            passes = torch.stack(
                [torch.sigmoid(self.forward(x)) for _ in range(max(1, mc_dropout_passes))]
            )

        self.train(was_training)
        return passes.mean(dim=0), passes.std(dim=0)
