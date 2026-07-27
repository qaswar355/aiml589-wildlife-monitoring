"""
EfficientNet-B3 binary classifier for invasive mammal detection.

Architecture:
  - Backbone: EfficientNet-B3 pretrained on ImageNet (torchvision)
  - Head: single logit output (mammal=1, bird=0) — use
    BCEWithLogitsLoss for training, torch.sigmoid(logit) for a
    probability at inference. Never apply sigmoid twice.

Uncertainty:
  - MC Dropout: predict_proba() with mc_dropout_passes > 1 keeps dropout
    active across N forward passes and returns (mean probability, std
    across passes) as an epistemic uncertainty score. With
    mc_dropout_passes=1 it's a normal deterministic eval-mode forward
    pass and uncertainty is exactly 0.

The model checkpoint and threshold_config.yaml are versioned together in
the MLflow Model Registry so they are always deployed as a pair.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3


class WildlifeClassifier(nn.Module):
    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.3) -> None:
        super().__init__()
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b3(weights=weights)

        in_features = backbone.classifier[-1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 1),
        )
        self.backbone = backbone

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
