"""
EfficientNet-B3 binary classifier for invasive mammal detection.

Architecture:
  - Backbone: EfficientNet-B3 pretrained on ImageNet (torchvision)
  - Head: single sigmoid output (mammal=1, bird=0)
  - Loss: BCEWithLogitsLoss with pos_weight for class imbalance correction

Uncertainty:
  - MC Dropout enabled at inference time (dropout layers kept active)
  - N forward passes → mean prediction + epistemic uncertainty score

The model checkpoint and threshold_config.yaml are versioned together
in the MLflow Model Registry so they are always deployed as a pair.
"""
