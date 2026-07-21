"""
PyTorch Lightning training loop for the wildlife classifier.

Covers:
  - AdamW optimiser + CosineAnnealingLR scheduler
  - WeightedRandomSampler for per-batch class balance
  - BCEWithLogitsLoss(pos_weight) for gradient correction
  - MLflow autologging (params, metrics, artifacts per epoch)
  - codecarbon EmissionsTracker wrapped around the fit() call
  - Optuna integration for HPO sweeps (called from train.py CLI)

Entry point: `python -m src.training.train --config-name default`
             (Hydra resolves configs/training/default.yaml)

HPC usage: ECS GPU servers are shared, cooperative-etiquette machines with
no job scheduler — run via `tmux`, pinned to a specific GPU, not submitted
to a queue.
"""
