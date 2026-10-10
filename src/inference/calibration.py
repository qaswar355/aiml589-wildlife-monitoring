"""
Post-hoc calibration for the served model -- temperature scaling
(confidence calibration) and MC Dropout uncertainty-cutoff analysis
(review-flagging calibration).

Both operate on the VALIDATION split only, never test: test is reserved
for the final, once-only reported metrics in evaluate.py. Neither of
these touches accuracy/precision/recall/ROC-AUC (see below for why), so
using test wouldn't bias those numbers -- but tuning any operational
parameter against it is worth avoiding on principle for a thesis under
scrutiny, and it keeps a consistent story with how the decision
threshold was calibrated (also validation-derived, in spirit -- computed
from the site-split test predictions that were already the reported
result, since no separate val-set PR analysis existed at the time).

Two genuinely different questions, on purpose (see predict.py's own
docstring reasoning for why the config keeps them as separate values):

  - Temperature scaling (run_temperature_calibration): is the model's
    stated confidence NUMBER trustworthy -- does "90% confident" really
    mean right 90% of the time? A single scalar T rescales the raw
    logit (sigmoid(z / T)) before it's shown to a user. T=1 is
    unchanged; T>1 softens overconfidence. This is provably
    rank-preserving -- dividing every logit by the same positive
    constant can't change which of two logits is bigger -- so it CANNOT
    affect accuracy, ROC-AUC, PR-AUC, or which class gets predicted; it
    only reshapes how loud the confidence number is. The decision
    threshold in configs/inference/threshold_config.yaml was tuned
    against RAW probabilities, so the predicted LABEL must keep using
    raw probability -- only the CONFIDENCE shown to a user should switch
    to the calibrated one. Conflating the two would silently invalidate
    the threshold work already done.

  - MC Dropout uncertainty-cutoff (run_uncertainty_analysis): is the
    review-flagging signal (WildlifeClassifier.predict_proba's std
    across N dropout passes) actually informative about whether a
    prediction is wrong -- and if so, what cutoff trades off review
    workload against how many real errors get caught? Reported as an
    ROC-AUC treating "was this prediction wrong" as the label and "MC
    Dropout std" as the score (is std informative about correctness at
    all, or just noise?), plus a workload/catch-rate table across
    candidate percentile cutoffs -- mirroring evaluate.py's own
    precision/recall-vs-threshold framing for the decision threshold.

Neither function edits threshold_config.yaml automatically -- both
report a recommendation with the evidence behind it and leave the
actual governance call to a human, the same pattern already used for
the decision threshold itself.

Run: python -m src.inference.calibration temperature [architecture]
     python -m src.inference.calibration uncertainty [architecture]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.models.classifier import load_checkpoint
from src.training.dataset import ShardDataset
from src.training.evaluate import DEFAULT_DATA_CONFIG, _load_yaml, _resolve_torch_device

DEFAULT_ARCHITECTURE = "efficientnet_b3"
DEFAULT_THRESHOLD_CONFIG = "configs/inference/threshold_config.yaml"
DEFAULT_MC_DROPOUT_PASSES = 20
DEFAULT_CANDIDATE_PERCENTILES = (80, 90, 95, 99)


def _val_loader(
    manifest_path: str,
    shards_dir: str,
    batch_size: int = 32,
    crop_to_box: bool = False,
    box_crop_padding: float = 0.15,
) -> tuple[DataLoader, ShardDataset]:
    manifest = pd.read_csv(manifest_path)
    val_ds = ShardDataset(
        manifest, shards_dir=shards_dir, split="val", crop_to_box=crop_to_box, box_crop_padding=box_crop_padding
    )
    if len(val_ds) == 0:
        raise RuntimeError(f"No validation images found in {shards_dir} -- has ingest finished?")
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0), val_ds


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Standard ECE: bucket predictions by confidence (max(p, 1-p)), and
    within each bucket compare average confidence to actual accuracy,
    weighted by bucket size. 0 = perfectly calibrated. Uses the
    conventional 0.5 decision rule for "correct" -- this measures
    whether the raw probability itself is a trustworthy probability,
    independent of whatever operational threshold is later chosen for a
    business decision. Pure numpy, unit-tested independently of any
    model (see tests/inference/test_calibration.py)."""
    confidences = np.maximum(probs, 1 - probs)
    predictions = (probs >= 0.5).astype(int)
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
    """Finds the single scalar T minimising binary cross-entropy of
    sigmoid(logits / T) against `labels`, via LBFGS (standard choice for
    this 1-parameter, smooth, near-convex problem -- see Guo et al.
    2017). Optimises log(T) rather than T directly so T = exp(log_T) is
    always positive regardless of the optimiser's steps."""
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return torch.exp(log_temperature).item()


def _collect_val_logits(
    architecture: str,
    checkpoint_path: str,
    manifest_path: str,
    shards_dir: str,
    batch_size: int = 32,
    crop_to_box: bool = False,
    box_crop_padding: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader, _ = _val_loader(manifest_path, shards_dir, batch_size, crop_to_box, box_crop_padding)
    device = _resolve_torch_device()
    model = load_checkpoint(checkpoint_path, architecture=architecture, device=device)

    logits, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits.append(model(x.to(device)).cpu())
            labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def run_temperature_calibration(
    architecture: str = DEFAULT_ARCHITECTURE,
    tag: str | None = None,
    checkpoint_path: str | None = None,
    manifest_path: str | None = None,
    shards_dir: str | None = None,
    n_bins: int = 15,
    crop_to_box: bool = False,
    box_crop_padding: float = 0.15,
) -> dict:
    tag = tag or architecture
    checkpoint_path = checkpoint_path or f"models/checkpoint_{tag}.pt"
    out_dir = Path("metrics") / tag / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = _load_yaml(DEFAULT_DATA_CONFIG)["data"]
    manifest_path = manifest_path or data_cfg["manifest_path"]
    shards_dir = shards_dir or data_cfg["shards_dir"]

    logits, labels = _collect_val_logits(
        architecture, checkpoint_path, manifest_path, shards_dir, crop_to_box=crop_to_box, box_crop_padding=box_crop_padding
    )
    raw_probs = torch.sigmoid(logits).numpy()
    y = labels.numpy()

    nll_before = nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    ece_before = expected_calibration_error(raw_probs, y, n_bins=n_bins)

    temperature = fit_temperature(logits, labels)

    calibrated_probs = torch.sigmoid(logits / temperature).numpy()
    nll_after = nn.functional.binary_cross_entropy_with_logits(logits / temperature, labels).item()
    ece_after = expected_calibration_error(calibrated_probs, y, n_bins=n_bins)

    result = {
        "tag": tag,
        "n_val_images": len(y),
        "temperature": temperature,
        "nll_before": nll_before,
        "nll_after": nll_after,
        "ece_before": ece_before,
        "ece_after": ece_after,
    }
    (out_dir / "temperature.json").write_text(json.dumps(result, indent=2))

    print(f"[{tag}] fitted temperature on {len(y)} validation images: T={temperature:.4f}")
    print(f"  NLL:  before={nll_before:.4f}  after={nll_after:.4f}")
    print(f"  ECE:  before={ece_before:.4f}  after={ece_after:.4f}")
    print(f"wrote {out_dir}/temperature.json")
    return result


# ---------------------------------------------------------------------------
# MC Dropout uncertainty-cutoff analysis
# ---------------------------------------------------------------------------


def compute_candidate_cutoffs(
    uncertainty: np.ndarray, is_error: np.ndarray, percentiles: tuple[int, ...] = DEFAULT_CANDIDATE_PERCENTILES
) -> list[dict]:
    """For each candidate percentile of the uncertainty distribution,
    reports what fraction of all predictions would be flagged, what
    fraction of the ACTUAL errors that flagging would catch, and what
    fraction of what's flagged is actually wrong -- the workload/catch
    rate tradeoff a human picks a cutoff from. Pure numpy, unit-tested
    independently of any model."""
    n = len(uncertainty)
    n_errors = int(is_error.sum())
    candidates = []
    for pct in percentiles:
        cutoff = float(np.percentile(uncertainty, pct))
        flagged = uncertainty >= cutoff
        n_flagged = int(flagged.sum())
        n_errors_caught = int((flagged & (is_error == 1)).sum())
        candidates.append(
            {
                "percentile": pct,
                "cutoff": cutoff,
                "flagged_fraction": n_flagged / n if n else None,
                "error_capture_rate": n_errors_caught / n_errors if n_errors else None,
                "precision_of_flagging": n_errors_caught / n_flagged if n_flagged else None,
            }
        )
    return candidates


def _collect_val_uncertainty(
    architecture: str,
    checkpoint_path: str,
    manifest_path: str,
    shards_dir: str,
    dropout_rate: float,
    mc_dropout_passes: int,
    threshold: float,
    batch_size: int = 32,
    crop_to_box: bool = False,
    box_crop_padding: float = 0.15,
) -> pd.DataFrame:
    loader, _ = _val_loader(manifest_path, shards_dir, batch_size, crop_to_box, box_crop_padding)
    device = _resolve_torch_device()
    model = load_checkpoint(checkpoint_path, architecture=architecture, device=device, dropout_rate=dropout_rate)

    rows = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # Deterministic pass decides pred_label, exactly matching
            # predict.py's own separation -- MC Dropout's spread is an
            # uncertainty signal only, never the label.
            det_prob, _ = model.predict_proba(x, mc_dropout_passes=1)
            _, std_prob = model.predict_proba(x, mc_dropout_passes=mc_dropout_passes)
            pred_label = (det_prob >= threshold).int()
            rows.append(
                pd.DataFrame(
                    {
                        "true_label": y.int().cpu().numpy(),
                        "pred_label": pred_label.cpu().numpy(),
                        "uncertainty": std_prob.cpu().numpy(),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def run_uncertainty_analysis(
    architecture: str = DEFAULT_ARCHITECTURE,
    tag: str | None = None,
    checkpoint_path: str | None = None,
    manifest_path: str | None = None,
    shards_dir: str | None = None,
    model_config_path: str | None = None,
    threshold: float | None = None,
    mc_dropout_passes: int | None = None,
    batch_size: int = 32,
    crop_to_box: bool = False,
    box_crop_padding: float = 0.15,
) -> dict:
    tag = tag or architecture
    checkpoint_path = checkpoint_path or f"models/checkpoint_{tag}.pt"
    out_dir = Path("metrics") / tag / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = _load_yaml(DEFAULT_DATA_CONFIG)["data"]
    manifest_path = manifest_path or data_cfg["manifest_path"]
    shards_dir = shards_dir or data_cfg["shards_dir"]

    model_config_path = model_config_path or f"configs/model/{architecture}.yaml"
    model_cfg = _load_yaml(model_config_path)["model"]
    dropout_rate = model_cfg["dropout_rate"]
    mc_dropout_passes = mc_dropout_passes or model_cfg.get("mc_dropout_passes", DEFAULT_MC_DROPOUT_PASSES)

    threshold_cfg = _load_yaml(DEFAULT_THRESHOLD_CONFIG)
    threshold = threshold if threshold is not None else threshold_cfg["thresholds"]["default"]

    df = _collect_val_uncertainty(
        architecture,
        checkpoint_path,
        manifest_path,
        shards_dir,
        dropout_rate,
        mc_dropout_passes,
        threshold,
        batch_size,
        crop_to_box=crop_to_box,
        box_crop_padding=box_crop_padding,
    )
    df["is_error"] = (df["true_label"] != df["pred_label"]).astype(int)
    df.to_csv(out_dir / "uncertainty_scores.csv", index=False)

    n = len(df)
    n_errors = int(df["is_error"].sum())
    auc = roc_auc_score(df["is_error"], df["uncertainty"]) if 0 < n_errors < n else None
    candidates = compute_candidate_cutoffs(df["uncertainty"].to_numpy(), df["is_error"].to_numpy())

    result = {
        "tag": tag,
        "n_val_images": n,
        "n_errors": n_errors,
        "error_rate": n_errors / n if n else None,
        "mc_dropout_passes": mc_dropout_passes,
        "threshold_used": threshold,
        "uncertainty_vs_error_auc": auc,
        "mean_uncertainty_correct": float(df.loc[df["is_error"] == 0, "uncertainty"].mean()) if n - n_errors else None,
        "mean_uncertainty_incorrect": float(df.loc[df["is_error"] == 1, "uncertainty"].mean()) if n_errors else None,
        "candidate_cutoffs": candidates,
    }
    (out_dir / "uncertainty.json").write_text(json.dumps(result, indent=2))

    print(f"[{tag}] MC Dropout uncertainty analysis on {n} validation images "
          f"({n_errors} errors, {result['error_rate']:.2%})")
    if n_errors:
        print(f"  mean uncertainty -- correct: {result['mean_uncertainty_correct']:.4f}  "
              f"incorrect: {result['mean_uncertainty_incorrect']:.4f}")
    print(f"  AUC (uncertainty predicting error): {auc:.4f}" if auc is not None else "  AUC: not computable")
    print("  candidate cutoffs:")
    for c in candidates:
        print(f"    p{c['percentile']}: cutoff={c['cutoff']:.4f}  flagged={c['flagged_fraction']:.2%}  "
              f"catches {c['error_capture_rate']:.2%} of errors  precision={c['precision_of_flagging']:.2%}")
    print(f"wrote {out_dir}/uncertainty.json, uncertainty_scores.csv")
    return result


def main() -> None:
    import sys

    if len(sys.argv) < 2 or sys.argv[1] not in ("temperature", "uncertainty"):
        raise SystemExit("usage: python -m src.inference.calibration <temperature|uncertainty> [architecture]")
    mode = sys.argv[1]
    architecture = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ARCHITECTURE
    if mode == "temperature":
        run_temperature_calibration(architecture=architecture)
    else:
        run_uncertainty_analysis(architecture=architecture)


if __name__ == "__main__":
    main()
