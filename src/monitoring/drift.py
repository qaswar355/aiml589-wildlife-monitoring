"""
ML drift detection using Evidently AI.

Three drift signals monitored:
  1. Confidence drift    — distribution of model confidence scores shifts
                           (DataDriftPreset on confidence column)
  2. Concept drift       — precision/recall degrades on labelled batches
                           (scipy.stats.ks_2samp on feature distributions)
  3. XAI centroid drift  — Grad-CAM attention regions shift across batches
                           (custom: mean heatmap centroid tracked over time)

When any signal exceeds its threshold, a GitHub Actions retraining workflow
is triggered automatically (retrain.yml). Datadog receives the drift metric
for alerting and dashboard display.

Drift reports are saved as Evidently HTML artifacts in MLflow.
"""
