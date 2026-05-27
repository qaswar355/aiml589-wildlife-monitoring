"""
Model evaluation: metrics, threshold sweep, and seasonal ablation.

Metrics reported (accuracy intentionally downweighted):
  - Precision, Recall, F1, AUC-ROC, AUC-PR
  - Confusion matrix in absolute counts (ecological interpretation)
  - ROC and PR curves saved as MLflow artifacts

Threshold sweep:
  precision_recall_curve → select operating threshold per deployment zone
  Result written to configs/inference/threshold_config.yaml

Seasonal ablation (three comparisons for thesis):
  1. Standard site-split baseline
  2. Seasonal split (Spring/Summer train → Autumn/Winter test)
  3. Post-retraining recovery

OOD detection:
  MC Dropout uncertainty flags images outside training distribution
  (e.g. deer, cattle). These surface in the Streamlit review queue
  with high uncertainty rather than silently misclassifying.
"""
