"""
Site-aware train/val/test splitting with a seasonal generalisation experiment.

Primary split:
  GroupShuffleSplit on camera_site_id — no site spans train and test,
  preventing spatial leakage. 70 / 15 / 15 site-level split.

Seasonal experiment (thesis contribution):
  COCO metadata timestamps parsed to extract season per image.
  Secondary split trains on Spring/Summer only and tests on Autumn/Winter —
  directly quantifies the seasonal generalisation gap and provides empirical
  justification for drift-triggered retraining.

Seasonal distribution is logged to MLflow for every data version.
"""
