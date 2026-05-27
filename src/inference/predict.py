"""
Core inference logic — model loading and batch prediction.

Design:
  - Model is loaded once at API startup (not per request)
  - Batch of images processed together for efficiency
  - MC Dropout: N forward passes per image → mean confidence + uncertainty
  - Threshold read from configs/inference/threshold_config.yaml at startup
  - Images flagged as requires_review when confidence < threshold
"""
