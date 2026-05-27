"""
Grad-CAM and Grad-CAM++ heatmap generation (pytorch-grad-cam).

Wraps the EfficientNet-B3 backbone to produce per-image visual explanations.
Heatmaps are overlaid on the original camera trap image and saved as artifacts.

Used in two contexts:
  1. Streamlit dashboard — shown per image in the practitioner review queue
  2. MLflow artifacts — logged for every evaluation run for academic review

Target layer: the last convolutional block of EfficientNet-B3.
"""
