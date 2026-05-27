"""
Integrated Gradients attribution (captum).

Provides pixel-level attribution as cross-validation against Grad-CAM.
Run on the test set after training to confirm that both methods agree
on which image regions drive the classification decision.

Result logged as an MLflow artifact for thesis documentation.
"""
