"""End-to-end smoke test of the /predict route: a tiny untrained model, a
real tiny JPEG, and create_app() pointed entirely at tmp_path configs so
this never touches the real models/ or configs/ directories."""
from pathlib import Path

import pandas as pd
import torch
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from src.inference.app import create_app
from src.models.classifier import WildlifeClassifier


def _build_fake_serving_setup(tmp_path):
    checkpoint_path = tmp_path / "ckpt.pt"
    model = WildlifeClassifier(architecture="efficientnet_b3", pretrained=False, dropout_rate=0.3)
    torch.save(model.state_dict(), checkpoint_path)

    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text(
        yaml.dump(
            {
                "model": {
                    "architecture": "efficientnet_b3",
                    "pretrained": False,
                    "num_classes": 1,
                    "dropout_rate": 0.3,
                    "mc_dropout_passes": 2,  # small on purpose -- test speed, not accuracy
                }
            }
        )
    )

    heatmap_dir = tmp_path / "heatmaps"
    prediction_log_path = tmp_path / "serving_log.csv"
    serving_config_path = tmp_path / "serving.yaml"
    serving_config_path.write_text(
        yaml.dump(
            {
                "serving": {
                    "tag": "test",
                    "architecture": "efficientnet_b3",
                    "checkpoint_path": str(checkpoint_path),
                    "model_config_path": str(model_config_path),
                    "heatmap_dir": str(heatmap_dir),
                    "prediction_log_path": str(prediction_log_path),
                }
            }
        )
    )

    threshold_config_path = tmp_path / "threshold.yaml"
    threshold_config_path.write_text(
        yaml.dump(
            {
                "thresholds": {"default": 0.5},
                "recall_floor": 0.9,
                "uncertainty_review_cutoff": 0.15,
            }
        )
    )

    image_path = tmp_path / "img.jpg"
    Image.new("RGB", (50, 50), color=(120, 80, 40)).save(image_path)

    return serving_config_path, threshold_config_path, image_path, heatmap_dir, prediction_log_path


def test_predict_endpoint_returns_prediction(tmp_path):
    serving_cfg, threshold_cfg, image_path, _heatmap_dir, log_path = _build_fake_serving_setup(tmp_path)
    app = create_app(serving_config_path=str(serving_cfg), threshold_config_path=str(threshold_cfg))

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"images": [{"image_path": str(image_path)}], "return_heatmaps": False},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "test"
    assert body["threshold_used"] == 0.5

    pred = body["predictions"][0]
    assert pred["label"] in (0, 1)
    # confidence is "how sure is the model in the label it picked", i.e.
    # max(p, 1-p) -- always >= 0.5 by construction. A regression back to
    # returning raw P(mammal) here would make this fail for bird
    # predictions (label=0) roughly half the time.
    assert 0.5 <= pred["confidence"] <= 1.0
    assert pred["uncertainty"] >= 0.0
    assert isinstance(pred["requires_review"], bool)
    assert pred["heatmap_path"] is None

    assert log_path.exists()
    log_df = pd.read_csv(log_path)
    assert len(log_df) == 1
    assert log_df.iloc[0]["label"] == pred["label"]


def test_predict_endpoint_with_heatmap_returns_a_real_file(tmp_path):
    serving_cfg, threshold_cfg, image_path, _heatmap_dir, _log_path = _build_fake_serving_setup(tmp_path)
    app = create_app(serving_config_path=str(serving_cfg), threshold_config_path=str(threshold_cfg))

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"images": [{"image_path": str(image_path)}], "return_heatmaps": True},
        )

    assert response.status_code == 200
    heatmap_path = response.json()["predictions"][0]["heatmap_path"]
    assert heatmap_path is not None
    assert Path(heatmap_path).exists()


def test_predict_endpoint_rejects_empty_images_list(tmp_path):
    serving_cfg, threshold_cfg, _image_path, _heatmap_dir, _log_path = _build_fake_serving_setup(tmp_path)
    app = create_app(serving_config_path=str(serving_cfg), threshold_config_path=str(threshold_cfg))

    with TestClient(app) as client:
        response = client.post("/predict", json={"images": [], "return_heatmaps": False})

    assert response.status_code == 422


def test_model_info_endpoint(tmp_path):
    serving_cfg, threshold_cfg, _image_path, _heatmap_dir, _log_path = _build_fake_serving_setup(tmp_path)
    app = create_app(serving_config_path=str(serving_cfg), threshold_config_path=str(threshold_cfg))

    with TestClient(app) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["tag"] == "test"
    assert body["architecture"] == "efficientnet_b3"
    assert body["uncertainty_review_cutoff"] == 0.15
