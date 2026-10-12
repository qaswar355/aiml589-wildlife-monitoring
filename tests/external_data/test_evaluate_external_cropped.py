"""Tests for the pure logic in evaluate_external_cropped.py -- loading the
boxes, cropping, and threshold metrics -- on tiny synthetic data."""
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.external_data.evaluate_external_cropped import (
    CroppedImageDataset,
    load_external_with_boxes,
    threshold_metrics,
)
from src.training.dataset import DEFAULT_TRANSFORM


def test_threshold_metrics_matches_hand_counts():
    y_true = [0, 0, 0, 1, 1, 1, 1]
    y_prob = [0.1, 0.2, 0.6, 0.9, 0.8, 0.7, 0.3]

    m = threshold_metrics(y_true, y_prob, 0.5)

    assert m["false_positives"] == 1
    assert m["false_negatives"] == 1
    assert m["accuracy"] == pytest.approx(5 / 7)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)


def test_threshold_metrics_uses_greater_or_equal():
    m = threshold_metrics([1, 0], [0.95, 0.94], 0.95)
    assert m["false_negatives"] == 0  # 0.95 counts as mammal
    assert m["false_positives"] == 0


def test_load_external_with_boxes_replaces_box_columns(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    boxes_path = tmp_path / "boxes.csv"
    pd.DataFrame({"image_id": ["a", "b"], "label": [1, 0], "has_box": [False, False], "bbox": ["", ""]}).to_csv(
        manifest_path, index=False
    )
    pd.DataFrame({"image_id": ["a", "b"], "has_box": [True, False], "bbox": ["[0.1, 0.1, 0.5, 0.5]", ""]}).to_csv(
        boxes_path, index=False
    )

    df = load_external_with_boxes(str(manifest_path), str(boxes_path))

    assert df["has_box"].tolist() == [True, False]
    assert df.loc[0, "bbox"] == "[0.1, 0.1, 0.5, 0.5]"


def test_load_external_with_boxes_raises_when_an_image_has_no_result(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    boxes_path = tmp_path / "boxes.csv"
    pd.DataFrame({"image_id": ["a", "b"], "label": [1, 0]}).to_csv(manifest_path, index=False)
    pd.DataFrame({"image_id": ["a"], "has_box": [True], "bbox": ["[0, 0, 1, 1]"]}).to_csv(boxes_path, index=False)

    with pytest.raises(ValueError, match="no MegaDetector result"):
        load_external_with_boxes(str(manifest_path), str(boxes_path))


def _two_colour_image(root):
    root.mkdir()
    pixels = np.zeros((100, 100, 3), dtype=np.uint8)
    pixels[:, :50] = [255, 0, 0]
    pixels[:, 50:] = [0, 0, 255]
    Image.fromarray(pixels).save(root / "x.png")
    return Image.open(root / "x.png").convert("RGB")


def test_cropped_dataset_crops_to_the_box(tmp_path):
    root = tmp_path / "images"
    image = _two_colour_image(root)
    df = pd.DataFrame({"file_name": ["x.png"], "has_box": [True], "bbox": ["[0.6, 0.1, 0.3, 0.8]"]})

    tensor = CroppedImageDataset(df, str(root), padding=0.0)[0]

    expected = DEFAULT_TRANSFORM(image.crop((60, 10, 90, 90)))
    assert tensor.shape == (3, 300, 300)
    assert torch.allclose(tensor, expected)


def test_cropped_dataset_uses_full_frame_without_a_box(tmp_path):
    root = tmp_path / "images"
    image = _two_colour_image(root)
    df = pd.DataFrame({"file_name": ["x.png"], "has_box": [False], "bbox": [""]})

    tensor = CroppedImageDataset(df, str(root), padding=0.0)[0]

    assert torch.allclose(tensor, DEFAULT_TRANSFORM(image))
