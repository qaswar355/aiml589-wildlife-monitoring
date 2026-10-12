"""Tests for detect_external_boxes.py using a fake detector, not the real
MegaDetector model."""
import json

import pandas as pd
from PIL import Image

from src.external_data.detect_external_boxes import detect_boxes


def _write_images(root, names):
    root.mkdir()
    for name in names:
        Image.new("RGB", (40, 30)).save(root / name)


def test_detect_boxes_records_box_or_no_box_per_image(tmp_path):
    images_root = tmp_path / "images"
    _write_images(images_root, ["a.jpg", "b.jpg"])
    manifest = pd.DataFrame({"image_id": ["a", "b"], "file_name": ["a.jpg", "b.jpg"]})

    # First image gets a box, second gets nothing.
    results = iter([[0.1, 0.2, 0.3, 0.4], None])
    boxes = detect_boxes(manifest, str(images_root), detector=None, detect_fn=lambda d, i, conf_threshold: next(results))

    assert boxes["image_id"].tolist() == ["a", "b"]
    assert boxes["has_box"].tolist() == [True, False]
    assert json.loads(boxes.loc[0, "bbox"]) == [0.1, 0.2, 0.3, 0.4]
    assert boxes.loc[1, "bbox"] == ""


def test_detect_boxes_leaves_bbox_empty_when_nothing_detected(tmp_path):
    images_root = tmp_path / "images"
    _write_images(images_root, ["a.jpg"])
    manifest = pd.DataFrame({"image_id": ["a"], "file_name": ["a.jpg"]})

    boxes = detect_boxes(manifest, str(images_root), detector=None, detect_fn=lambda d, i, conf_threshold: None)

    assert boxes.loc[0, "has_box"] == False  # noqa: E712 -- numpy bool
    assert boxes.loc[0, "bbox"] == ""


def test_detect_boxes_passes_the_confidence_threshold_through(tmp_path):
    images_root = tmp_path / "images"
    _write_images(images_root, ["a.jpg"])
    manifest = pd.DataFrame({"image_id": ["a"], "file_name": ["a.jpg"]})
    seen = []

    detect_boxes(
        manifest, str(images_root), detector=None, conf_threshold=0.2,
        detect_fn=lambda d, i, conf_threshold: seen.append(conf_threshold),
    )

    assert seen == [0.2]
