"""Tests for the pure logic in evaluate_external.py -- metrics
computation and PlainImageDataset -- using synthetic labels/predictions
or a tiny generated image, not the real ~1,900-image external set."""
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.external_data.evaluate_external import PlainImageDataset, _compute_metrics, run_external_evaluation
from src.models.classifier import WildlifeClassifier


def test_compute_metrics_matches_hand_counted_confusion_matrix():
    # 2 true birds (0) correctly called bird, 1 true bird wrongly called
    # mammal, 3 true mammals correctly called mammal, 1 true mammal
    # wrongly called bird.
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.6, 0.9, 0.8, 0.7, 0.3])

    metrics = _compute_metrics(y_true, y_pred, y_prob)

    assert metrics["n_images"] == 7
    assert metrics["confusion_matrix"] == [[2, 1], [1, 3]]
    assert metrics["accuracy"] == pytest.approx(5 / 7)
    assert metrics["recall"] == pytest.approx(3 / 4)  # 3 of 4 true mammals caught
    assert "roc_auc" in metrics  # both classes present


def test_compute_metrics_skips_auc_for_single_class_slice():
    # An event-level slice can end up all-one-class (e.g. every frame in
    # a reconstructed event is the same true label) -- AUC needs both
    # classes, so it should be omitted rather than raising or returning
    # a meaningless value.
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.4])

    metrics = _compute_metrics(y_true, y_pred, y_prob)
    assert "roc_auc" not in metrics
    assert "pr_auc" not in metrics
    assert metrics["accuracy"] == pytest.approx(2 / 3)


def test_plain_image_dataset_reads_real_file_and_applies_transform(tmp_path):
    images_root = tmp_path / "images"
    images_root.mkdir()
    Image.new("RGB", (50, 80), color=(10, 20, 30)).save(images_root / "sample.jpg")

    manifest = pd.DataFrame({"file_name": ["sample.jpg"], "label": [1]})
    ds = PlainImageDataset(manifest, str(images_root))

    assert len(ds) == 1
    image, label = ds[0]
    assert image.shape == (3, 300, 300)  # DEFAULT_TRANSFORM resizes to 300x300
    assert torch.equal(label, torch.tensor(1.0))


def test_plain_image_dataset_length_matches_manifest_rows(tmp_path):
    images_root = tmp_path / "images"
    images_root.mkdir()
    for i in range(3):
        Image.new("RGB", (20, 20)).save(images_root / f"img{i}.jpg")

    manifest = pd.DataFrame({"file_name": [f"img{i}.jpg" for i in range(3)], "label": [0, 1, 0]})
    ds = PlainImageDataset(manifest, str(images_root))
    assert len(ds) == 3


@pytest.fixture
def tiny_split_fixture(tmp_path):
    """A tiny real checkpoint + tiny real images + a manifest with a
    'split' column, for exercising run_external_evaluation's
    split_filter end-to-end (not just PlainImageDataset in isolation)."""
    images_root = tmp_path / "images"
    images_root.mkdir()
    for i in range(4):
        Image.new("RGB", (40, 40), color=(i * 10, i * 10, i * 10)).save(images_root / f"img{i}.jpg")

    manifest = pd.DataFrame(
        {
            "image_id": [f"img{i}.jpg" for i in range(4)],
            "file_name": [f"img{i}.jpg" for i in range(4)],
            "label": [0, 1, 0, 1],
            "split": ["train", "train", "test", "test"],
            "event_id": [f"evt{i}" for i in range(4)],
            "datetime": ["2026-01-01"] * 4,
            "species": ["kaka", "rat", "kaka", "rat"],
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    model = WildlifeClassifier(architecture="efficientnet_b3", pretrained=False)
    checkpoint_path = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), checkpoint_path)

    return {"manifest_path": str(manifest_path), "images_root": str(images_root), "checkpoint_path": str(checkpoint_path)}


def test_run_external_evaluation_split_filter_restricts_to_matching_rows(tiny_split_fixture, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result = run_external_evaluation(
        architecture="efficientnet_b3",
        tag="test_tag",
        checkpoint_path=tiny_split_fixture["checkpoint_path"],
        manifest_path=tiny_split_fixture["manifest_path"],
        images_root=tiny_split_fixture["images_root"],
        threshold=0.5,
        split_filter="test",
    )
    assert result["n_frames"] == 2  # only the two split="test" rows
    assert result["split_filter"] == "test"


def test_run_external_evaluation_no_split_filter_uses_every_row(tiny_split_fixture, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result = run_external_evaluation(
        architecture="efficientnet_b3",
        tag="test_tag",
        checkpoint_path=tiny_split_fixture["checkpoint_path"],
        manifest_path=tiny_split_fixture["manifest_path"],
        images_root=tiny_split_fixture["images_root"],
        threshold=0.5,
        split_filter=None,
    )
    assert result["n_frames"] == 4  # all rows, matching the original (pre-retrain) drift-eval use case


def test_run_external_evaluation_raises_on_split_filter_matching_nothing(tiny_split_fixture, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="0 rows"):
        run_external_evaluation(
            architecture="efficientnet_b3",
            tag="test_tag",
            checkpoint_path=tiny_split_fixture["checkpoint_path"],
            manifest_path=tiny_split_fixture["manifest_path"],
            images_root=tiny_split_fixture["images_root"],
            threshold=0.5,
            split_filter="val",  # no val rows in the fixture
        )
