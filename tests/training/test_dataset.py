"""Tests for crop_to_bbox -- the box-crop experiment's core preprocessing
step. Pure PIL/logic, no shards or a real model needed."""
import json

import pandas as pd
from PIL import Image

from src.training.dataset import crop_to_bbox


def _row(bbox=None, has_box=True):
    return pd.Series({"bbox": json.dumps(bbox) if bbox is not None else "", "has_box": has_box})


def test_crop_to_bbox_returns_full_image_when_has_box_false():
    image = Image.new("RGB", (100, 100))
    row = _row(bbox=[0.2, 0.2, 0.3, 0.3], has_box=False)
    assert crop_to_bbox(image, row) is image


def test_crop_to_bbox_returns_full_image_when_bbox_empty():
    image = Image.new("RGB", (100, 100))
    row = _row(bbox=None, has_box=True)
    assert crop_to_bbox(image, row) is image


def test_crop_to_bbox_returns_full_image_when_bbox_malformed():
    image = Image.new("RGB", (100, 100))
    row = pd.Series({"bbox": "not json", "has_box": True})
    assert crop_to_bbox(image, row) is image


def test_crop_to_bbox_returns_full_image_for_degenerate_box():
    image = Image.new("RGB", (100, 100))
    row = _row(bbox=[0.5, 0.5, 0.0, 0.0], has_box=True)
    assert crop_to_bbox(image, row) is image


def test_crop_to_bbox_applies_padding_and_crops_correctly():
    image = Image.new("RGB", (100, 100))
    # box spans [0.4, 0.6] x [0.4, 0.6]; padding=0.5 adds 0.1 on each side
    row = _row(bbox=[0.4, 0.4, 0.2, 0.2], has_box=True)
    cropped = crop_to_bbox(image, row, padding=0.5)
    # expected bounds: x0=0.3->30px, x1=0.7->70px (same for y) -> 40x40
    assert cropped.size == (40, 40)


def test_crop_to_bbox_clamps_to_image_bounds_near_edge():
    image = Image.new("RGB", (100, 100))
    # box right at the top-left corner; padding would push past 0
    row = _row(bbox=[0.0, 0.0, 0.1, 0.1], has_box=True)
    cropped = crop_to_bbox(image, row, padding=1.0)
    # x0 clamps to 0; x1 = 0.1 + 0.1*1.0 = 0.2 -> 20px
    assert cropped.size == (20, 20)
