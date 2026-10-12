"""
Run MegaDetector over the external "birds_and_mammals" images, the same
way the live API does (src/inference/predict.py): the highest-confidence
animal box at confidence >= 0.2, or no box if nothing clears it.

The external dataset ships with no MegaDetector boxes at all, so without
this every external image reaches a box-crop model as a full frame -- the
same train/inference mismatch the live API's own MegaDetector step was
added to fix. These boxes let evaluate_external_cropped.py test the
box-crop models the way they are actually served.

Output: src/external_data/metadata/external_md_boxes.csv
(image_id, has_box, bbox), bbox as JSON [x, y, w, h] normalised 0-1,
matching the main manifest's convention.

Run: python -m src.external_data.detect_external_boxes
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from src.external_data.evaluate_external import DEFAULT_IMAGES_ROOT, DEFAULT_MANIFEST_PATH

DEFAULT_BOXES_PATH = "src/external_data/metadata/external_md_boxes.csv"
DEFAULT_CONF_THRESHOLD = 0.2


def detect_boxes(
    manifest: pd.DataFrame,
    images_root: str,
    detector: Any,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    detect_fn=None,
) -> pd.DataFrame:
    """One row per manifest image. `detect_fn` defaults to the serving
    path's own detect_animal_box; it is a parameter so tests can pass a
    fake detector without loading MegaDetector."""
    if detect_fn is None:
        from src.inference.megadetector import detect_animal_box as detect_fn

    rows = []
    for row in manifest.itertuples(index=False):
        image = Image.open(Path(images_root) / row.file_name).convert("RGB")
        bbox = detect_fn(detector, image, conf_threshold=conf_threshold)
        rows.append(
            {
                "image_id": row.image_id,
                "has_box": bbox is not None,
                "bbox": json.dumps([round(v, 4) for v in bbox]) if bbox is not None else "",
            }
        )
    return pd.DataFrame(rows, columns=["image_id", "has_box", "bbox"])


def main() -> None:
    from src.inference.megadetector import load_megadetector

    manifest = pd.read_csv(DEFAULT_MANIFEST_PATH)
    boxes = detect_boxes(manifest, DEFAULT_IMAGES_ROOT, load_megadetector())
    boxes.to_csv(DEFAULT_BOXES_PATH, index=False)
    print(f"{boxes.has_box.mean():.1%} of {len(boxes)} external images have a box -> {DEFAULT_BOXES_PATH}")


if __name__ == "__main__":
    main()
