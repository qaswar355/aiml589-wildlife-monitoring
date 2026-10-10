"""
Live MegaDetector wrapper -- finds where the animal is in an uploaded
image before the classifier sees it, closing the gap serving.yaml
documents: the served checkpoint (efficientnet_b3_boxcrop_retrained) was
trained mostly on MegaDetector-cropped images, but predict_one() never
cropped a live upload because no live detector was wired in.

Wraps megadetector==5.0.3's detection.pytorch_detector.PTDetector (see
requirements.in for why this exact version is pinned -- newer releases
pull in a legacy sub-dependency that hard-conflicts with this project's
tensorboard/mlflow/opentelemetry stack) with two fixes over the raw
library, both scoped narrowly and documented inline:
  1. load_megadetector() temporarily monkeypatches torch.load to pass
     weights_only=False -- PyTorch >=2.6 defaults that to True, which
     refuses this checkpoint's pickled models.yolo.Model class. Safe here
     because the weights come from MegaDetector's own official GitHub
     release, not user input, and the patch is only live for the single
     rd.load_detector() call at startup.
  2. detect_animal_box() passes an already-loaded, EXIF-corrected PIL
     Image rather than a path string -- generate_detections_one_image
     silently produces a 0-d array and crashes downstream in letterbox()
     if given a path (this version's docstring says it wants a PIL Image;
     the path-based convenience wrapper other megadetector versions offer
     doesn't exist in 5.0.3).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import requests
import torch
from PIL import Image, ImageOps

# The exact MegaDetector version data/md_boxes.parquet's training-time
# boxes were generated with (see configs/data/default.yaml's
# md_results_json), not the newer v5a.0.1 patch -- keeps live detection
# and training-time detection on identical weights.
DEFAULT_MODEL_URL = "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt"
DEFAULT_MODEL_PATH = "models/md_v5a.0.0.pt"
ANIMAL_CATEGORY = "1"  # MegaDetector's output convention: 1=animal, 2=person, 3=vehicle


def ensure_model_downloaded(model_path: str = DEFAULT_MODEL_PATH, url: str = DEFAULT_MODEL_URL) -> str:
    """Downloads MegaDetector's official weights to model_path if not
    already present. Presence of the file is treated as sufficient --
    not re-verified against a checksum on every call."""
    path = Path(model_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        tmp_path.rename(path)
    return str(path)


def load_megadetector(model_path: str = DEFAULT_MODEL_PATH) -> Any:
    """Loads MegaDetector 5.0.3's PTDetector, downloading the weights
    first if needed. Call once at server startup, like the classifier
    checkpoint -- not per request."""
    resolved_path = ensure_model_downloaded(model_path)

    from detection import run_detector as rd

    original_torch_load = torch.load
    torch.load = lambda *a, **kw: original_torch_load(*a, **{**kw, "weights_only": False})
    try:
        return rd.load_detector(resolved_path)
    finally:
        torch.load = original_torch_load


def detect_animal_box(detector: Any, image: Image.Image, conf_threshold: float = 0.2) -> list[float] | None:
    """Highest-confidence animal detection's bbox ([x, y, w, h], normalised
    0-1, matching this project's manifest convention) whose confidence
    clears conf_threshold, or None if MegaDetector found nothing that
    confident -- matching training-time crop_to_bbox's fallback to the
    full, uncropped frame."""
    oriented = ImageOps.exif_transpose(image.convert("RGB"))
    result = detector.generate_detections_one_image(oriented, "live_upload", detection_threshold=conf_threshold)

    animal_detections = [d for d in result.get("detections", []) if d["category"] == ANIMAL_CATEGORY]
    if not animal_detections:
        return None
    return max(animal_detections, key=lambda d: d["conf"])["bbox"]
