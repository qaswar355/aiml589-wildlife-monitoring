"""
PyTorch Dataset reading resized JPEGs directly out of the tar shards
produced by src/data/ingest.py, joined against manifest.csv for labels
and split assignment.

Only shards with a `.done` sentinel are considered — a shard still being
written by a concurrent ingest run is skipped, not read, so training can
start against whatever prefix of shards has completed so far and pick up
more as ingest continues.
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def safe_member_name(image_id: str) -> str:
    """Must match src/data/ingest.py's _safe_member_name exactly."""
    return image_id.replace("/", "__").rsplit(".", 1)[0] + ".jpg"


def crop_to_bbox(image: Image.Image, row, padding: float = 0.15) -> Image.Image:
    """Crops `image` to its MegaDetector box (row['bbox'], a JSON
    [x, y, w, h] string normalised 0-1), expanded by `padding` as a
    fraction of the box's own width/height on each side, clamped to the
    image bounds.

    Falls back to the full, uncropped image whenever there's no
    trustworthy box: `has_box` is False (a low/negative-confidence
    detection -- see build_manifest.py, ~0.5% of the manifest), the bbox
    is missing/malformed, or the box has zero area. Every manifest row
    actually carries *some* bbox string, but has_box is the real "is this
    detection worth trusting" flag, so that's what gates the crop, not
    just "is bbox non-empty".
    """
    if not row.get("has_box", False):
        return image
    bbox_json = row.get("bbox")
    if not isinstance(bbox_json, str) or not bbox_json:
        return image
    try:
        x, y, w, h = json.loads(bbox_json)
    except (ValueError, TypeError):
        return image
    if w <= 0 or h <= 0:
        return image

    img_w, img_h = image.size
    pad_x, pad_y = w * padding, h * padding
    x0, y0 = max(0.0, x - pad_x), max(0.0, y - pad_y)
    x1, y1 = min(1.0, x + w + pad_x), min(1.0, y + h + pad_y)

    left, top = int(x0 * img_w), int(y0 * img_h)
    right, bottom = int(x1 * img_w), int(y1 * img_h)
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def index_shards(shards_dir: Path | str) -> tuple[dict[str, tarfile.TarInfo], dict[str, Path]]:
    """Scan every finished shard once and remember where each image lives.

    Tar files have no built-in index, so without this every single read
    would mean re-scanning the whole archive to find one file. We pay
    that cost once per shard here instead. Used by ShardReader below, and
    by anything else (extract_images.py, the XAI tooling) that needs to
    pull specific images out of the shards.
    """
    shards_dir = Path(shards_dir)
    member_info: dict[str, tarfile.TarInfo] = {}
    member_shard: dict[str, Path] = {}
    for done_file in sorted(shards_dir.glob("shard-*.done")):
        shard_path = done_file.with_suffix(".tar")
        if not shard_path.exists():
            continue
        with tarfile.open(shard_path) as tar:
            for info in tar.getmembers():
                member_info[info.name] = info
                member_shard[info.name] = shard_path
    return member_info, member_shard


class ShardReader:
    """Pulls raw image bytes out of indexed shards, keeping tar handles
    open across repeated reads rather than reopening a shard every time."""

    def __init__(self, shards_dir: Path | str) -> None:
        self.member_info, self.member_shard = index_shards(shards_dir)
        self._tar_handles: dict[Path, tarfile.TarFile] = {}

    def __contains__(self, member_name: str) -> bool:
        return member_name in self.member_info

    def read(self, member_name: str) -> bytes:
        shard_path = self.member_shard[member_name]
        tar = self._tar_handles.get(shard_path)
        if tar is None:
            tar = tarfile.open(shard_path)
            self._tar_handles[shard_path] = tar
        return tar.extractfile(self.member_info[member_name]).read()


class ShardDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        shards_dir: Path | str = "data/shards",
        split: str | None = None,
        transform=DEFAULT_TRANSFORM,
        crop_to_box: bool = False,
        box_crop_padding: float = 0.15,
    ) -> None:
        self.transform = transform
        self.crop_to_box = crop_to_box
        self.box_crop_padding = box_crop_padding
        self.reader = ShardReader(shards_dir)

        df = manifest if split is None else manifest[manifest["split"] == split]
        df = df.assign(_member_name=df["image_id"].map(safe_member_name))
        df = df[df["_member_name"].isin(self.reader.member_info)].reset_index(drop=True)
        self._rows = df

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int):
        row = self._rows.iloc[idx]
        data = self.reader.read(row["_member_name"])
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if self.crop_to_box:
            img = crop_to_bbox(img, row, padding=self.box_crop_padding)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(float(row["label"]))
        return img, label
