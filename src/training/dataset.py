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
    ) -> None:
        self.transform = transform
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
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(float(row["label"]))
        return img, label
