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


class ShardDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        shards_dir: Path | str = "data/shards",
        split: str | None = None,
        transform=DEFAULT_TRANSFORM,
    ) -> None:
        self.transform = transform
        self.shards_dir = Path(shards_dir)

        df = manifest if split is None else manifest[manifest["split"] == split]
        df = df.assign(_member_name=df["image_id"].map(safe_member_name))

        self._member_info, self._member_shard = self._index_shards()
        df = df[df["_member_name"].isin(self._member_info)].reset_index(drop=True)
        self._rows = df
        self._tar_handles: dict[Path, tarfile.TarFile] = {}

    def _index_shards(self) -> tuple[dict[str, tarfile.TarInfo], dict[str, Path]]:
        """One getmembers() scan per finished shard, cached for O(1) lookups
        in __getitem__ (tar has no native index, so this avoids an O(n)
        linear re-scan on every single item)."""
        member_info: dict[str, tarfile.TarInfo] = {}
        member_shard: dict[str, Path] = {}
        for done_file in sorted(self.shards_dir.glob("shard-*.done")):
            shard_path = done_file.with_suffix(".tar")
            if not shard_path.exists():
                continue
            with tarfile.open(shard_path) as tar:
                for info in tar.getmembers():
                    member_info[info.name] = info
                    member_shard[info.name] = shard_path
        return member_info, member_shard

    def __len__(self) -> int:
        return len(self._rows)

    def _tar_for(self, shard_path: Path) -> tarfile.TarFile:
        handle = self._tar_handles.get(shard_path)
        if handle is None:
            handle = tarfile.open(shard_path)
            self._tar_handles[shard_path] = handle
        return handle

    def __getitem__(self, idx: int):
        row = self._rows.iloc[idx]
        member_name = row["_member_name"]
        shard_path = self._member_shard[member_name]
        info = self._member_info[member_name]

        tar = self._tar_for(shard_path)
        data = tar.extractfile(info).read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(float(row["label"]))
        return img, label
