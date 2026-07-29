"""
Pull specific images out of the shard tars into a plain folder of loose
JPEGs -- for grabbing failure cases (e.g. filtered from
data/predictions/test_predictions.csv) for Grad-CAM++ figures without
unpacking every shard.

Run:
    python -m src.data.extract_images --ids <image_id> [<image_id> ...] --out <dir>
    python -m src.data.extract_images --ids-file <csv-with-image_id-column> --out <dir>
"""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import pandas as pd

from src.training.dataset import safe_member_name


def extract_images(image_ids: list[str], shards_dir: Path | str, out_dir: Path | str) -> list[str]:
    shards_dir = Path(shards_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    member_shard: dict[str, Path] = {}
    for done_file in sorted(shards_dir.glob("shard-*.done")):
        shard_path = done_file.with_suffix(".tar")
        if shard_path.exists():
            with tarfile.open(shard_path) as tar:
                for name in tar.getnames():
                    member_shard[name] = shard_path

    written: list[str] = []
    missing: list[str] = []
    for image_id in image_ids:
        member_name = safe_member_name(image_id)
        shard_path = member_shard.get(member_name)
        if shard_path is None:
            missing.append(image_id)
            continue
        with tarfile.open(shard_path) as tar:
            data = tar.extractfile(member_name).read()
        out_path = out_dir / member_name
        out_path.write_bytes(data)
        written.append(str(out_path))

    if missing:
        preview = missing[:10]
        suffix = " ..." if len(missing) > 10 else ""
        print(f"warning: {len(missing)} image_ids not found in any shard: {preview}{suffix}")
    print(f"wrote {len(written)} images to {out_dir}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="*", default=[], help="image_id values to extract")
    parser.add_argument(
        "--ids-file", type=str, default=None,
        help="CSV with an image_id column (e.g. a filtered predictions.csv slice)",
    )
    parser.add_argument("--shards-dir", type=str, default="data/shards")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    image_ids = list(args.ids)
    if args.ids_file:
        image_ids += pd.read_csv(args.ids_file)["image_id"].tolist()
    if not image_ids:
        raise SystemExit("no image_ids given -- pass --ids or --ids-file")

    extract_images(image_ids, args.shards_dir, args.out)


if __name__ == "__main__":
    main()
