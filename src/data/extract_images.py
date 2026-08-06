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
from pathlib import Path

import pandas as pd

from src.training.dataset import ShardReader, safe_member_name


def extract_images(image_ids: list[str], shards_dir: Path | str, out_dir: Path | str) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = ShardReader(shards_dir)

    written: list[str] = []
    missing: list[str] = []
    for image_id in image_ids:
        member_name = safe_member_name(image_id)
        if member_name not in reader:
            missing.append(image_id)
            continue
        out_path = out_dir / member_name
        out_path.write_bytes(reader.read(member_name))
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
