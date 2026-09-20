"""
Combines the original LILA-derived manifest (data/manifest.csv) with the
external "birds_and_mammals" manifest (build_external_manifest.py's
output) into one manifest ShardDataset/train.py can read completely
unmodified -- both datasets' images already live in the same
data/shards directory (see pack_external_shards.py), so ShardReader
finds either source's bytes by image_id alone regardless of which
manifest row it came from.

Written to a NEW file (data/manifest_combined.csv by default), never
overwriting data/manifest.csv -- every existing result in this project
(the baseline, box-crop, seasonal experiments, and every metrics/*
comparison) was produced against the original manifest, and overwriting
it would make those results silently unreproducible.

Column reconciliation (external -> combined, matching data/manifest.csv's
own MANIFEST_COLUMNS exactly so nothing downstream needs to know two
schemas exist):
  - project  <- external's `source` column ("external_birds_and_mammals")
               -- plays the same role as `project` does for LILA rows
               (which sub-collection an image came from).
  - box_conf <- 0.0 for every external row (no MegaDetector was run on
               this dataset; has_box is already False, which is the only
               thing ShardDataset's crop_to_bbox actually checks --
               box_conf itself is a build_manifest.py-time filtering
               field, never read again once a manifest exists).
  - file_name <- already set to the same value as image_id for external
               rows (build_external_manifest.py); harmless either way,
               since ShardDataset never reads file_name once images are
               already packed into shards -- only image_id is used to
               look a shard member up.
  - every other column already shares the same name and meaning across
    both manifests (image_id, species, taxon, label, native, wild,
    location, datetime, season, event_id, has_box, bbox, split).

Run: python -m src.external_data.combine_manifests
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_ORIGINAL_MANIFEST = "data/manifest.csv"
DEFAULT_EXTERNAL_MANIFEST = "src/external_data/metadata/external_manifest.csv"
DEFAULT_OUTPUT_PATH = "data/manifest_combined.csv"

COMBINED_COLUMNS = [
    "image_id", "file_name", "species", "taxon", "label", "native", "wild",
    "project", "location", "datetime", "season", "event_id", "has_box",
    "box_conf", "bbox", "split",
]


def combine_manifests(
    original_manifest_path: str = DEFAULT_ORIGINAL_MANIFEST,
    external_manifest_path: str = DEFAULT_EXTERNAL_MANIFEST,
) -> tuple[pd.DataFrame, dict]:
    original = pd.read_csv(original_manifest_path)
    external = pd.read_csv(external_manifest_path)

    overlap = set(original["image_id"]) & set(external["image_id"])
    if overlap:
        raise ValueError(
            f"{len(overlap)} image_id(s) appear in BOTH manifests -- this would corrupt "
            f"the shared shards directory's member lookup. First few: {sorted(overlap)[:5]}"
        )

    external_mapped = external.assign(project=external["source"], box_conf=0.0)[COMBINED_COLUMNS]
    combined = pd.concat([original[COMBINED_COLUMNS], external_mapped], ignore_index=True)

    stats = {
        "n_original": len(original),
        "n_external": len(external),
        "n_combined": len(combined),
        "split_counts": combined["split"].value_counts().to_dict(),
        "label_counts": combined["label"].value_counts().to_dict(),
    }
    return combined, stats


def main() -> None:
    combined, stats = combine_manifests()

    out_path = Path(DEFAULT_OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    print(f"Wrote {out_path} ({stats['n_combined']:,} rows: {stats['n_original']:,} original + "
          f"{stats['n_external']:,} external)")
    print(f"split counts: {stats['split_counts']}")
    print(f"label counts (1=mammal, 0=bird): {stats['label_counts']}")


if __name__ == "__main__":
    main()
