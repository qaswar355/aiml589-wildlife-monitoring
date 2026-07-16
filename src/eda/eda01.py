"""
EDA Step 1 — Flatten the LILA NZ trail-cam COCO metadata, then inventory categories.
 
Run once. Produces `nz_flat.parquet`, which every later EDA step reads instead of
re-parsing the JSON.
 
Usage:
    python eda_01_flatten_and_inventory.py /path/to/nz_trailcams.json
 
Outputs:
    nz_flat.parquet          one row per (image, annotation)
    category_inventory.csv   category name, id, image count, % of total
    ... and prints a paste-friendly summary to stdout.
"""
 
import json
import sys
from collections import Counter
from pathlib import Path
 
import pandas as pd
 
JSON_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "trail_camera_images_of_new_zealand_animals_1.00.json")
OUT_PARQUET = Path("data/nz_flat.parquet")
OUT_CSV = Path("data/category_inventory.csv")


def main() -> None:
    print(f"Loading {JSON_PATH} ...", flush=True)
    with JSON_PATH.open() as f:
        coco = json.load(f)
 
    # --- 0. What does this file actually contain? Don't assume. -------------
    print(f"\nTop-level keys: {sorted(coco.keys())}")
    for key in ("images", "annotations", "categories"):
        print(f"  {key:12s} n = {len(coco.get(key, [])):,}")
 
    if coco.get("images"):
        print(f"\nSample image record:\n  {json.dumps(coco['images'][0], indent=2)}")
    if coco.get("annotations"):
        print(f"\nSample annotation record:\n  {json.dumps(coco['annotations'][0], indent=2)}")
 
    # --- 1. Flatten ---------------------------------------------------------
    cats = {c["id"]: c["name"] for c in coco["categories"]}
 
    images = pd.DataFrame(coco["images"])
    annots = pd.DataFrame(coco["annotations"])
 
    # Keep only the columns we care about, tolerating absent ones.
    img_cols = [
        c for c in
        ("id", "file_name", "location", "datetime", "seq_id", "seq_num_frames", "frame_num")
        if c in images.columns
    ]
    ann_cols = [c for c in ("image_id", "category_id") if c in annots.columns]
 
    missing_img = {"location", "datetime", "seq_id"} - set(images.columns)
    if missing_img:
        print(f"\n!! WARNING: images lack expected columns: {sorted(missing_img)}")
        print("   (site / season / sequence analysis depends on these — flag this.)")
 
    df = annots[ann_cols].merge(
        images[img_cols], left_on="image_id", right_on="id", how="left", suffixes=("", "_img")
    )
    df["category"] = df["category_id"].map(cats)
    df = df.drop(columns=[c for c in ("id",) if c in df.columns])
 
    # --- 2. Multi-label check ----------------------------------------------
    # Some camera-trap images carry >1 annotation (two species in frame).
    # This matters: a naive image-level label would silently pick one.
    per_image = Counter(df["image_id"])
    multi = sum(1 for v in per_image.values() if v > 1)
    print(f"\nImages with >1 annotation: {multi:,} "
          f"({multi / max(len(per_image), 1):.2%} of {len(per_image):,} annotated images)")
 
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"\nWrote {OUT_PARQUET}  ({OUT_PARQUET.stat().st_size / 1e6:.1f} MB, {len(df):,} rows)")
 
    # --- 3. THE CATEGORY INVENTORY — this is the thing to paste back --------
    inv = (
        df.groupby("category")["image_id"]
        .nunique()
        .sort_values(ascending=False)
        .rename("n_images")
        .reset_index()
    )
    inv["pct"] = 100 * inv["n_images"] / inv["n_images"].sum()
    inv["cum_pct"] = inv["pct"].cumsum()
    inv.to_csv(OUT_CSV, index=False)
 
    print(f"\n{'=' * 64}\nCATEGORY INVENTORY  ({len(inv)} categories)\n{'=' * 64}")
    print(f"{'category':<32}{'n_images':>12}{'pct':>8}{'cum%':>8}")
    print("-" * 64)
    for r in inv.itertuples():
        print(f"{r.category:<32}{r.n_images:>12,}{r.pct:>7.2f}%{r.cum_pct:>7.1f}%")
    print("=" * 64)
    print(f"\nTotal annotated images: {inv['n_images'].sum():,}")
 
 
if __name__ == "__main__":
    main()