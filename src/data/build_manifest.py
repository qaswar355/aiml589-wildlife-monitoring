"""
Build data/manifest.csv from the LILA NZ trail-cam metadata.

Replaces the old filter_and_sample.py. Inputs are data/nz_flat.parquet
(flattened COCO metadata from src/eda/eda01.py) and data/md_boxes.parquet
(MegaDetector v5a RDE results from src/eda/eda02_box.py) — both metadata
artifacts only, no image bytes.

Pipeline:
  1. species -> taxon -> label via SpeciesMapper (raises on any unmapped
     species; drops rows whose taxon is excluded or, under
     mammal_scope: wild_only, whose species is not wild).
  2. Southern-hemisphere season assignment; rows with no timestamp get
     season=None and are logged, not dropped (they're still usable for the
     standard site-split experiment, just excluded from the seasonal one).
  3. Burst-event reconstruction (src/data/events.py).
  4. Join MegaDetector boxes; drop images MD failed to process entirely
     (corrupt/unreadable — would break ingest). This happens before event
     sampling so a failed image can never be chosen as an event's
     representative frame.
  5. Sample at event level with a symmetric per-species cap, preferring the
     highest-confidence frame per event.
  6. Site-aware train/val/test split (GroupShuffleSplit on location).
  7. Print a summary: class balance (both mammal_scope views), season and
     split distribution, events vs frames, rows dropped by reason.

Run: python -m src.data.build_manifest
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

from src.data.events import assign_events
from src.data.species_map import SpeciesMapper

DEFAULT_CONFIG_PATH = Path("configs/data/default.yaml")

_SEASON_BY_MONTH = {
    9: "spring", 10: "spring", 11: "spring",     # NZ spring
    12: "summer", 1: "summer", 2: "summer",      # NZ summer
    3: "autumn", 4: "autumn", 5: "autumn",       # NZ autumn
    6: "winter", 7: "winter", 8: "winter",       # NZ winter
}

MANIFEST_COLUMNS = [
    "image_id", "file_name", "species", "taxon", "label", "native", "wild",
    "project", "location", "datetime", "season", "event_id", "has_box",
    "box_conf", "bbox", "split",
]


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)["data"]


def assign_seasons(datetime_series: pd.Series) -> pd.Series:
    """Southern-hemisphere season from a datetime-like column.

    Spring: Sep/Oct/Nov  Summer: Dec/Jan/Feb
    Autumn: Mar/Apr/May  Winter: Jun/Jul/Aug

    Returns NaN (not a guess) where the timestamp is missing/unparseable.
    """
    parsed = pd.to_datetime(datetime_series, errors="coerce")
    return parsed.dt.month.map(_SEASON_BY_MONTH)


def load_md_failures(raw_json_path: Path | str, cache_path: Path | str | None = None) -> set[str]:
    """image_ids MegaDetector failed to process (corrupt/unreadable source
    image) — these would break streaming ingest and must be dropped, not
    just given a zero-detection box. Cached after the first (slow,
    streaming) pass over the raw results file."""
    raw_json_path = Path(raw_json_path)
    cache_path = Path(cache_path) if cache_path else raw_json_path.parent / "md_failures.csv"

    if cache_path.exists():
        return set(pd.read_csv(cache_path)["image_id"])

    if not raw_json_path.exists():
        raise FileNotFoundError(
            f"{raw_json_path} not found — run src/eda/eda02_box.py first to "
            "download and extract the MegaDetector RDE results file."
        )

    import ijson

    failures = []
    with raw_json_path.open("rb") as f:
        for img in ijson.items(f, "images.item"):
            reason = img.get("failure")
            if reason is not None:
                failures.append((img["file"], reason))

    out = pd.DataFrame(failures, columns=["image_id", "failure_reason"])
    out.to_csv(cache_path, index=False)
    return set(out["image_id"])


def sample_events_per_species(
    df: pd.DataFrame,
    max_images_per_species: int,
    frames_per_event: int,
    species_col: str = "species",
    event_col: str = "event_id",
    conf_col: str = "_select_conf",
) -> pd.DataFrame:
    """Cap each species at `max_images_per_species` images, sampling whole
    events (not raw frames) so a cap of N images reflects roughly N distinct
    animal encounters rather than N frames of a handful of bursts.

    Within an event, the top `frames_per_event` frames by confidence are
    kept. Events are then admitted in order of their best frame's
    confidence, highest first, until the cap is reached. The same rule
    (same cap, same tie-break) applies to every species — mammal or bird.
    """
    parts = []
    for _, group in df.groupby(species_col, sort=False):
        g = group.sort_values(conf_col, ascending=False)
        g = g.assign(_rank_in_event=g.groupby(event_col).cumcount())
        g = g[g["_rank_in_event"] < frames_per_event].drop(columns=["_rank_in_event"])

        event_best = g.groupby(event_col)[conf_col].max().sort_values(ascending=False)
        event_size = g.groupby(event_col).size().reindex(event_best.index)
        cum = event_size.cumsum()
        keep_events = event_size.index[cum <= max_images_per_species]

        parts.append(g[g[event_col].isin(keep_events)])
    return pd.concat(parts, ignore_index=False)


def site_split(
    df: pd.DataFrame,
    group_col: str = "location",
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> pd.Series:
    """Site-aware train/val/test split — no single group_col value spans
    two splits.

    Note: rows with group_col == "unknown" (camera location not recorded)
    are treated as one group for split purposes, so they all land in the
    same split together. This is conservative (guarantees no leakage) at
    the cost of losing some site diversity for that slice; they are a
    small fraction of the data (~1%) and this is logged, not hidden.
    """
    n = len(df)
    groups = df[group_col].astype(str).to_numpy()
    positions = np.arange(n)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_pos, test_pos = next(gss1.split(positions, groups=groups))

    relative_val = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed)
    train_pos2, val_pos2 = next(gss2.split(trainval_pos, groups=groups[trainval_pos]))

    train_pos = trainval_pos[train_pos2]
    val_pos = trainval_pos[val_pos2]

    split = np.empty(n, dtype=object)
    split[train_pos] = "train"
    split[val_pos] = "val"
    split[test_pos] = "test"
    return pd.Series(split, index=df.index, name="split")


def build_manifest(cfg: dict) -> tuple[pd.DataFrame, dict]:
    flat = pd.read_parquet(cfg["flat_metadata_path"])
    n_total_raw = len(flat)

    mapper = SpeciesMapper(cfg["taxonomy_path"], cfg["class_map_path"])
    mapper.validate_exhaustive(set(flat["category"].unique()))

    flat = flat.rename(columns={"category": "species"})

    # 1. species -> taxon -> label (raises on any unmapped species)
    flat = flat.assign(
        taxon=flat["species"].map(mapper.taxon_of),
        native=flat["species"].map(lambda s: mapper.taxonomy[s]["native"]),
        wild=flat["species"].map(lambda s: mapper.taxonomy[s]["wild"]),
        label=flat["species"].map(mapper.to_label),
    )

    n_excluded_other = int((flat["taxon"] == "other").sum())
    n_excluded_wild_only = int((flat["label"].isna() & (flat["taxon"] != "other")).sum())

    kept = flat[flat["label"].notna()].copy()
    kept["label"] = kept["label"].astype(int)
    kept["project"] = kept["file_name"].str.split("/").str[0]

    n_unknown_location = int((kept["location"] == "unknown").sum())

    # 2. season
    kept["season"] = assign_seasons(kept["datetime"])
    n_missing_season = int(kept["season"].isna().sum())
    missing_season_by_label = kept.loc[kept["season"].isna(), "label"].value_counts().to_dict()

    # 3. burst events
    kept = assign_events(kept, burst_window_seconds=cfg["burst_window_seconds"])

    # 4. MD boxes + drop images MD failed to process, before event sampling
    md_boxes = pd.read_parquet(cfg["md_boxes_path"])
    failures = load_md_failures(cfg["source"]["md_results_json"])
    n_md_failed = int(kept["image_id"].isin(failures).sum())
    kept = kept[~kept["image_id"].isin(failures)].copy()

    kept = kept.merge(md_boxes, on="image_id", how="left", validate="one_to_one")
    if kept["max_conf"].isna().any():
        n_missing = int(kept["max_conf"].isna().sum())
        raise ValueError(
            f"{n_missing} manifest rows have no MegaDetector result — "
            f"{cfg['md_boxes_path']} does not cover the filtered dataset"
        )
    kept = kept.rename(columns={"max_conf": "box_conf"})
    kept["has_box"] = kept["box_conf"] >= cfg["md_conf_threshold"]
    # Negative box_conf marks an RDE-suppressed detection (likely a static
    # false trigger) — never preferred as an event's representative frame.
    kept["_select_conf"] = kept["box_conf"].clip(lower=0)

    n_frames_before_cap = len(kept)
    n_events_before_cap = kept["event_id"].nunique()

    # 5. event-level, per-species-capped sampling
    kept = sample_events_per_species(
        kept,
        max_images_per_species=cfg["max_images_per_species"],
        frames_per_event=cfg["frames_per_event"],
    )

    n_frames_after_cap = len(kept)
    n_events_after_cap = kept["event_id"].nunique()

    # 6. site-aware split
    kept["split"] = site_split(
        kept,
        group_col=cfg["split"]["group_col"],
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        seed=cfg["split"]["seed"],
    )

    def _bbox_to_json(b):
        if b is None:
            return ""
        try:
            if len(b) == 0:
                return ""
        except TypeError:
            return ""
        return json.dumps([float(x) for x in b])

    kept["bbox"] = kept["bbox"].apply(_bbox_to_json)

    manifest = kept[MANIFEST_COLUMNS].reset_index(drop=True)

    stats = {
        "n_total_raw": n_total_raw,
        "n_excluded_other_taxon": n_excluded_other,
        "n_excluded_wild_only": n_excluded_wild_only,
        "n_unknown_location": n_unknown_location,
        "n_missing_season": n_missing_season,
        "missing_season_by_label": missing_season_by_label,
        "n_md_failed": n_md_failed,
        "n_events_before_cap": int(n_events_before_cap),
        "n_frames_before_cap": int(n_frames_before_cap),
        "n_events_after_cap": int(n_events_after_cap),
        "n_frames_after_cap": int(n_frames_after_cap),
    }
    return manifest, stats


def _print_summary(manifest: pd.DataFrame, stats: dict) -> None:
    print("=" * 70)
    print("MANIFEST BUILD SUMMARY")
    print("=" * 70)
    print(f"raw rows in nz_flat.parquet:            {stats['n_total_raw']:>10,}")
    print(f"  dropped — taxon 'other' (excluded):    {stats['n_excluded_other_taxon']:>10,}")
    print(f"  dropped — wild_only domestic exclusion:{stats['n_excluded_wild_only']:>10,}")
    print(f"  dropped — MD failed to process image:  {stats['n_md_failed']:>10,}")
    print(f"rows with location == 'unknown':         {stats['n_unknown_location']:>10,}  "
          "(grouped as one pseudo-site for the split — see site_split docstring)")
    print(f"rows with no parseable timestamp:        {stats['n_missing_season']:>10,}  "
          f"(kept in manifest, season=None, excluded from seasonal experiment; "
          f"by label: {stats['missing_season_by_label']})")
    print()
    print(f"events before per-species cap: {stats['n_events_before_cap']:>10,}   "
          f"frames before cap: {stats['n_frames_before_cap']:>10,}")
    print(f"events after per-species cap:  {stats['n_events_after_cap']:>10,}   "
          f"frames after cap:  {stats['n_frames_after_cap']:>10,}")
    print()
    print(f"FINAL MANIFEST ROWS: {len(manifest):,}")
    print()

    print("-- class balance (current class_map.yaml labelling) --")
    print(manifest["label"].value_counts().rename({1: "mammal (1)", 0: "bird (0)"}).to_string())
    print()

    print("-- class balance under mammal_scope: wild_only (derived, not a re-run) --")
    wild_only = manifest[manifest["wild"] == True]  # noqa: E712 — explicit elementwise compare
    print(wild_only["taxon"].value_counts().to_string())
    print(f"  (drops {int((manifest['wild'] == False).sum())} domestic/livestock rows: "  # noqa: E712
          f"{sorted(manifest.loc[manifest['wild'] == False, 'species'].unique().tolist())})")  # noqa: E712
    print()

    print("-- images per season --")
    print(manifest["season"].value_counts(dropna=False).to_string())
    print()

    print("-- images per split --")
    print(manifest["split"].value_counts().to_string())
    print()

    per_species = manifest["species"].value_counts()
    print(f"-- species retained: {per_species.size} --")
    print(f"species with < 200 images: {int((per_species < 200).sum())} "
          f"{sorted(per_species[per_species < 200].index.tolist())}")
    print(f"species with < 20 images:  {int((per_species < 20).sum())} "
          f"{sorted(per_species[per_species < 20].index.tolist())}")
    print("=" * 70)


def main() -> None:
    cfg = load_config()
    manifest, stats = build_manifest(cfg)

    out_path = Path(cfg["manifest_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(manifest):,} rows)\n")

    _print_summary(manifest, stats)


if __name__ == "__main__":
    main()
