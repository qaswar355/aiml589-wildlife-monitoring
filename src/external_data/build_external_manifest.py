"""
Build a manifest for the supervisor-provided "birds_and_mammals" external
test set -- a genuinely held-out evaluation set (Ryan pelifeed 2026,
different reserves/cameras/equipment than the LILA NZ trail-cam dataset
everything else was trained on), used to check how far real-world
performance drops when the model sees data it has never encountered in
any form.

Reuses the project's existing species -> taxon -> label resolution
(src.data.species_map.SpeciesMapper, configs/data/species_taxonomy.yaml +
class_map.yaml) rather than duplicating it, and reuses the existing
burst-event reconstruction (src.data.events.assign_events) for the same
reason burst reconstruction mattered for the main dataset: "rats: 933
rows" is almost certainly a handful of real encounters photographed many
times, not 933 independent test cases, and treating them as independent
would badly overstate this eval's real sample size.

Species-name normalisation (SPECIES_NAME_MAP below) is needed because
this external manifest's species column uses different conventions than
species_taxonomy.yaml (plural/mixed-case English names, and Maori names
for two species already present in the taxonomy under their English
name) -- see that dict for the specific reasoning per entry.
species_taxonomy.yaml itself is never touched: its own header comment is
explicit that it must match the LILA dataset's exact category strings,
so any translation for a different source dataset belongs here, not
there.

A train/val/test split is also assigned here (site_split, reused verbatim
from build_manifest.py -- same test_size/val_size/seed, read from
configs/data/default.yaml's `split` section, for a consistent
methodology across both datasets), grouped on the same compound
`location` (reserve + bait_station) used for event reconstruction below.
This matters for retraining on combined data: without a real held-out
slice of the external set, "did retraining on combined data fix the
external-distribution gap" could only ever be checked against images the
model was also trained on, which would be leakage, not a real answer.

Run: python -m src.external_data.build_external_manifest
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.data.build_manifest import assign_seasons, site_split
from src.data.events import assign_events
from src.data.species_map import SpeciesMapper

DEFAULT_METADATA_CSV = Path("src/external_data/metadata/manifest.csv")
DEFAULT_OUTPUT_PATH = Path("src/external_data/metadata/external_manifest.csv")
DEFAULT_DATA_CONFIG = Path("configs/data/default.yaml")
DEFAULT_BURST_WINDOW_SECONDS = 60
SOURCE_NAME = "external_birds_and_mammals"

# Raw `species` value (as it appears in the supervisor's manifest) ->
# the exact key species_taxonomy.yaml uses for the same animal.
#   - rats/possums/blackbirds/cats: plural -> singular, same species.
#   - Kaka: case only.
#   - Piwakawaka: the Maori name for fantail -- both refer to the same
#     bird (Rhipidura fuliginosa); species_taxonomy.yaml lists it as
#     "fantail" (the LILA dataset's category string), not "piwakawaka".
#   - popokotea: the Maori name for whitehead (Mohoua albicilla);
#     species_taxonomy.yaml lists it as "whitehead".
#   - tui, thrush, tieke: already match the taxonomy key as-is.
#   - CannotID: not a species at all (unidentifiable in the source
#     photo) -- handled separately as a drop, not a rename.
#   - hihi: present in this manifest's species_columns metadata (so it
#     COULD appear) but 0 rows in the actual data; not in
#     species_taxonomy.yaml at all (stitchbird has no LILA equivalent).
#     Left unmapped deliberately -- if it ever appears, it should be
#     reported and dropped, not silently guessed.
SPECIES_NAME_MAP: dict[str, str] = {
    "rats": "rat",
    "Kaka": "kaka",
    "possums": "possum",
    "blackbirds": "blackbird",
    "cats": "cat",
    "tui": "tui",
    "thrush": "thrush",
    "tieke": "tieke",
    "Piwakawaka": "fantail",
    "popokotea": "whitehead",
}

# Not a species at all -- dropped unconditionally, never passed to
# SpeciesMapper.
NOT_A_SPECIES = {"CannotID"}


def normalise_species(raw_species: pd.Series) -> pd.Series:
    """Maps raw manifest species strings onto species_taxonomy.yaml's
    keys via SPECIES_NAME_MAP; anything not in the map (including
    NOT_A_SPECIES and any genuinely new/unexpected value) passes through
    unchanged so build_external_manifest's caller can detect and report
    it explicitly, rather than this function silently guessing."""
    return raw_species.map(lambda s: SPECIES_NAME_MAP.get(s, s))


def build_external_manifest(
    metadata_csv: Path | str = DEFAULT_METADATA_CSV,
    burst_window_seconds: int = DEFAULT_BURST_WINDOW_SECONDS,
    mapper: SpeciesMapper | None = None,
    data_config_path: Path | str = DEFAULT_DATA_CONFIG,
) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(metadata_csv)
    n_total_raw = len(raw)

    mapper = mapper or SpeciesMapper()

    raw = raw.assign(orig_species=raw["species"], species=normalise_species(raw["species"]))

    not_a_species_mask = raw["orig_species"].isin(NOT_A_SPECIES)
    n_not_a_species = int(not_a_species_mask.sum())
    kept = raw[~not_a_species_mask].copy()

    taxonomy_species = set(mapper.taxonomy.keys())
    unmapped_mask = ~kept["species"].isin(taxonomy_species)
    unmapped_species = sorted(kept.loc[unmapped_mask, "species"].unique().tolist())
    n_unmapped = int(unmapped_mask.sum())
    kept = kept[~unmapped_mask].copy()

    kept["taxon"] = kept["species"].map(mapper.taxon_of)
    kept["native"] = kept["species"].map(lambda s: mapper.taxonomy[s]["native"])
    kept["wild"] = kept["species"].map(lambda s: mapper.taxonomy[s]["wild"])
    kept["label"] = kept["species"].map(mapper.to_label)

    n_excluded_by_class_map = int(kept["label"].isna().sum())
    kept = kept[kept["label"].notna()].copy()
    kept["label"] = kept["label"].astype(int)

    # Compound location: bait_station codes are reused across reserves
    # (the manifest's own naming_rule notes 810 original filenames
    # collide across reserves for this reason), so reserve alone or
    # bait_station alone would conflate genuinely different cameras.
    kept["location"] = kept["reserve"] + "_" + kept["bait_station"]
    kept["image_id"] = kept["new_relpath"]
    kept["file_name"] = kept["new_relpath"]
    kept["colour"] = kept["colour"].str.lower()

    kept = assign_events(
        kept,
        burst_window_seconds=burst_window_seconds,
        location_col="location",
        species_col="species",
        datetime_col="last_modified",
        id_col="image_id",
    )
    kept = kept.rename(columns={"last_modified": "datetime"})
    kept["season"] = assign_seasons(kept["datetime"])

    with open(data_config_path) as f:
        split_cfg = yaml.safe_load(f)["data"]["split"]
    kept["split"] = site_split(
        kept,
        group_col="location",
        test_size=split_cfg["test_size"],
        val_size=split_cfg["val_size"],
        seed=split_cfg["seed"],
    )

    n_events = kept["event_id"].nunique()

    manifest = kept[
        [
            "image_id", "file_name", "species", "orig_species", "taxon", "label", "native", "wild",
            "colour", "reserve", "bait_station", "location", "deployment", "datetime", "season",
            "event_id", "has_timestamp", "size_kb", "orig_filename", "split",
        ]
    ].reset_index(drop=True)
    manifest["source"] = SOURCE_NAME
    # No MegaDetector run against this dataset -- has_box=False/bbox=""
    # matches src.training.dataset.crop_to_bbox's own "no trustworthy
    # box" fallback (full frame), so box-crop checkpoints can still be
    # evaluated against this manifest without any special-casing.
    manifest["has_box"] = False
    manifest["bbox"] = ""

    stats = {
        "n_total_raw": n_total_raw,
        "n_not_a_species_dropped": n_not_a_species,
        "n_unmapped_species_dropped": n_unmapped,
        "unmapped_species": unmapped_species,
        "n_excluded_by_class_map": n_excluded_by_class_map,
        "n_final": len(manifest),
        "n_events": int(n_events),
    }
    return manifest, stats


def _print_summary(manifest: pd.DataFrame, stats: dict) -> None:
    print("=" * 70)
    print("EXTERNAL MANIFEST BUILD SUMMARY (birds_and_mammals)")
    print("=" * 70)
    print(f"raw rows in metadata manifest:          {stats['n_total_raw']:>8,}")
    print(f"  dropped - not a species (CannotID):   {stats['n_not_a_species_dropped']:>8,}")
    print(f"  dropped - unmapped species:            {stats['n_unmapped_species_dropped']:>8,}  "
          f"{stats['unmapped_species']}")
    print(f"  dropped - excluded by class_map.yaml:  {stats['n_excluded_by_class_map']:>8,}")
    print()
    print(f"FINAL EXTERNAL MANIFEST ROWS: {stats['n_final']:,}")
    print(f"reconstructed events (real encounters): {stats['n_events']:,}")
    print()

    print("-- label distribution --")
    print(manifest["label"].value_counts().rename({1: "mammal (1)", 0: "bird (0)"}).to_string())
    print()

    print("-- species --")
    print(manifest["species"].value_counts().to_string())
    print()

    print("-- colour (IR vs RGB) --")
    print(manifest["colour"].value_counts().to_string())
    print()

    print("-- reserve --")
    print(manifest["reserve"].value_counts().to_string())
    print()

    print("-- split (site-based, reusing configs/data/default.yaml's split section) --")
    print(manifest["split"].value_counts().to_string())
    print()

    frames_per_event = manifest.groupby("event_id").size()
    print("-- frames per reconstructed event --")
    print(f"  mean: {frames_per_event.mean():.2f}  median: {frames_per_event.median():.0f}  "
          f"max: {frames_per_event.max()}")
    print("=" * 70)


def main() -> None:
    manifest, stats = build_external_manifest()

    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(DEFAULT_OUTPUT_PATH, index=False)
    print(f"Wrote {DEFAULT_OUTPUT_PATH} ({len(manifest):,} rows)\n")

    _print_summary(manifest, stats)


if __name__ == "__main__":
    main()
