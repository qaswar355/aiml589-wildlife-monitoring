"""Tests for the pure logic in build_external_manifest.py -- species-name
normalisation and the end-to-end manifest build, using the project's
real taxonomy/class-map configs (no mocking of them, since the whole
point is that this maps correctly onto the actual mammal_vs_bird
scheme) but a small synthetic metadata CSV rather than the real
1,900-row one, so this stays fast and self-contained."""
import pandas as pd

from src.data.species_map import SpeciesMapper
from src.external_data.build_external_manifest import (
    NOT_A_SPECIES,
    SPECIES_NAME_MAP,
    build_external_manifest,
    normalise_species,
)


def test_normalise_species_maps_known_aliases():
    raw = pd.Series(["rats", "Kaka", "possums", "blackbirds", "cats", "Piwakawaka", "popokotea"])
    normalised = normalise_species(raw)
    assert normalised.tolist() == ["rat", "kaka", "possum", "blackbird", "cat", "fantail", "whitehead"]


def test_normalise_species_leaves_unknown_values_unchanged():
    # Anything not in SPECIES_NAME_MAP passes through as-is, so the
    # caller can detect and report it rather than this function
    # silently guessing at a mapping.
    raw = pd.Series(["tui", "some_never_seen_species"])
    normalised = normalise_species(raw)
    assert normalised.tolist() == ["tui", "some_never_seen_species"]


def test_species_name_map_targets_are_all_real_taxonomy_entries():
    """Every SPECIES_NAME_MAP value must actually exist in
    species_taxonomy.yaml -- a typo here would silently produce an
    "unmapped species" drop instead of the intended mapping."""
    mapper = SpeciesMapper()
    for raw_name, mapped_name in SPECIES_NAME_MAP.items():
        assert mapped_name in mapper.taxonomy, f"{raw_name!r} -> {mapped_name!r} is not in species_taxonomy.yaml"


def _synthetic_metadata(tmp_path):
    df = pd.DataFrame(
        {
            "orig_relpath": [f"orig/{i}.JPG" for i in range(6)],
            "new_relpath": [
                "rats/IR/POW_B6_dep1_001.JPG",
                "rats/IR/POW_B6_dep1_002.JPG",  # same station+species, close in time -> same event as row 0
                "Kaka/RGB/HP_G5_dep1_010.JPG",
                "CannotID/IR/HP_G5_dep1_011.JPG",  # dropped: not a species
                "cats/IR/WP_K7_dep2_020.JPG",
                "hihi/RGB/WP_K7_dep2_021.JPG",  # dropped: no taxonomy entry at all
            ],
            "species": ["rats", "rats", "Kaka", "CannotID", "cats", "hihi"],
            "count": [1] * 6,
            "colour": ["IR", "IR", "RGB", "IR", "IR", "RGB"],
            "reserve": ["POW", "POW", "HP", "HP", "WP", "WP"],
            "bait_station": ["B6", "B6", "G5", "G5", "K7", "K7"],
            "deployment": ["dep1"] * 4 + ["dep2"] * 2,
            "orig_filename": [f"{i}.JPG" for i in range(6)],
            "extension": [".jpg"] * 6,
            "size_kb": [500.0] * 6,
            "last_modified": [
                "2026-07-01 10:00:00",
                "2026-07-01 10:00:30",  # 30s after row 0 -> same burst event
                "2026-07-02 09:00:00",
                "2026-07-02 09:00:05",
                "2026-08-01 12:00:00",
                "2026-08-01 12:00:05",
            ],
            "notes": [None] * 6,
            "source_row": list(range(6)),
        }
    )
    path = tmp_path / "manifest.csv"
    df.to_csv(path, index=False)
    return path


def test_build_external_manifest_drops_cannotid_and_unmapped_species(tmp_path):
    metadata_csv = _synthetic_metadata(tmp_path)
    manifest, stats = build_external_manifest(metadata_csv)

    assert stats["n_total_raw"] == 6
    assert stats["n_not_a_species_dropped"] == 1  # CannotID
    assert stats["n_unmapped_species_dropped"] == 1  # hihi
    assert stats["unmapped_species"] == ["hihi"]
    assert stats["n_final"] == 4
    assert "CannotID" not in manifest["species"].tolist()
    assert "hihi" not in manifest["species"].tolist()


def test_build_external_manifest_assigns_correct_labels(tmp_path):
    metadata_csv = _synthetic_metadata(tmp_path)
    manifest, _ = build_external_manifest(metadata_csv)

    by_species = manifest.set_index("species")["label"].to_dict()
    assert by_species["rat"] == 1  # mammal
    assert by_species["cat"] == 1  # mammal
    assert by_species["kaka"] == 0  # bird


def test_build_external_manifest_reconstructs_bursts_within_a_station(tmp_path):
    metadata_csv = _synthetic_metadata(tmp_path)
    manifest, stats = build_external_manifest(metadata_csv, burst_window_seconds=60)

    rat_rows = manifest[manifest["species"] == "rat"]
    assert len(rat_rows) == 2
    # 30s apart, same station+species, within the 60s window -> one event
    assert rat_rows["event_id"].nunique() == 1

    # 4 kept rows (rat x2, kaka, cat), but the two rat rows collapse to one
    # event -> 3 distinct events total.
    assert stats["n_events"] == 3


def test_build_external_manifest_sets_has_box_false_for_every_row(tmp_path):
    """No MegaDetector was run on this dataset -- crop_to_bbox's own
    fallback (src.training.dataset) is gated on has_box, so this must be
    False for every row, not just missing/empty."""
    metadata_csv = _synthetic_metadata(tmp_path)
    manifest, _ = build_external_manifest(metadata_csv)
    assert (manifest["has_box"] == False).all()  # noqa: E712


def test_build_external_manifest_assigns_a_valid_split_to_every_row(tmp_path):
    metadata_csv = _synthetic_metadata(tmp_path)
    manifest, _ = build_external_manifest(metadata_csv)
    assert manifest["split"].isin(["train", "val", "test"]).all()
    assert manifest["split"].notna().all()


def test_build_external_manifest_split_has_no_location_leakage(tmp_path):
    """The whole point of a site-based split: no single location (camera
    station) may appear in more than one split, or a held-out 'test'
    evaluation would partly be testing on cameras the model trained on."""
    metadata_csv = _synthetic_metadata(tmp_path)
    manifest, _ = build_external_manifest(metadata_csv)

    locations_by_split = manifest.groupby("split")["location"].apply(set)
    all_splits = list(locations_by_split.index)
    for i, split_a in enumerate(all_splits):
        for split_b in all_splits[i + 1 :]:
            assert locations_by_split[split_a].isdisjoint(locations_by_split[split_b])
