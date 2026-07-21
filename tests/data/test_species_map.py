"""Tests for species -> taxon -> label resolution."""
from pathlib import Path

import pandas as pd
import pytest

from src.data.species_map import SpeciesMapper

FLAT_PATH = Path("data/nz_flat.parquet")


@pytest.fixture(scope="module")
def mapper() -> SpeciesMapper:
    return SpeciesMapper()


def test_unmapped_species_raises(mapper):
    """Regression test: the old scaffold's `else: label = 0` silently
    mislabelled any unlisted category as a bird. That must now be
    structurally impossible — an unknown species raises, never defaults."""
    with pytest.raises(ValueError):
        mapper.to_label("kangaroo")


def test_taxon_of_unmapped_species_raises(mapper):
    with pytest.raises(ValueError):
        mapper.taxon_of("kangaroo")


@pytest.mark.parametrize(
    ("species", "expected_label"),
    [
        ("mouse", 1),
        ("possum", 1),
        ("stoat", 1),
        ("kiwi", 0),
        ("robin", 0),
        ("sparrow", 0),   # introduced, non-predator — still label 0 (bird)
        ("harrier", 0),   # native bird that predates — still label 0 (bird)
        ("sealion", 1),   # native mammal — still label 1 (mammal)
    ],
)
def test_to_label_mammal_vs_bird(mapper, species, expected_label):
    assert mapper.to_label(species) == expected_label


@pytest.mark.parametrize("species", ["human", "moth", "lizard", "skink"])
def test_other_taxon_dropped(mapper, species):
    assert mapper.to_label(species) is None


def test_taxonomy_counts_partition_97():
    # SpeciesMapper() construction itself asserts this (raises on mismatch),
    # so a successful construction plus this recount is the assertion.
    m = SpeciesMapper()
    counts: dict[str, int] = {}
    for entry in m.taxonomy.values():
        counts[entry["taxon"]] = counts.get(entry["taxon"], 0) + 1
    assert counts == {"mammal": 22, "bird": 71, "other": 4}
    assert sum(counts.values()) == 97


@pytest.mark.skipif(
    not FLAT_PATH.exists(),
    reason="requires local data/nz_flat.parquet (DVC-tracked, not committed to git)",
)
def test_taxonomy_exhaustive_over_real_data(mapper):
    categories = set(pd.read_parquet(FLAT_PATH, columns=["category"])["category"].unique())
    assert len(categories) == 97
    mapper.validate_exhaustive(categories)  # raises ValueError on any mismatch


def test_validate_exhaustive_detects_unmapped(mapper):
    with pytest.raises(ValueError):
        mapper.validate_exhaustive({"mouse", "kangaroo"})


def test_validate_exhaustive_detects_phantom(mapper):
    # A category set that's a strict subset of the real 97 leaves every
    # other taxonomy species "phantom" (in the taxonomy, absent from data).
    with pytest.raises(ValueError):
        mapper.validate_exhaustive({"mouse"})


def test_wild_only_excludes_domestic_species(tmp_path):
    class_map_path = tmp_path / "class_map.yaml"
    class_map_path.write_text(
        "name: mammal_vs_bird\n"
        "positive_class: mammal\n"
        "negative_class: bird\n"
        "exclude: [other]\n"
        "mammal_scope: wild_only\n"
    )
    m = SpeciesMapper(class_map_path=class_map_path)

    for species in ("cow", "sheep", "horse", "dog", "chicken"):
        assert m.to_label(species) is None, species

    # genuinely wild species are unaffected by wild_only
    assert m.to_label("mouse") == 1
    assert m.to_label("stoat") == 1
    assert m.to_label("kiwi") == 0
