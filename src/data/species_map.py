"""
Species → taxonomic class → binary label resolution.

Design (per supervisor instruction): the manifest keeps the real species
name for every row. The binary label used for training is derived from a
*mapping applied on top*, split across two config files so the label
scheme can be changed without touching the biological ground truth:

  configs/data/species_taxonomy.yaml — species -> {mammal, bird, other},
      plus native/wild metadata. Stable; this is a biology fact, not an
      experiment choice.
  configs/data/class_map.yaml — taxon -> label for the *current*
      experiment (currently mammal_vs_bird). Swap this file to re-frame
      the task; species_taxonomy.yaml never needs to change.

"Mammal vs bird" is taxonomic, not ecological: sparrow (introduced,
non-predator), harrier (native, but does predate), and sealion (native
mammal) all make sense under mammal/bird but would break a naive
predator-vs-native-bird split. See class_map.yaml for the rationale.

Every category in the raw data must resolve to a taxon — there is no
silent fall-through. `to_label` raises ValueError for any species absent
from species_taxonomy.yaml (this is a regression test target: the old
scaffold's `else: label = 0` bug silently mislabelled unlisted species as
birds; that must be structurally impossible here).
"""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_TAXONOMY_PATH = Path("configs/data/species_taxonomy.yaml")
DEFAULT_CLASS_MAP_PATH = Path("configs/data/class_map.yaml")

VALID_TAXA = {"mammal", "bird", "other"}

# Ground truth partition sizes for species_taxonomy.yaml — asserted at load
# time, not just trusted from the spec that produced this file.
EXPECTED_TAXON_COUNTS = {"mammal": 22, "bird": 71, "other": 4}


class SpeciesMapper:
    """Resolves a raw species string to a taxon and, from there, a binary label."""

    def __init__(
        self,
        taxonomy_path: Path | str = DEFAULT_TAXONOMY_PATH,
        class_map_path: Path | str = DEFAULT_CLASS_MAP_PATH,
    ) -> None:
        self.taxonomy: dict[str, dict] = _load_taxonomy(Path(taxonomy_path))
        self.class_map: dict = _load_class_map(Path(class_map_path))
        self._validate_class_map()

    # -- validation ----------------------------------------------------------
    def _validate_class_map(self) -> None:
        cm = self.class_map
        for key in ("positive_class", "negative_class"):
            if cm.get(key) not in VALID_TAXA:
                raise ValueError(
                    f"class_map.yaml: {key}={cm.get(key)!r} is not a valid "
                    f"taxon ({sorted(VALID_TAXA)})"
                )
        if cm["positive_class"] == cm["negative_class"]:
            raise ValueError(
                "class_map.yaml: positive_class and negative_class must differ "
                f"(both are {cm['positive_class']!r})"
            )
        exclude = set(cm.get("exclude", []))
        unknown_excluded = exclude - VALID_TAXA
        if unknown_excluded:
            raise ValueError(
                f"class_map.yaml: exclude contains unknown taxa {sorted(unknown_excluded)}"
            )
        accounted = {cm["positive_class"], cm["negative_class"]} | exclude
        unaccounted = VALID_TAXA - accounted
        if unaccounted:
            raise ValueError(
                f"class_map.yaml: taxa {sorted(unaccounted)} are neither "
                "positive_class, negative_class, nor excluded — every taxon "
                "must be handled explicitly"
            )
        if cm.get("mammal_scope") not in ("all", "wild_only"):
            raise ValueError(
                f"class_map.yaml: mammal_scope={cm.get('mammal_scope')!r} must "
                "be 'all' or 'wild_only'"
            )

    def validate_exhaustive(self, categories: set[str]) -> None:
        """Raise if the taxonomy and a real category set don't match exactly."""
        taxonomy_species = set(self.taxonomy.keys())
        unmapped = categories - taxonomy_species
        phantom = taxonomy_species - categories
        if unmapped or phantom:
            raise ValueError(
                "species_taxonomy.yaml does not exactly partition the given "
                f"categories — unmapped (in data, missing from taxonomy): "
                f"{sorted(unmapped)}; phantom (in taxonomy, absent from data): "
                f"{sorted(phantom)}"
            )

    # -- lookups ---------------------------------------------------------
    def taxon_of(self, species: str) -> str:
        entry = self.taxonomy.get(species)
        if entry is None:
            raise ValueError(
                f"Unknown species {species!r} — not present in "
                f"{DEFAULT_TAXONOMY_PATH}. Refusing to guess a label."
            )
        return entry["taxon"]

    def is_native(self, species: str) -> bool | None:
        return self.taxonomy[species]["native"] if species in self.taxonomy else self._raise(species)

    def is_wild(self, species: str) -> bool | None:
        return self.taxonomy[species]["wild"] if species in self.taxonomy else self._raise(species)

    @staticmethod
    def _raise(species: str):
        raise ValueError(f"Unknown species {species!r} — not present in taxonomy")

    # -- the actual label ------------------------------------------------
    def to_label(self, species: str) -> int | None:
        """Resolve a species to its binary label.

        Returns 1 (positive_class), 0 (negative_class), or None (drop this
        row — excluded taxon, or a domestic/livestock species dropped under
        mammal_scope: wild_only). Raises ValueError for species absent from
        the taxonomy — never silently defaults.
        """
        taxon = self.taxon_of(species)  # raises ValueError if unmapped
        cm = self.class_map

        if cm["mammal_scope"] == "wild_only":
            wild = self.taxonomy[species]["wild"]
            if wild is False:
                return None

        if taxon in set(cm.get("exclude", [])):
            return None
        if taxon == cm["positive_class"]:
            return 1
        if taxon == cm["negative_class"]:
            return 0
        # _validate_class_map guarantees every taxon is positive, negative,
        # or excluded — reaching here means that invariant broke.
        raise ValueError(
            f"taxon {taxon!r} for species {species!r} is not handled by "
            "class_map.yaml (not positive_class, negative_class, or excluded)"
        )


def _load_taxonomy(path: Path) -> dict[str, dict]:
    with path.open() as f:
        raw = yaml.safe_load(f)
    species = raw.get("species", {})
    if not species:
        raise ValueError(f"{path}: no 'species' mapping found")

    counts: dict[str, int] = {}
    for name, entry in species.items():
        taxon = entry.get("taxon")
        if taxon not in VALID_TAXA:
            raise ValueError(f"{path}: species {name!r} has invalid taxon {taxon!r}")
        counts[taxon] = counts.get(taxon, 0) + 1

    if counts != EXPECTED_TAXON_COUNTS:
        raise ValueError(
            f"{path}: taxon counts {counts} do not match the expected "
            f"partition {EXPECTED_TAXON_COUNTS} (total "
            f"{sum(counts.values())} vs expected {sum(EXPECTED_TAXON_COUNTS.values())})"
        )
    return species


def _load_class_map(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


_default_mapper: SpeciesMapper | None = None


def get_default_mapper() -> SpeciesMapper:
    """Lazily-cached SpeciesMapper over the default config paths."""
    global _default_mapper
    if _default_mapper is None:
        _default_mapper = SpeciesMapper()
    return _default_mapper
