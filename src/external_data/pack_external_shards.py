"""
Packs the external "birds_and_mammals" images (already downloaded and
extracted locally, see build_external_manifest.py) into tar shards in
EXACTLY the same format src/data/ingest.py produces for the main LILA
dataset -- same member-naming convention (safe_member_name, shared with
src/training/dataset.py so both packers agree), same shard-NNNNN.tar +
shard-NNNNN.done sentinel structure.

Why this matters: ShardReader/ShardDataset don't care which shard file a
given image_id's bytes live in -- they index every shard-*.tar in a
directory into one flat lookup. Writing new shards directly into the
SAME shards_dir as the original data (default: data/shards) means the
existing training pipeline (train.py, boxcrop_experiment.py) needs zero
code changes to train on combined old+new data -- it just needs a
manifest whose rows happen to resolve into a directory that now has more
shards in it than before.

Shard numbering continues from whatever shard-NNNNN.tar files already
exist in the target directory (scanned, not hardcoded), so this is safe
to point at the same data/shards directory the main pipeline already
uses without colliding with existing shard indices.

Images are already on local disk (no network fetch, unlike ingest.py),
so this reads bytes directly rather than reusing ingest.py's
fetch_and_resize -- but the resize policy matches it exactly (longer
side capped at image_size, same LANCZOS resize, same JPEG quality) so
packed external images end up statistically consistent with the
originals, not a second, differently-processed image population hiding
inside the same shard pool.

Run: python -m src.external_data.pack_external_shards
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from tarfile import TarFile, TarInfo

import pandas as pd
from PIL import Image

from src.training.dataset import safe_member_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = "src/external_data/metadata/external_manifest.csv"
DEFAULT_IMAGES_ROOT = "src/external_data/birds_and_mammals"
DEFAULT_SHARDS_DIR = "data/shards"
DEFAULT_IMAGES_PER_SHARD = 2000
DEFAULT_IMAGE_SIZE = 300


def _next_shard_index(shards_dir: Path) -> int:
    existing = sorted(shards_dir.glob("shard-*.tar"))
    if not existing:
        return 0
    last = existing[-1].stem  # "shard-00040"
    return int(last.split("-")[1]) + 1


def _resize_like_ingest(image: Image.Image, image_size: int) -> bytes:
    """Mirrors src.data.ingest.fetch_and_resize's resize policy exactly:
    longer side capped at image_size, never upscaled, LANCZOS, JPEG q85."""
    image = image.convert("RGB")
    w, h = image.size
    scale = min(1.0, image_size / max(w, h))
    if scale < 1.0:
        image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=85)
    return out.getvalue()


def pack_external_shards(
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    images_root: str = DEFAULT_IMAGES_ROOT,
    shards_dir: str = DEFAULT_SHARDS_DIR,
    images_per_shard: int = DEFAULT_IMAGES_PER_SHARD,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> dict:
    manifest = pd.read_csv(manifest_path)
    manifest = manifest.sort_values("image_id").reset_index(drop=True)  # deterministic, matches ingest.py
    images_root = Path(images_root)
    shards_dir = Path(shards_dir)
    shards_dir.mkdir(parents=True, exist_ok=True)

    start_idx = _next_shard_index(shards_dir)
    chunks = [manifest.iloc[i : i + images_per_shard] for i in range(0, len(manifest), images_per_shard)]

    written_shards: list[str] = []
    n_written = 0
    n_missing = 0
    for offset, rows in enumerate(chunks):
        shard_idx = start_idx + offset
        shard_path = shards_dir / f"shard-{shard_idx:05d}.tar"
        done_path = shards_dir / f"shard-{shard_idx:05d}.done"

        if done_path.exists():
            log.info("shard %d already done, skipping", shard_idx)
            written_shards.append(shard_path.name)
            continue

        failures: list[tuple[str, str]] = []
        with TarFile.open(shard_path, "w") as tar:
            for row in rows.itertuples():
                src_path = images_root / row.file_name
                if not src_path.exists():
                    log.error("missing source image: %s", src_path)
                    failures.append((row.image_id, "file not found"))
                    n_missing += 1
                    continue
                try:
                    image = Image.open(src_path)
                    jpeg_bytes = _resize_like_ingest(image, image_size)
                except Exception as exc:  # noqa: BLE001 — one bad image shouldn't abort the whole shard
                    log.error("failed to process %s: %r", src_path, exc)
                    failures.append((row.image_id, repr(exc)))
                    continue

                info = TarInfo(name=safe_member_name(row.image_id))
                info.size = len(jpeg_bytes)
                tar.addfile(info, io.BytesIO(jpeg_bytes))
                n_written += 1

        if failures:
            failed_path = shards_dir / f"shard-{shard_idx:05d}.failed.csv"
            pd.DataFrame(failures, columns=["image_id", "error"]).to_csv(failed_path, index=False)
            log.warning("shard %d: %d/%d images failed (see %s)", shard_idx, len(failures), len(rows), failed_path)

        done_path.write_text(f"{len(rows) - len(failures)}/{len(rows)} images written\n")
        written_shards.append(shard_path.name)
        log.info("shard %d done: %d/%d images", shard_idx, len(rows) - len(failures), len(rows))

    total_bytes = sum((shards_dir / name).stat().st_size for name in written_shards)
    result = {
        "shards_written": written_shards,
        "n_images_written": n_written,
        "n_images_missing": n_missing,
        "total_bytes": total_bytes,
    }
    log.info(
        "packed %d external images into %d shard(s) (%.1f MB) starting at shard-%05d",
        n_written, len(written_shards), total_bytes / 1e6, start_idx,
    )
    return result


def main() -> None:
    pack_external_shards()


if __name__ == "__main__":
    main()
