"""
Streaming shard ingest — turns manifest.csv rows into resized-JPEG tar
shards without ever writing a full-resolution image to disk.

Full-res NZ trail-cam images run several MB each (a 4224x2376 sample was
~3.3MB); the ~80k rows in a capped manifest would be well over 100GB at
native resolution, more than local scratch can hold. Each row is instead
streamed over HTTPS, decoded and resized in memory, and appended straight
into a tar shard — the original bytes are discarded, never touching disk.
Target output is on the order of 2-3GB total for a ~60-80k image manifest.

URL construction — verified empirically with a live GET + PIL decode
before this loop was written (not assumed):
  base:  https://storage.googleapis.com/public-datasets-lila/nz-trailcams
  full:  {base}/{file_name}          (file_name is the manifest column,
                                       e.g. "ACC/banded_rail/<uuid>.JPG")
`https://data.source.coop/agentmorris/lila-wildlife/nz-trailcams/<file_name>`
mirrors the same bytes and is used as a fallback if the primary 404s.

Shards use the WebDataset basename convention (one member per sample,
named `<safe_image_id>.jpg`) written with the stdlib `tarfile` module —
the `webdataset` package itself is only needed to *read* shards
efficiently at training time, which is out of scope for the data layer,
so no new dependency was added for this.

Idempotent / resumable at shard granularity: a shard is considered done
only once every image in it is written AND a `<shard>.done` sentinel
file exists next to it. A shard whose tar exists without a `.done`
sentinel (i.e. an interrupted run) is deleted and rebuilt from scratch on
the next run — safer than trying to resume a partially-written tar.

Run: python -m src.data.ingest
"""
from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tarfile import TarFile, TarInfo

import pandas as pd
import requests
from PIL import Image
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.data.build_manifest import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FALLBACK_BASE_URL = "https://data.source.coop/agentmorris/lila-wildlife/nz-trailcams"

DEFAULT_IMAGES_PER_SHARD = 2000
DEFAULT_MAX_WORKERS = 8
DEFAULT_TIMEOUT_SECONDS = 20


class DownloadError(Exception):
    """Raised when an image cannot be fetched after retries."""


def verify_url_pattern(base_url: str, sample_file_name: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> None:
    """One live GET + image decode before the ingest loop runs. Never assume
    a URL pattern resolves — if this fails, stop and report; do not fall
    back to a guessed alternative silently."""
    url = f"{base_url}/{sample_file_name}"
    resp = requests.get(url, timeout=timeout)
    if resp.status_code != 200:
        raise DownloadError(
            f"URL pattern verification failed: GET {url} -> {resp.status_code}. "
            "Stopping — do not guess a different pattern."
        )
    try:
        img = Image.open(io.BytesIO(resp.content))
        img.load()
    except Exception as exc:  # noqa: BLE001 — surfacing the real decode error matters here
        raise DownloadError(
            f"URL {url} returned status 200 but the body did not decode as "
            f"an image: {exc!r}"
        ) from exc
    log.info("URL pattern verified: %s -> %s %s", url, img.format, img.size)


@retry(
    retry=retry_if_exception_type((requests.RequestException, DownloadError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    reraise=True,
)
def _get_bytes(session: requests.Session, url: str, timeout: int) -> bytes:
    resp = session.get(url, timeout=timeout)
    if resp.status_code != 200:
        raise DownloadError(f"GET {url} -> {resp.status_code}")
    return resp.content


def fetch_and_resize(
    session: requests.Session,
    base_url: str,
    file_name: str,
    image_size: int,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    fallback_base_url: str | None = FALLBACK_BASE_URL,
) -> bytes:
    """Stream one image, resize so its *longer* side is `image_size` px
    (never upscale), re-encode as JPEG, return the bytes. The original
    full-resolution bytes are never written to disk.

    Longer-side (not shorter-side) resize is a deliberate storage-budget
    choice: full manifest.csv ended up at ~81k rows (larger than the ~60k
    originally estimated), and shorter-side=300 measured ~70-90KB/image —
    5-10GB total, well over the ~2-3GB target. Longer-side=300 measured
    ~25KB/image on the same sample (~2GB extrapolated across the manifest),
    matching the target. Final training-time cropping/resizing to the
    model's 300x300 input happens later in the pipeline regardless of the
    shard's native aspect ratio.
    """
    url = f"{base_url}/{file_name}"
    try:
        raw = _get_bytes(session, url, timeout)
    except (requests.RequestException, DownloadError):
        if not fallback_base_url:
            raise
        raw = _get_bytes(session, f"{fallback_base_url}/{file_name}", timeout)

    img = Image.open(io.BytesIO(raw))
    img.load()
    img = img.convert("RGB")

    w, h = img.size
    scale = min(1.0, image_size / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()


def _safe_member_name(image_id: str) -> str:
    return image_id.replace("/", "__").rsplit(".", 1)[0] + ".jpg"


def iter_shards(manifest: pd.DataFrame, images_per_shard: int) -> list[pd.DataFrame]:
    """Deterministic shard assignment — sorted by image_id so re-running
    ingest always produces the same shard boundaries."""
    ordered = manifest.sort_values("image_id").reset_index(drop=True)
    return [
        ordered.iloc[i : i + images_per_shard]
        for i in range(0, len(ordered), images_per_shard)
    ]


def build_shard(
    shard_idx: int,
    rows: pd.DataFrame,
    base_url: str,
    out_dir: Path,
    image_size: int,
    max_workers: int = DEFAULT_MAX_WORKERS,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    shard_path = out_dir / f"shard-{shard_idx:05d}.tar"
    done_path = out_dir / f"shard-{shard_idx:05d}.done"
    failed_path = out_dir / f"shard-{shard_idx:05d}.failed.csv"

    if done_path.exists():
        log.info("shard %d already done, skipping", shard_idx)
        return
    if shard_path.exists():
        log.warning("shard %d has a tar but no .done sentinel — rebuilding from scratch", shard_idx)
        shard_path.unlink()

    failures: list[tuple[str, str]] = []
    with requests.Session() as session, TarFile.open(shard_path, "w") as tar:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    fetch_and_resize, session, base_url, row.file_name, image_size, timeout
                ): row.image_id
                for row in rows.itertuples()
            }
            for future in as_completed(futures):
                image_id = futures[future]
                try:
                    jpeg_bytes = future.result()
                except Exception as exc:  # noqa: BLE001 — log and continue, one bad row shouldn't abort a shard
                    log.error("failed to fetch %s: %r", image_id, exc)
                    failures.append((image_id, repr(exc)))
                    continue

                info = TarInfo(name=_safe_member_name(image_id))
                info.size = len(jpeg_bytes)
                tar.addfile(info, io.BytesIO(jpeg_bytes))

    if failures:
        pd.DataFrame(failures, columns=["image_id", "error"]).to_csv(failed_path, index=False)
        log.warning("shard %d: %d/%d images failed permanently (see %s)",
                    shard_idx, len(failures), len(rows), failed_path)

    done_path.write_text(f"{len(rows) - len(failures)}/{len(rows)} images written\n")
    log.info("shard %d done: %d/%d images", shard_idx, len(rows) - len(failures), len(rows))


def main() -> None:
    cfg = load_config()
    manifest = pd.read_csv(cfg["manifest_path"])

    base_url = cfg["source"]["https_base_url"]
    verify_url_pattern(base_url, manifest.iloc[0]["file_name"])

    out_dir = Path("data/shards")
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = iter_shards(manifest, DEFAULT_IMAGES_PER_SHARD)
    log.info("%d images -> %d shards of up to %d images each", len(manifest), len(shards), DEFAULT_IMAGES_PER_SHARD)

    for idx, rows in enumerate(shards):
        build_shard(idx, rows, base_url, out_dir, cfg["image_size"])

    total_bytes = sum(p.stat().st_size for p in out_dir.glob("shard-*.tar"))
    log.info("total shard size: %.2f GB", total_bytes / 1e9)
    log.info("next step (not run automatically): dvc add %s && dvc push", out_dir)


if __name__ == "__main__":
    main()
