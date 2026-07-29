"""Tests for pulling specific images out of shard tars into a loose folder."""
import tarfile
from pathlib import Path

from src.data.extract_images import extract_images


def _make_shard(tmp_path: Path, shard_name: str, members: dict[str, bytes]) -> None:
    shard_path = tmp_path / f"{shard_name}.tar"
    with tarfile.open(shard_path, "w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, __import__("io").BytesIO(data))
    (tmp_path / f"{shard_name}.done").write_text("done\n")


def test_extract_images_pulls_requested_ids(tmp_path):
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    _make_shard(
        shards_dir, "shard-00000",
        {"ACC__banded_rail__abc.jpg": b"fake-jpeg-bytes-1"},
    )
    _make_shard(
        shards_dir, "shard-00001",
        {"EFH__mouse__def.jpg": b"fake-jpeg-bytes-2"},
    )

    out_dir = tmp_path / "out"
    written = extract_images(
        ["ACC/banded_rail/abc.JPG", "EFH/mouse/def.JPG"],
        shards_dir=shards_dir,
        out_dir=out_dir,
    )

    assert len(written) == 2
    assert (out_dir / "ACC__banded_rail__abc.jpg").read_bytes() == b"fake-jpeg-bytes-1"
    assert (out_dir / "EFH__mouse__def.jpg").read_bytes() == b"fake-jpeg-bytes-2"


def test_extract_images_reports_missing_ids_without_raising(tmp_path, capsys):
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    _make_shard(shards_dir, "shard-00000", {"ACC__banded_rail__abc.jpg": b"data"})

    out_dir = tmp_path / "out"
    written = extract_images(
        ["ACC/banded_rail/abc.JPG", "does/not/exist.JPG"],
        shards_dir=shards_dir,
        out_dir=out_dir,
    )

    assert len(written) == 1
    captured = capsys.readouterr()
    assert "does/not/exist.JPG" in captured.out


def test_extract_images_skips_unfinished_shards(tmp_path):
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    # a .tar with no matching .done sentinel should be ignored entirely
    shard_path = shards_dir / "shard-00000.tar"
    with tarfile.open(shard_path, "w") as tar:
        info = tarfile.TarInfo(name="ACC__banded_rail__abc.jpg")
        data = b"data"
        info.size = len(data)
        tar.addfile(info, __import__("io").BytesIO(data))

    out_dir = tmp_path / "out"
    written = extract_images(["ACC/banded_rail/abc.JPG"], shards_dir=shards_dir, out_dir=out_dir)
    assert written == []
