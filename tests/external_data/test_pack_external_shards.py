"""Tests for pack_external_shards.py -- shard numbering, and an
end-to-end pack against real tiny synthetic images, verified by reading
the result back through the real ShardReader (not just checking the tar
file exists)."""
import pandas as pd
import pytest
from PIL import Image

from src.external_data.pack_external_shards import _next_shard_index, pack_external_shards
from src.training.dataset import ShardReader, safe_member_name


def test_next_shard_index_continues_from_existing_shards(tmp_path):
    (tmp_path / "shard-00000.tar").touch()
    (tmp_path / "shard-00003.tar").touch()
    assert _next_shard_index(tmp_path) == 4


def test_next_shard_index_starts_at_zero_for_an_empty_directory(tmp_path):
    assert _next_shard_index(tmp_path) == 0


@pytest.fixture
def tiny_external_dataset(tmp_path):
    images_root = tmp_path / "images"
    (images_root / "rat" / "ir").mkdir(parents=True)
    (images_root / "kaka" / "rgb").mkdir(parents=True)

    Image.new("RGB", (400, 300), color=(120, 60, 10)).save(images_root / "rat" / "ir" / "a.jpg")
    Image.new("RGB", (300, 400), color=(10, 200, 10)).save(images_root / "kaka" / "rgb" / "b.jpg")
    # No file on disk for this row -- exercises the missing-file path.
    manifest = pd.DataFrame(
        {
            "image_id": ["rat/ir/a.jpg", "kaka/rgb/b.jpg", "cat/ir/missing.jpg"],
            "file_name": ["rat/ir/a.jpg", "kaka/rgb/b.jpg", "cat/ir/missing.jpg"],
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    shards_dir = tmp_path / "shards"
    return {"manifest_path": manifest_path, "images_root": images_root, "shards_dir": shards_dir}


def test_pack_external_shards_writes_a_readable_shard(tiny_external_dataset):
    result = pack_external_shards(
        manifest_path=str(tiny_external_dataset["manifest_path"]),
        images_root=str(tiny_external_dataset["images_root"]),
        shards_dir=str(tiny_external_dataset["shards_dir"]),
        images_per_shard=2000,
        image_size=300,
    )
    assert result["n_images_written"] == 2
    assert result["n_images_missing"] == 1
    assert result["shards_written"] == ["shard-00000.tar"]

    reader = ShardReader(str(tiny_external_dataset["shards_dir"]))
    assert safe_member_name("rat/ir/a.jpg") in reader
    assert safe_member_name("kaka/rgb/b.jpg") in reader
    assert safe_member_name("cat/ir/missing.jpg") not in reader


def test_pack_external_shards_resizes_the_longer_side_down_to_image_size(tiny_external_dataset):
    pack_external_shards(
        manifest_path=str(tiny_external_dataset["manifest_path"]),
        images_root=str(tiny_external_dataset["images_root"]),
        shards_dir=str(tiny_external_dataset["shards_dir"]),
        images_per_shard=2000,
        image_size=200,  # smaller than the 400x300 source, so it must shrink
    )
    reader = ShardReader(str(tiny_external_dataset["shards_dir"]))
    import io

    raw = reader.read(safe_member_name("rat/ir/a.jpg"))
    img = Image.open(io.BytesIO(raw))
    assert max(img.size) == 200


def test_pack_external_shards_continues_numbering_from_an_existing_shard(tiny_external_dataset):
    shards_dir = tiny_external_dataset["shards_dir"]
    shards_dir.mkdir(parents=True)
    (shards_dir / "shard-00000.tar").touch()
    (shards_dir / "shard-00000.done").touch()

    result = pack_external_shards(
        manifest_path=str(tiny_external_dataset["manifest_path"]),
        images_root=str(tiny_external_dataset["images_root"]),
        shards_dir=str(shards_dir),
        images_per_shard=2000,
        image_size=300,
    )
    assert result["shards_written"] == ["shard-00001.tar"]
