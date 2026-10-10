"""Tests for the pure logic in src/inference/megadetector.py -- box
selection and download caching -- using fake detectors/HTTP responses so
nothing here touches the network or loads the real ~280MB model. Loading
the real detector and running it on a real image is verified separately
by hand (see REPORT_NOTES.txt), matching this project's existing
convention of keeping the automated suite hermetic (test_classifier.py's
load_checkpoint tests use synthetic, freshly-saved checkpoints rather than
real pretrained ones for the same reason)."""
from PIL import Image

from src.inference.megadetector import detect_animal_box, ensure_model_downloaded


class _FakeDetector:
    def __init__(self, detections):
        self._detections = detections
        self.calls = []

    def generate_detections_one_image(self, image, image_id, detection_threshold):
        self.calls.append((image, image_id, detection_threshold))
        return {"file": image_id, "detections": self._detections}


def _image():
    return Image.new("RGB", (20, 20), color=(10, 20, 30))


def test_detect_animal_box_returns_highest_confidence_animal_bbox():
    detector = _FakeDetector(
        [
            {"category": "1", "conf": 0.4, "bbox": [0.1, 0.1, 0.2, 0.2]},
            {"category": "1", "conf": 0.9, "bbox": [0.3, 0.3, 0.2, 0.2]},
        ]
    )
    assert detect_animal_box(detector, _image()) == [0.3, 0.3, 0.2, 0.2]


def test_detect_animal_box_ignores_non_animal_categories():
    detector = _FakeDetector(
        [
            {"category": "2", "conf": 0.99, "bbox": [0.1, 0.1, 0.2, 0.2]},  # person
            {"category": "1", "conf": 0.5, "bbox": [0.5, 0.5, 0.1, 0.1]},
        ]
    )
    assert detect_animal_box(detector, _image()) == [0.5, 0.5, 0.1, 0.1]


def test_detect_animal_box_returns_none_when_no_animal_detected():
    detector = _FakeDetector([{"category": "2", "conf": 0.9, "bbox": [0.1, 0.1, 0.2, 0.2]}])
    assert detect_animal_box(detector, _image()) is None


def test_detect_animal_box_returns_none_for_empty_detections():
    detector = _FakeDetector([])
    assert detect_animal_box(detector, _image()) is None


def test_detect_animal_box_passes_conf_threshold_through_to_detector():
    detector = _FakeDetector([{"category": "1", "conf": 0.5, "bbox": [0.0, 0.0, 0.1, 0.1]}])
    detect_animal_box(detector, _image(), conf_threshold=0.35)
    assert detector.calls[0][2] == 0.35


def test_ensure_model_downloaded_skips_download_if_file_exists(tmp_path, monkeypatch):
    model_path = tmp_path / "md.pt"
    model_path.write_bytes(b"already here")

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("should not attempt a network call when the file already exists")

    monkeypatch.setattr("src.inference.megadetector.requests.get", _fail_if_called)

    result = ensure_model_downloaded(str(model_path), url="https://example.invalid/md.pt")

    assert result == str(model_path)
    assert model_path.read_bytes() == b"already here"


def test_ensure_model_downloaded_writes_response_content(tmp_path, monkeypatch):
    model_path = tmp_path / "nested" / "md.pt"

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"fake-weights-bytes"

    def _fake_get(url, stream, timeout):
        assert url == "https://example.invalid/md.pt"
        return _FakeResponse()

    monkeypatch.setattr("src.inference.megadetector.requests.get", _fake_get)

    result = ensure_model_downloaded(str(model_path), url="https://example.invalid/md.pt")

    assert result == str(model_path)
    assert model_path.read_bytes() == b"fake-weights-bytes"
