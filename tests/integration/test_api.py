"""Integration tests for the Eidon REST API.

The pipeline is injected via FastAPI's dependency_overrides — no ONNX models
or gallery files are required.  The lifespan skips model loading when
app.state.pipeline is already set.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from eidon.detection.alignment import ARCFACE_DST
from eidon.matching import Gallery
from eidon.pipeline import EidonPipeline
from eidon.serving.api import app, get_pipeline
from eidon.types import Detection, RecognitionResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 512


def _unit_vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random(_DIM, dtype=np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _make_jpeg(h: int = 120, w: int = 120) -> bytes:
    """Return a valid JPEG byte-string from a blank image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return bytes(buf)


def _make_detection() -> Detection:
    return Detection(
        bbox=np.array([10.0, 10.0, 100.0, 100.0], dtype=np.float32),
        landmarks=ARCFACE_DST.copy(),
        score=0.95,
    )


def _make_result(identity: str = "Alice", matched: bool = True) -> RecognitionResult:
    return RecognitionResult(
        detection=_make_detection(),
        identity=identity,
        similarity=0.85 if matched else 0.15,
        matched=matched,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def gallery(tmp_path) -> Gallery:
    g = Gallery()
    g.add("Alice", _unit_vec(0))
    g.add("Alice", _unit_vec(1))
    g.add("Bob",   _unit_vec(2))
    return g


@pytest.fixture()
def mock_pipeline(gallery: Gallery) -> MagicMock:
    p = MagicMock(spec=EidonPipeline)
    p.gallery = gallery
    p.recognize.return_value = [_make_result("Alice")]
    p.detector.detect_largest.return_value = _make_detection()
    p.embedder.embed.return_value = _unit_vec(0)
    return p


@pytest.fixture()
def client(mock_pipeline: MagicMock) -> TestClient:
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_ok(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_reports_gallery_stats(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["gallery_identities"] == 2   # Alice + Bob
        assert data["gallery_embeddings"] == 3   # 2 for Alice, 1 for Bob

    def test_no_auth_required(self, client: TestClient) -> None:
        """Health is always accessible, even when API key is configured."""
        r = client.get("/health")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /recognize
# ---------------------------------------------------------------------------

class TestRecognize:
    def test_returns_matched_face(self, client: TestClient) -> None:
        r = client.post("/recognize", files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 1
        assert body["faces"][0]["identity"] == "Alice"
        assert body["faces"][0]["matched"] is True

    def test_returns_empty_when_no_faces(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        mock_pipeline.recognize.return_value = []
        r = client.post("/recognize", files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")})
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_returns_multiple_faces(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        mock_pipeline.recognize.return_value = [
            _make_result("Alice"),
            _make_result("unknown", matched=False),
        ]
        r = client.post("/recognize", files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")})
        assert r.json()["count"] == 2

    def test_rejects_invalid_image(self, client: TestClient) -> None:
        r = client.post("/recognize", files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")})
        assert r.status_code == 422
        assert "decode" in r.json()["detail"].lower()

    def test_response_includes_processing_time(self, client: TestClient) -> None:
        r = client.post("/recognize", files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")})
        assert r.json()["processing_time_ms"] >= 0

    def test_bbox_fields_present(self, client: TestClient) -> None:
        face = client.post(
            "/recognize", files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")}
        ).json()["faces"][0]
        assert all(k in face["bbox"] for k in ("x1", "y1", "x2", "y2"))


# ---------------------------------------------------------------------------
# POST /enroll
# ---------------------------------------------------------------------------

class TestEnroll:
    def test_enrolls_new_identity(
        self, client: TestClient, mock_pipeline: MagicMock, tmp_path
    ) -> None:
        mock_pipeline.gallery.add = MagicMock()
        mock_pipeline.gallery.save = MagicMock()
        mock_pipeline.gallery.embeddings_for = MagicMock(
            return_value=np.zeros((1, _DIM), dtype=np.float32)
        )
        with patch("eidon.serving.api.settings") as mock_settings:
            mock_settings.gallery_path = tmp_path / "g.npz"
            mock_settings.api_key = None
            r = client.post(
                "/enroll",
                files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
                data={"identity": "Charlie"},
            )
        assert r.status_code == 200
        assert r.json()["identity"] == "Charlie"

    def test_422_when_no_face_detected(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        mock_pipeline.detector.detect_largest.return_value = None
        r = client.post(
            "/enroll",
            files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
            data={"identity": "NoFace"},
        )
        assert r.status_code == 422
        assert "No face" in r.json()["detail"]

    def test_422_for_invalid_image(self, client: TestClient) -> None:
        r = client.post(
            "/enroll",
            files={"file": ("bad.jpg", b"garbage", "image/jpeg")},
            data={"identity": "Test"},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /identities
# ---------------------------------------------------------------------------

class TestIdentities:
    def test_lists_enrolled_identities(self, client: TestClient) -> None:
        r = client.get("/identities")
        assert r.status_code == 200
        body = r.json()
        names = [i["name"] for i in body["identities"]]
        assert "Alice" in names
        assert "Bob" in names
        assert body["total"] == 2

    def test_embedding_counts_correct(self, client: TestClient) -> None:
        identities = {
            i["name"]: i["embeddings"]
            for i in client.get("/identities").json()["identities"]
        }
        assert identities["Alice"] == 2
        assert identities["Bob"] == 1


# ---------------------------------------------------------------------------
# DELETE /identities/{name}
# ---------------------------------------------------------------------------

class TestDeleteIdentity:
    def test_deletes_existing_identity(
        self, client: TestClient, mock_pipeline: MagicMock, tmp_path
    ) -> None:
        mock_pipeline.gallery.save = MagicMock()
        with patch("eidon.serving.api.settings") as mock_settings:
            mock_settings.gallery_path = tmp_path / "g.npz"
            mock_settings.api_key = None
            r = client.delete("/identities/Bob")
        assert r.status_code == 200
        assert r.json()["identity"] == "Bob"
        assert "Bob" not in mock_pipeline.gallery

    def test_404_for_unknown_identity(self, client: TestClient) -> None:
        r = client.delete("/identities/Ghost")
        assert r.status_code == 404
        assert "Ghost" in r.json()["detail"]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TestAuth:
    def test_no_auth_when_key_not_configured(self, client: TestClient) -> None:
        """All endpoints accessible when EIDON_API_KEY is not set."""
        r = client.post("/recognize", files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")})
        assert r.status_code == 200

    def test_403_when_key_required_and_missing(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        with patch("eidon.serving.api.settings") as mock_settings:
            mock_settings.api_key = "secret-key"
            r = client.post(
                "/recognize", files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")}
            )
        assert r.status_code == 403

    def test_200_with_correct_key(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        with patch("eidon.serving.api.settings") as mock_settings:
            mock_settings.api_key = "secret-key"
            r = client.post(
                "/recognize",
                files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
                headers={"X-API-Key": "secret-key"},
            )
        assert r.status_code == 200

    def test_403_with_wrong_key(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        with patch("eidon.serving.api.settings") as mock_settings:
            mock_settings.api_key = "secret-key"
            r = client.post(
                "/recognize",
                files={"file": ("img.jpg", _make_jpeg(), "image/jpeg")},
                headers={"X-API-Key": "wrong-key"},
            )
        assert r.status_code == 403
