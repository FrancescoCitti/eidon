"""Unit tests for the webcam live-recognition module.

cv2 is fully mocked — no camera or display required.
The pipeline is mocked — no ONNX models required.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from eidon.detection.alignment import ARCFACE_DST
from eidon.types import Detection, RecognitionResult
from eidon.webcam.live import LiveRecognition, _FPSCounter, annotate_frame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 512


def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_result(
    *,
    matched: bool = True,
    identity: str = "Alice",
    similarity: float = 0.85,
    bbox: list[float] | None = None,
) -> RecognitionResult:
    return RecognitionResult(
        detection=Detection(
            bbox=np.array(bbox or [50.0, 50.0, 200.0, 200.0], dtype=np.float32),
            landmarks=ARCFACE_DST.copy(),
            score=0.95,
        ),
        identity=identity,
        similarity=similarity,
        matched=matched,
    )


def _make_mock_pipeline(results: list[RecognitionResult] | None = None) -> MagicMock:
    pipeline = MagicMock()
    pipeline.recognize.return_value = results or []
    return pipeline


# ---------------------------------------------------------------------------
# _FPSCounter
# ---------------------------------------------------------------------------

class TestFPSCounter:
    def test_first_tick_returns_zero(self) -> None:
        assert _FPSCounter().tick() == 0.0

    def test_two_ticks_return_positive_fps(self) -> None:
        fps_counter = _FPSCounter()
        fps_counter.tick()
        time.sleep(0.01)
        fps = fps_counter.tick()
        assert fps > 0.0

    def test_fps_is_reasonable(self) -> None:
        fps_counter = _FPSCounter()
        for _ in range(5):
            fps_counter.tick()
            time.sleep(0.01)
        fps = fps_counter.tick()
        # Should be roughly 100 FPS for 10ms sleeps (generous bounds)
        assert 10 < fps < 1000

    def test_window_size_is_respected(self) -> None:
        fps_counter = _FPSCounter(window=3)
        for _ in range(10):
            fps_counter.tick()
        # Internal deque should not exceed the window size
        assert len(fps_counter._times) == 3  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# annotate_frame
# ---------------------------------------------------------------------------

class TestAnnotateFrame:
    def test_output_shape_matches_input(self) -> None:
        frame = _make_frame()
        out = annotate_frame(frame, [])
        assert out.shape == frame.shape

    def test_output_dtype_is_uint8(self) -> None:
        assert annotate_frame(_make_frame(), []).dtype == np.uint8

    def test_does_not_modify_original_frame(self) -> None:
        frame = _make_frame()
        original = frame.copy()
        annotate_frame(frame, [_make_result()])
        np.testing.assert_array_equal(frame, original)

    def test_empty_results_runs_without_error(self) -> None:
        annotate_frame(_make_frame(), [], fps=30.0)

    def test_matched_result_runs_without_error(self) -> None:
        annotate_frame(_make_frame(), [_make_result(matched=True)])

    def test_unmatched_result_runs_without_error(self) -> None:
        annotate_frame(_make_frame(), [_make_result(matched=False, identity="unknown")])

    def test_multiple_results_run_without_error(self) -> None:
        results = [
            _make_result(matched=True, bbox=[10.0, 10.0, 120.0, 120.0]),
            _make_result(matched=False, bbox=[200.0, 10.0, 310.0, 120.0]),
        ]
        annotate_frame(_make_frame(), results, fps=25.5)

    def test_bbox_at_top_edge_runs_without_error(self) -> None:
        """Label above a face near the top of the frame should flip to draw below."""
        result = _make_result(bbox=[10.0, 0.0, 120.0, 80.0])
        annotate_frame(_make_frame(), [result])

    def test_bbox_clipped_to_image_bounds(self) -> None:
        """Bounding boxes outside the frame should not raise."""
        result = _make_result(bbox=[-50.0, -50.0, 700.0, 560.0])
        annotate_frame(_make_frame(), [result])

    def test_output_differs_from_input_when_results_present(self) -> None:
        """Annotating with results should change at least some pixels."""
        frame = _make_frame()
        out = annotate_frame(frame, [_make_result()])
        assert not np.array_equal(frame, out)


# ---------------------------------------------------------------------------
# LiveRecognition — init
# ---------------------------------------------------------------------------

class TestLiveRecognitionInit:
    def test_raises_on_skip_frames_zero(self) -> None:
        with pytest.raises(ValueError, match="skip_frames"):
            LiveRecognition(_make_mock_pipeline(), skip_frames=0)

    def test_raises_on_negative_skip_frames(self) -> None:
        with pytest.raises(ValueError, match="skip_frames"):
            LiveRecognition(_make_mock_pipeline(), skip_frames=-1)

    def test_valid_construction(self) -> None:
        live = LiveRecognition(_make_mock_pipeline(), skip_frames=2)
        assert live._skip_frames == 2  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# LiveRecognition — run() (cv2 fully mocked)
# ---------------------------------------------------------------------------

def _make_cap_mock(frames: list[np.ndarray], *, opened: bool = True) -> MagicMock:
    cap = MagicMock()
    cap.isOpened.return_value = opened
    cap.read.side_effect = [(True, f) for f in frames] + [(False, None)]
    return cap


@pytest.fixture()
def mock_cv2():
    with patch("eidon.webcam.live.cv2") as m:
        m.WND_PROP_VISIBLE = 1
        m.FILLED = -1
        m.LINE_AA = 8
        m.FONT_HERSHEY_SIMPLEX = 0
        m.getTextSize.return_value = ((80, 16), 4)
        m.getWindowProperty.return_value = 1.0  # window is visible
        yield m


class TestLiveRecognitionRun:
    def test_raises_when_camera_cannot_open(self, mock_cv2: MagicMock) -> None:
        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = cap

        live = LiveRecognition(_make_mock_pipeline(), webcam_index=0)
        with pytest.raises(RuntimeError, match="Cannot open webcam"):
            live.run()

    def test_exits_on_q_keypress(self, mock_cv2: MagicMock) -> None:
        frames = [_make_frame() for _ in range(5)]
        cap = _make_cap_mock(frames)
        mock_cv2.VideoCapture.return_value = cap
        # Return 'q' on the 3rd waitKey call
        mock_cv2.waitKey.side_effect = [255, 255, ord("q"), 255, 255]

        live = LiveRecognition(_make_mock_pipeline(), skip_frames=1)
        live.run()

        assert mock_cv2.destroyAllWindows.called
        assert cap.release.called

    def test_exits_when_frame_read_fails(self, mock_cv2: MagicMock) -> None:
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.waitKey.return_value = 0

        live = LiveRecognition(_make_mock_pipeline(), skip_frames=1)
        live.run()

        cap.release.assert_called_once()

    def test_inference_called_every_skip_frames(self, mock_cv2: MagicMock) -> None:
        n_frames = 6
        skip = 2
        frames = [_make_frame() for _ in range(n_frames)]
        cap = _make_cap_mock(frames)
        mock_cv2.VideoCapture.return_value = cap
        # Exit only after all frames are consumed (read returns False)
        mock_cv2.waitKey.return_value = 0

        pipeline = _make_mock_pipeline()
        live = LiveRecognition(pipeline, skip_frames=skip)
        live.run()

        # Frames 0, 2, 4 trigger inference → 3 calls
        assert pipeline.recognize.call_count == n_frames // skip

    def test_inference_not_called_on_skip_frames(self, mock_cv2: MagicMock) -> None:
        frames = [_make_frame() for _ in range(3)]
        cap = _make_cap_mock(frames)
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.waitKey.return_value = 0

        pipeline = _make_mock_pipeline()
        LiveRecognition(pipeline, skip_frames=3).run()

        # Only frame 0 triggers inference
        assert pipeline.recognize.call_count == 1

    def test_camera_always_released_on_exception(self, mock_cv2: MagicMock) -> None:
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.side_effect = RuntimeError("unexpected camera error")
        mock_cv2.VideoCapture.return_value = cap

        live = LiveRecognition(_make_mock_pipeline(), skip_frames=1)
        with pytest.raises(RuntimeError):
            live.run()

        cap.release.assert_called_once()
        mock_cv2.destroyAllWindows.assert_called_once()

    def test_exits_when_window_closed(self, mock_cv2: MagicMock) -> None:
        # Guard fires only when frame_index > 5 (i.e. from the 7th frame onward).
        # Return invisible on the 3rd visibility check → exits after processing 8 frames.
        frames = [_make_frame() for _ in range(15)]
        cap = _make_cap_mock(frames)
        mock_cv2.VideoCapture.return_value = cap
        mock_cv2.waitKey.return_value = 0
        mock_cv2.getWindowProperty.side_effect = [1.0, 1.0, 0.0] + [1.0] * 20

        pipeline = _make_mock_pipeline()
        LiveRecognition(pipeline, skip_frames=1).run()

        # Exits before consuming all 15 frames — visibility triggered early stop
        assert pipeline.recognize.call_count < 15
