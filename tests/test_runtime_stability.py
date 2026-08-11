import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import web_app


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.released = False

    def isOpened(self):
        return True

    def get(self, _prop):
        return 30.0

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


def test_analyze_video_keeps_only_the_best_candidate(monkeypatch):
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
    capture = FakeCapture(frames)
    detector = object()
    wrist_heights = iter([30.0, 10.0, 20.0])
    metrics = [
        web_app.ShotMetrics(100.0, 110.0, 120.0, 130.0),
        web_app.ShotMetrics(101.0, 111.0, 121.0, 131.0),
        web_app.ShotMetrics(102.0, 112.0, 122.0, 132.0),
    ]
    metric_iter = iter(metrics)

    monkeypatch.setattr(web_app.cv2, "VideoCapture", lambda _path: capture)
    monkeypatch.setattr(web_app, "create_pose_detector", lambda fps: detector)
    monkeypatch.setattr(web_app, "close_detector", lambda value: None)
    monkeypatch.setattr(web_app, "resize_to_height", lambda frame, _height: frame)
    monkeypatch.setattr(web_app, "detect_primary_pose", lambda frame, value: object())
    monkeypatch.setattr(
        web_app,
        "collect_metrics",
        lambda landmarks, width, height: (next(metric_iter), next(wrist_heights)),
    )
    monkeypatch.setattr(web_app, "draw_pose", lambda frame, landmarks: frame)

    result = web_app.analyze_video(Path("unused.mp4"), max_frames=3, render_height=4)

    assert result.ok
    assert result.release_frame_index == 1
    assert result.metrics == metrics[1]
    assert result.frames_scanned == 3
    assert capture.released

    source = inspect.getsource(web_app.analyze_video)
    assert "best_candidate" in source
    assert "candidates.append" not in source


def test_slow_motion_uses_one_timebase_adjustment():
    assert web_app.slow_motion_fps(30.0, 3) == 10.0
    assert web_app.slow_motion_fps(24.0, 1) == 24.0
    assert web_app.slow_motion_fps(0.0, 3) == 1.0

    for function in (
        web_app.make_saved_model_comparison_video,
        web_app.make_reference_clip_comparison_video,
    ):
        source = inspect.getsource(function)
        assert "writer.write(combined)" in source
        assert "frames.append" not in source
        assert "for _ in range(slow_factor)" not in source


def test_runtime_status_reports_missing_tasks_api(monkeypatch):
    monkeypatch.setattr(web_app, "mp", object())
    monkeypatch.setattr(web_app, "MEDIAPIPE_IMPORT_ERROR", None)
    web_app.runtime_status.cache_clear()
    try:
        ok, detail = web_app.runtime_status()
    finally:
        web_app.runtime_status.cache_clear()

    assert not ok
    assert "Tasks API is unavailable" in detail


def test_runtime_status_surfaces_model_startup_failure(monkeypatch, tmp_path):
    model_path = tmp_path / "pose.task"
    model_path.write_bytes(b"model")

    class FakeOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class BrokenLandmarker:
        @staticmethod
        def create_from_options(_options):
            raise RuntimeError("invalid model")

    fake_vision = SimpleNamespace(
        PoseLandmarkerOptions=FakeOptions,
        PoseLandmarker=BrokenLandmarker,
        RunningMode=SimpleNamespace(VIDEO="video"),
    )
    fake_tasks = SimpleNamespace(
        vision=fake_vision,
        BaseOptions=lambda **kwargs: kwargs,
    )
    fake_mp = SimpleNamespace(tasks=fake_tasks)

    monkeypatch.setattr(web_app, "mp", fake_mp)
    monkeypatch.setattr(web_app, "MEDIAPIPE_IMPORT_ERROR", None)
    monkeypatch.setattr(web_app, "TASK_MODEL_PATH", model_path)
    web_app.runtime_status.cache_clear()
    try:
        ok, detail = web_app.runtime_status()
    finally:
        web_app.runtime_status.cache_clear()

    assert not ok
    assert "Tasks startup failed" in detail
    assert "invalid model" in detail
