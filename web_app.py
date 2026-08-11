import json
import math
import tempfile
import traceback
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    import mediapipe as mp
except Exception as exc:
    mp = None
    MEDIAPIPE_IMPORT_ERROR = exc
else:
    MEDIAPIPE_IMPORT_ERROR = None


APP_DIR = Path(__file__).resolve().parent
PLAYER_MODELS_PATH = APP_DIR / "models" / "nba_player_models.json"
TASK_MODEL_PATH = APP_DIR / "models" / "pose_landmarker_full.task"
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv"]
METRIC_ORDER = ["Elbow", "Shoulder", "Hip", "Knee"]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32),
    (27, 31), (28, 32),
]


@dataclass
class _PoseLandmarksContainer:
    landmark: List


@dataclass
class _PoseProcessResult:
    pose_landmarks: Optional[_PoseLandmarksContainer]


class _TasksPoseAdapter:
    def __init__(self, landmarker, fps: float):
        self._landmarker = landmarker
        self._timestamp_ms = 0
        self._interval_ms = max(1, int(round(1000.0 / max(fps, 1e-6))))

    def process(self, frame_rgb: np.ndarray) -> _PoseProcessResult:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(image, self._timestamp_ms)
        self._timestamp_ms += self._interval_ms
        if result.pose_landmarks:
            return _PoseProcessResult(_PoseLandmarksContainer(result.pose_landmarks[0]))
        return _PoseProcessResult(None)

    def close(self) -> None:
        self._landmarker.close()


@dataclass
class ShotMetrics:
    elbow: float
    shoulder: float
    hip: float
    knee: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "Elbow": self.elbow,
            "Shoulder": self.shoulder,
            "Hip": self.hip,
            "Knee": self.knee,
        }

    def to_model_dict(self) -> Dict[str, float]:
        return {
            "Elbow angle": self.elbow,
            "Shoulder angle": self.shoulder,
            "Hip angle": self.hip,
            "Knee angle": self.knee,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "ShotMetrics":
        return cls(
            float(data.get("Elbow", data.get("Elbow angle", 0.0))),
            float(data.get("Shoulder", data.get("Shoulder angle", 0.0))),
            float(data.get("Hip", data.get("Hip angle", 0.0))),
            float(data.get("Knee", data.get("Knee angle", 0.0))),
        )


@dataclass
class AnalysisResult:
    metrics: Optional[ShotMetrics] = None
    release_frame: Optional[np.ndarray] = None
    release_frame_index: int = -1
    frames_scanned: int = 0
    error: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.metrics is not None and not self.error


@lru_cache(maxsize=1)
def _runtime_status_probe() -> Tuple[bool, str]:
    if MEDIAPIPE_IMPORT_ERROR is not None:
        return False, f"MediaPipe import failed: {MEDIAPIPE_IMPORT_ERROR}"
    if mp is None:
        return False, "MediaPipe is unavailable."
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        return True, "MediaPipe solutions.pose"

    try:
        vision = mp.tasks.vision
        base_options = mp.tasks.BaseOptions
    except Exception as exc:
        return False, f"MediaPipe Tasks API is unavailable: {type(exc).__name__}: {exc}"

    if not TASK_MODEL_PATH.is_file():
        return False, f"Missing pose model: {TASK_MODEL_PATH}"

    try:
        options = vision.PoseLandmarkerOptions(
            base_options=base_options(model_asset_path=str(TASK_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)
    except Exception as exc:
        return False, f"MediaPipe Tasks startup failed: {type(exc).__name__}: {exc}"

    try:
        landmarker.close()
    except Exception as exc:
        return False, f"MediaPipe Tasks shutdown check failed: {type(exc).__name__}: {exc}"

    return True, "MediaPipe Tasks fallback (model verified)"


def runtime_status() -> Tuple[bool, str]:
    result = _runtime_status_probe()
    if not result[0]:
        _runtime_status_probe.cache_clear()
    return result


def clear_runtime_status_cache() -> None:
    _runtime_status_probe.cache_clear()

def create_pose_detector(fps: float):
    if mp is None:
        raise RuntimeError(f"MediaPipe is not available: {MEDIAPIPE_IMPORT_ERROR}")
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        return mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    vision = mp.tasks.vision
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(TASK_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=1,
    )
    return _TasksPoseAdapter(vision.PoseLandmarker.create_from_options(options), fps=fps)


def close_detector(detector) -> None:
    close = getattr(detector, "close", None)
    if callable(close):
        close()


def load_player_models(path: Path = PLAYER_MODELS_PATH) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def save_player_models(models: Dict[str, Dict], path: Path = PLAYER_MODELS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(models, indent=2), encoding="utf-8")


def save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if height == target_height:
        return frame
    target_width = max(1, int(round(width * target_height / height)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def landmark_xy(landmarks, idx: int, width: int, height: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x * width, lm.y * height], dtype=np.float32)


def mean_visibility(landmarks, indices: List[int]) -> float:
    values = [float(getattr(landmarks[i], "visibility", 1.0)) for i in indices if i < len(landmarks)]
    return float(np.mean(values)) if values else 0.0


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom <= 1e-6:
        return float("nan")
    cosine = float(np.dot(ba, bc) / denom)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def draw_pose(frame: np.ndarray, landmarks) -> np.ndarray:
    height, width = frame.shape[:2]
    points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
    for start, end in POSE_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], (30, 186, 143), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 3, (233, 95, 70), -1, cv2.LINE_AA)
    return frame


def detect_primary_pose(frame_bgr: np.ndarray, detector):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)
    if result.pose_landmarks:
        return result.pose_landmarks.landmark
    return None


def collect_metrics(landmarks, width: int, height: int) -> Tuple[Optional[ShotMetrics], float]:
    if landmarks is None or len(landmarks) < 29:
        return None, float("inf")

    left_score = mean_visibility(landmarks, [11, 13, 15, 23, 25, 27])
    right_score = mean_visibility(landmarks, [12, 14, 16, 24, 26, 28])
    indices = (11, 13, 15, 23, 25, 27) if left_score > right_score else (12, 14, 16, 24, 26, 28)
    if mean_visibility(landmarks, list(indices)) < 0.35:
        return None, float("inf")

    shoulder = landmark_xy(landmarks, indices[0], width, height)
    elbow = landmark_xy(landmarks, indices[1], width, height)
    wrist = landmark_xy(landmarks, indices[2], width, height)
    hip = landmark_xy(landmarks, indices[3], width, height)
    knee = landmark_xy(landmarks, indices[4], width, height)
    ankle = landmark_xy(landmarks, indices[5], width, height)

    metrics = ShotMetrics(
        elbow=calculate_angle(shoulder, elbow, wrist),
        shoulder=calculate_angle(elbow, shoulder, hip),
        hip=calculate_angle(shoulder, hip, knee),
        knee=calculate_angle(hip, knee, ankle),
    )
    if any(math.isnan(v) or math.isinf(v) for v in metrics.to_dict().values()):
        return None, float("inf")
    return metrics, float(wrist[1])


def analyze_video(video_path: Path, max_frames: int = 240, render_height: int = 720) -> AnalysisResult:
    cap = None
    detector = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return AnalysisResult(error="The uploaded video could not be opened.")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        detector = create_pose_detector(fps=fps)
        best_candidate: Optional[Tuple[float, int, ShotMetrics, np.ndarray]] = None
        frames_scanned = 0

        while frames_scanned < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_to_height(frame, render_height)
            landmarks = detect_primary_pose(frame, detector)
            if landmarks is not None:
                height, width = frame.shape[:2]
                metrics, wrist_y = collect_metrics(landmarks, width, height)
                if metrics is not None and (best_candidate is None or wrist_y < best_candidate[0]):
                    best_candidate = (
                        wrist_y,
                        frames_scanned,
                        metrics,
                        draw_pose(frame.copy(), landmarks),
                    )
            frames_scanned += 1

        if best_candidate is None:
            return AnalysisResult(error="No reliable pose was detected.", frames_scanned=frames_scanned)

        _, release_frame_index, metrics, release_frame = best_candidate
        return AnalysisResult(
            metrics=metrics,
            release_frame=release_frame,
            release_frame_index=release_frame_index,
            frames_scanned=frames_scanned,
        )
    except Exception as exc:
        return AnalysisResult(error=f"{type(exc).__name__}: {exc}", detail=traceback.format_exc(limit=8))
    finally:
        if cap is not None:
            cap.release()
        if detector is not None:
            close_detector(detector)


def aggregate_metrics(samples: List[ShotMetrics]) -> ShotMetrics:
    values = np.array([[m.elbow, m.shoulder, m.hip, m.knee] for m in samples], dtype=np.float32)
    med = np.median(values, axis=0)
    return ShotMetrics(float(med[0]), float(med[1]), float(med[2]), float(med[3]))


def compare_rows(user: ShotMetrics, reference: ShotMetrics, reference_name: str) -> pd.DataFrame:
    user_values = user.to_dict()
    reference_values = reference.to_dict()
    rows = []
    for metric in METRIC_ORDER:
        user_value = user_values[metric]
        ref_value = reference_values[metric]
        rows.append(
            {
                "Metric": metric,
                "Your shot": round(user_value, 1),
                reference_name: round(ref_value, 1),
                "Delta": round(user_value - ref_value, 1),
                "Gap": round(abs(user_value - ref_value), 1),
            }
        )
    return pd.DataFrame(rows)


def safe_metric_text(metrics: ShotMetrics, reference_name: str) -> List[str]:
    values = metrics.to_dict()
    return [
        f"{reference_name}",
        f"Elbow: {values['Elbow']:.1f} deg",
        f"Shoulder: {values['Shoulder']:.1f} deg",
        f"Hip: {values['Hip']:.1f} deg",
        f"Knee: {values['Knee']:.1f} deg",
    ]


def draw_overlay_lines(frame: np.ndarray, lines: List[str], origin: Tuple[int, int]) -> None:
    x, y = origin
    cv2.rectangle(frame, (x - 12, y - 32), (x + 330, y + 26 * len(lines)), (20, 26, 24), -1)
    for i, line in enumerate(lines):
        color = (255, 255, 255) if i == 0 else (225, 238, 232)
        cv2.putText(frame, line, (x, y + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def slow_motion_fps(source_fps: float, slow_factor: int) -> float:
    """Return a slowed playback rate without also duplicating frames."""
    return max(1.0, float(source_fps) / max(1, int(slow_factor)))


def make_saved_model_comparison_video(
    user_path: Path,
    user_result: AnalysisResult,
    reference_metrics: ShotMetrics,
    reference_name: str,
    max_frames: int,
    window: int = 24,
    slow_factor: int = 3,
    render_height: int = 540,
) -> Tuple[Optional[Path], str]:
    cap = None
    detector = None
    writer = None
    out_path: Optional[Path] = None
    frames_written = 0
    succeeded = False
    try:
        cap = cv2.VideoCapture(str(user_path))
        if not cap.isOpened():
            return None, "Could not open your clip for video rendering."

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        detector = create_pose_detector(fps=fps)
        frame_index = 0
        center = user_result.release_frame_index if user_result.release_frame_index >= 0 else 0
        start = max(0, center - window)
        end = min(max_frames, center + window + 1)

        while frame_index < end:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index >= start:
                frame = resize_to_height(frame, render_height)
                landmarks = detect_primary_pose(frame, detector)
                user_metrics = None
                if landmarks is not None:
                    height, width = frame.shape[:2]
                    user_metrics, _ = collect_metrics(landmarks, width, height)
                    draw_pose(frame, landmarks)

                left = frame.copy()
                right = (frame.astype(np.float32) * 0.72).astype(np.uint8)

                if user_metrics is not None:
                    draw_overlay_lines(left, safe_metric_text(user_metrics, "Your shot"), (24, 48))
                draw_overlay_lines(right, safe_metric_text(reference_metrics, reference_name), (24, 48))
                cv2.putText(left, "YOUR RELEASE MOTION", (24, left.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(right, "METRIC TARGET - SAME MOTION", (24, right.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
                combined = np.hstack([left, right])

                if writer is None:
                    out_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name)
                    height, width = combined.shape[:2]
                    writer = cv2.VideoWriter(
                        str(out_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        slow_motion_fps(fps, slow_factor),
                        (width, height),
                    )
                    if not writer.isOpened():
                        return None, "Could not open video writer."

                writer.write(combined)
                frames_written += 1
            frame_index += 1

        if frames_written == 0:
            return None, "No frames were available for rendering."
        writer.release()
        writer = None
        if out_path is None or not out_path.exists() or out_path.stat().st_size == 0:
            return None, "The rendered video file is empty."
        succeeded = True
        return out_path, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        if writer is not None:
            writer.release()
        if not succeeded and out_path is not None:
            out_path.unlink(missing_ok=True)
        if cap is not None:
            cap.release()
        if detector is not None:
            close_detector(detector)


def comparison_offset_bounds(
    user_center_seconds: float,
    reference_center_seconds: float,
    user_fps: float,
    reference_fps: float,
    sample_fps: float,
    max_frames: int,
    window: int,
) -> Tuple[int, int]:
    lower = max(
        -window,
        math.ceil(-user_center_seconds * sample_fps),
        math.ceil(-reference_center_seconds * sample_fps),
    )
    user_limit = max_frames / max(user_fps, 1e-6)
    reference_limit = max_frames / max(reference_fps, 1e-6)
    upper = min(
        window,
        math.floor((user_limit - user_center_seconds) * sample_fps),
        math.floor((reference_limit - reference_center_seconds) * sample_fps),
    )
    return int(lower), int(upper)


def make_reference_clip_comparison_video(
    user_path: Path,
    user_result: AnalysisResult,
    reference_path: Path,
    reference_result: AnalysisResult,
    max_frames: int,
    window: int = 24,
    slow_factor: int = 3,
    render_height: int = 540,
) -> Tuple[Optional[Path], str]:
    user_cap = None
    reference_cap = None
    user_detector = None
    reference_detector = None
    writer = None
    out_path: Optional[Path] = None
    frames_written = 0
    succeeded = False
    try:
        user_cap = cv2.VideoCapture(str(user_path))
        reference_cap = cv2.VideoCapture(str(reference_path))
        if not user_cap.isOpened() or not reference_cap.isOpened():
            return None, "Could not open one or both clips for rendering."

        user_fps = float(user_cap.get(cv2.CAP_PROP_FPS) or 30.0)
        reference_fps = float(reference_cap.get(cv2.CAP_PROP_FPS) or 30.0)
        sample_fps = max(1.0, min(user_fps, reference_fps, 30.0))
        user_detector = create_pose_detector(fps=sample_fps)
        reference_detector = create_pose_detector(fps=sample_fps)

        user_center_seconds = max(0, user_result.release_frame_index) / user_fps
        reference_center_seconds = max(0, reference_result.release_frame_index) / reference_fps
        first_offset, last_offset = comparison_offset_bounds(
            user_center_seconds,
            reference_center_seconds,
            user_fps,
            reference_fps,
            sample_fps,
            max_frames,
            window,
        )

        for offset in range(first_offset, last_offset + 1):
            user_seconds = user_center_seconds + offset / sample_fps
            reference_seconds = reference_center_seconds + offset / sample_fps
            user_cap.set(cv2.CAP_PROP_POS_MSEC, user_seconds * 1000.0)
            reference_cap.set(cv2.CAP_PROP_POS_MSEC, reference_seconds * 1000.0)
            user_ok, left = user_cap.read()
            reference_ok, right = reference_cap.read()
            if not user_ok or not reference_ok:
                continue

            left = resize_to_height(left, render_height)
            right = resize_to_height(right, render_height)
            left_landmarks = detect_primary_pose(left, user_detector)
            right_landmarks = detect_primary_pose(right, reference_detector)
            if left_landmarks is not None:
                draw_pose(left, left_landmarks)
            if right_landmarks is not None:
                draw_pose(right, right_landmarks)

            cv2.putText(left, "YOUR CLIP", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(right, "REFERENCE CLIP", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            combined = np.hstack([left, right])

            if writer is None:
                out_path = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name)
                height, width = combined.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    slow_motion_fps(sample_fps, slow_factor),
                    (width, height),
                )
                if not writer.isOpened():
                    return None, "Could not open video writer."

            writer.write(combined)
            frames_written += 1

        if frames_written == 0:
            return None, "No frames were available for rendering."
        writer.release()
        writer = None
        if out_path is None or not out_path.exists() or out_path.stat().st_size == 0:
            return None, "The rendered video file is empty."
        succeeded = True
        return out_path, ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        if writer is not None:
            writer.release()
        if not succeeded and out_path is not None:
            out_path.unlink(missing_ok=True)
        if user_cap is not None:
            user_cap.release()
        if reference_cap is not None:
            reference_cap.release()
        if user_detector is not None:
            close_detector(user_detector)
        if reference_detector is not None:
            close_detector(reference_detector)


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
          --bg: #f7f7f3;
          --ink: #202623;
          --muted: #67716d;
          --line: #dce3de;
          --panel: #ffffff;
          --soft: #eef4f0;
          --green: #136f63;
          --coral: #c95132;
        }
        .stApp {
          background:
            linear-gradient(120deg, rgba(19,111,99,.08), transparent 30%),
            linear-gradient(300deg, rgba(201,81,50,.08), transparent 26%),
            var(--bg);
        }
        [data-testid="stSidebar"] {
          background: #ffffff;
          border-right: 1px solid var(--line);
        }
        h1, h2, h3 {
          color: var(--ink);
          letter-spacing: 0;
        }
        .topbar {
          display: flex;
          justify-content: space-between;
          gap: 16px;
          align-items: end;
          padding: 10px 0 18px 0;
          border-bottom: 1px solid var(--line);
          margin-bottom: 18px;
        }
        .title {
          font-size: 30px;
          line-height: 1.05;
          font-weight: 760;
          color: var(--ink);
        }
        .chips {
          display: flex;
          gap: 8px;
          flex-wrap: wrap;
          justify-content: flex-end;
        }
        .chip {
          border: 1px solid var(--line);
          border-radius: 999px;
          padding: 6px 10px;
          background: rgba(255,255,255,.8);
          color: var(--muted);
          font-size: 13px;
        }
        .chip strong {
          color: var(--ink);
        }
        .panel {
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 16px;
          background: rgba(255,255,255,.88);
          margin-bottom: 14px;
        }
        .metric-grid {
          display: grid;
          grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 10px;
          margin: 8px 0 14px 0;
        }
        .metric {
          border: 1px solid var(--line);
          border-radius: 8px;
          background: var(--soft);
          padding: 10px 12px;
        }
        .metric .label {
          color: var(--muted);
          font-size: 12px;
        }
        .metric .value {
          color: var(--ink);
          font-size: 22px;
          font-weight: 760;
        }
        .stButton > button {
          border-radius: 8px;
          border: 1px solid #a53e25;
          background: var(--coral);
          color: #fff;
          font-weight: 720;
        }
        @media (max-width: 760px) {
          .topbar { flex-direction: column; align-items: start; }
          .chips { justify-content: flex-start; }
          .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_error(message: str, detail: str = "") -> None:
    st.error(message)
    if detail:
        with st.expander("Debug detail"):
            st.code(detail)


def topbar(models: Dict[str, Dict]) -> None:
    ok, runtime = runtime_status()
    status = "Ready" if ok else "Needs setup"
    st.markdown(
        f"""
        <div class="topbar">
          <div class="title">Shooting Form Studio</div>
          <div class="chips">
            <div class="chip">Runtime: <strong>{status}</strong></div>
            <div class="chip">Saved models: <strong>{len(models)}</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not ok:
        st.warning(runtime)


def metric_cards(df: pd.DataFrame) -> None:
    html = ['<div class="metric-grid">']
    for row in df.to_dict("records"):
        html.append(
            f'<div class="metric"><div class="label">{row["Metric"]}</div>'
            f'<div class="value">{row["Gap"]:.1f} deg</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_results(user_result: AnalysisResult, reference_result: AnalysisResult, reference_name: str) -> None:
    df = compare_rows(user_result.metrics, reference_result.metrics, reference_name)
    st.markdown("### Comparison")
    metric_cards(df)
    st.dataframe(df, width="stretch", hide_index=True)
    st.bar_chart(df[["Metric", "Gap"]].set_index("Metric"))

    left, right = st.columns(2)
    with left:
        if user_result.release_frame is not None:
            st.markdown("#### Your release frame")
            st.image(cv2.cvtColor(user_result.release_frame, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
    with right:
        if reference_result.release_frame is not None:
            st.markdown("#### Reference release frame")
            st.image(cv2.cvtColor(reference_result.release_frame, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")


def analyze_tab(models: Dict[str, Dict], max_frames: int) -> None:
    left, right = st.columns([1.05, 0.95])
    with left:
        st.markdown('<div class="panel"><h3>Your clip</h3>', unsafe_allow_html=True)
        user_video = st.file_uploader(
            "Your clip",
            type=VIDEO_TYPES,
            label_visibility="collapsed",
            key="analyze_user_video",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><h3>Reference</h3>', unsafe_allow_html=True)
        source = st.radio(
            "Reference source",
            ["Saved model", "Reference clip"],
            horizontal=True,
            label_visibility="collapsed",
            key="analyze_reference_source",
        )
        selected_model = None
        reference_video = None
        if source == "Saved model":
            names = sorted(models.keys())
            selected_model = st.selectbox("Player model", names if names else ["<none>"], key="analyze_model_select")
        else:
            reference_video = st.file_uploader("Reference video", type=VIDEO_TYPES, key="analyze_reference_video")
        st.markdown("</div>", unsafe_allow_html=True)

    render_video = st.checkbox("Create slow-motion comparison video", value=True, key="render_comparison_video")
    if not st.button("Analyze form", type="primary", key="analyze_form_button"):
        return
    if user_video is None:
        st.warning("Upload your clip first.")
        return

    with st.spinner("Analyzing video frames..."):
        user_path = save_upload(user_video)
        user_result = analyze_video(user_path, max_frames=max_frames)

        if source == "Saved model":
            if not selected_model or selected_model == "<none>" or selected_model not in models:
                show_error("Choose a saved model or upload a reference clip.")
                return
            reference_metrics = ShotMetrics.from_dict(models[selected_model]["metrics"])
            reference_result = AnalysisResult(metrics=reference_metrics)
            reference_name = selected_model
            reference_path = None
        else:
            if reference_video is None:
                show_error("Upload a reference clip.")
                return
            reference_path = save_upload(reference_video)
            reference_result = analyze_video(reference_path, max_frames=max_frames)
            reference_name = "Reference clip"

    if not user_result.ok:
        show_error("No reliable pose was detected in your clip.", user_result.detail or user_result.error)
        return
    if not reference_result.ok:
        show_error("No reliable pose was detected in the reference.", reference_result.detail or reference_result.error)
        return

    render_results(user_result, reference_result, reference_name)
    if render_video:
        with st.spinner("Rendering slow-motion comparison video..."):
            if source == "Saved model":
                comparison_path, render_error = make_saved_model_comparison_video(
                    user_path,
                    user_result,
                    reference_result.metrics,
                    reference_name,
                    max_frames=max_frames,
                )
            else:
                comparison_path, render_error = make_reference_clip_comparison_video(
                    user_path,
                    user_result,
                    reference_path,
                    reference_result,
                    max_frames=max_frames,
                )
        if render_error:
            show_error("The analysis succeeded, but the comparison video could not be rendered.", render_error)
        elif comparison_path is not None and comparison_path.exists():
            video_bytes = comparison_path.read_bytes()
            st.markdown("### Slow-motion comparison video")
            st.video(video_bytes, format="video/mp4")
            st.download_button(
                "Download comparison video",
                data=video_bytes,
                file_name="shooting_form_comparison.mp4",
                mime="video/mp4",
                key="download_comparison_video",
            )


def models_tab(models: Dict[str, Dict], max_frames: int) -> None:
    left, right = st.columns([1.0, 1.0])
    with left:
        st.markdown('<div class="panel"><h3>Build model</h3>', unsafe_allow_html=True)
        name = st.text_input("Player name", value="Stephen Curry", key="model_player_name")
        uploads = st.file_uploader("Model clips", type=VIDEO_TYPES, accept_multiple_files=True, key="model_clips")
        build = st.button("Build model", type="primary", key="build_model_button")
        st.markdown("</div>", unsafe_allow_html=True)

    if build:
        if not name.strip() or not uploads:
            st.warning("Enter a player name and upload at least one clip.")
        else:
            samples: List[ShotMetrics] = []
            errors: List[str] = []
            with st.spinner("Building player model..."):
                for upload in uploads:
                    result = analyze_video(save_upload(upload), max_frames=max_frames)
                    if result.ok:
                        samples.append(result.metrics)
                    else:
                        errors.append(f"{upload.name}: {result.error}")
            if not samples:
                show_error("Could not build a model from those clips.", "\n".join(errors))
            else:
                models[name.strip()] = {
                    "sample_count": len(samples),
                    "metrics": aggregate_metrics(samples).to_model_dict(),
                }
                save_player_models(models)
                st.success("Model saved.")
                st.rerun()

    with right:
        st.markdown('<div class="panel"><h3>Saved models</h3>', unsafe_allow_html=True)
        if not models:
            st.info("No saved models yet.")
        else:
            rows = []
            for player, data in sorted(models.items()):
                row = {"Player": player, "Clips": int(data.get("sample_count", 0))}
                row.update({k: round(v, 1) for k, v in ShotMetrics.from_dict(data.get("metrics", {})).to_dict().items()})
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)


def run_app() -> None:
    st.set_page_config(page_title="Shooting Form Studio", layout="wide", initial_sidebar_state="expanded")
    inject_theme()
    models = load_player_models()
    st.sidebar.markdown("### Settings")
    max_frames = st.sidebar.slider("Max scan frames", 60, 600, 240, step=30)
    ok, runtime = runtime_status()
    st.sidebar.caption(f"Runtime: {runtime}")
    topbar(models)

    tab_analyze, tab_models = st.tabs(["Analyze", "Models"])
    with tab_analyze:
        analyze_tab(models, max_frames)
    with tab_models:
        models_tab(models, max_frames)


if __name__ == "__main__":
    run_app()
