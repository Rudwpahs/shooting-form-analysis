import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
try:
    import mediapipe as mp
except Exception as exc:
    raise RuntimeError(
        "Failed to import 'mediapipe'. Install the official package with: pip install mediapipe"
    ) from exc

import numpy as np
import pandas as pd
import streamlit as st

PLAYER_MODELS_PATH = Path(__file__).resolve().parent / "models" / "nba_player_models.json"
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (24, 26),
    (25, 27), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32),
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
        self._frame_interval_ms = max(1, int(round(1000.0 / max(fps, 1e-6))))

    def process(self, frame_rgb: np.ndarray) -> _PoseProcessResult:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(image, self._timestamp_ms)
        self._timestamp_ms += self._frame_interval_ms

        if result.pose_landmarks:
            return _PoseProcessResult(_PoseLandmarksContainer(result.pose_landmarks[0]))
        return _PoseProcessResult(None)

    def close(self):
        self._landmarker.close()


def _create_solutions_pose_detector():
    pose_cls = mp.solutions.pose.Pose
    return pose_cls(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _create_tasks_pose_detector(fps: float):
    try:
        vision = mp.tasks.vision
        base_options = mp.tasks.BaseOptions
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe is installed but neither solutions.pose nor tasks APIs are usable."
        ) from exc

    model_path = Path(__file__).resolve().parent / "models" / "pose_landmarker_full.task"
    if not model_path.exists():
        raise RuntimeError(
            "PoseLandmarker model file not found at "
            f"{model_path}. Add the .task model or install a mediapipe build that includes solutions.pose."
        )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    return _TasksPoseAdapter(landmarker, fps=fps)


def create_pose_detector(fps: float = 30.0):
    """Create a MediaPipe Pose detector across package variants."""
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        return _create_solutions_pose_detector()
    return _create_tasks_pose_detector(fps=fps)


@dataclass
class ShotMetrics:
    elbow_angle: float
    shoulder_angle: float
    hip_angle: float
    knee_angle: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "Elbow angle": self.elbow_angle,
            "Shoulder angle": self.shoulder_angle,
            "Hip angle": self.hip_angle,
            "Knee angle": self.knee_angle,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "ShotMetrics":
        return cls(data["Elbow angle"], data["Shoulder angle"], data["Hip angle"], data["Knee angle"])


def aggregate_metrics(samples: List["ShotMetrics"]) -> "ShotMetrics":
    arr = np.array(
        [[s.elbow_angle, s.shoulder_angle, s.hip_angle, s.knee_angle] for s in samples],
        dtype=np.float32,
    )
    med = np.median(arr, axis=0)
    return ShotMetrics(float(med[0]), float(med[1]), float(med[2]), float(med[3]))


def load_player_models(path: Path = PLAYER_MODELS_PATH) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_player_models(models: Dict[str, Dict], path: Path = PLAYER_MODELS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(models, indent=2), encoding="utf-8")


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return float("nan")
    cos_angle = float(np.dot(ba, bc) / denom)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def landmark_xy(landmarks, idx: int, w: int, h: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def get_shooting_side(landmarks) -> str:
    left_vis = landmarks[15].visibility + landmarks[13].visibility + landmarks[11].visibility
    right_vis = landmarks[16].visibility + landmarks[14].visibility + landmarks[12].visibility
    return "left" if left_vis > right_vis else "right"


def collect_metrics_for_frame(landmarks, w: int, h: int) -> Tuple[ShotMetrics, float]:
    side = get_shooting_side(landmarks)
    s_idx, e_idx, w_idx, h_idx, k_idx, a_idx = (11, 13, 15, 23, 25, 27) if side == "left" else (12, 14, 16, 24, 26, 28)

    shoulder = landmark_xy(landmarks, s_idx, w, h)
    elbow = landmark_xy(landmarks, e_idx, w, h)
    wrist = landmark_xy(landmarks, w_idx, w, h)
    hip = landmark_xy(landmarks, h_idx, w, h)
    knee = landmark_xy(landmarks, k_idx, w, h)
    ankle = landmark_xy(landmarks, a_idx, w, h)

    metrics = ShotMetrics(
        elbow_angle=calculate_angle(shoulder, elbow, wrist),
        shoulder_angle=calculate_angle(elbow, shoulder, hip),
        hip_angle=calculate_angle(shoulder, hip, knee),
        knee_angle=calculate_angle(hip, knee, ankle),
    )
    return metrics, float(wrist[1])


def draw_pose(frame: np.ndarray, landmarks, draw_indices: bool = False) -> np.ndarray:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in POSE_CONNECTIONS:
        if 0 <= a < len(pts) and 0 <= b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)

    for i, (x, y) in enumerate(pts):
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
        if draw_indices:
            cv2.putText(frame, str(i), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return frame


def detect_primary_pose(frame_bgr: np.ndarray, pose_detector):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = pose_detector.process(rgb)
    if result.pose_landmarks:
        return result.pose_landmarks.landmark
    return None


def collect_pose_metrics(landmarks, w: int, h: int) -> Optional[Dict[str, float]]:
    if landmarks is None or len(landmarks) < 29:
        return None
    metrics, _ = collect_metrics_for_frame(landmarks, w, h)
    metric_dict = {
        "Elbow": metrics.elbow_angle,
        "Shoulder": metrics.shoulder_angle,
        "Hip": metrics.hip_angle,
        "Knee": metrics.knee_angle,
    }
    if any(np.isnan(list(metric_dict.values()))):
        return None
    return metric_dict


def resize_to_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == target_h:
        return frame
    new_w = max(1, int(round(w * target_h / h)))
    return cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_AREA)


def combine_side_by_side(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    target_h = max(frame_a.shape[0], frame_b.shape[0])
    left = resize_to_height(frame_a, target_h)
    right = resize_to_height(frame_b, target_h)
    return np.hstack([left, right])


def _draw_alpha_rect(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    alpha: float = 0.65,
):
    h, w = canvas.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = canvas[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


def draw_diff_overlay(
    canvas: np.ndarray,
    m1: Optional[Dict[str, float]],
    m2: Optional[Dict[str, float]],
    offset: int,
    window: int,
    is_release: bool,
):
    h, w = canvas.shape[:2]
    half = w // 2
    font_scale = max(0.52, min(0.9, h / 930.0))
    line_h = int(round(34 * font_scale))
    thick = 2

    # Top header bars for each side.
    bar_h = int(max(52, h * 0.078))
    _draw_alpha_rect(canvas, 0, 0, half, bar_h, (38, 78, 64), alpha=0.72)
    _draw_alpha_rect(canvas, half, 0, w, bar_h, (82, 53, 34), alpha=0.72)
    cv2.putText(canvas, "VIDEO A", (20, int(bar_h * 0.66)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220, 255, 236), thick)
    cv2.putText(canvas, "VIDEO B", (half + 20, int(bar_h * 0.66)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (235, 230, 220), thick)

    # Timeline chip at center.
    chip_w = int(max(170, w * 0.19))
    chip_h = int(max(36, h * 0.055))
    chip_x1 = (w - chip_w) // 2
    chip_y1 = int(bar_h * 0.18)
    chip_color = (18, 132, 116) if is_release else (36, 36, 36)
    _draw_alpha_rect(canvas, chip_x1, chip_y1, chip_x1 + chip_w, chip_y1 + chip_h, chip_color, alpha=0.82)
    cv2.putText(
        canvas,
        f"t={offset:+d}",
        (chip_x1 + 18, chip_y1 + int(chip_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.93,
        (255, 255, 255),
        thick,
    )

    # Metrics card.
    card_x = 14
    card_y = bar_h + 10
    card_w = int(min(500, max(360, w * 0.31)))
    lines = ["ABS ANGLE DIFFERENCE"]
    if m1 is None or m2 is None:
        lines.append("Landmarks not detected in one/both frames")
    else:
        for key in ["Elbow", "Shoulder", "Hip", "Knee"]:
            lines.append(f"{key:8s} {abs(m1[key] - m2[key]):5.1f} deg")
    card_h = 14 + line_h * len(lines)
    _draw_alpha_rect(canvas, card_x, card_y, card_x + card_w, card_y + card_h, (22, 24, 27), alpha=0.78)
    for i, text in enumerate(lines):
        color = (255, 214, 140) if i == 0 else (232, 242, 255)
        cv2.putText(
            canvas,
            text,
            (card_x + 14, card_y + 28 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * (0.88 if i == 0 else 0.82),
            color,
            thick,
        )

    # Bottom progress bar across synchronized window.
    prog_y = h - int(max(18, h * 0.03))
    cv2.line(canvas, (26, prog_y), (w - 26, prog_y), (80, 80, 80), 6)
    progress = (offset + window) / max(1.0, float(2 * window))
    px = int(26 + progress * (w - 52))
    cv2.line(canvas, (26, prog_y), (px, prog_y), (74, 188, 158), 6)

    if is_release:
        badge = "SYNCHRONIZED RELEASE"
        badge_w = int(max(270, w * 0.27))
        badge_h = int(max(42, h * 0.06))
        bx1 = (w - badge_w) // 2
        by1 = h - badge_h - 26
        _draw_alpha_rect(canvas, bx1, by1, bx1 + badge_w, by1 + badge_h, (0, 145, 123), alpha=0.82)
        cv2.putText(
            canvas,
            badge,
            (bx1 + 16, by1 + int(badge_h * 0.72)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.78,
            (255, 255, 255),
            thick,
        )


def _collect_video_pose_track(
    video_path: Path,
    max_frames: int = 300,
    render_height: int = 720,
) -> Tuple[List[np.ndarray], List[Optional[Dict[str, float]]], int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], [], -1, 30.0

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    pose = create_pose_detector(fps=fps)
    frames: List[np.ndarray] = []
    metrics_track: List[Optional[Dict[str, float]]] = []
    release_idx = -1
    best_wrist_y = float("inf")
    frame_idx = 0

    try:
        while max_frames <= 0 or frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            vis = resize_to_height(frame, render_height)
            lms = detect_primary_pose(vis, pose)
            metric_dict = None

            if lms is not None:
                draw_pose(vis, lms)
                h, w = vis.shape[:2]
                shot, wrist_y = collect_metrics_for_frame(lms, w, h)
                candidate = {
                    "Elbow": shot.elbow_angle,
                    "Shoulder": shot.shoulder_angle,
                    "Hip": shot.hip_angle,
                    "Knee": shot.knee_angle,
                }
                if not any(np.isnan(list(candidate.values()))):
                    metric_dict = candidate
                    if wrist_y < best_wrist_y:
                        best_wrist_y = wrist_y
                        release_idx = frame_idx

            frames.append(vis)
            metrics_track.append(metric_dict)
            frame_idx += 1
    finally:
        cap.release()
        pose.close()

    return frames, metrics_track, release_idx, fps


def _frame_or_blank(frames: List[np.ndarray], idx: int) -> np.ndarray:
    if 0 <= idx < len(frames):
        return frames[idx].copy()
    return np.zeros_like(frames[0])


def compare_videos(
    video_a: Path,
    video_b: Path,
    max_frames: int = 300,
    window: int = 24,
    slow_factor: int = 3,
    render_height: int = 720,
) -> Tuple[Optional[Path], Optional[pd.DataFrame], int, Optional[np.ndarray], List[np.ndarray]]:
    frames1, metrics1, rel1, fps1 = _collect_video_pose_track(
        video_a,
        max_frames=max_frames,
        render_height=render_height,
    )
    frames2, metrics2, rel2, fps2 = _collect_video_pose_track(
        video_b,
        max_frames=max_frames,
        render_height=render_height,
    )
    if not frames1 or not frames2 or rel1 < 0 or rel2 < 0:
        return None, None, 0, None, []

    out_fps = min(fps1, fps2) if fps1 > 0 and fps2 > 0 else 30.0
    slow_factor = max(1, int(slow_factor))
    window = max(1, int(window))

    timeline_frames: List[np.ndarray] = []
    diffs: List[Dict[str, float]] = []

    for offset in range(-window, window + 1):
        idx1 = rel1 + offset
        idx2 = rel2 + offset

        vis1 = _frame_or_blank(frames1, idx1)
        vis2 = _frame_or_blank(frames2, idx2)
        m1 = metrics1[idx1] if 0 <= idx1 < len(metrics1) else None
        m2 = metrics2[idx2] if 0 <= idx2 < len(metrics2) else None

        combined = combine_side_by_side(vis1, vis2)
        draw_diff_overlay(
            combined,
            m1,
            m2,
            offset=offset,
            window=window,
            is_release=(offset == 0),
        )

        timeline_frames.append(combined)

        if m1 is not None and m2 is not None:
            diffs.append({
                "Elbow": abs(m1["Elbow"] - m2["Elbow"]),
                "Shoulder": abs(m1["Shoulder"] - m2["Shoulder"]),
                "Hip": abs(m1["Hip"] - m2["Hip"]),
                "Knee": abs(m1["Knee"] - m2["Knee"]),
            })

    release_frame = timeline_frames[window] if timeline_frames else None

    out_h, out_w = timeline_frames[0].shape[:2]
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = Path(tmp_out.name)
    tmp_out.close()
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (out_w, out_h))

    frame_count = 0
    if writer.isOpened():
        for frame in timeline_frames:
            for _ in range(slow_factor):
                writer.write(frame)
                frame_count += 1
        writer.release()
    else:
        out_path = None

    summary_df = None
    if diffs:
        ddf = pd.DataFrame(diffs)
        summary_df = pd.DataFrame(
            {
                "Metric": ["Elbow", "Shoulder", "Hip", "Knee"],
                "Mean abs diff (deg)": [ddf["Elbow"].mean(), ddf["Shoulder"].mean(), ddf["Hip"].mean(), ddf["Knee"].mean()],
                "Median abs diff (deg)": [ddf["Elbow"].median(), ddf["Shoulder"].median(), ddf["Hip"].median(), ddf["Knee"].median()],
            }
        )

    return out_path, summary_df, frame_count, release_frame, timeline_frames


def analyze_video(video_path: Path, max_frames: int = 300) -> Optional[ShotMetrics]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    pose = create_pose_detector(fps=fps)

    per_frame: List[Tuple[float, ShotMetrics]] = []
    frame_idx = 0

    try:
        while frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                metrics, wrist_y = collect_metrics_for_frame(result.pose_landmarks.landmark, w, h)
                if not any(np.isnan(list(metrics.to_dict().values()))):
                    per_frame.append((wrist_y, metrics))

            frame_idx += 1
    finally:
        cap.release()
        pose.close()

    if not per_frame:
        return None

    per_frame.sort(key=lambda item: item[0])
    return per_frame[0][1]


def save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def render_result(player_name: str, user_metrics: ShotMetrics, nba_metrics: ShotMetrics):
    user_dict = user_metrics.to_dict()
    nba_dict = nba_metrics.to_dict()

    rows = []
    for name, user_value in user_dict.items():
        nba_value = nba_dict[name]
        rows.append({
            "Metric": name,
            "Your shot (°)": round(user_value, 1),
            f"{player_name} (°)": round(nba_value, 1),
            "Difference (°)": round(user_value - nba_value, 1),
        })

    st.subheader("Comparison")
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Absolute angle gap")
    chart_df = pd.DataFrame({
        "Metric": [row["Metric"] for row in rows],
        "Gap": [abs(row["Difference (°)"]) for row in rows],
    }).set_index("Metric")
    st.bar_chart(chart_df)


def main():
    st.set_page_config(page_title="Basketball Shooting Form Analyzer", layout="wide")
    st.title("🏀 Basketball Shooting Form Analyzer")
    st.write("Upload your video and compare your release posture against an NBA player model built from real clips.")

    models = load_player_models()
    available_players = sorted(models.keys())
    if not available_players:
        st.info("No built-in NBA player models found yet. Build one below from uploaded NBA clips.")

    st.subheader("Build or update NBA player model")
    model_player_name = st.text_input("Player name for model", value="Stephen Curry")
    model_clips = st.file_uploader(
        "Upload one or more NBA reference clips",
        type=["mp4", "mov", "avi", "mkv"],
        accept_multiple_files=True,
        key="model_clips",
        help="Use clear side-angle shooting clips. More clips improve robustness.",
    )
    if st.button("Build model from uploaded clips"):
        if not model_player_name.strip():
            st.error("Enter a player name for the model.")
            return
        if not model_clips:
            st.error("Upload at least one NBA reference clip.")
            return

        sample_metrics: List[ShotMetrics] = []
        with st.spinner("Analyzing NBA clips and building model..."):
            for clip in model_clips:
                clip_path = save_upload(clip)
                clip_metrics = analyze_video(clip_path)
                if clip_metrics is not None:
                    sample_metrics.append(clip_metrics)

        if not sample_metrics:
            st.error("Could not extract landmarks from the uploaded clips.")
            return

        models[model_player_name.strip()] = {
            "sample_count": len(sample_metrics),
            "metrics": aggregate_metrics(sample_metrics).to_dict(),
        }
        save_player_models(models)
        st.success(f"Saved model for {model_player_name.strip()} using {len(sample_metrics)} clip(s).")
        st.rerun()

    player_name = st.selectbox(
        "Choose NBA player model",
        available_players if available_players else ["<none>"],
    )
    compare_mode = st.radio(
        "Reference type",
        ["Use built-in NBA profile", "Upload custom NBA reference clip"],
        horizontal=True,
    )

    nba_video = None
    if compare_mode == "Upload custom NBA reference clip":
        nba_video = st.file_uploader(
            f"Upload reference video for {player_name}",
            type=["mp4", "mov", "avi", "mkv"],
            key="nba",
            help="Use a clear side-angle shooting clip.",
        )

    user_video = st.file_uploader(
        "Upload your shooting video",
        type=["mp4", "mov", "avi", "mkv"],
        key="user",
        help="Use a similar camera angle to improve comparison accuracy.",
    )

    if st.button("Analyze form", type="primary"):
        if not user_video:
            st.error("Please upload your video first.")
            return

        with st.spinner("Analyzing pose and extracting release-frame angles..."):
            user_path = save_upload(user_video)
            user_metrics = analyze_video(user_path)

            if compare_mode == "Upload custom NBA reference clip":
                if not nba_video:
                    st.error("Please upload an NBA reference video or switch to built-in profile mode.")
                    return
                nba_path = save_upload(nba_video)
                nba_metrics = analyze_video(nba_path)
            else:
                if player_name == "<none>" or player_name not in models:
                    st.error("No built-in model selected. Build one above or upload a custom reference clip.")
                    return
                nba_metrics = ShotMetrics.from_dict(models[player_name]["metrics"])

        if user_metrics is None:
            st.error("Could not detect body landmarks from your video. Try a clearer side view.")
            return
        if nba_metrics is None:
            st.error("Could not detect landmarks from NBA reference video. Try another clip or built-in mode.")
            return

        render_result(player_name, user_metrics, nba_metrics)

    st.divider()
    st.subheader("Two-video pose comparison viewer")
    st.write("Upload two videos to visualize pose estimation side-by-side and frame-level angle differences.")

    col_a, col_b = st.columns(2)
    with col_a:
        video_a = st.file_uploader(
            "Video A",
            type=["mp4", "mov", "avi", "mkv"],
            key="compare_a",
        )
    with col_b:
        video_b = st.file_uploader(
            "Video B",
            type=["mp4", "mov", "avi", "mkv"],
            key="compare_b",
        )

    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        compare_max_frames = st.number_input("Max frames", min_value=0, value=300, step=50)
    with cfg2:
        compare_window = st.number_input("Release window", min_value=5, value=24, step=1)
    with cfg3:
        compare_slow = st.number_input("Slow-mo x", min_value=1, value=3, step=1)

    with st.expander("Advanced compare settings"):
        render_height = st.number_input("Render height (px)", min_value=360, max_value=1440, value=720, step=60)

    if st.button("Generate pose comparison video"):
        if not video_a or not video_b:
            st.error("Upload both videos first.")
            return

        with st.spinner("Generating side-by-side pose comparison..."):
            path_a = save_upload(video_a)
            path_b = save_upload(video_b)
            compare_path, summary_df, frame_count, release_frame, timeline_frames = compare_videos(
                path_a,
                path_b,
                max_frames=int(compare_max_frames),
                window=int(compare_window),
                slow_factor=int(compare_slow),
                render_height=int(render_height),
            )

        if release_frame is None or not timeline_frames:
            st.error("Could not find a clear release frame in one or both videos.")
            return

        st.success("Release-synchronized comparison generated.")
        st.subheader("Synchronized release frame")
        st.image(cv2.cvtColor(release_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        if compare_path is not None and frame_count > 0:
            video_bytes = compare_path.read_bytes()
            st.video(video_bytes, format="video/mp4")
            st.download_button(
                "Download comparison video",
                data=video_bytes,
                file_name="pose_comparison_release_sync_slowmo.mp4",
                mime="video/mp4",
            )
        else:
            st.warning("Video codec playback is unavailable in this environment. Use the frame scrubber below.")

        st.subheader("Frame scrubber")
        scrub_idx = st.slider("Synchronized frame index", min_value=0, max_value=len(timeline_frames) - 1, value=len(timeline_frames) // 2)
        st.image(cv2.cvtColor(timeline_frames[scrub_idx], cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        if summary_df is not None:
            st.subheader("Angle difference summary")
            st.dataframe(
                summary_df.style.format({"Mean abs diff (deg)": "{:.1f}", "Median abs diff (deg)": "{:.1f}"}),
                use_container_width=True,
            )


I18N = {
    "en": {
        "app_title": "Shooting Form Studio",
        "hero_desc": "Build real NBA pose models, compare your shot to a reference, and run synchronized release-frame analysis.",
        "saved_models": "Saved NBA Models",
        "compare_modes": "Compare Modes",
        "release_sync": "Release Sync",
        "enabled": "Enabled",
        "tab_models": "NBA Model Builder",
        "tab_single": "Single Video Compare",
        "tab_sync": "Release Sync Compare",
        "lang": "Language",
        "build_update": "Build or Update Player Model",
        "player_name": "Player name",
        "nba_clips": "NBA reference clips",
        "build_model": "Build model",
        "saved_models_hdr": "Saved Models",
        "no_models": "No models yet.",
        "clips": "Clips",
        "enter_name": "Enter a player name.",
        "upload_one_clip": "Upload at least one NBA clip.",
        "building_model": "Building model from uploaded clips...",
        "landmark_fail": "Could not extract reliable landmarks from the clips.",
        "saved_model_msg": "Saved model for {player} using {count} clip(s).",
        "compare_hdr": "Compare Your Shot Against Reference",
        "ref_player": "Reference player model",
        "ref_source": "Reference source",
        "saved_model_src": "Use saved player model",
        "custom_clip_src": "Upload custom reference clip",
        "your_video": "Your shooting video",
        "ref_video": "Reference video",
        "run_single": "Run single-video comparison",
        "upload_your_video": "Upload your shooting video first.",
        "extracting": "Extracting release-frame pose metrics...",
        "user_landmark_fail": "Could not detect body landmarks from your video.",
        "ref_fail": "Could not resolve reference metrics. Choose a saved model or upload a clearer reference clip.",
        "sync_subfeature": "Release Sync Sub-feature",
        "sync_needs_clip": "Release sync runs automatically when a custom reference clip is uploaded.",
        "custom_ref": "Custom Reference",
        "sync_hdr": "Synchronized Release-Frame Comparison",
        "sync_caption": "Both videos are aligned at full release, then rendered in slow motion.",
        "video_a": "Video A",
        "video_b": "Video B",
        "max_scan": "Max scan frames",
        "window": "Window (+/- frames)",
        "slow_motion": "Slow motion",
        "render_height": "Render height",
        "gen_sync": "Generate synchronized comparison",
        "upload_both": "Upload both videos first.",
        "running_sync": "Running synchronized release-frame comparison...",
        "no_release": "Could not find a clear release frame in one or both videos.",
        "sync_ready": "Synchronized comparison is ready.",
        "release_frame_aligned": "Release Frame (Aligned)",
        "download_sync": "Download synchronized video",
        "video_playback_warn": "Video playback is unavailable in this runtime. Use frame scrubber below.",
        "frame_scrubber": "Frame Scrubber",
        "sync_idx": "Synchronized frame index",
        "angle_diff_summary": "Angle Difference Summary",
        "angle_comparison": "Angle Comparison",
        "your_shot_deg": "Your shot (deg)",
        "difference_deg": "Difference (deg)",
        "abs_gap": "Absolute Angle Gap",
        "gap_deg": "Gap (deg)",
    },
    "ko": {
        "app_title": "슛 폼 스튜디오",
        "hero_desc": "실제 NBA 포즈 모델을 만들고, 내 슛 폼을 비교하며, 릴리즈 프레임 동기화 분석을 수행합니다.",
        "saved_models": "저장된 NBA 모델",
        "compare_modes": "비교 모드",
        "release_sync": "릴리즈 동기화",
        "enabled": "활성화",
        "tab_models": "NBA 모델 생성",
        "tab_single": "단일 영상 비교",
        "tab_sync": "릴리즈 동기화 비교",
        "lang": "언어",
        "build_update": "선수 모델 생성/업데이트",
        "player_name": "선수 이름",
        "nba_clips": "NBA 기준 영상",
        "build_model": "모델 생성",
        "saved_models_hdr": "저장된 모델",
        "no_models": "아직 모델이 없습니다.",
        "clips": "클립 수",
        "enter_name": "선수 이름을 입력하세요.",
        "upload_one_clip": "NBA 영상 1개 이상 업로드하세요.",
        "building_model": "업로드한 영상으로 모델 생성 중...",
        "landmark_fail": "영상에서 랜드마크를 안정적으로 추출하지 못했습니다.",
        "saved_model_msg": "{player} 모델 저장 완료 ({count}개 클립).",
        "compare_hdr": "내 슛과 기준 비교",
        "ref_player": "기준 선수 모델",
        "ref_source": "기준 소스",
        "saved_model_src": "저장된 선수 모델 사용",
        "custom_clip_src": "사용자 기준 영상 업로드",
        "your_video": "내 슈팅 영상",
        "ref_video": "기준 영상",
        "run_single": "단일 영상 비교 실행",
        "upload_your_video": "먼저 내 슈팅 영상을 업로드하세요.",
        "extracting": "릴리즈 프레임 포즈 지표 추출 중...",
        "user_landmark_fail": "내 영상에서 랜드마크를 찾지 못했습니다.",
        "ref_fail": "기준 지표를 불러오지 못했습니다. 저장 모델을 선택하거나 더 선명한 기준 영상을 업로드하세요.",
        "sync_subfeature": "릴리즈 동기화 하위 기능",
        "sync_needs_clip": "사용자 기준 영상을 업로드하면 릴리즈 동기화가 자동 실행됩니다.",
        "custom_ref": "사용자 기준",
        "sync_hdr": "릴리즈 프레임 동기화 비교",
        "sync_caption": "두 영상을 릴리즈 시점에 맞춰 정렬하고 슬로모션으로 표시합니다.",
        "video_a": "영상 A",
        "video_b": "영상 B",
        "max_scan": "최대 스캔 프레임",
        "window": "구간 (+/- 프레임)",
        "slow_motion": "슬로모션 배수",
        "render_height": "렌더 높이",
        "gen_sync": "동기화 비교 생성",
        "upload_both": "두 영상을 모두 업로드하세요.",
        "running_sync": "릴리즈 동기화 비교 생성 중...",
        "no_release": "한쪽 또는 양쪽 영상에서 명확한 릴리즈 프레임을 찾지 못했습니다.",
        "sync_ready": "동기화 비교가 준비되었습니다.",
        "release_frame_aligned": "동기화된 릴리즈 프레임",
        "download_sync": "동기화 영상 다운로드",
        "video_playback_warn": "현재 환경에서 영상 재생이 제한됩니다. 아래 프레임 스크러버를 사용하세요.",
        "frame_scrubber": "프레임 스크러버",
        "sync_idx": "동기화 프레임 인덱스",
        "angle_diff_summary": "각도 차이 요약",
        "angle_comparison": "각도 비교",
        "your_shot_deg": "내 슛 (deg)",
        "difference_deg": "차이 (deg)",
        "abs_gap": "절대 각도 차이",
        "gap_deg": "차이 (deg)",
    },
}


def tr(lang: str, key: str) -> str:
    return I18N.get(lang, I18N["en"]).get(key, I18N["en"].get(key, key))


def render_result_clean(player_name: str, user_metrics: ShotMetrics, nba_metrics: ShotMetrics, lang: str = "en"):
    user_dict = user_metrics.to_dict()
    nba_dict = nba_metrics.to_dict()

    rows = []
    for name, user_value in user_dict.items():
        nba_value = nba_dict[name]
        rows.append(
            {
                "Metric": name,
                tr(lang, "your_shot_deg"): round(user_value, 1),
                f"{player_name} (deg)": round(nba_value, 1),
                tr(lang, "difference_deg"): round(user_value - nba_value, 1),
            }
        )

    st.subheader(tr(lang, "angle_comparison"))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader(tr(lang, "abs_gap"))
    chart_df = pd.DataFrame(
        {
            "Metric": [row["Metric"] for row in rows],
            tr(lang, "gap_deg"): [abs(row[tr(lang, "difference_deg")]) for row in rows],
        }
    ).set_index("Metric")
    st.bar_chart(chart_df)


def inject_ui_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@400;600;700&display=swap');
        :root {
            --bg: #f4f1e8;
            --paper: #fffdf8;
            --ink: #1f2a2e;
            --accent: #c7512f;
            --muted: #637176;
            --stroke: #e7dfcf;
        }
        .stApp {
            background:
                radial-gradient(1200px 500px at 0% -20%, #efe6d4 10%, transparent 60%),
                radial-gradient(900px 420px at 110% 10%, #d7e8df 8%, transparent 55%),
                var(--bg);
        }
        h1, h2, h3, .stTabs [data-baseweb="tab"] {
            font-family: "Space Grotesk", sans-serif !important;
            color: var(--ink);
            letter-spacing: 0.01em;
        }
        p, label, .stMarkdown, .stTextInput, .stFileUploader, .stSelectbox, .stRadio, .stNumberInput {
            font-family: "Manrope", sans-serif !important;
            color: var(--ink);
        }
        .hero {
            border: 1px solid var(--stroke);
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            background: linear-gradient(145deg, #fffef9 0%, #f8f3e8 60%, #f1ece0 100%);
            box-shadow: 0 8px 24px rgba(35, 40, 45, 0.06);
            margin-bottom: 0.8rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2rem;
        }
        .hero p {
            margin: 0.4rem 0 0 0;
            color: var(--muted);
            font-size: 1rem;
        }
        .kpi {
            border: 1px solid var(--stroke);
            border-radius: 14px;
            padding: 0.7rem 0.9rem;
            background: var(--paper);
        }
        .kpi .label {
            color: var(--muted);
            font-size: 0.82rem;
        }
        .kpi .value {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--ink);
        }
        .stButton > button {
            border-radius: 12px;
            border: 1px solid #b44729;
            background: linear-gradient(180deg, #d0603c 0%, #bf4d2e 100%);
            color: white;
            font-weight: 700;
            padding: 0.5rem 0.95rem;
        }
        .stDownloadButton > button {
            border-radius: 12px;
            border: 1px solid #2f7567;
            background: linear-gradient(180deg, #2e8b77 0%, #236e5f 100%);
            color: white;
            font-weight: 700;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            border: 1px solid var(--stroke);
            padding: 8px 14px;
            background: #fcf8ef;
        }
        .stTabs [aria-selected="true"] {
            background: #fce6d6 !important;
            border-color: #e7b18e !important;
            color: #7d2f19 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_app():
    st.set_page_config(page_title="Shooting Form Studio", layout="wide", initial_sidebar_state="collapsed")
    inject_ui_theme()

    top_l, top_r = st.columns([0.8, 0.2])
    with top_r:
        lang_label = st.selectbox("Language", ["English", "한국어"], index=0, key="ui_lang_select")
    lang = "ko" if lang_label == "한국어" else "en"

    models = load_player_models()
    available_players = sorted(models.keys())

    st.markdown(
        f"""
        <div class="hero">
          <h1>{tr(lang, "app_title")}</h1>
          <p>{tr(lang, "hero_desc")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            f'<div class="kpi"><div class="label">{tr(lang, "saved_models")}</div><div class="value">{len(available_players)}</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(f'<div class="kpi"><div class="label">{tr(lang, "compare_modes")}</div><div class="value">1</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi"><div class="label">{tr(lang, "release_sync")}</div><div class="value">{tr(lang, "enabled")}</div></div>', unsafe_allow_html=True)

    tab_models, tab_single = st.tabs([tr(lang, "tab_models"), tr(lang, "tab_single")])

    with tab_models:
        st.markdown(f"### {tr(lang, 'build_update')}")
        left, right = st.columns([1.4, 1.0])
        with left:
            model_player_name = st.text_input(tr(lang, "player_name"), value="Stephen Curry", key="new_model_player_name")
            model_clips = st.file_uploader(
                tr(lang, "nba_clips"),
                type=["mp4", "mov", "avi", "mkv"],
                accept_multiple_files=True,
                key="new_model_clips",
                help="Upload side-angle shooting clips. More clips produce a stronger model.",
            )
            if st.button(tr(lang, "build_model"), key="new_build_model_btn"):
                if not model_player_name.strip():
                    st.error(tr(lang, "enter_name"))
                elif not model_clips:
                    st.error(tr(lang, "upload_one_clip"))
                else:
                    sample_metrics: List[ShotMetrics] = []
                    with st.spinner(tr(lang, "building_model")):
                        for clip in model_clips:
                            clip_path = save_upload(clip)
                            clip_metrics = analyze_video(clip_path)
                            if clip_metrics is not None:
                                sample_metrics.append(clip_metrics)

                    if not sample_metrics:
                        st.error(tr(lang, "landmark_fail"))
                    else:
                        models[model_player_name.strip()] = {
                            "sample_count": len(sample_metrics),
                            "metrics": aggregate_metrics(sample_metrics).to_dict(),
                        }
                        save_player_models(models)
                        st.success(tr(lang, "saved_model_msg").format(player=model_player_name.strip(), count=len(sample_metrics)))
                        st.rerun()
        with right:
            st.markdown(f"### {tr(lang, 'saved_models_hdr')}")
            if not available_players:
                st.info(tr(lang, "no_models"))
            else:
                rows = [{"Player": p, tr(lang, "clips"): int(models[p].get("sample_count", 0))} for p in available_players]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_single:
        st.markdown(f"### {tr(lang, 'compare_hdr')}")
        c1, c2 = st.columns(2)
        with c1:
            player_name = st.selectbox(
                tr(lang, "ref_player"),
                available_players if available_players else ["<none>"],
                key="new_single_player_name",
            )
            compare_mode = st.radio(
                tr(lang, "ref_source"),
                [tr(lang, "saved_model_src"), tr(lang, "custom_clip_src")],
                horizontal=True,
                key="new_single_compare_mode",
            )
        with c2:
            user_video = st.file_uploader(
                tr(lang, "your_video"),
                type=["mp4", "mov", "avi", "mkv"],
                key="new_user_video",
            )
            nba_video = None
            if compare_mode == tr(lang, "custom_clip_src"):
                nba_video = st.file_uploader(
                    tr(lang, "ref_video"),
                    type=["mp4", "mov", "avi", "mkv"],
                    key="new_nba_video",
                )

        st.markdown(f"#### {tr(lang, 'sync_subfeature')}")
        s1, s2, s3 = st.columns(3)
        with s1:
            sync_max_frames = st.number_input(tr(lang, "max_scan"), min_value=0, value=300, step=50, key="single_sync_max")
        with s2:
            sync_window = st.number_input(tr(lang, "window"), min_value=5, value=24, step=1, key="single_sync_window")
        with s3:
            sync_slow = st.number_input(tr(lang, "slow_motion"), min_value=1, value=3, step=1, key="single_sync_slow")
        sync_render_height = st.number_input(
            tr(lang, "render_height"),
            min_value=360,
            max_value=1440,
            value=720,
            step=60,
            key="single_sync_render_h",
        )

        if st.button(tr(lang, "run_single"), type="primary", key="new_single_compare_btn"):
            if not user_video:
                st.error(tr(lang, "upload_your_video"))
            else:
                nba_path = None
                with st.spinner(tr(lang, "extracting")):
                    user_path = save_upload(user_video)
                    user_metrics = analyze_video(user_path)

                    if compare_mode == tr(lang, "custom_clip_src"):
                        if not nba_video:
                            nba_metrics = None
                        else:
                            nba_path = save_upload(nba_video)
                            nba_metrics = analyze_video(nba_path)
                            player_name = tr(lang, "custom_ref")
                    else:
                        if player_name == "<none>" or player_name not in models:
                            nba_metrics = None
                        else:
                            nba_metrics = ShotMetrics.from_dict(models[player_name]["metrics"])

                if user_metrics is None:
                    st.error(tr(lang, "user_landmark_fail"))
                elif nba_metrics is None:
                    st.error(tr(lang, "ref_fail"))
                else:
                    render_result_clean(player_name, user_metrics, nba_metrics, lang=lang)
                    st.markdown("---")
                    st.markdown(f"### {tr(lang, 'sync_hdr')}")
                    if nba_path is None:
                        st.info(tr(lang, "sync_needs_clip"))
                    else:
                        with st.spinner(tr(lang, "running_sync")):
                            compare_path, summary_df, frame_count, release_frame, timeline_frames = compare_videos(
                                user_path,
                                nba_path,
                                max_frames=int(sync_max_frames),
                                window=int(sync_window),
                                slow_factor=int(sync_slow),
                                render_height=int(sync_render_height),
                            )

                        if release_frame is None or not timeline_frames:
                            st.error(tr(lang, "no_release"))
                        else:
                            st.success(tr(lang, "sync_ready"))
                            st.markdown(f"#### {tr(lang, 'release_frame_aligned')}")
                            st.image(cv2.cvtColor(release_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                            if compare_path is not None and frame_count > 0:
                                video_bytes = compare_path.read_bytes()
                                st.video(video_bytes, format="video/mp4")
                                st.download_button(
                                    tr(lang, "download_sync"),
                                    data=video_bytes,
                                    file_name="pose_comparison_release_sync_slowmo.mp4",
                                    mime="video/mp4",
                                    key="new_sync_download_btn",
                                )
                            else:
                                st.warning(tr(lang, "video_playback_warn"))

                            st.markdown(f"#### {tr(lang, 'frame_scrubber')}")
                            scrub_idx = st.slider(
                                tr(lang, "sync_idx"),
                                min_value=0,
                                max_value=len(timeline_frames) - 1,
                                value=len(timeline_frames) // 2,
                                key="new_sync_scrubber",
                            )
                            st.image(
                                cv2.cvtColor(timeline_frames[scrub_idx], cv2.COLOR_BGR2RGB),
                                channels="RGB",
                                use_container_width=True,
                            )

                            if summary_df is not None:
                                st.markdown(f"#### {tr(lang, 'angle_diff_summary')}")
                                st.dataframe(
                                    summary_df.style.format({"Mean abs diff (deg)": "{:.1f}", "Median abs diff (deg)": "{:.1f}"}),
                                    use_container_width=True,
                                    hide_index=True,
                                )


if __name__ == "__main__":
    run_app()
