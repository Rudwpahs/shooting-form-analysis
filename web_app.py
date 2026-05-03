import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
try:
    import mediapipe as mp
except Exception:
    mp = None

import numpy as np
import pandas as pd
import streamlit as st




def create_pose_detector():
    """Create a MediaPipe Pose detector across package variants."""
    if mp is None:
        return None, "`mediapipe` is not installed. Install with: pip install mediapipe"

    # 1) Standard API
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        return mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ), None

    # 2) Fallback import path
    try:
        from mediapipe.python.solutions.pose import Pose as PoseCls  # type: ignore

        return PoseCls(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ), None
    except Exception:
        return None, (
            "Could not initialize MediaPipe Pose from this `mediapipe` installation. "
            "Please reinstall the official package: `pip uninstall mediapipe -y && pip install mediapipe`."
        )


NBA_BASELINES = {
    "Stephen Curry": {"Elbow angle": 86.0, "Shoulder angle": 61.0, "Hip angle": 168.0, "Knee angle": 157.0},
    "Klay Thompson": {"Elbow angle": 89.0, "Shoulder angle": 58.0, "Hip angle": 170.0, "Knee angle": 160.0},
    "Ray Allen": {"Elbow angle": 90.0, "Shoulder angle": 57.0, "Hip angle": 166.0, "Knee angle": 154.0},
    "Damian Lillard": {"Elbow angle": 84.0, "Shoulder angle": 63.0, "Hip angle": 165.0, "Knee angle": 152.0},
    "Kyrie Irving": {"Elbow angle": 87.0, "Shoulder angle": 60.0, "Hip angle": 164.0, "Knee angle": 150.0},
}


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


def analyze_video(video_path: Path, max_frames: int = 300) -> Tuple[Optional[ShotMetrics], Optional[str]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, "Failed to open the uploaded video file."

    pose, pose_error = create_pose_detector()
    if pose is None:
        cap.release()
        return None, pose_error

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
        return None, "No pose landmarks were detected in the video. Try a clearer side view."

    per_frame.sort(key=lambda item: item[0])
    return per_frame[0][1], None


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
    st.write("Upload your video and compare your release posture with the NBA player you choose.")

    player_name = st.selectbox("Choose NBA player", list(NBA_BASELINES.keys()))
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
            user_metrics, user_error = analyze_video(user_path)

            if compare_mode == "Upload custom NBA reference clip":
                if not nba_video:
                    st.error("Please upload an NBA reference video or switch to built-in profile mode.")
                    return
                nba_path = save_upload(nba_video)
                nba_metrics, nba_error = analyze_video(nba_path)
            else:
                nba_metrics = ShotMetrics.from_dict(NBA_BASELINES[player_name])
                nba_error = None

        if user_metrics is None:
            st.error(user_error or "Could not detect body landmarks from your video. Try a clearer side view.")
            return
        if nba_metrics is None:
            st.error(nba_error or "Could not detect landmarks from NBA reference video. Try another clip or built-in mode.")
            return

        render_result(player_name, user_metrics, nba_metrics)


if __name__ == "__main__":
    main()
