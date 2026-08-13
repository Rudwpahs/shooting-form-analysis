"""Video → 3D joint-angle profiles (degrees only)."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .angles import AngleSnapshot, angles_from_landmarks, median_angles
from .pose import PoseCandidate, close_detector, create_pose_detector


VIEW_TAGS = ("front", "side", "oblique")


@dataclass
class PersonPreview:
    person_index: int
    bbox: Tuple[float, float, float, float]
    thumb_jpeg_b64: str


@dataclass
class ViewAnalysis:
    view: str
    release_frame_index: int
    frames_scanned: int
    space: str
    hand: str
    release_angles: Dict[str, float]
    sequence_angles: Dict[str, List[float]] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SessionAnalysis:
    views: List[ViewAnalysis]
    person_index: int
    release_angles_merged: Dict[str, float]
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "person_index": self.person_index,
            "release_angles_merged": self.release_angles_merged,
            "views": [v.to_dict() for v in self.views],
            "error": self.error,
        }


def _resize(frame: np.ndarray, height: int = 720) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == height:
        return frame
    width = max(1, int(round(w * height / h)))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _encode_thumb(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> str:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = bbox
    pad = 0.05
    xa = max(0, int((x0 - pad) * w))
    ya = max(0, int((y0 - pad) * h))
    xb = min(w, int((x1 + pad) * w))
    yb = min(h, int((y1 + pad) * h))
    crop = frame[ya:yb, xa:xb] if xb > xa and yb > ya else frame
    crop = _resize(crop, 180)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _pick_pose(poses: List[PoseCandidate], person_index: int) -> Optional[PoseCandidate]:
    if not poses:
        return None
    for pose in poses:
        if pose.person_index == person_index:
            return pose
    return poses[0] if person_index == 0 else None


def _angles_for_pose(pose: PoseCandidate, hand: Optional[str]) -> Optional[AngleSnapshot]:
    if pose.world_landmarks:
        snap = angles_from_landmarks(pose.world_landmarks, hand, space="3d")
        if snap is not None:
            return snap
    return angles_from_landmarks(pose.image_landmarks, hand, space="2d")


def _wrist_image_y(pose: PoseCandidate, hand: str) -> float:
    idx = 15 if hand == "left" else 16
    lms = pose.image_landmarks
    if idx >= len(lms):
        return float("inf")
    return float(lms[idx].y)


def list_people_in_video(
    video_path: Path,
    sample_time_sec: float = 0.4,
    num_poses: int = 3,
    render_height: int = 720,
) -> List[PersonPreview]:
    """Return detectable people near an early frame for user selection."""
    cap = cv2.VideoCapture(str(video_path))
    detector = None
    try:
        if not cap.isOpened():
            return []
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_idx = max(0, int(sample_time_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            return []
        frame = _resize(frame, render_height)
        detector = create_pose_detector(fps=fps, num_poses=num_poses)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        poses = detector.detect(rgb)
        return [
            PersonPreview(
                person_index=p.person_index,
                bbox=p.bbox,
                thumb_jpeg_b64=_encode_thumb(frame, p.bbox),
            )
            for p in poses
        ]
    finally:
        cap.release()
        if detector is not None:
            close_detector(detector)


def analyze_view(
    video_path: Path,
    view: str = "side",
    person_index: int = 0,
    hand: Optional[str] = None,
    max_frames: int = 240,
    num_poses: int = 3,
    render_height: int = 720,
    keep_sequence: bool = True,
) -> ViewAnalysis:
    cap = cv2.VideoCapture(str(video_path))
    detector = None
    try:
        if not cap.isOpened():
            return ViewAnalysis(
                view=view,
                release_frame_index=-1,
                frames_scanned=0,
                space="",
                hand="",
                release_angles={},
                error="Could not open video.",
            )

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        detector = create_pose_detector(fps=fps, num_poses=num_poses)

        candidates: List[Tuple[float, int, AngleSnapshot]] = []
        sequence: List[AngleSnapshot] = []
        frames_scanned = 0

        while frames_scanned < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame = _resize(frame, render_height)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            poses = detector.detect(rgb)
            pose = _pick_pose(poses, person_index)
            if pose is not None:
                snap = _angles_for_pose(pose, hand)
                if snap is not None:
                    wrist_y = _wrist_image_y(pose, snap.hand)
                    candidates.append((wrist_y, frames_scanned, snap))
                    if keep_sequence:
                        sequence.append(snap)
            frames_scanned += 1

        if not candidates:
            return ViewAnalysis(
                view=view,
                release_frame_index=-1,
                frames_scanned=frames_scanned,
                space="",
                hand="",
                release_angles={},
                error="No reliable pose for the selected person.",
            )

        _, release_idx, release_snap = min(candidates, key=lambda item: item[0])
        seq_dict: Dict[str, List[float]] = {}
        if keep_sequence and sequence:
            for key in ("elbow", "shoulder", "hip", "knee"):
                seq_dict[key] = [float(s.as_dict()[key]) for s in sequence]

        return ViewAnalysis(
            view=view,
            release_frame_index=release_idx,
            frames_scanned=frames_scanned,
            space=release_snap.space,
            hand=release_snap.hand,
            release_angles=release_snap.as_dict(),
            sequence_angles=seq_dict,
        )
    except Exception as exc:
        return ViewAnalysis(
            view=view,
            release_frame_index=-1,
            frames_scanned=0,
            space="",
            hand="",
            release_angles={},
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        cap.release()
        if detector is not None:
            close_detector(detector)


def analyze_session(
    videos: Sequence[Tuple[Path, str]],
    person_index: int = 0,
    hand: Optional[str] = None,
    max_frames: int = 240,
) -> SessionAnalysis:
    """Analyze 1–3 videos tagged with view names; merge release angles by median."""
    views: List[ViewAnalysis] = []
    for path, view in videos:
        tag = view if view in VIEW_TAGS else "side"
        views.append(
            analyze_view(
                path,
                view=tag,
                person_index=person_index,
                hand=hand,
                max_frames=max_frames,
            )
        )

    ok_snaps = [
        AngleSnapshot(
            elbow=v.release_angles["elbow"],
            shoulder=v.release_angles["shoulder"],
            hip=v.release_angles["hip"],
            knee=v.release_angles["knee"],
            hand=v.hand or "right",
            space=v.space or "3d",
        )
        for v in views
        if v.release_angles and not v.error
    ]
    merged = median_angles(ok_snaps)
    if merged is None:
        return SessionAnalysis(
            views=views,
            person_index=person_index,
            release_angles_merged={},
            error="No view produced usable angles.",
        )
    return SessionAnalysis(
        views=views,
        person_index=person_index,
        release_angles_merged=merged.as_dict(),
    )
