"""MediaPipe pose detection — supports multiple people and world (3D) landmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Install mediapipe: pip install mediapipe") from exc

ROOT = Path(__file__).resolve().parent.parent
TASK_MODEL_PATH = ROOT / "models" / "pose_landmarker_full.task"


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float = 1.0


@dataclass
class PoseCandidate:
    person_index: int
    image_landmarks: List[Landmark]
    world_landmarks: Optional[List[Landmark]]
    bbox: tuple  # xmin, ymin, xmax, ymax in normalized image coords


def _to_landmarks(raw) -> List[Landmark]:
    out: List[Landmark] = []
    for lm in raw:
        out.append(
            Landmark(
                x=float(lm.x),
                y=float(lm.y),
                z=float(getattr(lm, "z", 0.0)),
                visibility=float(getattr(lm, "visibility", getattr(lm, "presence", 1.0))),
            )
        )
    return out


def _bbox_from_landmarks(landmarks: Sequence[Landmark]) -> tuple:
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return (min(xs), min(ys), max(xs), max(ys))


class _TasksPoseAdapter:
    def __init__(self, landmarker, fps: float):
        self._landmarker = landmarker
        self._timestamp_ms = 0
        self._interval_ms = max(1, int(round(1000.0 / max(fps, 1e-6))))

    def detect(self, frame_rgb: np.ndarray) -> List[PoseCandidate]:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(image, self._timestamp_ms)
        self._timestamp_ms += self._interval_ms
        poses: List[PoseCandidate] = []
        image_list = result.pose_landmarks or []
        world_list = result.pose_world_landmarks or []
        for i, image_lms in enumerate(image_list):
            image_landmarks = _to_landmarks(image_lms)
            world_landmarks = _to_landmarks(world_list[i]) if i < len(world_list) else None
            poses.append(
                PoseCandidate(
                    person_index=i,
                    image_landmarks=image_landmarks,
                    world_landmarks=world_landmarks,
                    bbox=_bbox_from_landmarks(image_landmarks),
                )
            )
        return poses

    def close(self) -> None:
        self._landmarker.close()


class _SolutionsPoseAdapter:
    def __init__(self, pose):
        self._pose = pose

    def detect(self, frame_rgb: np.ndarray) -> List[PoseCandidate]:
        result = self._pose.process(frame_rgb)
        if not result.pose_landmarks:
            return []
        image_landmarks = _to_landmarks(result.pose_landmarks.landmark)
        world_landmarks = None
        if getattr(result, "pose_world_landmarks", None):
            world_landmarks = _to_landmarks(result.pose_world_landmarks.landmark)
        return [
            PoseCandidate(
                person_index=0,
                image_landmarks=image_landmarks,
                world_landmarks=world_landmarks,
                bbox=_bbox_from_landmarks(image_landmarks),
            )
        ]

    def close(self) -> None:
        self._pose.close()


def create_pose_detector(fps: float = 30.0, num_poses: int = 3):
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose") and num_poses <= 1:
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=False,
        )
        return _SolutionsPoseAdapter(pose)

    if not TASK_MODEL_PATH.exists():
        raise RuntimeError(f"Missing pose model: {TASK_MODEL_PATH}")

    vision = mp.tasks.vision
    options = vision.PoseLandmarkerOptions(
        # MediaPipe's native Windows path loader can fail on non-ASCII paths.
        # Supplying bytes keeps projects under folders such as "문서" usable.
        base_options=mp.tasks.BaseOptions(model_asset_buffer=TASK_MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.45,
        min_pose_presence_confidence=0.45,
        min_tracking_confidence=0.45,
        num_poses=max(1, num_poses),
    )
    return _TasksPoseAdapter(vision.PoseLandmarker.create_from_options(options), fps=fps)


def close_detector(detector) -> None:
    close = getattr(detector, "close", None)
    if callable(close):
        close()
