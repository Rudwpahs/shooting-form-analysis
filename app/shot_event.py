"""Basketball-specific evidence for accepting a pose sequence as a shot.

Pose-only wrist peaks are useful candidates but do not prove that a basketball
shot occurred.  This module exposes a conservative, dependency-free baseline:
an orange/circular ball detector, a continuity-aware ball track, and a
hand-to-ball separation check around the pose-derived release frame.  Missing
or weak evidence returns an explicit rejection reason rather than inventing a
shot event.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class BallObservation:
    frame: int
    x: float
    y: float
    radius: float
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ShooterTrackQuality:
    observations: int
    continuity_ratio: float
    identity_switch_suspected: bool
    max_gap_frames: int

    @property
    def valid(self) -> bool:
        return (
            self.observations >= 6
            and self.continuity_ratio >= 0.75
            and not self.identity_switch_suspected
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["valid"] = self.valid
        return data


@dataclass(frozen=True)
class ShotEventEvidence:
    status: str
    verified: bool
    release_frame: int
    ball_observations: int
    pre_release_hand_ball_distance: float | None
    post_release_hand_ball_distance: float | None
    upward_ball_motion: float | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        return data


def detect_basketball(frame_bgr: np.ndarray, frame_index: int) -> Optional[BallObservation]:
    """Return one high-confidence orange/circular basketball candidate.

    This is deliberately only a baseline.  Its backend is named in the API so
    callers can distinguish detected evidence from an ML basketball detector.
    A real product should replace or augment this with a sport-ball detector
    trained on the deployed courts and ball appearance.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    height, width = frame_bgr.shape[:2]
    if height < 20 or width < 20:
        return None
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # Indoor basketball hues span orange through orange-red.  The detector is
    # intentionally conservative: a poor color/circle match is no evidence.
    lower = np.array([3, 85, 75], dtype=np.uint8)
    upper = np.array([28, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(12.0, 0.000015 * width * height)
    max_area = max(min_area * 2.0, 0.03 * width * height)
    best: Optional[BallObservation] = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not min_area <= area <= max_area:
            continue
        (cx, cy), radius_px = cv2.minEnclosingCircle(contour)
        if radius_px < 2:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
        fill = area / max(np.pi * radius_px * radius_px, 1e-6)
        confidence = float(np.clip(0.55 * circularity + 0.45 * min(fill, 1.0), 0.0, 1.0))
        if confidence < 0.48:
            continue
        candidate = BallObservation(
            frame=int(frame_index),
            x=float(cx / width),
            y=float(cy / height),
            radius=float(radius_px / max(width, height)),
            confidence=confidence,
        )
        if best is None or candidate.confidence > best.confidence:
            best = candidate
    return best


def link_ball_track(
    observations: Iterable[BallObservation],
    *,
    max_frame_gap: int = 4,
    max_step_distance: float = 0.22,
) -> list[BallObservation]:
    """Keep the most plausible continuous ball track from frame observations."""
    ordered = sorted(observations, key=lambda item: (item.frame, -item.confidence))
    if not ordered:
        return []
    track: list[BallObservation] = []
    for observation in ordered:
        if not track:
            track.append(observation)
            continue
        previous = track[-1]
        if observation.frame == previous.frame:
            if observation.confidence > previous.confidence:
                track[-1] = observation
            continue
        frame_gap = observation.frame - previous.frame
        distance = _distance(observation.x, observation.y, previous.x, previous.y)
        allowed = max_step_distance * max(1, frame_gap)
        if 0 < frame_gap <= max_frame_gap and distance <= allowed:
            track.append(observation)
    return track


def shooter_track_quality(
    bboxes: Sequence[tuple[int, tuple[float, float, float, float]]], *,
    expected_frames: int,
) -> ShooterTrackQuality:
    """Quantify whether the selected shooter remained the same person."""
    if not bboxes or expected_frames <= 0:
        return ShooterTrackQuality(0, 0.0, True, expected_frames)
    ordered = sorted(bboxes, key=lambda item: item[0])
    switches = False
    max_gap = 0
    previous_frame, previous_box = ordered[0]
    for frame, box in ordered[1:]:
        max_gap = max(max_gap, frame - previous_frame)
        if _bbox_iou(previous_box, box) < 0.08:
            switches = True
        previous_frame, previous_box = frame, box
    return ShooterTrackQuality(
        observations=len(ordered),
        continuity_ratio=min(1.0, len(ordered) / max(1, expected_frames)),
        identity_switch_suspected=switches,
        max_gap_frames=max_gap,
    )


def verify_shot_event(
    *,
    release_frame: int,
    wrist_positions: Sequence[tuple[int, float, float]],
    ball_track: Sequence[BallObservation],
    shooter_quality: ShooterTrackQuality,
    fps: float,
) -> ShotEventEvidence:
    """Confirm a shot only when the ball is near the shooting hand then separates upward."""
    fps = max(float(fps or 0.0), 24.0)
    if not shooter_quality.valid:
        return ShotEventEvidence(
            status="rejected",
            verified=False,
            release_frame=release_frame,
            ball_observations=len(ball_track),
            pre_release_hand_ball_distance=None,
            post_release_hand_ball_distance=None,
            upward_ball_motion=None,
            reasons=("shooter track is discontinuous or too short",),
        )
    if len(ball_track) < 4:
        return ShotEventEvidence(
            status="pose_only_unverified",
            verified=False,
            release_frame=release_frame,
            ball_observations=len(ball_track),
            pre_release_hand_ball_distance=None,
            post_release_hand_ball_distance=None,
            upward_ball_motion=None,
            reasons=("insufficient basketball observations",),
        )

    wrists = {int(frame): (float(x), float(y)) for frame, x, y in wrist_positions}
    pre_window = max(2, int(round(0.22 * fps)))
    post_window = max(2, int(round(0.28 * fps)))
    nearby_pre: list[float] = []
    nearby_post: list[float] = []
    post_y: list[float] = []
    pre_y: list[float] = []
    for ball in ball_track:
        wrist = wrists.get(ball.frame)
        if wrist is None:
            continue
        distance = _distance(ball.x, ball.y, wrist[0], wrist[1])
        delta = ball.frame - release_frame
        if -pre_window <= delta <= 0:
            nearby_pre.append(distance)
            pre_y.append(ball.y)
        if 1 <= delta <= post_window:
            nearby_post.append(distance)
            post_y.append(ball.y)

    if not nearby_pre or not nearby_post:
        return ShotEventEvidence(
            status="pose_only_unverified",
            verified=False,
            release_frame=release_frame,
            ball_observations=len(ball_track),
            pre_release_hand_ball_distance=None,
            post_release_hand_ball_distance=None,
            upward_ball_motion=None,
            reasons=("missing hand-ball evidence around release",),
        )
    pre_distance = float(np.median(nearby_pre))
    post_distance = float(np.median(nearby_post))
    upward_motion = float(np.median(pre_y) - np.median(post_y)) if pre_y and post_y else 0.0
    reasons: list[str] = []
    if pre_distance > 0.16:
        reasons.append("ball was not near the shooting hand before release")
    if post_distance < pre_distance + 0.035:
        reasons.append("hand-ball separation was not observed")
    if upward_motion < 0.025:
        reasons.append("upward ball trajectory was not observed")
    if reasons:
        return ShotEventEvidence(
            status="pose_only_unverified",
            verified=False,
            release_frame=release_frame,
            ball_observations=len(ball_track),
            pre_release_hand_ball_distance=round(pre_distance, 4),
            post_release_hand_ball_distance=round(post_distance, 4),
            upward_ball_motion=round(upward_motion, 4),
            reasons=tuple(reasons),
        )
    return ShotEventEvidence(
        status="verified",
        verified=True,
        release_frame=release_frame,
        ball_observations=len(ball_track),
        pre_release_hand_ball_distance=round(pre_distance, 4),
        post_release_hand_ball_distance=round(post_distance, 4),
        upward_ball_motion=round(upward_motion, 4),
        reasons=(),
    )


def _distance(ax: float, ay: float, bx: float, by: float) -> float:
    return float(np.hypot(ax - bx, ay - by))


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = left
    bx0, by0, bx1, by1 = right
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    union = max(0.0, (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection)
    return intersection / union
