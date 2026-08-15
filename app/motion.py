"""Pose-sequence normalization, quality control, prototype learning, and matching.

The public API still exposes release angles for backwards compatibility, but
player identity is better represented by the whole catch-to-follow-through
trajectory.  This module keeps that representation camera- and body-scale
normalized so YouTube clips can be compared without treating pixel locations
as biomechanics measurements.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


LANDMARK_COUNT = 33
POSE_SIZE = 4  # normalized x, y, z, visibility
BODY_LANDMARKS = (0, 11, 12, 13, 14, 15, 16, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
MATCH_LANDMARKS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32)
JOINT_KEYS = ("elbow", "shoulder", "hip", "knee")
PHASE_ANCHORS = {
    "catch": 0.0,
    "dip": 0.25,
    "rise": 0.48,
    "release": 0.75,
    "follow_through": 1.0,
}


def _point(raw: Any) -> Optional[Tuple[float, float, float, float]]:
    try:
        if isinstance(raw, Mapping):
            return (
                float(raw["x"]),
                float(raw["y"]),
                float(raw.get("z", 0.0)),
                float(raw.get("visibility", 1.0)),
            )
        if len(raw) >= 3:
            return (
                float(raw[0]),
                float(raw[1]),
                float(raw[2]),
                float(raw[3]) if len(raw) > 3 else 1.0,
            )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return None


def normalize_landmarks(landmarks: Sequence[Any], hand: str = "right") -> List[List[float]]:
    """Normalize MediaPipe landmarks around the pelvis and torso.

    Image y is inverted so +y points upward.  The torso is rotated upright in
    the image plane, scale is the pelvis-to-shoulder distance, and left-handed
    shooters are mirrored to the same canonical handedness as right-handers.
    Depth remains an estimate and is deliberately downweighted in matching.
    """
    if len(landmarks) < LANDMARK_COUNT:
        return []
    points = [_point(value) for value in landmarks[:LANDMARK_COUNT]]
    if any(points[index] is None for index in (11, 12, 23, 24)):
        return []

    left_shoulder, right_shoulder = points[11], points[12]
    left_hip, right_hip = points[23], points[24]
    assert left_shoulder and right_shoulder and left_hip and right_hip
    shoulder = np.array(
        [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2, (left_shoulder[2] + right_shoulder[2]) / 2],
        dtype=np.float64,
    )
    pelvis = np.array(
        [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2, (left_hip[2] + right_hip[2]) / 2],
        dtype=np.float64,
    )
    torso = shoulder - pelvis
    scale = float(np.linalg.norm(torso[:2]))
    if not math.isfinite(scale) or scale < 0.035:
        return []

    # Rotate the pelvis->shoulder vector to image-up (0, -1), then invert y.
    theta = math.atan2(float(torso[0]), float(-torso[1]))
    c, s = math.cos(theta), math.sin(theta)
    mirror = -1.0 if hand == "left" else 1.0
    out: List[List[float]] = []
    for item in points:
        if item is None:
            out.append([0.0, 0.0, 0.0, 0.0])
            continue
        dx = (item[0] - pelvis[0]) / scale
        dy = (item[1] - pelvis[1]) / scale
        x = (c * dx - s * dy) * mirror
        y = -(s * dx + c * dy)
        z = ((item[2] - pelvis[2]) / scale) * mirror
        visibility = max(0.0, min(1.0, float(item[3])))
        out.append([round(x, 5), round(y, 5), round(z, 5), round(visibility, 4)])
    return out


def estimate_view(samples: Sequence[Mapping[str, Any]]) -> str:
    """Estimate front/oblique/side from normalized shoulder and hip widths."""
    ratios: List[float] = []
    for sample in samples:
        pose = sample.get("pose") or []
        if len(pose) < LANDMARK_COUNT:
            continue
        try:
            shoulder = abs(float(pose[12][0]) - float(pose[11][0]))
            hip = abs(float(pose[24][0]) - float(pose[23][0]))
            ratios.append((shoulder + hip) / 2.0)
        except (IndexError, TypeError, ValueError):
            continue
    if not ratios:
        return "side"
    width = float(statistics.median(ratios))
    if width < 0.34:
        return "side"
    if width > 0.72:
        return "front"
    return "oblique"


def _phase_progress(samples: Sequence[Mapping[str, Any]]) -> List[float]:
    n = len(samples)
    if n <= 1:
        return [0.0] * n
    indices: Dict[str, int] = {}
    for index, sample in enumerate(samples):
        phase = str(sample.get("phase") or "")
        if phase == "catch" and "catch" not in indices:
            indices["catch"] = index
        elif phase == "dip":
            indices.setdefault("dip", index)
        elif phase == "release":
            indices.setdefault("release", index)
        elif phase == "follow_through":
            indices["follow_through"] = index
    anchors = [
        (indices.get("catch", 0), 0.0),
        (indices.get("dip", max(0, int(round((n - 1) * 0.25)))), 0.25),
        (indices.get("release", max(0, int(round((n - 1) * 0.75)))), 0.75),
        (indices.get("follow_through", n - 1), 1.0),
    ]
    anchors.sort(key=lambda item: item[0])
    clean: List[Tuple[int, float]] = []
    for index, value in anchors:
        index = max(0, min(n - 1, int(index)))
        if clean and index == clean[-1][0]:
            clean[-1] = (index, max(value, clean[-1][1]))
        else:
            clean.append((index, value))
    progress = [0.0] * n
    for segment in range(len(clean) - 1):
        i0, p0 = clean[segment]
        i1, p1 = clean[segment + 1]
        width = max(1, i1 - i0)
        for index in range(i0, i1 + 1):
            progress[index] = p0 + (p1 - p0) * (index - i0) / width
    for index in range(0, clean[0][0]):
        progress[index] = clean[0][1]
    for index in range(clean[-1][0], n):
        progress[index] = clean[-1][1]
    return progress


def _phase_at(progress: float) -> str:
    if progress < 0.22:
        return "catch"
    if progress < 0.30:
        return "dip"
    if progress < 0.735:
        return "rise"
    if progress < 0.765:
        return "release"
    return "follow_through"


def resample_timeline(samples: Sequence[Mapping[str, Any]], frame_count: int = 48) -> List[Dict[str, Any]]:
    """Phase-align a pose timeline to a fixed number of frames."""
    usable = [sample for sample in samples if len(sample.get("pose") or []) >= LANDMARK_COUNT]
    if len(usable) < 2 or frame_count < 2:
        return []
    progress = np.asarray(_phase_progress(usable), dtype=np.float64)
    # np.interp expects strictly increasing x; tiny offsets preserve repeated phases.
    progress = np.maximum.accumulate(progress + np.arange(len(progress)) * 1e-7)
    target = np.linspace(0.0, 1.0, frame_count)

    pose_array = np.asarray([sample["pose"][:LANDMARK_COUNT] for sample in usable], dtype=np.float64)
    if pose_array.shape != (len(usable), LANDMARK_COUNT, POSE_SIZE):
        return []
    out_pose = np.empty((frame_count, LANDMARK_COUNT, POSE_SIZE), dtype=np.float64)
    for landmark in range(LANDMARK_COUNT):
        for axis in range(POSE_SIZE):
            out_pose[:, landmark, axis] = np.interp(target, progress, pose_array[:, landmark, axis])

    out_angles: Dict[str, np.ndarray] = {}
    for key in JOINT_KEYS:
        try:
            values = np.asarray([float(sample[key]) for sample in usable], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            values = np.zeros(len(usable), dtype=np.float64)
        out_angles[key] = np.interp(target, progress, values)

    duration = max(0.4, float(usable[-1].get("t", 1.0)) - float(usable[0].get("t", 0.0)))
    frames: List[Dict[str, Any]] = []
    for index, phase_progress in enumerate(target):
        frame = {
            "t": round(duration * index / (frame_count - 1), 4),
            "phase": _phase_at(float(phase_progress)),
            "pose": np.round(out_pose[index], 5).tolist(),
        }
        frame.update({key: round(float(out_angles[key][index]), 2) for key in JOINT_KEYS})
        frames.append(frame)
    return frames


def smooth_timeline(samples: Sequence[Mapping[str, Any]], window: int = 5) -> List[Dict[str, Any]]:
    """Median-filter isolated pose-estimation spikes while preserving metadata."""
    if len(samples) < 3 or window < 3:
        return [dict(sample) for sample in samples]
    half = window // 2
    out: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples):
        left = max(0, index - half)
        right = min(len(samples), index + half + 1)
        neighborhood = samples[left:right]
        clean = dict(sample)
        for key in JOINT_KEYS:
            values = [float(item[key]) for item in neighborhood if item.get(key) is not None]
            if values:
                clean[key] = round(float(statistics.median(values)), 2)
        pose = sample.get("pose") or []
        if len(pose) >= LANDMARK_COUNT:
            filtered_pose = []
            for landmark in range(LANDMARK_COUNT):
                values = [
                    item["pose"][landmark]
                    for item in neighborhood
                    if len(item.get("pose") or []) >= LANDMARK_COUNT
                ]
                if not values:
                    filtered_pose.append(list(pose[landmark]))
                    continue
                filtered_pose.append(
                    [
                        round(float(statistics.median(value[axis] for value in values)), 5)
                        for axis in range(4)
                    ]
                )
            clean["pose"] = filtered_pose
        out.append(clean)
    return out


def timeline_quality(samples: Sequence[Mapping[str, Any]], fps: float = 30.0) -> Dict[str, Any]:
    """Score tracking completeness and continuity without judging shooting style."""
    if not samples:
        return {"score": 0.0, "valid": False, "reason": "empty timeline"}
    pose_samples = [sample for sample in samples if len(sample.get("pose") or []) >= LANDMARK_COUNT]
    coverage = len(pose_samples) / max(1, len(samples))
    if not pose_samples:
        return {"score": 0.0, "valid": False, "reason": "no landmark timeline", "coverage": 0.0}

    visibility_values: List[float] = []
    for sample in pose_samples:
        pose = sample["pose"]
        visibility_values.extend(float(pose[index][3]) for index in MATCH_LANDMARKS)
    visibility = float(np.mean(visibility_values)) if visibility_values else 0.0

    frame_gaps: List[int] = []
    angular_speeds: List[float] = []
    pose_speeds: List[float] = []
    for left, right in zip(pose_samples, pose_samples[1:]):
        gap = max(1, int(right.get("frame", 0)) - int(left.get("frame", 0)))
        frame_gaps.append(gap)
        dt = gap / max(float(fps), 1.0)
        for key in JOINT_KEYS:
            try:
                angular_speeds.append(abs(float(right[key]) - float(left[key])) / dt)
            except (KeyError, TypeError, ValueError):
                pass
        a = np.asarray([left["pose"][idx][:3] for idx in MATCH_LANDMARKS], dtype=np.float64)
        b = np.asarray([right["pose"][idx][:3] for idx in MATCH_LANDMARKS], dtype=np.float64)
        pose_speeds.append(float(np.median(np.linalg.norm(b - a, axis=1))) / dt)

    max_gap = max(frame_gaps, default=1)
    p95_angular_speed = float(np.percentile(angular_speeds, 95)) if angular_speeds else 0.0
    p95_pose_speed = float(np.percentile(pose_speeds, 95)) if pose_speeds else 0.0
    duration = float(pose_samples[-1].get("t", 0.0)) - float(pose_samples[0].get("t", 0.0))

    score = 100.0
    score -= max(0.0, 0.92 - coverage) * 90.0
    score -= max(0.0, 0.72 - visibility) * 80.0
    score -= max(0, max_gap - 2) * 7.0
    score -= max(0.0, p95_angular_speed - 720.0) / 30.0
    score -= max(0.0, p95_pose_speed - 7.0) * 2.0
    if not 0.4 <= duration <= 2.1:
        score -= 25.0
    score = max(0.0, min(100.0, score))
    valid = coverage >= 0.75 and visibility >= 0.35 and max_gap <= 6 and score >= 45.0
    reason = "ok" if valid else "low coverage, visibility, or temporal continuity"
    return {
        "score": round(score, 2),
        "valid": bool(valid),
        "reason": reason,
        "coverage": round(coverage, 3),
        "mean_visibility": round(visibility, 3),
        "max_frame_gap": int(max_gap),
        "p95_angular_speed_deg_s": round(p95_angular_speed, 1),
        "p95_pose_speed_torso_s": round(p95_pose_speed, 2),
        "duration_sec": round(duration, 3),
    }


def build_motion_prototype(
    timelines: Sequence[Sequence[Mapping[str, Any]]],
    *,
    frame_count: int = 48,
    min_quality: float = 45.0,
) -> Dict[str, Any]:
    """Learn a robust median player trajectory from quality-controlled shots."""
    sequences: List[List[Dict[str, Any]]] = []
    source_quality: List[float] = []
    for timeline in timelines:
        quality = timeline_quality(timeline)
        if float(quality["score"]) < min_quality or not quality["valid"]:
            continue
        sequence = resample_timeline(timeline, frame_count=frame_count)
        if sequence:
            sequences.append(sequence)
            source_quality.append(float(quality["score"]))
    if not sequences:
        return {"samples": [], "source_count": 0, "quality": 0.0}

    pose_stack = np.asarray(
        [[frame["pose"] for frame in sequence] for sequence in sequences], dtype=np.float64
    )
    median_pose = np.median(pose_stack, axis=0)
    angle_stacks = {
        key: np.asarray([[frame[key] for frame in sequence] for sequence in sequences], dtype=np.float64)
        for key in JOINT_KEYS
    }
    median_angles = {key: np.median(values, axis=0) for key, values in angle_stacks.items()}
    durations = [float(sequence[-1]["t"]) for sequence in sequences]
    duration = float(statistics.median(durations))
    samples: List[Dict[str, Any]] = []
    for index in range(frame_count):
        progress = index / (frame_count - 1)
        frame: Dict[str, Any] = {
            "t": round(duration * progress, 4),
            "phase": _phase_at(progress),
            "pose": np.round(median_pose[index], 5).tolist(),
        }
        frame.update({key: round(float(median_angles[key][index]), 2) for key in JOINT_KEYS})
        samples.append(frame)
    return {
        "samples": samples,
        "source_count": len(sequences),
        "quality": round(float(statistics.median(source_quality)), 2),
        "method": "phase_aligned_coordinate_median_v1",
        "frame_count": frame_count,
    }


def _frame_cost(left: Mapping[str, Any], right: Mapping[str, Any]) -> Tuple[float, float]:
    try:
        a = np.asarray([left["pose"][idx] for idx in MATCH_LANDMARKS], dtype=np.float64)
        b = np.asarray([right["pose"][idx] for idx in MATCH_LANDMARKS], dtype=np.float64)
    except (KeyError, IndexError, TypeError, ValueError):
        return float("inf"), 0.0
    visibility = np.minimum(a[:, 3], b[:, 3])
    # Endpoint frames frequently contain temporarily occluded wrists/ankles.
    # The coordinates still exist, so use a softer per-frame mask and let the
    # returned coverage lower confidence instead of making the entire DTW path
    # impossible.  The dataset-level quality gate remains stricter (0.35).
    mask = visibility >= 0.20
    if int(mask.sum()) < 6:
        strongest = np.argsort(visibility)[-6:]
        if float(np.mean(visibility[strongest])) < 0.10:
            return float("inf"), float(mask.mean())
        mask = np.zeros_like(visibility, dtype=bool)
        mask[strongest] = True
    delta = a[mask, :3] - b[mask, :3]
    # Monocular z is useful for orientation but much noisier than image x/y.
    delta[:, 2] *= 0.25
    spatial = np.sqrt(np.sum(delta * delta, axis=1))
    spatial_cost = float(np.average(spatial, weights=np.maximum(visibility[mask], 0.10)))
    angle_delta = []
    for key in JOINT_KEYS:
        try:
            angle_delta.append(abs(float(left[key]) - float(right[key])) / 90.0)
        except (KeyError, TypeError, ValueError):
            pass
    angle_cost = float(np.mean(angle_delta)) if angle_delta else spatial_cost
    return 0.78 * spatial_cost + 0.22 * angle_cost, float(mask.mean())


def motion_distance(
    query: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    *,
    frame_count: int = 48,
    window: int = 6,
) -> Tuple[float, float]:
    """Return band-limited DTW distance and visible-landmark coverage."""
    left = resample_timeline(query, frame_count=frame_count)
    right = resample_timeline(reference, frame_count=frame_count)
    if not left or not right:
        return float("inf"), 0.0
    n, m = len(left), len(right)
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    steps = np.zeros((n + 1, m + 1), dtype=np.int32)
    coverage = np.zeros((n + 1, m + 1), dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m, i + window) + 1):
            cost, frame_coverage = _frame_cost(left[i - 1], right[j - 1])
            if not math.isfinite(cost):
                continue
            choices = ((dp[i - 1, j], i - 1, j), (dp[i, j - 1], i, j - 1), (dp[i - 1, j - 1], i - 1, j - 1))
            previous, pi, pj = min(choices, key=lambda item: item[0])
            if not math.isfinite(previous):
                continue
            dp[i, j] = previous + cost
            steps[i, j] = steps[pi, pj] + 1
            coverage[i, j] = coverage[pi, pj] + frame_coverage
    count = int(steps[n, m])
    if count <= 0 or not math.isfinite(float(dp[n, m])):
        return float("inf"), 0.0
    return float(dp[n, m] / count), float(coverage[n, m] / count)


def motion_similarity_score(distance: float, coverage: float = 1.0) -> float:
    if not math.isfinite(distance):
        return 0.0
    base = 100.0 * math.exp(-max(0.0, distance) / 0.42)
    confidence = max(0.55, min(1.0, float(coverage)))
    return max(0.0, min(100.0, base * confidence))
