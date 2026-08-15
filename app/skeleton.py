"""Canonical 3D skeleton reconstruction from shooting-form joint angles.

The source profiles contain scale-invariant angles rather than athlete body
measurements.  This module therefore uses fixed, ordinary adult proportions
and inverse kinematics so players and users are visualized on the same body.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

Point3 = Tuple[float, float, float]

MODEL_NAME = "canonical_human_v1"
CANONICAL_HEIGHT_M = 1.75
LANDMARK_ORDER = (
    "head",
    "neck",
    "shoulder_l",
    "shoulder_r",
    "elbow_l",
    "elbow_r",
    "wrist_l",
    "wrist_r",
    "spine",
    "pelvis",
    "hip_l",
    "hip_r",
    "knee_l",
    "knee_r",
    "ankle_l",
    "ankle_r",
    "toe_l",
    "toe_r",
)

BONES = (
    ("head", "neck"),
    ("neck", "shoulder_l"),
    ("neck", "shoulder_r"),
    ("shoulder_l", "shoulder_r"),
    ("shoulder_l", "elbow_l"),
    ("elbow_l", "wrist_l"),
    ("shoulder_r", "elbow_r"),
    ("elbow_r", "wrist_r"),
    ("neck", "spine"),
    ("spine", "pelvis"),
    ("pelvis", "hip_l"),
    ("pelvis", "hip_r"),
    ("hip_l", "hip_r"),
    ("hip_l", "knee_l"),
    ("knee_l", "ankle_l"),
    ("ankle_l", "toe_l"),
    ("hip_r", "knee_r"),
    ("knee_r", "ankle_r"),
    ("ankle_r", "toe_r"),
)

REQUIRED_ANGLES = ("elbow", "shoulder", "hip", "knee")


def _add(a: Point3, b: Point3) -> Point3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Point3, b: Point3) -> Point3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(point: Point3, value: float) -> Point3:
    return (point[0] * value, point[1] * value, point[2] * value)


def _norm(point: Point3) -> float:
    return math.sqrt(sum(value * value for value in point))


def _unit(point: Point3) -> Point3:
    length = _norm(point)
    if length <= 1e-9:
        return (0.0, 1.0, 0.0)
    return _scale(point, 1.0 / length)


def _clamped_angle(value: Any) -> float:
    return max(15.0, min(179.0, float(value)))


def _distal_candidates(
    joint: Point3,
    proximal: Point3,
    angle_deg: float,
    length: float,
) -> Tuple[Point3, Point3]:
    """Two sagittal-plane points that preserve the requested joint angle."""
    toward_proximal = _unit(_sub(proximal, joint))
    perpendicular = _unit((0.0, -toward_proximal[2], toward_proximal[1]))
    radians = math.radians(_clamped_angle(angle_deg))
    base = _scale(toward_proximal, math.cos(radians))
    bend = _scale(perpendicular, math.sin(radians))
    return (
        _add(joint, _scale(_add(base, bend), length)),
        _add(joint, _scale(_sub(base, bend), length)),
    )


def _arm_chain(shoulder: Point3, hip: Point3, angles: Mapping[str, float]) -> Tuple[Point3, Point3]:
    elbow_options = _distal_candidates(shoulder, hip, angles["shoulder"], 0.31)
    elbow = max(elbow_options, key=lambda point: point[1] + 0.35 * point[2])
    wrist_options = _distal_candidates(elbow, shoulder, angles["elbow"], 0.27)
    wrist = max(wrist_options, key=lambda point: point[1] + 0.2 * point[2])
    return elbow, wrist


def _leg_chain(hip: Point3, shoulder: Point3, angles: Mapping[str, float]) -> Tuple[Point3, Point3, Point3]:
    knee_options = _distal_candidates(hip, shoulder, angles["hip"], 0.45)
    knee = max(knee_options, key=lambda point: point[2])
    ankle_options = _distal_candidates(knee, hip, angles["knee"], 0.45)
    ankle = min(ankle_options, key=lambda point: point[1] + 0.8 * abs(point[2]))
    toe = (ankle[0], ankle[1] - 0.025, ankle[2] + 0.25)
    return knee, ankle, toe


def canonical_landmarks(angles: Mapping[str, float], hand: str = "right") -> Dict[str, List[float]]:
    """Build a fixed-proportion 3D body whose shooting-side angles match input."""
    clean = {name: _clamped_angle(angles[name]) for name in REQUIRED_ANGLES}
    points: Dict[str, Point3] = {
        "head": (0.0, 1.75, 0.0),
        "neck": (0.0, 1.58, 0.0),
        "shoulder_l": (-0.21, 1.45, 0.0),
        "shoulder_r": (0.21, 1.45, 0.0),
        "spine": (0.0, 1.22, 0.0),
        "pelvis": (0.0, 0.95, 0.0),
        "hip_l": (-0.15, 0.95, 0.0),
        "hip_r": (0.15, 0.95, 0.0),
    }

    for suffix in ("l", "r"):
        shoulder = points[f"shoulder_{suffix}"]
        hip = points[f"hip_{suffix}"]
        elbow, wrist = _arm_chain(shoulder, hip, clean)
        knee, ankle, toe = _leg_chain(hip, shoulder, clean)
        points[f"elbow_{suffix}"] = elbow
        points[f"wrist_{suffix}"] = wrist
        points[f"knee_{suffix}"] = knee
        points[f"ankle_{suffix}"] = ankle
        points[f"toe_{suffix}"] = toe

    # Keep the lowest foot on the floor while preserving every segment length.
    floor_y = min(points["toe_l"][1], points["toe_r"][1], points["ankle_l"][1], points["ankle_r"][1])
    points = {name: (point[0], point[1] - floor_y, point[2]) for name, point in points.items()}
    return {
        name: [round(value, 5) for value in points[name]]
        for name in LANDMARK_ORDER
    }


def build_skeleton_timeline(
    samples: Sequence[Mapping[str, Any]],
    *,
    hand: str = "right",
    view: str = "side",
    source_space: str = "3d",
) -> Dict[str, Any]:
    """Convert angle samples into an API-ready canonical 3D animation."""
    frames: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples):
        try:
            angles = {name: float(sample[name]) for name in REQUIRED_ANGLES}
            landmarks = canonical_landmarks(angles, hand=hand)
        except (KeyError, TypeError, ValueError):
            continue
        frames.append(
            {
                "t": round(float(sample.get("t", index / 30.0)), 4),
                "frame": sample.get("frame"),
                "phase": str(sample.get("phase") or "release"),
                "angles": {name: round(value, 2) for name, value in angles.items()},
                "landmarks": landmarks,
            }
        )

    return {
        "model": MODEL_NAME,
        "space": "canonical_3d",
        "source_space": str(source_space or "3d"),
        "units": "meters",
        "canonical_height_m": CANONICAL_HEIGHT_M,
        "proportions": "general_adult",
        "hand": hand if hand in ("left", "right") else "right",
        "view": view,
        "duration": round(float(frames[-1]["t"]), 4) if frames else 0.0,
        "landmark_order": list(LANDMARK_ORDER),
        "bones": [list(bone) for bone in BONES],
        "frames": frames,
    }


def release_skeleton(
    angles: Mapping[str, float],
    *,
    hand: str = "right",
    view: str = "merged",
    source_space: str = "3d",
) -> Dict[str, Any]:
    """Create a one-frame fallback profile when no motion timeline exists."""
    sample = {"t": 0.0, "phase": "release", **angles}
    return build_skeleton_timeline(
        [sample],
        hand=hand,
        view=view,
        source_space=source_space,
    )
