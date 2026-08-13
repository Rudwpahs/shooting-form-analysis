"""Scale-invariant joint angles from 3D (or 2D) landmarks.

Comparison must use degrees only — never bone length, height, or absolute distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

JOINT_KEYS = ("elbow", "shoulder", "hip", "knee")

# MediaPipe Pose indices
SIDE_INDICES = {
    "left": (11, 13, 15, 23, 25, 27),   # shoulder, elbow, wrist, hip, knee, ankle
    "right": (12, 14, 16, 24, 26, 28),
}


@dataclass(frozen=True)
class AngleSnapshot:
    """Joint angles in degrees. No length fields by design."""

    elbow: float
    shoulder: float
    hip: float
    knee: float
    hand: str
    space: str  # "3d" | "2d"

    def as_dict(self) -> Dict[str, float]:
        return {
            "elbow": float(self.elbow),
            "shoulder": float(self.shoulder),
            "hip": float(self.hip),
            "knee": float(self.knee),
        }

    def vector(self, keys: Sequence[str] = JOINT_KEYS) -> np.ndarray:
        data = self.as_dict()
        return np.array([data[k] for k in keys], dtype=np.float64)


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC at vertex B. Scale-invariant (unit-free)."""
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom <= 1e-9:
        return float("nan")
    cosine = float(np.dot(ba, bc) / denom)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _as_xyz(point) -> np.ndarray:
    if isinstance(point, np.ndarray):
        arr = point.astype(np.float64).ravel()
        if arr.size >= 3:
            return arr[:3]
        if arr.size == 2:
            return np.array([arr[0], arr[1], 0.0], dtype=np.float64)
        raise ValueError("point must have 2 or 3 components")
    x = float(getattr(point, "x", point[0]))
    y = float(getattr(point, "y", point[1]))
    z = float(getattr(point, "z", 0.0))
    return np.array([x, y, z], dtype=np.float64)


def visibility_mean(landmarks: Sequence, indices: Iterable[int]) -> float:
    values = []
    for i in indices:
        if i >= len(landmarks):
            continue
        values.append(float(getattr(landmarks[i], "visibility", 1.0)))
    return float(np.mean(values)) if values else 0.0


def choose_hand(landmarks: Sequence, preferred: Optional[str] = None) -> str:
    if preferred in ("left", "right"):
        return preferred
    left = visibility_mean(landmarks, SIDE_INDICES["left"])
    right = visibility_mean(landmarks, SIDE_INDICES["right"])
    return "left" if left >= right else "right"


def angles_from_landmarks(
    landmarks: Sequence,
    hand: Optional[str] = None,
    *,
    space: str = "3d",
) -> Optional[AngleSnapshot]:
    """Compute shooting-side joint angles. Prefer world (3d) landmarks when available."""
    if landmarks is None or len(landmarks) < 29:
        return None

    side = choose_hand(landmarks, hand)
    indices = SIDE_INDICES[side]
    if visibility_mean(landmarks, indices) < 0.35:
        return None

    shoulder, elbow, wrist, hip, knee, ankle = [_as_xyz(landmarks[i]) for i in indices]
    if space == "2d":
        shoulder[2] = elbow[2] = wrist[2] = hip[2] = knee[2] = ankle[2] = 0.0

    snap = AngleSnapshot(
        elbow=angle_degrees(shoulder, elbow, wrist),
        shoulder=angle_degrees(elbow, shoulder, hip),
        hip=angle_degrees(shoulder, hip, knee),
        knee=angle_degrees(hip, knee, ankle),
        hand=side,
        space=space,
    )
    if any(np.isnan(v) or np.isinf(v) for v in snap.as_dict().values()):
        return None
    return snap


def median_angles(samples: Sequence[AngleSnapshot]) -> Optional[AngleSnapshot]:
    if not samples:
        return None
    mat = np.vstack([s.vector() for s in samples])
    med = np.median(mat, axis=0)
    return AngleSnapshot(
        elbow=float(med[0]),
        shoulder=float(med[1]),
        hip=float(med[2]),
        knee=float(med[3]),
        hand=samples[0].hand,
        space=samples[0].space,
    )


def angle_distance(
    a: Mapping[str, float],
    b: Mapping[str, float],
    weights: Optional[Mapping[str, float]] = None,
    keys: Sequence[str] = JOINT_KEYS,
) -> float:
    """Weighted RMSE in degrees — length-free similarity core."""
    w = weights or {k: 1.0 for k in keys}
    total_w = 0.0
    acc = 0.0
    for key in keys:
        if key not in a or key not in b:
            continue
        wk = float(w.get(key, 1.0))
        diff = float(a[key]) - float(b[key])
        acc += wk * diff * diff
        total_w += wk
    if total_w <= 0:
        return float("inf")
    return float(np.sqrt(acc / total_w))


def similarity_score(distance_deg: float, scale: float = 12.0) -> float:
    """Map angle RMSE to 0–100 (higher = closer)."""
    if not np.isfinite(distance_deg):
        return 0.0
    return float(max(0.0, min(100.0, 100.0 * np.exp(-distance_deg / scale))))


def normalize_angle_dict(raw: Mapping[str, float]) -> Dict[str, float]:
    """Accept legacy keys like 'Elbow angle'."""
    aliases = {
        "elbow": ("elbow", "Elbow", "Elbow angle", "elbow_angle"),
        "shoulder": ("shoulder", "Shoulder", "Shoulder angle", "shoulder_angle"),
        "hip": ("hip", "Hip", "Hip angle", "hip_angle"),
        "knee": ("knee", "Knee", "Knee angle", "knee_angle"),
    }
    out: Dict[str, float] = {}
    for key, names in aliases.items():
        for name in names:
            if name in raw:
                out[key] = float(raw[name])
                break
    return out
