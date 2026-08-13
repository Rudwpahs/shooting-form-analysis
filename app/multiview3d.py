"""Multi-view 3D joint reconstruction from uncalibrated tagged cameras.

Phone uploads are not calibrated, but view tags (front / side / oblique)
give relative yaw. Joint *angles* are scale-invariant, so an approximate
camera distance cancels out — we only need consistent orientation.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .angles import (
    SIDE_INDICES,
    AngleSnapshot,
    angle_degrees,
    angles_plausible,
    choose_hand,
)

# Assumed yaw around vertical axis (radians). Shooter faces +Z.
VIEW_YAW = {
    "front": 0.0,
    "side": np.pi / 2.0,
    "oblique": np.pi / 4.0,
}

JOINT_NAMES = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")


def _rot_y(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _landmark_uv(lm) -> Tuple[float, float]:
    """Normalized image coords → centered, y-up."""
    if isinstance(lm, dict):
        return float(lm["x"]) - 0.5, 0.5 - float(lm["y"])
    if hasattr(lm, "x"):
        return float(lm.x) - 0.5, 0.5 - float(lm.y)
    return float(lm[0]) - 0.5, 0.5 - float(lm[1])


def camera_ray(
    landmark,
    view: str,
    *,
    focal: float = 1.15,
    cam_dist: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (camera_center, unit_ray_dir) in a shared world frame."""
    yaw = VIEW_YAW.get(view, 0.0)
    R = _rot_y(yaw)
    u, v = _landmark_uv(landmark)
    # Pinhole ray in camera coords (looking along +Z_cam).
    ray_cam = np.array([u / focal, v / focal, 1.0], dtype=np.float64)
    ray_cam /= max(np.linalg.norm(ray_cam), 1e-9)
    # Camera sits on a circle looking at origin.
    center = R @ np.array([0.0, 0.0, -cam_dist], dtype=np.float64)
    direction = R @ ray_cam
    return center, direction


def triangulate_point(rays: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Optional[np.ndarray]:
    """Least-squares intersection of 3D rays (Midpoint / DLT-style)."""
    if len(rays) < 2:
        return None
    # Solve min_p Σ || (p - c) × d ||^2  →  (I - dd^T) p = (I - dd^T) c
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    for center, direction in rays:
        d = direction / max(np.linalg.norm(direction), 1e-9)
        proj = np.eye(3) - np.outer(d, d)
        A += proj
        b += proj @ center
    try:
        p = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(p)):
        return None
    return p


def triangulate_side_joints(
    views: Mapping[str, Sequence],
    hand: Optional[str] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Triangulate shooting-side joints from ≥2 tagged views.

    views: {view_tag: image_landmarks (33 MediaPipe points)}
    """
    usable = {k: v for k, v in views.items() if v is not None and len(v) >= 29 and k in VIEW_YAW}
    if len(usable) < 2:
        return None

    # Pick hand from the first available view.
    first_lms = next(iter(usable.values()))
    side = choose_hand(first_lms, hand)
    indices = SIDE_INDICES[side]

    points: Dict[str, np.ndarray] = {}
    for name, idx in zip(JOINT_NAMES, indices):
        rays = []
        for view_tag, lms in usable.items():
            if idx >= len(lms):
                continue
            rays.append(camera_ray(lms[idx], view_tag))
        if len(rays) < 2:
            return None
        p = triangulate_point(rays)
        if p is None:
            return None
        points[name] = p
    return points


def angles_from_triangulated(
    points: Mapping[str, np.ndarray],
    hand: str = "right",
) -> Optional[AngleSnapshot]:
    needed = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")
    if not all(k in points for k in needed):
        return None
    snap = AngleSnapshot(
        elbow=angle_degrees(points["shoulder"], points["elbow"], points["wrist"]),
        shoulder=angle_degrees(points["elbow"], points["shoulder"], points["hip"]),
        hip=angle_degrees(points["shoulder"], points["hip"], points["knee"]),
        knee=angle_degrees(points["hip"], points["knee"], points["ankle"]),
        hand=hand,
        space="3d_mv",
    )
    if not angles_plausible(snap):
        return None
    return snap


def merge_multiview_release(
    view_landmarks: Mapping[str, Sequence],
    hand: Optional[str] = None,
) -> Optional[AngleSnapshot]:
    """Build release angles from multi-view triangulation when possible."""
    points = triangulate_side_joints(view_landmarks, hand=hand)
    if points is None:
        return None
    first = next(iter(view_landmarks.values()))
    side = choose_hand(first, hand)
    return angles_from_triangulated(points, hand=side)
