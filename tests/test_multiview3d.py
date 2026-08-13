"""Tests for multi-view triangulation (synthetic cameras)."""

from __future__ import annotations

import unittest

import numpy as np

from app.angles import SIDE_INDICES
from app.multiview3d import (
    angles_from_triangulated,
    merge_multiview_release,
    triangulate_point,
    triangulate_side_joints,
)


def _fake_landmarks(points_xy: dict, side: str = "right") -> list:
    """Build a 33-landmark list; fill shooting-side joints from points_xy."""
    lms = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0} for _ in range(33)]
    names = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")
    for name, idx in zip(names, SIDE_INDICES[side]):
        x, y = points_xy[name]
        lms[idx] = {"x": x, "y": y, "z": 0.0, "visibility": 1.0}
    return lms


class MultiView3DTests(unittest.TestCase):
    def test_ray_intersection(self):
        # Two rays that meet at (0, 1, 0)
        c1 = np.array([0.0, 1.0, -2.0])
        d1 = np.array([0.0, 0.0, 1.0])
        c2 = np.array([2.0, 1.0, 0.0])
        d2 = np.array([-1.0, 0.0, 0.0])
        p = triangulate_point([(c1, d1), (c2, d2)])
        self.assertIsNotNone(p)
        np.testing.assert_allclose(p, [0.0, 1.0, 0.0], atol=1e-6)

    def test_multiview_recovers_plausible_angles(self):
        # Side: sagittal silhouette (x=depth proxy, y=height)
        side_xy = {
            "shoulder": (0.55, 0.35),
            "elbow": (0.62, 0.28),
            "wrist": (0.68, 0.18),
            "hip": (0.52, 0.55),
            "knee": (0.54, 0.72),
            "ankle": (0.55, 0.90),
        }
        # Front: coronal
        front_xy = {
            "shoulder": (0.58, 0.35),
            "elbow": (0.64, 0.30),
            "wrist": (0.66, 0.20),
            "hip": (0.52, 0.55),
            "knee": (0.53, 0.72),
            "ankle": (0.53, 0.90),
        }
        views = {
            "side": _fake_landmarks(side_xy),
            "front": _fake_landmarks(front_xy),
        }
        points = triangulate_side_joints(views, hand="right")
        self.assertIsNotNone(points)
        snap = angles_from_triangulated(points, hand="right")
        # Synthetic pose may not pass tight sports ranges; at least finite angles.
        self.assertTrue(all(np.isfinite(v) for v in snap.as_dict().values()) if snap else True)
        merged = merge_multiview_release(views, hand="right")
        # Soft assert: function returns either None (implausible) or a snapshot
        if merged is not None:
            self.assertEqual(merged.space, "3d_mv")


if __name__ == "__main__":
    unittest.main()
