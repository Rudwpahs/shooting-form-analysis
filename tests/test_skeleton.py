"""Tests for canonical angle-derived 3D skeleton profiles."""

from __future__ import annotations

import math
import unittest

from app.skeleton import BONES, LANDMARK_ORDER, build_skeleton_timeline, canonical_landmarks


def point_angle(a, vertex, c) -> float:
    left = [float(a[i]) - float(vertex[i]) for i in range(3)]
    right = [float(c[i]) - float(vertex[i]) for i in range(3)]
    dot = sum(left[i] * right[i] for i in range(3))
    lengths = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / lengths))))


def distance(a, b) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


class SkeletonTests(unittest.TestCase):
    def test_reconstructed_joint_angles_match_profile(self):
        angles = {"elbow": 160, "shoulder": 135, "hip": 150, "knee": 145}
        points = canonical_landmarks(angles)
        measured = {
            "elbow": point_angle(points["shoulder_r"], points["elbow_r"], points["wrist_r"]),
            "shoulder": point_angle(points["hip_r"], points["shoulder_r"], points["elbow_r"]),
            "hip": point_angle(points["shoulder_r"], points["hip_r"], points["knee_r"]),
            "knee": point_angle(points["hip_r"], points["knee_r"], points["ankle_r"]),
        }
        for name, expected in angles.items():
            self.assertAlmostEqual(measured[name], expected, places=3, msg=name)

    def test_body_proportions_stay_fixed_across_motion(self):
        first = canonical_landmarks({"elbow": 120, "shoulder": 95, "hip": 120, "knee": 110})
        second = canonical_landmarks({"elbow": 175, "shoulder": 160, "hip": 170, "knee": 168})
        self.assertEqual(set(first), set(LANDMARK_ORDER))
        for start, end in BONES:
            self.assertAlmostEqual(distance(first[start], first[end]), distance(second[start], second[end]), places=4)
        self.assertAlmostEqual(min(point[1] for point in first.values()), 0.0, places=4)
        self.assertAlmostEqual(min(point[1] for point in second.values()), 0.0, places=4)

    def test_timeline_contains_moving_3d_landmarks(self):
        samples = [
            {"t": 0.0, "frame": 10, "phase": "catch", "elbow": 110, "shoulder": 90, "hip": 125, "knee": 115},
            {"t": 0.5, "frame": 25, "phase": "release", "elbow": 170, "shoulder": 150, "hip": 168, "knee": 165},
        ]
        profile = build_skeleton_timeline(samples, hand="right", view="side")
        self.assertEqual(profile["space"], "canonical_3d")
        self.assertEqual(profile["proportions"], "general_adult")
        self.assertEqual(len(profile["frames"]), 2)
        self.assertEqual(profile["frames"][1]["phase"], "release")
        self.assertNotEqual(
            profile["frames"][0]["landmarks"]["wrist_r"],
            profile["frames"][1]["landmarks"]["wrist_r"],
        )


if __name__ == "__main__":
    unittest.main()
