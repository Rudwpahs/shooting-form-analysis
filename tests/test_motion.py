"""Tests for normalized full-motion player profiles."""

from __future__ import annotations

import unittest

from app.motion import (
    build_motion_prototype,
    motion_distance,
    normalize_landmarks,
    timeline_quality,
)
from app.similarity import match_views


def raw_pose(offset_x: float = 0.0, scale: float = 1.0):
    pose = []
    for index in range(33):
        pose.append(
            {
                "x": offset_x + scale * (0.4 + (index % 3 - 1) * 0.04),
                "y": 0.2 + scale * (index / 50.0),
                "z": scale * ((index % 2) * 0.02),
                "visibility": 0.95,
            }
        )
    pose[11].update(x=offset_x + scale * 0.42, y=0.2 + scale * 0.15)
    pose[12].update(x=offset_x + scale * 0.58, y=0.2 + scale * 0.15)
    pose[23].update(x=offset_x + scale * 0.45, y=0.2 + scale * 0.42)
    pose[24].update(x=offset_x + scale * 0.55, y=0.2 + scale * 0.42)
    return pose


def timeline(shift: float = 0.0):
    samples = []
    for index in range(21):
        if index < 5:
            phase = "catch"
        elif index == 5:
            phase = "dip"
        elif index < 15:
            phase = "rise"
        elif index == 15:
            phase = "release"
        else:
            phase = "follow_through"
        pose = []
        for landmark in range(33):
            pose.append([
                (landmark % 4) * 0.08 + shift * index / 20.0,
                0.8 + (landmark % 6) * 0.05 + index * 0.01,
                (landmark % 2) * 0.03,
                0.95,
            ])
        samples.append(
            {
                "t": index * 0.05,
                "frame": index,
                "phase": phase,
                "elbow": 110 + index * 2.5 + shift * 15,
                "shoulder": 90 + index * 2.5,
                "hip": 135 + index,
                "knee": 120 + index * 1.5,
                "pose": pose,
            }
        )
    return samples


class MotionTests(unittest.TestCase):
    def test_normalization_removes_translation_and_scale(self):
        first = normalize_landmarks(raw_pose(offset_x=0.0, scale=1.0))
        second = normalize_landmarks(raw_pose(offset_x=0.3, scale=1.7))
        self.assertEqual(len(first), 33)
        for landmark in (11, 12, 23, 24):
            for axis in range(3):
                self.assertAlmostEqual(first[landmark][axis], second[landmark][axis], places=4)

    def test_quality_and_prototype_accept_continuous_motion(self):
        source = timeline()
        quality = timeline_quality(source, fps=20.0)
        self.assertTrue(quality["valid"], quality)
        prototype = build_motion_prototype([source, timeline(0.02)])
        self.assertEqual(prototype["source_count"], 2)
        self.assertEqual(len(prototype["samples"]), 48)
        self.assertEqual(len(prototype["samples"][0]["pose"]), 33)

    def test_motion_distance_prefers_same_trajectory(self):
        source = timeline()
        same_distance, same_coverage = motion_distance(source, source)
        other_distance, _ = motion_distance(source, timeline(0.8))
        self.assertAlmostEqual(same_distance, 0.0, places=6)
        self.assertGreater(same_coverage, 0.9)
        self.assertGreater(other_distance, same_distance)

    def test_similarity_uses_motion_before_release_angle_fallback(self):
        query = timeline()
        angles = {"elbow": 160, "shoulder": 145, "hip": 155, "knee": 150}
        catalog = [
            {"player_key": "near", "display_name": "Near", "view": "side", "space": "3d", "angles": angles, "timeline": query},
            {"player_key": "far", "display_name": "Far", "view": "side", "space": "3d", "angles": angles, "timeline": timeline(0.8)},
        ]
        matches = match_views([{"view": "side", "space": "3d", "angles": angles, "timeline": query}], catalog)
        self.assertEqual(matches[0].player_key, "near")
        self.assertEqual(matches[0].method, "motion_dtw_v1")
        self.assertGreater(matches[0].score, matches[1].score)


if __name__ == "__main__":
    unittest.main()
