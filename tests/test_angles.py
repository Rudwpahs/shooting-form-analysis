"""Unit tests for scale-invariant angle matching (no video/GPU required)."""

from __future__ import annotations

import unittest

import numpy as np

from app.angles import angle_degrees, angle_distance, normalize_angle_dict, similarity_score
from app.similarity import match_angles


class AngleTests(unittest.TestCase):
    def test_scale_invariant(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([1.0, 1.0, 0.0])
        self.assertAlmostEqual(angle_degrees(a, b, c), 90.0, places=5)
        self.assertAlmostEqual(angle_degrees(a * 4.2, b * 4.2, c * 4.2), 90.0, places=5)

    def test_normalize_legacy_keys(self):
        raw = {"Elbow angle": 170.0, "Shoulder angle": 145.0, "Hip angle": 168.0, "Knee angle": 165.0}
        out = normalize_angle_dict(raw)
        self.assertEqual(out["elbow"], 170.0)
        self.assertEqual(set(out), {"elbow", "shoulder", "hip", "knee"})

    def test_match_ranks_closer_profile(self):
        query = {"elbow": 179.0, "shoulder": 152.0, "hip": 176.0, "knee": 177.0}
        catalog = [
            {"player_key": "far", "display_name": "Far", "angles": {"elbow": 140, "shoulder": 110, "hip": 140, "knee": 140}},
            {"player_key": "near", "display_name": "Near", "angles": {"elbow": 178, "shoulder": 151, "hip": 175, "knee": 176}},
        ]
        matches = match_angles(query, catalog, top_k=2)
        self.assertEqual(matches[0].player_key, "near")
        self.assertLess(matches[0].distance_deg, matches[1].distance_deg)
        self.assertGreater(matches[0].score, matches[1].score)

    def test_distance_zero_when_equal(self):
        a = {"elbow": 170, "shoulder": 145, "hip": 168, "knee": 165}
        self.assertEqual(angle_distance(a, a), 0.0)
        self.assertEqual(similarity_score(0.0), 100.0)


if __name__ == "__main__":
    unittest.main()
