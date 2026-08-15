"""Unit tests for scale-invariant angle matching (no video/GPU required)."""

from __future__ import annotations

import unittest

import numpy as np

from app.angles import (
    AngleSnapshot,
    angle_degrees,
    angle_distance,
    angles_plausible,
    normalize_angle_dict,
    similarity_score,
)
from app.similarity import match_angles, match_player, match_views


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

    def test_side_query_uses_side_profile_not_merged(self):
        query = {"elbow": 159.7, "shoulder": 160.7, "hip": 159.6, "knee": 129.0}
        catalog = [
            {
                "player_key": "stephen_curry",
                "display_name": "Stephen Curry",
                "view": "merged",
                "angles": {"elbow": 161.4, "shoulder": 116.2, "hip": 129.9, "knee": 132.3},
            },
            {
                "player_key": "stephen_curry",
                "display_name": "Stephen Curry",
                "view": "side",
                "angles": {"elbow": 159.7, "shoulder": 160.7, "hip": 159.6, "knee": 129.0},
            },
            {
                "player_key": "jamal_murray",
                "display_name": "Jamal Murray",
                "view": "merged",
                "angles": {"elbow": 147.5, "shoulder": 133.8, "hip": 150.6, "knee": 139.1},
            },
        ]
        matches = match_angles(query, catalog, top_k=2, query_view="side")
        self.assertEqual(matches[0].player_key, "stephen_curry")
        self.assertEqual(matches[0].matched_view, "side")
        self.assertLess(matches[0].distance_deg, 1.0)
        self.assertEqual(matches[1].player_key, "jamal_murray")

    def test_clip_sample_beats_other_player_median(self):
        query = {"elbow": 154.3, "shoulder": 148.3, "hip": 150.0, "knee": 160.0}
        catalog = [
            {
                "player_key": "devin_booker",
                "display_name": "Devin Booker",
                "view": "merged",
                "angles": {"elbow": 140.8, "shoulder": 125.5, "hip": 153.5, "knee": 159.6},
            },
            {
                "player_key": "devin_booker",
                "display_name": "Devin Booker",
                "view": "clip:9cz8R4x1DHw",
                "angles": {"elbow": 154.3, "shoulder": 148.3, "hip": 150.0, "knee": 160.0},
            },
            {
                "player_key": "jamal_murray",
                "display_name": "Jamal Murray",
                "view": "merged",
                "angles": {"elbow": 147.5, "shoulder": 133.8, "hip": 150.6, "knee": 139.1},
            },
        ]
        matches = match_angles(query, catalog, top_k=2, query_view="side")
        self.assertEqual(matches[0].player_key, "devin_booker")
        self.assertTrue(str(matches[0].matched_view).startswith("clip:"))
        self.assertLess(matches[0].distance_deg, 1.0)

    def test_distance_zero_when_equal(self):
        a = {"elbow": 170, "shoulder": 145, "hip": 168, "knee": 165}
        self.assertEqual(angle_distance(a, a), 0.0)
        self.assertEqual(similarity_score(0.0), 100.0)

    def test_angles_plausible_rejects_collapsed_shoulder(self):
        bad = AngleSnapshot(elbow=150, shoulder=12, hip=160, knee=150, hand="right", space="3d")
        good = AngleSnapshot(elbow=150, shoulder=140, hip=160, knee=150, hand="right", space="3d")
        self.assertFalse(angles_plausible(bad))
        self.assertTrue(angles_plausible(good))

    def test_3d_query_does_not_match_2d_profile(self):
        query = {"elbow": 170, "shoulder": 145, "hip": 168, "knee": 165}
        catalog = [
            {"player_key": "two_d", "display_name": "2D", "space": "2d", "view": "side", "angles": query},
            {
                "player_key": "three_d",
                "display_name": "3D",
                "space": "3d",
                "view": "side",
                "angles": {"elbow": 168, "shoulder": 143, "hip": 166, "knee": 163},
            },
        ]
        matches = match_views([{"view": "side", "space": "3d", "angles": query}], catalog)
        self.assertEqual([match.player_key for match in matches], ["three_d"])
        self.assertEqual(matches[0].matched_space, "3d")

    def test_multiview_3d_query_matches_3d_profile(self):
        query = {"elbow": 170, "shoulder": 145, "hip": 168, "knee": 165}
        catalog = [
            {
                "player_key": "three_d",
                "display_name": "3D",
                "space": "3d",
                "view": "merged",
                "angles": query,
            }
        ]
        matches = match_views([{"view": "merged", "space": "3d_mv", "angles": query}], catalog)
        self.assertEqual([match.player_key for match in matches], ["three_d"])
        self.assertEqual(matches[0].matched_space, "3d")

    def test_selected_player_is_compared_even_when_not_closest(self):
        query = {"elbow": 170, "shoulder": 145, "hip": 168, "knee": 165}
        catalog = [
            {"player_key": "near", "display_name": "Near", "space": "3d", "view": "side", "angles": query},
            {
                "player_key": "chosen",
                "display_name": "Chosen",
                "space": "3d",
                "view": "side",
                "angles": {"elbow": 150, "shoulder": 125, "hip": 150, "knee": 145},
            },
        ]
        selected = match_player(
            [{"view": "side", "space": "3d", "angles": query}],
            catalog,
            "chosen",
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected.player_key, "chosen")
        self.assertLess(selected.score, 100)


if __name__ == "__main__":
    unittest.main()
