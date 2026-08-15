"""Smoke tests for the Flask angle-only app."""

from __future__ import annotations

import unittest

from app.server import app


class FlaskSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["compare"], "motion_dtw_with_angle_fallback")

    def test_index_html(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        body = res.get_data(as_text=True)
        self.assertIn("Shooting Form Analysis", body)
        self.assertIn("/static/styles.css", body)
        self.assertIn('id="target_player"', body)
        self.assertIn('id="selected_compare"', body)
        self.assertIn('id="user_skeleton_canvas"', body)
        self.assertIn('id="player_skeleton_canvas"', body)
        self.assertIn('/static/skeleton3d.js', body)

    def test_players_list(self):
        res = self.client.get("/api/players")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("players", data)
        self.assertGreaterEqual(len(data["players"]), 1)
        self.assertTrue(all(player["space"] == "3d" for player in data["players"]))
        self.assertNotIn("pro_baseline", {player["player_key"] for player in data["players"]})

    def test_every_player_has_a_skeleton_profile(self):
        players = self.client.get("/api/players").get_json()["players"]
        self.assertGreaterEqual(len(players), 1)
        for player in players:
            res = self.client.get(f"/api/players/{player['player_key']}/skeleton")
            self.assertEqual(res.status_code, 200, player["display_name"])
            data = res.get_json()
            self.assertEqual(data["skeleton"]["space"], "canonical_3d")
            self.assertGreaterEqual(len(data["skeleton"]["frames"]), 1)

    def test_unknown_player_skeleton_is_404(self):
        res = self.client.get("/api/players/not-a-real-player/skeleton")
        self.assertEqual(res.status_code, 404)
