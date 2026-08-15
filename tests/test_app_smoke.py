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
        self.assertEqual(data["compare"], "angles_deg_only")

    def test_index_html(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        body = res.get_data(as_text=True)
        self.assertIn("Shooting Form Analysis", body)
        self.assertIn("/static/styles.css", body)
        self.assertIn('id="target_player"', body)
        self.assertIn('id="selected_compare"', body)

    def test_players_list(self):
        res = self.client.get("/api/players")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("players", data)
        self.assertGreaterEqual(len(data["players"]), 1)
        self.assertTrue(all(player["space"] == "3d" for player in data["players"]))
        self.assertNotIn("pro_baseline", {player["player_key"] for player in data["players"]})
