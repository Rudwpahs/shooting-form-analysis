"""Smoke tests for the Flask angle-only app."""

from __future__ import annotations

import unittest
from unittest import mock

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

    def test_paris_usa_frontend_scope_lists_only_five_players(self):
        res = self.client.get("/api/players?scope=paris_2024_usa")
        self.assertEqual(res.status_code, 200)
        keys = {player["player_key"] for player in res.get_json()["players"]}
        self.assertEqual(
            keys,
            {
                "stephen_curry",
                "devin_booker",
                "kevin_durant",
                "anthony_edwards",
                "lebron_james",
            },
        )

    def test_frontend_requests_paris_scope_and_labels_3d_quality(self):
        res = self.client.get("/static/app.js")
        self.assertEqual(res.status_code, 200)
        body = res.get_data(as_text=True)
        self.assertIn("/api/players?scope=paris_2024_usa", body)
        self.assertIn("Multi-view 3D", body)
        self.assertIn("Single-view estimated 3D", body)

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

    def test_usa_olympic_endpoints_prefer_validated_canonical_model(self):
        canonical = {
            "model": "player_multiview_bundle_v1",
            "space": "canonical_3d",
            "quality_mode": "multi_view_3d",
            "frames": [{"t": 0.0, "landmarks": {}}],
        }
        olympians = (
            "stephen_curry",
            "devin_booker",
            "kevin_durant",
            "anthony_edwards",
            "lebron_james",
        )
        for player_key in olympians:
            with self.subTest(player_key=player_key), mock.patch(
                "app.server.load_canonical_player_skeleton",
                return_value=canonical,
            ) as loader:
                response = self.client.get(
                    f"/api/players/{player_key}/skeleton"
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["skeleton"], canonical)
            loader.assert_called_once_with(player_key)

    def test_non_olympic_player_does_not_enter_canonical_loader(self):
        with mock.patch(
            "app.server.load_canonical_player_skeleton",
        ) as loader:
            response = self.client.get(
                "/api/players/donovan_mitchell/skeleton"
            )

        self.assertEqual(response.status_code, 200)
        loader.assert_not_called()
