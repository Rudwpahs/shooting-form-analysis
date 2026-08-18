"""Smoke tests for the provenance-gated shooting analysis API."""

from __future__ import annotations

import unittest
from unittest import mock

from app.server import app


class FlaskSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health_reports_verified_reference_policy(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["compare"], "verified_motion_reference_only")
        self.assertIn("verified_2d_profiles", data)
        self.assertIn("verified_3d_profiles", data)

    def test_index_html_describes_validation_boundary(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        body = res.get_data(as_text=True)
        self.assertIn("승인된 출처", body)
        self.assertIn("검증된 선수 3D", body)
        self.assertIn('id="target_player"', body)
        self.assertIn('id="user_skeleton_canvas"', body)
        self.assertIn('id="player_skeleton_canvas"', body)

    def test_default_players_endpoint_returns_only_matchable_profiles(self):
        res = self.client.get("/api/players")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("publication_policy", data)
        self.assertTrue(
            all(player["verification_status"] in {"verified_2d", "verified_3d"} for player in data["players"])
        )

    def test_paris_scope_can_show_unverified_profiles_without_publishing_them(self):
        res = self.client.get("/api/players?scope=paris_2024_usa&include_unverified=1")
        self.assertEqual(res.status_code, 200)
        players = res.get_json()["players"]
        self.assertEqual(
            {player["player_key"] for player in players},
            {"stephen_curry", "devin_booker", "kevin_durant", "anthony_edwards", "lebron_james"},
        )
        self.assertTrue(all("verification_reasons" in player for player in players))
        self.assertTrue(all("provenance" in player for player in players))

    def test_frontend_requests_review_state_but_only_enables_verified_profiles(self):
        res = self.client.get("/static/app.js")
        self.assertEqual(res.status_code, 200)
        body = res.get_data(as_text=True)
        self.assertIn("include_unverified=1", body)
        self.assertIn("verified_2d", body)
        self.assertIn("verified_3d", body)
        self.assertIn("3D 검증 대기", body)
        self.assertIn("not validated 3D", body)

    def test_unverified_player_skeleton_is_blocked(self):
        response = self.client.get("/api/players/stephen_curry/skeleton")
        self.assertEqual(response.status_code, 409)
        data = response.get_json()
        self.assertEqual(data["verification_status"], "unverified_legacy")
        self.assertIn("verification_reasons", data)

    def test_verified_profile_can_return_published_canonical_model(self):
        profile = {
            "player_key": "stephen_curry",
            "display_name": "Stephen Curry",
            "verification_status": "verified_3d",
            "verification_reasons": [],
            "provenance": {"approved_clip_count": 3},
        }
        canonical = {
            "model": "player_multiview_bundle_v2",
            "space": "canonical_3d",
            "quality_mode": "calibrated_multi_view_3d",
            "frames": [{"t": 0.0, "landmarks": {}}],
        }
        with mock.patch("app.server.get_player_skeleton_source", return_value=profile), mock.patch(
            "app.server.load_canonical_player_skeleton", return_value=canonical
        ) as loader:
            response = self.client.get("/api/players/stephen_curry/skeleton")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["skeleton"], canonical)
        loader.assert_called_once_with("stephen_curry")

    def test_verified_status_without_published_model_is_blocked(self):
        profile = {
            "player_key": "stephen_curry",
            "display_name": "Stephen Curry",
            "verification_status": "verified_3d",
            "verification_reasons": [],
            "provenance": {},
        }
        with mock.patch("app.server.get_player_skeleton_source", return_value=profile), mock.patch(
            "app.server.load_canonical_player_skeleton", return_value=None
        ):
            response = self.client.get("/api/players/stephen_curry/skeleton")
        self.assertEqual(response.status_code, 409)
        self.assertIn("published canonical model", response.get_json()["error"])

    def test_unknown_player_skeleton_is_404(self):
        res = self.client.get("/api/players/not-a-real-player/skeleton")
        self.assertEqual(res.status_code, 404)
