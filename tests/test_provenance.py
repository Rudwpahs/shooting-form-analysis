"""Tests for provenance-gated player reference publication."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.db import connect, refresh_player_verification, upsert_player, upsert_reference_clip
from app.provenance import ClipProvenance, PROFILE_UNVERIFIED, PROFILE_VERIFIED_2D, PROFILE_VERIFIED_3D


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "profiles.sqlite"
        self.conn = connect(self.db_path)
        upsert_player(
            self.conn,
            "stephen_curry",
            "Stephen Curry",
            {"elbow": 160, "shoulder": 140, "hip": 160, "knee": 150},
            space="3d",
        )

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    @staticmethod
    def approved_clip(index: int) -> ClipProvenance:
        return ClipProvenance.from_mapping(
            {
                "clip_id": f"curry-{index}",
                "player_key": "stephen_curry",
                "source_url": f"https://example.org/curry-shot-{index}",
                "footage_type": "real",
                "identity_status": "verified",
                "shot_status": "verified",
                "review_status": "approved",
                "reviewer": "coach-a",
                "catch_frame": 10,
                "release_frame": 20,
                "followthrough_end_frame": 35,
                "fps": 60,
                "view": "side",
                "ball_visible_ratio": 0.9,
                "occlusion_ratio": 0.05,
            }
        )

    def test_incomplete_or_synthetic_clip_cannot_be_approved(self):
        clip = ClipProvenance.from_mapping(
            {
                "clip_id": "synthetic-1",
                "player_key": "stephen_curry",
                "source_url": "https://example.org/gameplay",
                "footage_type": "synthetic",
                "identity_status": "verified",
                "shot_status": "verified",
                "review_status": "approved",
                "reviewer": "coach-a",
                "catch_frame": 10,
                "release_frame": 20,
                "followthrough_end_frame": 35,
            }
        )
        errors = upsert_reference_clip(self.conn, clip)
        self.assertIn("footage_type must be real", errors)
        status, reasons = refresh_player_verification(self.conn, "stephen_curry")
        self.assertEqual(status, PROFILE_UNVERIFIED)
        self.assertTrue(reasons)

    def test_three_independent_reviewed_clips_publish_only_verified_2d(self):
        for index in range(3):
            self.assertEqual(upsert_reference_clip(self.conn, self.approved_clip(index)), [])
        status, reasons = refresh_player_verification(self.conn, "stephen_curry")
        self.assertEqual(status, PROFILE_VERIFIED_2D)
        self.assertEqual(reasons, [])

    def test_verified_3d_requires_explicit_canonical_gate(self):
        for index in range(3):
            upsert_reference_clip(self.conn, self.approved_clip(index))
        status, reasons = refresh_player_verification(
            self.conn,
            "stephen_curry",
            canonical_3d_verified=True,
        )
        self.assertEqual(status, PROFILE_VERIFIED_3D)
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
