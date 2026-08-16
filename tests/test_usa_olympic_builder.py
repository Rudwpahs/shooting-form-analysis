"""Tests for the Paris 2024 USA canonical-model batch."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from app.reconstruction3d import (
    CANONICAL_MODEL_NAME,
    FRAME_COUNT,
    SUPPORTED_CANONICAL_PLAYER_KEYS,
    build_model_entry,
    load_canonical_player_skeleton,
    write_validated_model,
)
from scripts.build_curry_canonical3d import current_view_urls
from scripts.build_usa_olympic_canonical3d import (
    USA_OLYMPIC_PLAYERS,
    catalog_urls_for_player,
    catalog_urls_for_view,
)


EXPECTED_PLAYERS = {
    "stephen_curry": "Stephen Curry",
    "devin_booker": "Devin Booker",
    "kevin_durant": "Kevin Durant",
    "anthony_edwards": "Anthony Edwards",
    "lebron_james": "LeBron James",
}


class UsaOlympicBuilderTests(unittest.TestCase):
    def test_supported_players_are_exact_roster_intersection(self):
        self.assertEqual(dict(USA_OLYMPIC_PLAYERS), EXPECTED_PLAYERS)
        self.assertEqual(set(SUPPORTED_CANONICAL_PLAYER_KEYS), set(EXPECTED_PLAYERS))

    def test_existing_profile_urls_reject_wrong_player_identity(self):
        data = {
            "Anthony Edwards": {
                "views": {
                    "side": {
                        "clips": [
                            {
                                "youtube_url": "https://youtu.be/wrong",
                                "title": "Anthony Davis Shooting Form Slow Motion",
                            },
                            {
                                "youtube_url": "https://youtu.be/right",
                                "title": "Anthony Edwards Shooting Form Slow Motion",
                            },
                            {
                                "youtube_url": "https://youtu.be/surname",
                                "title": "Edwards slow-motion jumper",
                            },
                        ]
                    }
                }
            }
        }

        self.assertEqual(
            current_view_urls(data, "Anthony Edwards", "side"),
            (
                "https://youtu.be/right",
                "https://youtu.be/surname",
            ),
        )

    def test_catalog_separates_front_and_side_and_checks_identity(self):
        catalog = {
            "Anthony Edwards": [
                {
                    "youtube_url": "https://youtu.be/front",
                    "title": "Anthony Edwards front jumper",
                    "query": "Anthony Edwards shooting form front view",
                },
                {
                    "youtube_url": "https://youtu.be/side",
                    "title": "Edwards shooting form",
                    "query": "Anthony Edwards shooting form side angle",
                },
                {
                    "youtube_url": "https://youtu.be/wrong",
                    "title": "Anthony Davis shooting form",
                    "query": "Anthony Edwards shooting form front view",
                },
            ]
        }

        self.assertEqual(
            catalog_urls_for_view(catalog, "Anthony Edwards", "front"),
            ("https://youtu.be/front",),
        )
        self.assertEqual(
            catalog_urls_for_view(catalog, "Anthony Edwards", "side"),
            ("https://youtu.be/side",),
        )
        self.assertEqual(
            catalog_urls_for_player(catalog, "Anthony Edwards"),
            ("https://youtu.be/front", "https://youtu.be/side"),
        )

    def test_generic_model_round_trip_for_booker(self):
        xyz = np.zeros((FRAME_COUNT, 33, 3), dtype=float)
        xyz[:, :, 1] = np.linspace(1.0, 0.0, 33)
        entry = build_model_entry(
            player_key="devin_booker",
            xyz=xyz,
            dispersion=np.zeros_like(xyz),
            confidence=np.ones((FRAME_COUNT, 33)),
            phases=tuple(["rise"] * FRAME_COUNT),
            clips=[],
            bone_lengths={},
            optimizer={"success": True},
            validation={"passed": True, "reasons": []},
        )
        self.assertEqual(entry["model"], CANONICAL_MODEL_NAME)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "canonical.json"
            write_validated_model(path, entry)
            skeleton = load_canonical_player_skeleton("devin_booker", path=path)

        self.assertIsNotNone(skeleton)
        self.assertEqual(skeleton["model"], CANONICAL_MODEL_NAME)
        self.assertEqual(len(skeleton["frames"]), FRAME_COUNT)

    def test_loader_keeps_legacy_curry_model_compatible(self):
        xyz = np.zeros((FRAME_COUNT, 33, 3), dtype=float)
        entry = build_model_entry(
            player_key="stephen_curry",
            xyz=xyz,
            dispersion=np.zeros_like(xyz),
            confidence=np.ones((FRAME_COUNT, 33)),
            phases=tuple(["rise"] * FRAME_COUNT),
            clips=[],
            bone_lengths={},
            optimizer={"success": True},
            validation={"passed": True},
        )
        entry["model"] = "curry_multiview_bundle_v1"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "canonical.json"
            path.write_text(
                json.dumps({"stephen_curry": entry}), encoding="utf-8"
            )
            skeleton = load_canonical_player_skeleton("stephen_curry", path=path)

        self.assertIsNotNone(skeleton)


if __name__ == "__main__":
    unittest.main()
