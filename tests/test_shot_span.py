"""Tests for catch-to-follow-through shot span detection."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.shot_span import (
    MAX_TIMELINE_DURATION_SEC,
    ShotSpan,
    detect_shot_span,
    phase_label,
    shot_span_is_complete,
    summarize_phases,
    timeline_duration_is_plausible,
)


class ShotSpanTests(unittest.TestCase):
    def test_catch_dip_release_followthrough(self):
        y = []
        y.extend([0.40] * 10)
        y.extend([0.40 + 0.025 * i for i in range(1, 16)])
        y.extend([0.775 - 0.035 * i for i in range(1, 16)])
        y.extend([0.25 + 0.012 * i for i in range(1, 13)])
        span = detect_shot_span(y, fps=30.0)
        self.assertLess(span.catch_index, span.dip_index)
        self.assertLess(span.dip_index, span.release_index)
        self.assertGreaterEqual(span.followthrough_index, span.release_index)
        self.assertGreater(y[span.dip_index], y[span.release_index])
        self.assertEqual(phase_label(span.catch_index, span), "catch")
        self.assertEqual(phase_label(span.release_index, span), "release")
        self.assertEqual(phase_label(span.followthrough_index, span), "follow_through")

    def test_summarize_phases_keeps_every_phase(self):
        samples = [
            {"t": 0.0, "frame": 1, "phase": "catch", "elbow": 110, "shoulder": 90, "hip": 160, "knee": 140},
            {"t": 0.1, "frame": 4, "phase": "dip", "elbow": 120, "shoulder": 80, "hip": 150, "knee": 125},
            {"t": 0.2, "frame": 7, "phase": "rise", "elbow": 140, "shoulder": 120, "hip": 155, "knee": 130},
            {"t": 0.3, "frame": 10, "phase": "release", "elbow": 160, "shoulder": 150, "hip": 158, "knee": 135},
            {"t": 0.4, "frame": 13, "phase": "follow_through", "elbow": 165, "shoulder": 148, "hip": 170, "knee": 120},
        ]
        rows = summarize_phases(samples)
        self.assertEqual([r["phase"] for r in rows], ["catch", "dip", "rise", "release", "follow_through"])
        self.assertEqual(rows[0]["count"], 1)
        self.assertEqual(rows[3]["angles"]["elbow"], 160)

    def test_sparse_pose_frames_use_video_time_not_sample_count(self):
        frames = [0, 40, 80, 120, 160, 190, 208]
        wrist_y = [0.4, 0.42, 0.45, 0.5, 0.78, 0.2, 0.38]

        span = detect_shot_span(
            wrist_y,
            fps=30.0,
            release_index=5,
            frame_indices=frames,
        )

        duration = (frames[span.followthrough_index] - frames[span.catch_index]) / 30.0
        self.assertLessEqual(duration, MAX_TIMELINE_DURATION_SEC)
        self.assertLessEqual(frames[span.release_index] - frames[span.catch_index], 45)
        self.assertLessEqual(frames[span.followthrough_index] - frames[span.release_index], 17)

    def test_rejects_too_short_and_too_long_timelines(self):
        self.assertFalse(timeline_duration_is_plausible([{"t": 0.0}]))
        self.assertFalse(timeline_duration_is_plausible([{"t": 0.0}, {"t": 0.1}]))
        self.assertFalse(timeline_duration_is_plausible([{"t": 0.0}, {"t": 0.2}]))
        self.assertFalse(timeline_duration_is_plausible([{"t": 0.0}, {"t": 6.93}]))
        self.assertTrue(timeline_duration_is_plausible([{"t": 0.0}, {"t": 0.7}]))

    def test_requires_motion_before_and_after_release(self):
        self.assertTrue(shot_span_is_complete(detect_shot_span(
            [0.4, 0.6, 0.3, 0.2, 0.4], fps=30.0, release_index=3
        )))
        self.assertFalse(shot_span_is_complete(ShotSpan(0, 0, 0, 3)))

    def test_all_stored_player_timelines_are_valid(self):
        model_path = Path(__file__).resolve().parents[1] / "models" / "nba_player_models.json"
        players = json.loads(model_path.read_text(encoding="utf-8"))
        checked = 0
        for name, player in players.items():
            for view_name, view in (player.get("views") or {}).items():
                timeline = (view or {}).get("timeline") or {}
                samples = timeline.get("samples") or []
                if not samples:
                    continue
                phases = timeline.get("phases") or {}
                message = f"{name}/{view_name}"
                self.assertLess(phases["catch"], phases["release"], message)
                self.assertLess(phases["dip"], phases["release"], message)
                self.assertLess(phases["release"], phases["follow_through"], message)
                self.assertTrue(
                    timeline_duration_is_plausible(samples, fps=float(timeline.get("fps") or 30)),
                    message,
                )
                checked += 1
        self.assertEqual(checked, 16)


if __name__ == "__main__":
    unittest.main()
