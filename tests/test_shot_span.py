"""Tests for catch-to-follow-through shot span detection."""

from __future__ import annotations

import unittest

from app.shot_span import detect_shot_span, phase_label, summarize_phases


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


if __name__ == "__main__":
    unittest.main()
