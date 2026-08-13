"""Tests for catch-to-follow-through shot span detection."""

from __future__ import annotations

import unittest

from app.shot_span import detect_shot_span, phase_label


class ShotSpanTests(unittest.TestCase):
    def test_catch_dip_release_followthrough(self):
        y = []
        y.extend([0.40] * 10)  # hold / catch
        y.extend([0.40 + 0.025 * i for i in range(1, 16)])  # dip down
        y.extend([0.775 - 0.035 * i for i in range(1, 16)])  # rise to release
        y.extend([0.25 + 0.012 * i for i in range(1, 13)])  # follow-through drop
        span = detect_shot_span(y, fps=30.0)
        self.assertLess(span.catch_index, span.dip_index)
        self.assertLess(span.dip_index, span.release_index)
        self.assertGreaterEqual(span.followthrough_index, span.release_index)
        self.assertGreater(y[span.dip_index], y[span.release_index])
        self.assertEqual(phase_label(span.catch_index, span), "catch")
        self.assertEqual(phase_label(span.release_index, span), "release")
        self.assertEqual(phase_label(span.followthrough_index, span), "follow_through")


if __name__ == "__main__":
    unittest.main()
