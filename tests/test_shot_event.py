"""Tests for basketball evidence around a pose-derived release."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from app.shot_event import (
    BallObservation,
    detect_basketball,
    link_ball_track,
    shooter_track_quality,
    verify_shot_event,
)


class ShotEventTests(unittest.TestCase):
    def test_orange_circular_baseline_detector_finds_ball_candidate(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(image, (160, 120), 16, (0, 140, 255), -1)
        result = detect_basketball(image, frame_index=4)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.frame, 4)
        self.assertGreater(result.confidence, 0.48)
        self.assertAlmostEqual(result.x, 0.5, delta=0.04)

    def test_verified_event_requires_near_hand_then_upward_separation(self):
        bboxes = [(frame, (0.3, 0.1, 0.7, 0.9)) for frame in range(7, 13)]
        quality = shooter_track_quality(bboxes, expected_frames=6)
        ball_track = link_ball_track(
            [
                BallObservation(8, 0.50, 0.50, 0.02, 0.8),
                BallObservation(9, 0.51, 0.49, 0.02, 0.8),
                BallObservation(10, 0.52, 0.48, 0.02, 0.8),
                BallObservation(11, 0.63, 0.36, 0.02, 0.8),
                BallObservation(12, 0.71, 0.28, 0.02, 0.8),
            ]
        )
        wrists = [(frame, 0.50, 0.50) for frame in range(7, 13)]
        event = verify_shot_event(
            release_frame=10,
            wrist_positions=wrists,
            ball_track=ball_track,
            shooter_quality=quality,
            fps=60,
        )
        self.assertTrue(event.verified)
        self.assertEqual(event.status, "verified")
        self.assertGreater(event.upward_ball_motion or 0, 0.025)

    def test_pose_only_peak_is_not_a_verified_shot(self):
        bboxes = [(frame, (0.3, 0.1, 0.7, 0.9)) for frame in range(7, 13)]
        quality = shooter_track_quality(bboxes, expected_frames=6)
        event = verify_shot_event(
            release_frame=10,
            wrist_positions=[(frame, 0.50, 0.50) for frame in range(7, 13)],
            ball_track=[],
            shooter_quality=quality,
            fps=60,
        )
        self.assertFalse(event.verified)
        self.assertEqual(event.status, "pose_only_unverified")
        self.assertIn("insufficient basketball observations", event.reasons)

    def test_discontinuous_shooter_track_blocks_shot_even_with_ball_track(self):
        bboxes = [
            (7, (0.1, 0.1, 0.3, 0.9)),
            (8, (0.7, 0.1, 0.9, 0.9)),
            (9, (0.1, 0.1, 0.3, 0.9)),
            (10, (0.7, 0.1, 0.9, 0.9)),
            (11, (0.1, 0.1, 0.3, 0.9)),
            (12, (0.7, 0.1, 0.9, 0.9)),
        ]
        quality = shooter_track_quality(bboxes, expected_frames=6)
        event = verify_shot_event(
            release_frame=10,
            wrist_positions=[(frame, 0.50, 0.50) for frame in range(7, 13)],
            ball_track=[BallObservation(frame, 0.5, 0.5, 0.02, 0.9) for frame in range(7, 12)],
            shooter_quality=quality,
            fps=60,
        )
        self.assertFalse(event.verified)
        self.assertEqual(event.status, "rejected")


if __name__ == "__main__":
    unittest.main()
