"""Tests for calibrated multi-view quality gates."""

from __future__ import annotations

import unittest

import numpy as np

from app.calibration import CalibrationBundle, parse_calibration_payload
from app.sync import validate_sync_payload
from app.verified_multiview import triangulate_release


class CalibrationPipelineTests(unittest.TestCase):
    @staticmethod
    def payload():
        intrinsic = [[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]]
        return {
            "capture_id": "pilot-001",
            "cameras": {
                "side": {
                    "intrinsic": intrinsic,
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "translation": [0.0, 0.0, 0.0],
                    "reprojection_error_px": 0.4,
                    "image_width": 640,
                    "image_height": 480,
                },
                "front": {
                    "intrinsic": intrinsic,
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "translation": [-1.0, 0.0, 0.0],
                    "reprojection_error_px": 0.5,
                    "image_width": 640,
                    "image_height": 480,
                },
            },
        }

    @staticmethod
    def body_points():
        points = np.asarray([[0.03 * (i % 5), 0.05 * (i % 6), 5.0 + 0.01 * i] for i in range(33)], dtype=np.float64)
        # Right shooting side: shoulder, elbow, wrist, hip, knee, ankle.
        points[12] = [0.0, 1.0, 5.0]
        points[14] = [0.5, 1.1, 5.0]
        points[16] = [1.0, 1.3, 5.0]
        points[24] = [0.0, 0.0, 5.0]
        points[26] = [0.1, -1.0, 5.0]
        points[28] = [0.1, -2.0, 5.0]
        # Left side remains plausible if hand selection is changed later.
        points[11] = [-0.0, 1.0, 5.0]
        points[13] = [-0.5, 1.1, 5.0]
        points[15] = [-1.0, 1.3, 5.0]
        points[23] = [-0.0, 0.0, 5.0]
        points[25] = [-0.1, -1.0, 5.0]
        points[27] = [-0.1, -2.0, 5.0]
        return points

    @staticmethod
    def projected_landmarks(bundle, camera_id, points):
        camera = bundle.cameras[camera_id]
        output = []
        for point in points:
            x, y = camera.project(point)
            output.append({"x": x / camera.image_width, "y": y / camera.image_height, "visibility": 1.0})
        return output

    def test_calibration_contract_rejects_missing_resolution(self):
        payload = self.payload()
        del payload["cameras"]["side"]["image_width"]
        bundle, report = parse_calibration_payload(payload)
        self.assertIsNone(bundle)
        self.assertFalse(report["valid"])

    def test_sync_requires_explicit_high_confidence_offsets(self):
        report = validate_sync_payload(
            {
                "offsets": {
                    "side": {"offset_frames": 0, "confidence": 0.99, "method": "audio_clap"},
                    "front": {"offset_frames": 1, "confidence": 0.96, "method": "audio_clap"},
                }
            },
            required_camera_ids={"side", "front"},
        )
        self.assertTrue(report["valid"])
        rejected = validate_sync_payload(
            {"offsets": {"side": {"offset_frames": 8, "confidence": 0.1, "method": "unknown"}}},
            required_camera_ids={"side", "front"},
        )
        self.assertFalse(rejected["valid"])
        self.assertTrue(rejected["reasons"])

    def test_exact_two_camera_geometry_passes_calibrated_triangulation(self):
        bundle = CalibrationBundle.from_mapping(self.payload())
        points = self.body_points()
        result = triangulate_release(
            {
                "side": self.projected_landmarks(bundle, "side", points),
                "front": self.projected_landmarks(bundle, "front", points),
            },
            bundle,
            hand="right",
        )
        self.assertIsNotNone(result.angles)
        self.assertTrue(result.quality["valid"], result.quality)
        self.assertEqual(result.quality["status"], "verified")
        self.assertLess(max(result.quality["reprojection_rmse_px"].values()), 0.001)


if __name__ == "__main__":
    unittest.main()
