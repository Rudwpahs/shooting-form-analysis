"""Calibrated release-frame triangulation for the verified 3D path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from .angles import AngleSnapshot, angles_from_landmarks, angles_plausible
from .calibration import CalibrationBundle
from .quality3d import reconstruction_quality_report


@dataclass(frozen=True)
class VerifiedMultiViewResult:
    angles: Optional[AngleSnapshot]
    points: Optional[np.ndarray]
    quality: dict


def triangulate_release(
    landmark_views: Mapping[str, Sequence[Mapping[str, float]]],
    calibration: CalibrationBundle,
    *,
    hand: str | None = None,
) -> VerifiedMultiViewResult:
    """Triangulate a 33-point release pose using supplied calibration matrices."""
    report = calibration.validation_report()
    if not report["valid"]:
        return VerifiedMultiViewResult(None, None, report)
    usable = {
        camera_id: landmarks
        for camera_id, landmarks in landmark_views.items()
        if camera_id in calibration.cameras and len(landmarks) >= 29
    }
    if len(usable) < 2:
        return VerifiedMultiViewResult(
            None,
            None,
            {
                "status": "rejected",
                "valid": False,
                "reasons": ["at least two calibrated views with 29 landmarks are required"],
            },
        )

    point_rows = []
    for landmark_index in range(33):
        rays = []
        for camera_id, landmarks in usable.items():
            if landmark_index >= len(landmarks):
                continue
            landmark = landmarks[landmark_index]
            visibility = float(landmark.get("visibility", 1.0))
            if visibility < 0.35:
                continue
            camera = calibration.cameras[camera_id]
            x = float(landmark["x"]) * camera.image_width
            y = float(landmark["y"]) * camera.image_height
            rays.append((camera.projection, x, y))
        point = _dlt_point(rays)
        if point is None:
            return VerifiedMultiViewResult(
                None,
                None,
                {
                    "status": "rejected",
                    "valid": False,
                    "reasons": [f"landmark {landmark_index} lacks two usable calibrated observations"],
                },
            )
        point_rows.append(point)
    points = np.asarray(point_rows, dtype=np.float64)
    angles = angles_from_landmarks(points, hand=hand, space="3d_mv_calibrated")
    if angles is None or not angles_plausible(angles):
        return VerifiedMultiViewResult(
            angles,
            points,
            {
                "status": "rejected",
                "valid": False,
                "reasons": ["triangulated release angles are not plausible"],
            },
        )

    observations = {}
    for camera_id, landmarks in usable.items():
        camera = calibration.cameras[camera_id]
        observations[camera_id] = [
            np.asarray(
                [float(item["x"]) * camera.image_width, float(item["y"]) * camera.image_height],
                dtype=np.float64,
            )
            for item in landmarks[:33]
        ]
    quality = reconstruction_quality_report(
        [points],
        calibration=calibration,
        observations_by_camera={camera_id: [np.asarray(values)] for camera_id, values in observations.items()},
    )
    return VerifiedMultiViewResult(angles, points, quality)


def _dlt_point(observations: Sequence[tuple[np.ndarray, float, float]]) -> Optional[np.ndarray]:
    if len(observations) < 2:
        return None
    rows = []
    for projection, x, y in observations:
        rows.append(x * projection[2] - projection[0])
        rows.append(y * projection[2] - projection[1])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    homogeneous = vh[-1]
    if abs(float(homogeneous[3])) < 1e-9:
        return None
    point = homogeneous[:3] / homogeneous[3]
    return point if np.all(np.isfinite(point)) else None
