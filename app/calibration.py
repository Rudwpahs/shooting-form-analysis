"""Camera-calibration contracts for verified multi-view reconstruction.

The previous view-tag approach assumed a fixed focal length, camera distance,
and yaw.  It is retained nowhere in the verified path: callers must supply
per-camera intrinsics and extrinsics that are tied to the capture session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    camera_id: str
    intrinsic: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    reprojection_error_px: float
    image_width: int
    image_height: int

    @classmethod
    def from_mapping(cls, camera_id: str, data: Mapping[str, Any]) -> "CameraCalibration":
        intrinsic = np.asarray(data.get("intrinsic"), dtype=np.float64)
        rotation = np.asarray(data.get("rotation"), dtype=np.float64)
        translation = np.asarray(data.get("translation"), dtype=np.float64).reshape(-1)
        if intrinsic.shape != (3, 3):
            raise ValueError(f"{camera_id}: intrinsic must be 3x3")
        if rotation.shape != (3, 3):
            raise ValueError(f"{camera_id}: rotation must be 3x3")
        if translation.shape != (3,):
            raise ValueError(f"{camera_id}: translation must contain 3 values")
        if not np.all(np.isfinite(intrinsic)) or not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError(f"{camera_id}: calibration values must be finite")
        if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            raise ValueError(f"{camera_id}: focal lengths must be positive")
        error = float(data.get("reprojection_error_px", float("inf")))
        image_width = int(data.get("image_width", 0))
        image_height = int(data.get("image_height", 0))
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"{camera_id}: image_width and image_height must be positive")
        return cls(camera_id, intrinsic, rotation, translation, error, image_width, image_height)

    @property
    def projection(self) -> np.ndarray:
        extrinsic = np.concatenate([self.rotation, self.translation.reshape(3, 1)], axis=1)
        return self.intrinsic @ extrinsic

    def project(self, point_xyz: np.ndarray) -> tuple[float, float]:
        hom = np.append(np.asarray(point_xyz, dtype=np.float64).reshape(3), 1.0)
        image = self.projection @ hom
        if abs(float(image[2])) < 1e-9:
            raise ValueError("point projects to infinity")
        return float(image[0] / image[2]), float(image[1] / image[2])


@dataclass(frozen=True)
class CalibrationBundle:
    cameras: Mapping[str, CameraCalibration]
    capture_id: str = ""
    version: str = "calibration_v1"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CalibrationBundle":
        raw_cameras = payload.get("cameras") or {}
        if not isinstance(raw_cameras, Mapping):
            raise ValueError("cameras must be an object")
        cameras = {
            str(camera_id): CameraCalibration.from_mapping(str(camera_id), item or {})
            for camera_id, item in raw_cameras.items()
        }
        return cls(
            cameras=cameras,
            capture_id=str(payload.get("capture_id") or ""),
            version=str(payload.get("version") or "calibration_v1"),
        )

    def validation_report(self, *, max_reprojection_error_px: float = 2.5) -> dict:
        reasons: list[str] = []
        if len(self.cameras) < 2:
            reasons.append("requires calibration for at least two cameras")
        errors = {}
        for name, camera in self.cameras.items():
            error = float(camera.reprojection_error_px)
            errors[name] = error
            if not np.isfinite(error):
                reasons.append(f"{name}: missing reprojection_error_px")
            elif error > max_reprojection_error_px:
                reasons.append(f"{name}: reprojection error exceeds {max_reprojection_error_px:.1f}px")
        return {
            "status": "verified" if not reasons else "rejected",
            "valid": not reasons,
            "capture_id": self.capture_id,
            "version": self.version,
            "camera_count": len(self.cameras),
            "reprojection_errors_px": errors,
            "max_reprojection_error_px": max_reprojection_error_px,
            "reasons": reasons,
        }


def parse_calibration_payload(payload: Mapping[str, Any] | None) -> tuple[CalibrationBundle | None, dict]:
    """Parse untrusted request data into a bundle plus a safe validation report."""
    if not payload:
        return None, {
            "status": "not_provided",
            "valid": False,
            "reasons": ["calibration payload was not provided"],
        }
    try:
        bundle = CalibrationBundle.from_mapping(payload)
    except (TypeError, ValueError) as exc:
        return None, {"status": "rejected", "valid": False, "reasons": [str(exc)]}
    return bundle, bundle.validation_report()
