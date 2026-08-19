"""Quality gates for calibrated multi-view 3D reconstructions."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .calibration import CalibrationBundle

MAJOR_BONES = (
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 25), (25, 27),
    (24, 26), (26, 28),
)


def reconstruction_quality_report(
    points_by_frame: Sequence[np.ndarray],
    *,
    calibration: CalibrationBundle | None,
    observations_by_camera: Mapping[str, Sequence[np.ndarray]] | None = None,
    max_reprojection_error_px: float = 4.0,
    max_bone_cv: float = 0.12,
    max_velocity_spike_ratio: float = 5.0,
) -> dict:
    """Return an auditable quality report, never a silently repaired 3D result."""
    reasons: list[str] = []
    if calibration is None:
        return {
            "status": "not_available",
            "valid": False,
            "reasons": ["calibration is required for verified 3D"],
        }
    calibration_report = calibration.validation_report()
    if not calibration_report["valid"]:
        return {
            "status": "rejected",
            "valid": False,
            "calibration": calibration_report,
            "reasons": calibration_report["reasons"],
        }
    if not points_by_frame:
        return {
            "status": "rejected",
            "valid": False,
            "calibration": calibration_report,
            "reasons": ["no reconstructed 3D frames"],
        }
    points = np.asarray(points_by_frame, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (33, 3) or not np.all(np.isfinite(points)):
        return {
            "status": "rejected",
            "valid": False,
            "calibration": calibration_report,
            "reasons": ["3D frames must have finite shape [frame, 33, 3]"],
        }

    bone_cvs = {}
    for left, right in MAJOR_BONES:
        lengths = np.linalg.norm(points[:, left] - points[:, right], axis=1)
        median = float(np.median(lengths))
        cv = float(np.std(lengths) / max(median, 1e-9))
        bone_cvs[f"{left}-{right}"] = cv
    max_observed_bone_cv = max(bone_cvs.values(), default=0.0)
    if max_observed_bone_cv > max_bone_cv:
        reasons.append(f"bone-length coefficient of variation exceeds {max_bone_cv:.2f}")

    pelvis = (points[:, 23] + points[:, 24]) * 0.5
    velocity = np.linalg.norm(np.diff(pelvis, axis=0), axis=1) if len(pelvis) > 1 else np.array([0.0])
    median_velocity = float(np.median(velocity))
    peak_velocity = float(np.max(velocity))
    spike_ratio = peak_velocity / max(median_velocity, 1e-6)
    if len(velocity) > 3 and spike_ratio > max_velocity_spike_ratio:
        reasons.append(f"temporal velocity spike ratio exceeds {max_velocity_spike_ratio:.1f}")

    reprojection = _reprojection_errors(points, calibration, observations_by_camera or {})
    max_reprojection = max(reprojection.values(), default=float("inf"))
    if not reprojection:
        reasons.append("no 2D observations supplied for reprojection validation")
    elif max_reprojection > max_reprojection_error_px:
        reasons.append(f"reprojection RMSE exceeds {max_reprojection_error_px:.1f}px")

    return {
        "status": "verified" if not reasons else "rejected",
        "valid": not reasons,
        "frame_count": int(points.shape[0]),
        "calibration": calibration_report,
        "reprojection_rmse_px": reprojection,
        "max_reprojection_error_px": max_reprojection_error_px,
        "bone_length_cv": bone_cvs,
        "max_bone_cv": max_observed_bone_cv,
        "max_allowed_bone_cv": max_bone_cv,
        "pelvis_velocity_peak": peak_velocity,
        "pelvis_velocity_median": median_velocity,
        "velocity_spike_ratio": spike_ratio,
        "reasons": reasons,
    }


def _reprojection_errors(
    points: np.ndarray,
    calibration: CalibrationBundle,
    observations_by_camera: Mapping[str, Sequence[np.ndarray]],
) -> dict[str, float]:
    report: dict[str, float] = {}
    for camera_id, frames_2d in observations_by_camera.items():
        camera = calibration.cameras.get(camera_id)
        if camera is None or len(frames_2d) != len(points):
            continue
        errors = []
        for frame_points, observed in zip(points, frames_2d):
            target = np.asarray(observed, dtype=np.float64)
            if target.shape != (33, 2):
                continue
            projected = np.asarray([camera.project(point) for point in frame_points], dtype=np.float64)
            mask = np.all(np.isfinite(target), axis=1)
            if np.any(mask):
                errors.extend(np.linalg.norm(projected[mask] - target[mask], axis=1).tolist())
        if errors:
            report[camera_id] = round(float(np.sqrt(np.mean(np.square(errors)))), 4)
    return report
