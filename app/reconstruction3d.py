"""Robust multi-view canonical 3D reconstruction for player motion profiles.

The runtime loader in this module is intentionally independent from the
offline SciPy optimizer so serving a prebuilt model does not pull SciPy into
the Render image.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import math

import numpy as np

from .motion import LANDMARK_COUNT, resample_timeline


FRAME_COUNT = 48
CANONICAL_MODEL_NAME = "player_multiview_bundle_v1"
LEGACY_CANONICAL_MODEL_NAMES = frozenset({"curry_multiview_bundle_v1"})
SUPPORTED_CANONICAL_PLAYER_KEYS = (
    "stephen_curry",
    "devin_booker",
    "kevin_durant",
    "anthony_edwards",
    "lebron_james",
)
MIN_EFFECTIVE_CLIP_COVERAGE = 0.35
VIEW_YAW = {
    "front": 0.0,
    "side": math.pi / 2.0,
    "oblique": math.pi / 4.0,
}
CANONICAL_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "canonical_3d_models.json"
)
MAJOR_BONES = (
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (11, 23),
    (12, 24),
    (11, 12),
    (23, 24),
)
PAIRED_MAJOR_BONES = {
    (11, 13): (12, 14),
    (12, 14): (11, 13),
    (13, 15): (14, 16),
    (14, 16): (13, 15),
    (23, 25): (24, 26),
    (24, 26): (23, 25),
    (25, 27): (26, 28),
    (26, 28): (25, 27),
    (11, 23): (12, 24),
    (12, 24): (11, 23),
}
ANGLE_TRIPLETS = (
    (12, 14, 16),
    (14, 12, 24),
    (12, 24, 26),
    (24, 26, 28),
)
MP_LANDMARK_NAMES = (
    "nose",
    "eye_inner_l",
    "eye_l",
    "eye_outer_l",
    "eye_inner_r",
    "eye_r",
    "eye_outer_r",
    "ear_l",
    "ear_r",
    "mouth_l",
    "mouth_r",
    "shoulder_l",
    "shoulder_r",
    "elbow_l",
    "elbow_r",
    "wrist_l",
    "wrist_r",
    "pinky_l",
    "pinky_r",
    "index_l",
    "index_r",
    "thumb_l",
    "thumb_r",
    "hip_l",
    "hip_r",
    "knee_l",
    "knee_r",
    "ankle_l",
    "ankle_r",
    "heel_l",
    "heel_r",
    "foot_index_l",
    "foot_index_r",
)
MP_BONES = (
    ("nose", "eye_inner_l"),
    ("eye_inner_l", "eye_l"),
    ("eye_l", "eye_outer_l"),
    ("eye_outer_l", "ear_l"),
    ("nose", "eye_inner_r"),
    ("eye_inner_r", "eye_r"),
    ("eye_r", "eye_outer_r"),
    ("eye_outer_r", "ear_r"),
    ("mouth_l", "mouth_r"),
    ("shoulder_l", "shoulder_r"),
    ("shoulder_l", "elbow_l"),
    ("elbow_l", "wrist_l"),
    ("wrist_l", "pinky_l"),
    ("wrist_l", "index_l"),
    ("wrist_l", "thumb_l"),
    ("pinky_l", "index_l"),
    ("shoulder_r", "elbow_r"),
    ("elbow_r", "wrist_r"),
    ("wrist_r", "pinky_r"),
    ("wrist_r", "index_r"),
    ("wrist_r", "thumb_r"),
    ("pinky_r", "index_r"),
    ("shoulder_l", "hip_l"),
    ("shoulder_r", "hip_r"),
    ("hip_l", "hip_r"),
    ("hip_l", "knee_l"),
    ("knee_l", "ankle_l"),
    ("ankle_l", "heel_l"),
    ("heel_l", "foot_index_l"),
    ("ankle_l", "foot_index_l"),
    ("hip_r", "knee_r"),
    ("knee_r", "ankle_r"),
    ("ankle_r", "heel_r"),
    ("heel_r", "foot_index_r"),
    ("ankle_r", "foot_index_r"),
)


@dataclass(frozen=True)
class ClipObservation:
    clip_id: str
    view: str
    yaw: float
    quality: float
    uv: np.ndarray
    visibility: np.ndarray
    world_angles: np.ndarray
    frame_scale: np.ndarray
    phases: tuple[str, ...]
    source_url: str = ""
    duration: float = 1.0


@dataclass(frozen=True)
class OptimizationConfig:
    projection_weight: float = 1.0
    bone_weight: float = 3.0
    temporal_weight: float = 0.20
    world_angle_weight: float = 0.05
    anchor_weight: float = 5.0
    f_scale: float = 0.03
    max_nfev: int = 120


@dataclass(frozen=True)
class OptimizationReport:
    success: bool
    message: str
    cost: float
    nfev: int
    projection_rmse_before: float
    projection_rmse_after: float


_SEGMENT_INDICES = {
    "torso": (11, 12, 23, 24),
    "shoulder_width": (11, 12),
    "hip_width": (23, 24),
    "thigh_l": (23, 25),
    "thigh_r": (24, 26),
    "shin_l": (25, 27),
    "shin_r": (26, 28),
}


def _segment_lengths(image_xy: np.ndarray) -> dict[str, np.ndarray]:
    shoulder = (image_xy[:, 11] + image_xy[:, 12]) * 0.5
    pelvis = (image_xy[:, 23] + image_xy[:, 24]) * 0.5
    return {
        "torso": np.linalg.norm(shoulder - pelvis, axis=1),
        "shoulder_width": np.linalg.norm(
            image_xy[:, 11] - image_xy[:, 12], axis=1
        ),
        "hip_width": np.linalg.norm(image_xy[:, 23] - image_xy[:, 24], axis=1),
        "thigh_l": np.linalg.norm(image_xy[:, 23] - image_xy[:, 25], axis=1),
        "thigh_r": np.linalg.norm(image_xy[:, 24] - image_xy[:, 26], axis=1),
        "shin_l": np.linalg.norm(image_xy[:, 25] - image_xy[:, 27], axis=1),
        "shin_r": np.linalg.norm(image_xy[:, 26] - image_xy[:, 28], axis=1),
    }


def compute_fixed_scale(
    image_xy: np.ndarray,
    visibility: np.ndarray,
    *,
    view: str,
    min_visibility: float = 0.35,
) -> float:
    """Return one robust body scale for an entire shot clip."""
    xy = np.asarray(image_xy, dtype=np.float64)
    vis = np.asarray(visibility, dtype=np.float64)
    if xy.ndim != 3 or xy.shape[1:] != (LANDMARK_COUNT, 2):
        raise ValueError("image_xy must have shape [frames,33,2]")
    if vis.shape != xy.shape[:2]:
        raise ValueError("visibility must have shape [frames,33]")
    if view not in VIEW_YAW:
        raise ValueError(f"Unsupported view: {view}")

    weighted: list[float] = []
    for name, values in _segment_lengths(xy).items():
        valid = np.all(vis[:, _SEGMENT_INDICES[name]] >= min_visibility, axis=1)
        clean = values[valid & np.isfinite(values) & (values > 1e-5)]
        if not clean.size:
            continue
        center = float(np.median(clean))
        mad = float(np.median(np.abs(clean - center)))
        if mad > 1e-9:
            clean = clean[np.abs(clean - center) <= 3.5 * 1.4826 * mad]
        if not clean.size:
            continue
        weight = 0.2 if view == "side" and "width" in name else 1.0
        weighted.extend([float(np.median(clean))] * max(1, round(weight * 5)))
    if not weighted:
        raise ValueError("No reliable body segments for clip scale")
    return float(np.median(weighted))


def normalize_clip_observation(
    clip_id: str,
    samples: Sequence[Mapping[str, Any]],
    *,
    view: str,
    yaw: float | None = None,
    quality: float = 100.0,
    source_url: str = "",
) -> ClipObservation:
    """Normalize one complete shot with one scale and a catch-pelvis origin."""
    if len(samples) < 2:
        raise ValueError("At least two shot samples are required")
    image = np.asarray(
        [sample["image_landmarks"] for sample in samples], dtype=np.float64
    )
    if image.shape != (len(samples), LANDMARK_COUNT, 4):
        raise ValueError("image_landmarks must have shape [frames,33,4]")
    xy = image[:, :, :2]
    visibility = np.clip(image[:, :, 3], 0.0, 1.0)
    scale = compute_fixed_scale(xy, visibility, view=view)
    catch_pelvis = (xy[0, 23] + xy[0, 24]) * 0.5

    pose_samples: list[dict[str, Any]] = []
    for sample, points, vis in zip(samples, xy, visibility):
        normalized = np.zeros((LANDMARK_COUNT, 4), dtype=np.float64)
        normalized[:, 0] = (points[:, 0] - catch_pelvis[0]) / scale
        normalized[:, 1] = (catch_pelvis[1] - points[:, 1]) / scale
        normalized[:, 3] = vis
        pose_samples.append({**sample, "pose": normalized.tolist()})

    resampled = resample_timeline(pose_samples, frame_count=FRAME_COUNT)
    if len(resampled) != FRAME_COUNT:
        raise ValueError("Shot timeline could not be resampled to 48 frames")
    pose = np.asarray([frame["pose"] for frame in resampled], dtype=np.float64)
    world_angles = np.asarray(
        [
            [frame[key] for key in ("elbow", "shoulder", "hip", "knee")]
            for frame in resampled
        ],
        dtype=np.float64,
    )
    return ClipObservation(
        clip_id=str(clip_id),
        view=view,
        yaw=float(VIEW_YAW[view] if yaw is None else yaw),
        quality=float(quality),
        uv=pose[:, :, :2],
        visibility=pose[:, :, 3],
        world_angles=world_angles,
        frame_scale=np.full(FRAME_COUNT, scale, dtype=np.float64),
        phases=tuple(str(frame["phase"]) for frame in resampled),
        source_url=str(source_url),
        duration=float(resampled[-1]["t"]),
    )


def filter_observations(
    clips: Sequence[ClipObservation],
    *,
    min_visibility: float = 0.35,
    hampel_threshold: float = 3.5,
    minimum_jump: float = 0.08,
) -> list[ClipObservation]:
    """Disable low-confidence, velocity-spike, and bad-geometry observations."""
    filtered: list[ClipObservation] = []
    for clip in clips:
        visibility = np.asarray(clip.visibility, dtype=np.float64).copy()
        uv = np.asarray(clip.uv, dtype=np.float64)
        visibility[visibility < min_visibility] = 0.0
        for landmark in range(LANDMARK_COUNT):
            velocity = np.diff(uv[:, landmark], axis=0)
            speed = np.linalg.norm(velocity, axis=1)
            center = float(np.median(speed))
            mad = float(np.median(np.abs(speed - center)))
            limit = max(
                minimum_jump,
                center + hampel_threshold * 1.4826 * max(mad, 1e-9),
            )
            bad_edges = speed > limit
            for frame in range(1, FRAME_COUNT - 1):
                reversing_spike = (
                    bad_edges[frame - 1]
                    and bad_edges[frame]
                    and float(np.dot(velocity[frame - 1], velocity[frame])) < 0.0
                )
                if reversing_spike:
                    visibility[frame, landmark] = 0.0

        for start, end in MAJOR_BONES:
            length = np.linalg.norm(uv[:, start] - uv[:, end], axis=1)
            reliable = (visibility[:, start] > 0) & (visibility[:, end] > 0)
            clean = length[reliable]
            if clean.size < 5:
                continue
            center = float(np.median(clean))
            if center < 0.03:
                continue
            mad = float(np.median(np.abs(clean - center)))
            limit = max(
                center * 1.8,
                center + hampel_threshold * 1.4826 * max(mad, 1e-9),
            )
            bad_geometry = reliable & (length > limit)
            visibility[bad_geometry, start] = 0.0
            visibility[bad_geometry, end] = 0.0

        for left, right in (
            ((11, 13), (12, 14)),
            ((13, 15), (14, 16)),
            ((23, 25), (24, 26)),
            ((25, 27), (26, 28)),
        ):
            left_length = np.linalg.norm(uv[:, left[0]] - uv[:, left[1]], axis=1)
            right_length = np.linalg.norm(
                uv[:, right[0]] - uv[:, right[1]], axis=1
            )
            reliable = np.all(
                visibility[:, [left[0], left[1], right[0], right[1]]] > 0.0,
                axis=1,
            )
            collapsed_left = reliable & (right_length > 0.08) & (
                left_length < 0.15 * right_length
            )
            collapsed_right = reliable & (left_length > 0.08) & (
                right_length < 0.15 * left_length
            )
            visibility[collapsed_left, left[1]] = 0.0
            visibility[collapsed_right, right[1]] = 0.0
        filtered.append(replace(clip, visibility=visibility))
    return filtered


def reproject_xyz(xyz: np.ndarray, yaw: float) -> np.ndarray:
    """Orthographically project canonical XYZ into a yawed camera view."""
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("xyz must have shape [frames,landmarks,3]")
    horizontal = points[:, :, 0] * math.cos(yaw) + points[:, :, 2] * math.sin(
        yaw
    )
    return np.stack((horizontal, points[:, :, 1]), axis=-1)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not values.size or values.shape != weights.shape:
        raise ValueError("weighted median requires equally sized observations")
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = 0.5 * float(ordered_weights.sum())
    index = int(np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _mad_mask(values: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    if mad <= 1e-9:
        return np.abs(values - center) <= 1e-6
    return np.abs(values - center) <= threshold * 1.4826 * mad


def filter_cross_clip_residuals(
    clips: Sequence[ClipObservation],
    xyz: np.ndarray,
    *,
    minimum_residual: float = 0.08,
    mad_threshold: float = 3.5,
) -> list[ClipObservation]:
    """Mask observations far from same-yaw peers and the robust XYZ seed."""
    projected_residuals = [
        clip.uv - reproject_xyz(xyz, clip.yaw) for clip in clips
    ]
    visibility = [np.asarray(clip.visibility, dtype=np.float64).copy() for clip in clips]
    groups: dict[float, list[int]] = {}
    for index, clip in enumerate(clips):
        groups.setdefault(round(float(clip.yaw), 3), []).append(index)
    for indices in groups.values():
        if len(indices) < 3:
            continue
        for frame in range(FRAME_COUNT):
            for landmark in range(LANDMARK_COUNT):
                available = [
                    index
                    for index in indices
                    if visibility[index][frame, landmark] > 0.0
                ]
                if len(available) < 3:
                    continue
                residual = np.asarray(
                    [projected_residuals[index][frame, landmark] for index in available]
                )
                center = np.median(residual, axis=0)
                distance = np.linalg.norm(residual - center, axis=1)
                distance_center = float(np.median(distance))
                mad = float(np.median(np.abs(distance - distance_center)))
                limit = max(
                    minimum_residual,
                    distance_center
                    + mad_threshold * 1.4826 * max(mad, 1e-9),
                )
                for index, value in zip(available, distance):
                    if value > limit:
                        visibility[index][frame, landmark] = 0.0
    return [
        replace(clip, visibility=clip_visibility)
        for clip, clip_visibility in zip(clips, visibility)
    ]


def initialize_xyz(
    clips: Sequence[ClipObservation],
    *,
    min_visibility: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a robust XYZ seed from two or more distinct camera yaws.

    Horizontal observations are filtered within each yaw group before the
    camera equations are solved. This prevents a valid side observation from
    being rejected merely because its horizontal coordinate differs from the
    front view.
    """
    yaw_groups = {round(float(clip.yaw), 3) for clip in clips}
    if len(yaw_groups) < 2:
        raise ValueError("At least two distinct camera yaw groups are required")

    xyz = np.full((FRAME_COUNT, LANDMARK_COUNT, 3), np.nan, dtype=np.float64)
    confidence = np.zeros((FRAME_COUNT, LANDMARK_COUNT), dtype=np.float64)
    solved_xz = np.zeros((FRAME_COUNT, LANDMARK_COUNT), dtype=bool)
    observed_y = np.zeros((FRAME_COUNT, LANDMARK_COUNT), dtype=bool)
    for frame in range(FRAME_COUNT):
        for landmark in range(LANDMARK_COUNT):
            grouped: dict[float, list[tuple[float, float, float]]] = {}
            vertical: list[float] = []
            vertical_weights: list[float] = []
            for clip in clips:
                visibility = float(clip.visibility[frame, landmark])
                if visibility < min_visibility:
                    continue
                weight = visibility * max(0.05, min(1.0, clip.quality / 100.0))
                group_yaw = round(float(clip.yaw), 3)
                grouped.setdefault(group_yaw, []).append(
                    (float(clip.uv[frame, landmark, 0]), weight, float(clip.yaw))
                )
                vertical.append(float(clip.uv[frame, landmark, 1]))
                vertical_weights.append(weight)

            rows: list[tuple[float, float]] = []
            horizontal: list[float] = []
            horizontal_weights: list[float] = []
            kept_weights: list[float] = []
            for observations in grouped.values():
                values = np.asarray([item[0] for item in observations])
                group_weights = np.asarray([item[1] for item in observations])
                group_yaws = np.asarray([item[2] for item in observations])
                keep = _mad_mask(values)
                representative_yaw = float(
                    np.average(group_yaws[keep], weights=group_weights[keep])
                )
                rows.append(
                    (math.cos(representative_yaw), math.sin(representative_yaw))
                )
                horizontal.append(
                    _weighted_median(values[keep], group_weights[keep])
                )
                horizontal_weights.append(float(group_weights[keep].sum()))
                kept_weights.extend(group_weights[keep].tolist())

            if rows:
                camera = np.asarray(rows, dtype=np.float64)
                horizontal_values = np.asarray(horizontal, dtype=np.float64)
                group_weights = np.asarray(horizontal_weights, dtype=np.float64)
                root_weights = np.sqrt(group_weights)[:, None]
                xz, *_ = np.linalg.lstsq(
                    camera * root_weights,
                    horizontal_values * root_weights[:, 0],
                    rcond=None,
                )
                xyz[frame, landmark, (0, 2)] = xz
                solved_xz[frame, landmark] = np.linalg.matrix_rank(camera) >= 2

            if vertical:
                vertical_values = np.asarray(vertical, dtype=np.float64)
                vertical_weight_values = np.asarray(
                    vertical_weights, dtype=np.float64
                )
                keep_vertical = _mad_mask(vertical_values)
                xyz[frame, landmark, 1] = _weighted_median(
                    vertical_values[keep_vertical],
                    vertical_weight_values[keep_vertical],
                )
                observed_y[frame, landmark] = True
                kept_weights.extend(
                    vertical_weight_values[keep_vertical].tolist()
                )
            if kept_weights:
                yaw_support = 1.0 if solved_xz[frame, landmark] else 0.3
                confidence[frame, landmark] = min(
                    1.0, float(np.mean(kept_weights)) * yaw_support
                )

    frame_axis = np.arange(FRAME_COUNT, dtype=np.float64)
    for landmark in range(LANDMARK_COUNT):
        for axis in (0, 2):
            reliable = solved_xz[:, landmark] & np.isfinite(xyz[:, landmark, axis])
            if reliable.any():
                xyz[:, landmark, axis] = np.interp(
                    frame_axis,
                    frame_axis[reliable],
                    xyz[reliable, landmark, axis],
                )
            else:
                provisional = np.isfinite(xyz[:, landmark, axis])
                if provisional.any():
                    xyz[:, landmark, axis] = np.interp(
                        frame_axis,
                        frame_axis[provisional],
                        xyz[provisional, landmark, axis],
                    )
                else:
                    xyz[:, landmark, axis] = 0.0
        reliable_y = observed_y[:, landmark]
        if reliable_y.any():
            xyz[:, landmark, 1] = np.interp(
                frame_axis,
                frame_axis[reliable_y],
                xyz[reliable_y, landmark, 1],
            )
        else:
            xyz[:, landmark, 1] = 0.0
    return xyz, confidence


def observation_dispersion(
    clips: Sequence[ClipObservation], xyz: np.ndarray
) -> np.ndarray:
    """Return per-axis median absolute projection residuals."""
    if not clips:
        raise ValueError("At least one clip is required")
    residuals = np.stack(
        [clip.uv - reproject_xyz(xyz, clip.yaw) for clip in clips]
    )
    uv_median = np.median(residuals, axis=0)
    uv_mad = np.median(np.abs(residuals - uv_median), axis=0)
    dispersion = np.zeros((FRAME_COUNT, LANDMARK_COUNT, 3), dtype=np.float64)
    front = [
        index
        for index, clip in enumerate(clips)
        if abs(math.cos(clip.yaw)) >= 0.7
    ]
    side = [
        index
        for index, clip in enumerate(clips)
        if abs(math.sin(clip.yaw)) >= 0.7
    ]
    if front:
        dispersion[:, :, 0] = np.median(
            np.abs(residuals[front, :, :, 0]), axis=0
        )
    if side:
        dispersion[:, :, 2] = np.median(
            np.abs(residuals[side, :, :, 0]), axis=0
        )
    dispersion[:, :, 1] = uv_mad[:, :, 1]
    return dispersion


def major_bone_lengths(
    xyz: np.ndarray, confidence: np.ndarray
) -> dict[str, float]:
    """Estimate one robust canonical length for every major body segment."""
    points = np.asarray(xyz, dtype=np.float64)
    weights = np.asarray(confidence, dtype=np.float64)
    lengths: dict[str, float] = {}
    fallback: dict[tuple[int, int], np.ndarray] = {}
    preferred: dict[tuple[int, int], bool] = {}
    for start, end in MAJOR_BONES:
        values = np.linalg.norm(points[:, start] - points[:, end], axis=1)
        reliable = np.minimum(weights[:, start], weights[:, end]) >= 0.35
        finite = np.isfinite(values) & (values > 1e-6)
        clean = values[reliable & finite]
        fallback[(start, end)] = values[finite]
        preferred[(start, end)] = bool(clean.size)
        if clean.size:
            keep = _mad_mask(clean)
            lengths[f"{start}-{end}"] = float(np.median(clean[keep]))

    for start, end in MAJOR_BONES:
        key = f"{start}-{end}"
        if key in lengths:
            continue
        paired = PAIRED_MAJOR_BONES.get((start, end))
        paired_key = f"{paired[0]}-{paired[1]}" if paired else ""
        if paired and preferred.get(paired) and paired_key in lengths:
            lengths[key] = lengths[paired_key]
            continue
        clean = fallback[(start, end)]
        if not clean.size:
            raise ValueError(f"No finite seed for bone {start}-{end}")
        keep = _mad_mask(clean)
        lengths[key] = float(np.median(clean[keep]))
    return lengths


def clip_observation_coverage(clip: ClipObservation) -> float:
    visibility = np.asarray(clip.visibility, dtype=np.float64)
    return float(np.count_nonzero(visibility >= 0.35) / visibility.size)


def _angle_degrees(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float:
    left = a - vertex
    right = c - vertex
    denominator = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-9)
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _world_angle_targets(
    clips: Sequence[ClipObservation],
) -> tuple[np.ndarray, np.ndarray]:
    stack = np.asarray([clip.world_angles for clip in clips], dtype=np.float64)
    if stack.shape != (len(clips), FRAME_COUNT, len(ANGLE_TRIPLETS)):
        raise ValueError("world_angles must have shape [clips,48,4]")
    targets = np.zeros((FRAME_COUNT, len(ANGLE_TRIPLETS)), dtype=np.float64)
    weights = np.zeros_like(targets)
    for frame in range(FRAME_COUNT):
        for angle in range(len(ANGLE_TRIPLETS)):
            values = stack[:, frame, angle]
            finite = values[np.isfinite(values)]
            if finite.size < 3:
                continue
            keep = _mad_mask(finite)
            targets[frame, angle] = float(np.median(finite[keep]))
            weights[frame, angle] = min(1.0, float(keep.sum()) / 3.0)
    return targets, weights


def _point_columns(frame: int, landmark: int) -> tuple[int, int, int]:
    base = (frame * LANDMARK_COUNT + landmark) * 3
    return base, base + 1, base + 2


def build_jacobian_sparsity(clips: Sequence[ClipObservation]):
    """Describe exact point dependencies for SciPy's finite differences."""
    from scipy.sparse import lil_matrix

    projection_rows = len(clips) * FRAME_COUNT * LANDMARK_COUNT * 2
    bone_rows = len(MAJOR_BONES) * FRAME_COUNT
    temporal_rows = (FRAME_COUNT - 2) * LANDMARK_COUNT * 3
    anchor_rows = 3
    angle_rows = FRAME_COUNT * len(ANGLE_TRIPLETS)
    total_rows = (
        projection_rows
        + bone_rows
        + temporal_rows
        + anchor_rows
        + angle_rows
    )
    total_columns = FRAME_COUNT * LANDMARK_COUNT * 3
    sparse = lil_matrix((total_rows, total_columns), dtype=np.int8)
    row = 0
    for _clip in clips:
        for frame in range(FRAME_COUNT):
            for landmark in range(LANDMARK_COUNT):
                x, y, z = _point_columns(frame, landmark)
                sparse[row, [x, z]] = 1
                row += 1
                sparse[row, y] = 1
                row += 1
    for start, end in MAJOR_BONES:
        for frame in range(FRAME_COUNT):
            sparse[
                row,
                list(_point_columns(frame, start))
                + list(_point_columns(frame, end)),
            ] = 1
            row += 1
    for frame in range(1, FRAME_COUNT - 1):
        for landmark in range(LANDMARK_COUNT):
            for axis in range(3):
                sparse[
                    row,
                    [
                        _point_columns(frame - 1, landmark)[axis],
                        _point_columns(frame, landmark)[axis],
                        _point_columns(frame + 1, landmark)[axis],
                    ],
                ] = 1
                row += 1
    for axis in range(3):
        sparse[
            row,
            [
                _point_columns(0, 23)[axis],
                _point_columns(0, 24)[axis],
            ],
        ] = 1
        row += 1
    for frame in range(FRAME_COUNT):
        for a, vertex, c in ANGLE_TRIPLETS:
            sparse[
                row,
                list(_point_columns(frame, a))
                + list(_point_columns(frame, vertex))
                + list(_point_columns(frame, c)),
            ] = 1
            row += 1
    if row != total_rows:
        raise AssertionError("Jacobian sparsity row count does not match residuals")
    return sparse.tocsr()


def projection_rmse(clips: Sequence[ClipObservation], xyz: np.ndarray) -> float:
    squared_sum = 0.0
    weight_sum = 0.0
    for clip in clips:
        delta = reproject_xyz(xyz, clip.yaw) - clip.uv
        weights = np.clip(clip.visibility, 0.0, 1.0)[..., None]
        squared_sum += float(np.sum(delta * delta * weights))
        weight_sum += float(np.sum(np.broadcast_to(weights, delta.shape)))
    return float(math.sqrt(squared_sum / max(weight_sum, 1e-9)))


def optimize_canonical_motion(
    clips: Sequence[ClipObservation],
    initial: np.ndarray,
    confidence: np.ndarray,
    config: OptimizationConfig = OptimizationConfig(),
) -> tuple[np.ndarray, OptimizationReport]:
    """Jointly fit projections, fixed bones, continuity, and world angles."""
    from scipy.optimize import least_squares

    seed = np.asarray(initial, dtype=np.float64)
    if seed.shape != (FRAME_COUNT, LANDMARK_COUNT, 3):
        raise ValueError("initial must have shape [48,33,3]")
    lengths = major_bone_lengths(seed, confidence)
    catch_pelvis = (seed[0, 23] + seed[0, 24]) * 0.5
    angle_targets, angle_weights = _world_angle_targets(clips)

    def residual(flat: np.ndarray) -> np.ndarray:
        xyz = flat.reshape(FRAME_COUNT, LANDMARK_COUNT, 3)
        values: list[float] = []
        for clip in clips:
            projected = reproject_xyz(xyz, clip.yaw)
            weight = np.sqrt(
                np.clip(clip.visibility, 0.0, 1.0)
                * max(0.05, min(1.0, clip.quality / 100.0))
            )
            values.extend(
                (
                    config.projection_weight
                    * weight[..., None]
                    * (projected - clip.uv)
                ).ravel()
            )
        for start, end in MAJOR_BONES:
            target = lengths[f"{start}-{end}"]
            current = np.linalg.norm(xyz[:, start] - xyz[:, end], axis=1)
            values.extend(config.bone_weight * (current - target))
        acceleration = xyz[2:] - 2.0 * xyz[1:-1] + xyz[:-2]
        values.extend((config.temporal_weight * acceleration).ravel())
        pelvis = (xyz[0, 23] + xyz[0, 24]) * 0.5
        values.extend(config.anchor_weight * (pelvis - catch_pelvis))
        for frame in range(FRAME_COUNT):
            for index, (a, vertex, c) in enumerate(ANGLE_TRIPLETS):
                current = _angle_degrees(
                    xyz[frame, a], xyz[frame, vertex], xyz[frame, c]
                )
                normalized = (current - angle_targets[frame, index]) / 90.0
                values.append(
                    config.world_angle_weight
                    * angle_weights[frame, index]
                    * normalized
                )
        return np.asarray(values, dtype=np.float64)

    before = projection_rmse(clips, seed)
    result = least_squares(
        residual,
        seed.ravel(),
        jac_sparsity=build_jacobian_sparsity(clips),
        loss="soft_l1",
        f_scale=config.f_scale,
        max_nfev=config.max_nfev,
        method="trf",
    )
    optimized = result.x.reshape(FRAME_COUNT, LANDMARK_COUNT, 3)
    return optimized, OptimizationReport(
        success=bool(result.success),
        message=str(result.message),
        cost=float(result.cost),
        nfev=int(result.nfev),
        projection_rmse_before=before,
        projection_rmse_after=projection_rmse(clips, optimized),
    )


def projection_rmse_by_view(
    clips: Sequence[ClipObservation], xyz: np.ndarray
) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for view in ("front", "side", "oblique"):
        selected = [clip for clip in clips if clip.view == view]
        if selected:
            grouped[view] = projection_rmse(selected, xyz)
    return grouped


def validate_canonical_motion(
    xyz: np.ndarray,
    clips: Sequence[ClipObservation],
    report: OptimizationReport,
    confidence: np.ndarray,
) -> dict[str, Any]:
    """Apply canonical-motion quality gates before model publication."""
    points = np.asarray(xyz, dtype=np.float64)
    reasons: list[str] = []
    if points.shape != (FRAME_COUNT, LANDMARK_COUNT, 3) or not np.isfinite(
        points
    ).all():
        reasons.append("invalid 48x33x3 shape or non-finite coordinate")
        return {"passed": False, "reasons": reasons}

    view_counts = {
        view: sum(
            clip.view == view
            and clip_observation_coverage(clip) >= MIN_EFFECTIVE_CLIP_COVERAGE
            for clip in clips
        )
        for view in ("front", "side")
    }
    if view_counts["front"] < 3 or view_counts["side"] < 3:
        reasons.append("requires at least 3 front and 3 side clips")

    bone_cv: dict[str, float] = {}
    for start, end in MAJOR_BONES:
        values = np.linalg.norm(points[:, start] - points[:, end], axis=1)
        bone_cv[f"{start}-{end}"] = float(
            np.std(values) / max(float(np.mean(values)), 1e-9)
        )
    if max(bone_cv.values(), default=1.0) > 0.01:
        reasons.append("major bone coefficient of variation exceeds 1%")

    per_view_rmse = projection_rmse_by_view(clips, points)
    if any(
        per_view_rmse.get(view, float("inf")) > 0.08
        for view in ("front", "side")
    ):
        reasons.append("front or side projection RMSE exceeds 0.08")

    speed = np.linalg.norm(np.diff(points, axis=0), axis=2).ravel()
    spike_ratio = float(speed.max() / max(float(np.percentile(speed, 95)), 1e-9))
    if spike_ratio > 3.0:
        reasons.append("temporal velocity spike ratio exceeds 3")

    phases = clips[0].phases
    release_index = next(
        (index for index, phase in enumerate(phases) if phase == "release"), 36
    )
    targets, target_weights = _world_angle_targets(clips)
    angle_errors: dict[str, float] = {}
    angle_names = ("elbow", "shoulder", "hip", "knee")
    for index, (a, vertex, c) in enumerate(ANGLE_TRIPLETS):
        measured = _angle_degrees(
            points[release_index, a],
            points[release_index, vertex],
            points[release_index, c],
        )
        if target_weights[release_index, index] > 0:
            angle_errors[angle_names[index]] = abs(
                measured - float(targets[release_index, index])
            )
    release_angle_mae = (
        float(np.mean(list(angle_errors.values())))
        if len(angle_errors) == 4
        else float("inf")
    )
    if release_angle_mae > 8.0:
        reasons.append("release joint-angle MAE exceeds 8 degrees")

    paired = (
        ((11, 13), (12, 14)),
        ((13, 15), (14, 16)),
        ((23, 25), (24, 26)),
        ((25, 27), (26, 28)),
    )
    symmetry: dict[str, float] = {}
    for left, right in paired:
        left_length = float(
            np.median(
                np.linalg.norm(points[:, left[0]] - points[:, left[1]], axis=1)
            )
        )
        right_length = float(
            np.median(
                np.linalg.norm(points[:, right[0]] - points[:, right[1]], axis=1)
            )
        )
        symmetry[f"{left[0]}-{left[1]}:{right[0]}-{right[1]}"] = abs(
            left_length - right_length
        ) / max((left_length + right_length) * 0.5, 1e-9)
    if max(symmetry.values(), default=0.0) > 0.10:
        reasons.append("left/right paired-bone difference exceeds 10%")

    if not report.success:
        reasons.append("bundle adjustment did not converge")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "view_counts": view_counts,
        "bone_cv": bone_cv,
        "projection_rmse": per_view_rmse,
        "velocity_p95": float(np.percentile(speed, 95)),
        "velocity_max": float(speed.max()),
        "velocity_spike_ratio": spike_ratio,
        "release_angle_errors_deg": angle_errors,
        "release_angle_mae_deg": release_angle_mae,
        "left_right_bone_difference": symmetry,
        "mean_confidence": float(np.mean(confidence)),
        "optimizer_success": report.success,
    }


def _global_floor_shift(
    xyz: np.ndarray, phases: Sequence[str], confidence: np.ndarray
) -> np.ndarray:
    """Set one catch-phase floor for the whole motion, preserving jump height."""
    points = np.asarray(xyz, dtype=np.float64).copy()
    catch_frames = [
        index for index, phase in enumerate(phases) if str(phase) == "catch"
    ]
    if not catch_frames:
        catch_frames = [0]
    foot_landmarks = (27, 28, 29, 30, 31, 32)
    catch_index = np.asarray(catch_frames)
    foot_y = points[catch_index][:, foot_landmarks, 1]
    foot_confidence = np.asarray(confidence, dtype=np.float64)[catch_index][
        :, foot_landmarks
    ]
    reliable = foot_y[foot_confidence >= 0.35]
    floor_y = float(np.median(reliable if reliable.size else foot_y))
    points[:, :, 1] -= floor_y
    return points


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def build_model_entry(
    *,
    player_key: str,
    xyz: np.ndarray,
    dispersion: np.ndarray,
    confidence: np.ndarray,
    phases: Sequence[str],
    clips: Sequence[ClipObservation | Mapping[str, Any]],
    bone_lengths: Mapping[str, float],
    optimizer: OptimizationReport | Mapping[str, Any],
    validation: Mapping[str, Any],
    duration: float = 1.0,
    provenance: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Serialize an optimized motion without per-frame body/floor alignment."""
    points = np.asarray(xyz, dtype=np.float64)
    spread = np.asarray(dispersion, dtype=np.float64)
    certainty = np.asarray(confidence, dtype=np.float64)
    if points.shape != (FRAME_COUNT, LANDMARK_COUNT, 3):
        raise ValueError("xyz must have shape [48,33,3]")
    if spread.shape != points.shape:
        raise ValueError("dispersion must have shape [48,33,3]")
    if certainty.shape != (FRAME_COUNT, LANDMARK_COUNT):
        raise ValueError("confidence must have shape [48,33]")
    if len(phases) != FRAME_COUNT:
        raise ValueError("phases must contain 48 values")
    if not all(
        np.isfinite(values).all() for values in (points, spread, certainty)
    ):
        raise ValueError("model arrays must contain finite values")

    shifted = _global_floor_shift(points, phases, certainty)
    source_clips: list[dict[str, Any]] = []
    for clip in clips:
        if isinstance(clip, ClipObservation):
            source_clips.append(
                {
                    "clip_id": clip.clip_id,
                    "view": clip.view,
                    "yaw": round(float(clip.yaw), 6),
                    "quality": round(float(clip.quality), 4),
                    "source_url": clip.source_url,
                }
            )
        else:
            source_clips.append(_jsonable(clip))

    return {
        "player_key": str(player_key),
        "model": CANONICAL_MODEL_NAME,
        "quality_mode": "multi_view_3d",
        "space": "canonical_3d",
        "source_space": "multi_view_2d_bundle_adjustment",
        "units": "body_scale",
        "duration": float(duration),
        "progress": np.round(np.linspace(0.0, 1.0, FRAME_COUNT), 6).tolist(),
        "phases": [str(phase) for phase in phases],
        "xyz": np.round(shifted, 6).tolist(),
        "dispersion_mad": np.round(spread, 6).tolist(),
        "confidence": np.round(certainty, 6).tolist(),
        "bone_lengths": {
            str(key): round(float(value), 6)
            for key, value in bone_lengths.items()
        },
        "source_clips": source_clips,
        "source_provenance": _jsonable(provenance),
        "observation_count": int(sum(np.count_nonzero(clip.visibility) for clip in clips)),
        "optimizer": _jsonable(optimizer),
        "validation": _jsonable(validation),
    }


def write_validated_model(path: Path | str, entry: Mapping[str, Any]) -> None:
    """Atomically publish one validated player while preserving other entries."""
    target = Path(path)
    if entry.get("validation", {}).get("passed") is not True:
        raise ValueError("Canonical model failed validation and cannot be published")
    player_key = str(entry.get("player_key") or "")
    if not player_key:
        raise ValueError("Canonical model requires player_key")

    existing: dict[str, Any] = {}
    if target.exists():
        loaded = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("Canonical model store must be a JSON object")
        existing = loaded
    existing[player_key] = dict(entry)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def _fit_renderer_coordinates(
    xyz: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one full-timeline display transform for the legacy canvas camera."""
    points = np.asarray(xyz, dtype=np.float64)
    minimum = np.min(points, axis=(0, 1))
    maximum = np.max(points, axis=(0, 1))
    span = maximum - minimum
    limits = (1.15, 1.65, 1.15)
    scale_candidates = [
        limit / float(axis_span)
        for limit, axis_span in zip(limits, span)
        if axis_span > 1e-9
    ]
    scale = min(scale_candidates, default=1.0)
    scaled = points * scale
    scaled_minimum = np.min(scaled, axis=(0, 1))
    scaled_maximum = np.max(scaled, axis=(0, 1))
    translation = np.asarray(
        [
            -(scaled_minimum[0] + scaled_maximum[0]) * 0.5,
            -scaled_minimum[1],
            -(scaled_minimum[2] + scaled_maximum[2]) * 0.5,
        ],
        dtype=np.float64,
    )
    transformed = scaled + translation
    total_scale = scale
    for _iteration in range(20):
        if _renderer_bounds_fit(transformed):
            break
        transformed *= 0.9
        translation *= 0.9
        total_scale *= 0.9
    return transformed, {
        "type": "global_timeline_fit_v1",
        "scale": round(float(total_scale), 8),
        "translation": np.round(translation, 8).tolist(),
    }


def _renderer_bounds_fit(points: np.ndarray, margin: float = 8.0) -> bool:
    for width, height in ((640.0, 430.0), (320.0, 320.0)):
        yaw = -0.55
        pitch = -0.12
        x = points[:, :, 0]
        y = points[:, :, 1] - 0.88
        z = points[:, :, 2]
        rotated_x = x * math.cos(yaw) + z * math.sin(yaw)
        rotated_z = -x * math.sin(yaw) + z * math.cos(yaw)
        rotated_y = y * math.cos(pitch) - rotated_z * math.sin(pitch)
        depth = y * math.sin(pitch) + rotated_z * math.cos(pitch)
        perspective = 3.3 / np.maximum(1.8, 3.3 + depth)
        canvas_scale = min(width, height) * 0.46
        screen_x = width * 0.5 + rotated_x * canvas_scale * perspective
        screen_y = height * 0.54 - rotated_y * canvas_scale * perspective
        if (
            float(np.min(screen_x)) < margin
            or float(np.max(screen_x)) > width - margin
            or float(np.min(screen_y)) < margin
            or float(np.max(screen_y)) > height - margin
        ):
            return False
    return True


def load_canonical_player_skeleton(
    player_key: str, *, path: Path | str = CANONICAL_MODEL_PATH
) -> dict[str, Any] | None:
    """Load a validated 33-point profile, otherwise signal legacy fallback."""
    source = Path(path)
    try:
        store = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(store, Mapping):
        return None
    entry = store.get(str(player_key))
    if not isinstance(entry, Mapping):
        return None
    validation = entry.get("validation")
    if not isinstance(validation, Mapping) or validation.get("passed") is not True:
        return None
    if entry.get("model") not in {
        CANONICAL_MODEL_NAME,
        *LEGACY_CANONICAL_MODEL_NAMES,
    }:
        return None
    if entry.get("player_key") != str(player_key):
        return None
    phases = entry.get("phases")
    if not isinstance(phases, list) or len(phases) != FRAME_COUNT:
        return None
    try:
        coordinates = np.asarray(entry.get("xyz"), dtype=np.float64)
        duration = float(entry.get("duration", 1.0))
    except (TypeError, ValueError, OverflowError):
        return None
    if coordinates.shape != (FRAME_COUNT, LANDMARK_COUNT, 3):
        return None
    if not np.isfinite(coordinates).all():
        return None
    if not math.isfinite(duration) or duration < 0.0:
        return None

    display_coordinates, display_transform = _fit_renderer_coordinates(coordinates)
    frames = []
    for index, frame_coordinates in enumerate(display_coordinates):
        frames.append(
            {
                "t": round(duration * index / (FRAME_COUNT - 1), 4),
                "phase": str(phases[index]),
                "landmarks": {
                    name: [float(value) for value in frame_coordinates[landmark]]
                    for landmark, name in enumerate(MP_LANDMARK_NAMES)
                },
            }
        )
    return {
        "model": CANONICAL_MODEL_NAME,
        "space": "canonical_3d",
        "source_space": "multi_view_2d_bundle_adjustment",
        "quality_mode": "multi_view_3d",
        "units": "display_units",
        "canonical_units": "body_scale",
        "display_transform": display_transform,
        "hand": "right",
        "view": "merged",
        "duration": duration,
        "landmark_order": list(MP_LANDMARK_NAMES),
        "bones": [list(bone) for bone in MP_BONES],
        "frames": frames,
        "validation": dict(validation),
    }
