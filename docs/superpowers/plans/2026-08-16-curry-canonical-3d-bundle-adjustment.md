# Curry Canonical 3D Bundle Adjustment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, and safely serve a Stephen Curry-only 48-frame x 33-landmark canonical XYZ shooting motion reconstructed from at least three front and three side clips.

**Architecture:** Add an isolated `app.reconstruction3d` module that performs fixed-scale observation normalization, robust multi-view initialization, and SciPy sparse bundle adjustment. Generate a versioned Curry sidecar model offline, then make the existing skeleton endpoint prefer it only when strict validation passes; every current DB, matching, user-analysis, and fallback path remains intact.

**Tech Stack:** Python 3.12, NumPy, SciPy `least_squares`, MediaPipe, OpenCV, Flask, unittest, yt-dlp, JSON.

## Global Constraints

- Scope is Stephen Curry only; do not rebuild or change other player models.
- Output shape is exactly 48 normalized frames x 33 MediaPipe landmarks x XYZ.
- Require at least three accepted front clips and three accepted side clips before activation.
- Use one fixed scale per clip; never use a per-frame body-box or torso scale.
- Use the catch pelvis as the fixed clip origin and apply at most one global floor offset to the final timeline.
- Treat `world_landmarks` as an angle/confidence consistency signal, never as multi-view depth ground truth.
- Keep the existing player matcher, user analysis, SQLite schema, renderer, and Render runtime requirements unchanged unless a failing regression proves a minimal compatibility edit is necessary.
- Keep `requirements.txt` unchanged; pin SciPy in `requirements-data.txt` and install it only in data/CI workflows.
- Do not activate a generated model unless every validation gate passes.
- Existing skeleton generation is the mandatory fallback for Curry and the unchanged path for all other players.

---

## File map

- `app/reconstruction3d.py`: all reconstruction math, robust filtering, sparse optimization, validation, model loading, and skeleton-schema adaptation.
- `tests/test_reconstruction3d.py`: synthetic reconstruction, projection, scale, outlier, kinematic, temporal, serialization, and fallback tests.
- `app/analyze.py`: optional raw image/world observation retention for the offline builder; default API behavior unchanged.
- `scripts/youtube_profile.py`: optional certificate flag for reproducible source retrieval; default remains certificate verification enabled.
- `scripts/build_curry_canonical3d.py`: Curry-only source selection, download, observation extraction, optimization, validation, metrics, and atomic sidecar generation.
- `models/canonical_3d_models.json`: generated and validation-passed Curry model only.
- `app/server.py`: prefer a valid Curry sidecar skeleton, then execute the existing DB fallback unchanged.
- `tests/test_app_smoke.py`: preference and fallback regression contract.
- `requirements-data.txt`: SciPy offline/CI dependency.
- `.github/workflows/ci.yml`, `.github/workflows/quality.yml`: install offline test requirements without changing Render.
- `docs/superpowers/specs/2026-08-16-curry-canonical-3d-bundle-adjustment-design.md`: keep dependency placement consistent with implementation.

---

### Task 1: Fixed-scale clip observations and phase resampling

**Files:**
- Create: `app/reconstruction3d.py`
- Create: `tests/test_reconstruction3d.py`
- Modify: `requirements-data.txt`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/quality.yml`

**Interfaces:**
- Consumes: raw shot samples shaped as dictionaries with `phase`, `t`, `frame`, `image_landmarks[33]`, optional `world_landmarks[33]`, and four joint-angle keys.
- Produces: `ClipObservation`, `compute_fixed_scale()`, and `normalize_clip_observation()` for later reconstruction tasks.

- [ ] **Step 1: Write failing tests for fixed scale and preserved vertical motion**

Add these imports and fixtures to `tests/test_reconstruction3d.py`:

```python
import unittest
import numpy as np

from app.reconstruction3d import (
    FRAME_COUNT,
    ClipObservation,
    compute_fixed_scale,
    normalize_clip_observation,
)


def synthetic_image_timeline(frame_count=21, jump=0.18):
    samples = []
    for frame in range(frame_count):
        progress = frame / (frame_count - 1)
        phase = (
            "catch" if progress < 0.22 else
            "dip" if progress < 0.30 else
            "rise" if progress < 0.74 else
            "release" if progress < 0.78 else
            "follow_through"
        )
        vertical = jump * max(0.0, 1.0 - abs(progress - 0.78) / 0.35)
        points = np.zeros((33, 4), dtype=float)
        points[:, 3] = 0.95
        points[11, :2] = (0.42, 0.30 - vertical)
        points[12, :2] = (0.58, 0.30 - vertical)
        points[23, :2] = (0.46, 0.55 - vertical)
        points[24, :2] = (0.54, 0.55 - vertical)
        points[25, :2] = (0.46, 0.72 - vertical)
        points[26, :2] = (0.54, 0.72 - vertical)
        points[27, :2] = (0.46, 0.90 - vertical)
        points[28, :2] = (0.54, 0.90 - vertical)
        for index in range(33):
            if points[index, 3] == 0.95 and not points[index, :2].any():
                points[index, :2] = (0.50, 0.45 - vertical)
        samples.append({
            "t": frame / 30.0,
            "frame": frame,
            "phase": phase,
            "image_landmarks": points.tolist(),
            "world_landmarks": [],
            "elbow": 110 + 50 * progress,
            "shoulder": 90 + 55 * progress,
            "hip": 140 + 20 * progress,
            "knee": 125 + 35 * progress,
        })
    return samples


class Reconstruction3DTests(unittest.TestCase):
    def test_one_fixed_scale_preserves_pelvis_vertical_motion(self):
        samples = synthetic_image_timeline()
        image = np.asarray([s["image_landmarks"] for s in samples], dtype=float)
        scale = compute_fixed_scale(image[:, :, :2], image[:, :, 3], view="front")
        clip = normalize_clip_observation(
            "front-a", samples, view="front", yaw=0.0, quality=90.0
        )
        self.assertGreater(scale, 0.1)
        self.assertEqual(clip.uv.shape, (FRAME_COUNT, 33, 2))
        self.assertEqual(len(set(np.round(clip.frame_scale, 8))), 1)
        pelvis_y = clip.uv[:, [23, 24], 1].mean(axis=1)
        self.assertGreater(float(np.ptp(pelvis_y)), 0.10)
```

- [ ] **Step 2: Run the focused test and verify the missing module failure**

Run:

```bash
python -m unittest tests.test_reconstruction3d.Reconstruction3DTests.test_one_fixed_scale_preserves_pelvis_vertical_motion -v
```

Expected: `ModuleNotFoundError: No module named 'app.reconstruction3d'`.

- [ ] **Step 3: Add SciPy only to offline/CI dependencies**

Append this exact line to `requirements-data.txt`:

```text
scipy>=1.16.0,<2
```

Change both workflow install steps from `python -m pip install -r requirements.txt` or `pip install -r requirements.txt` to:

```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    python -m pip install -r requirements-data.txt
```

Do not modify `requirements.txt` or `Dockerfile`.

- [ ] **Step 4: Implement fixed-scale observation normalization**

Create `app/reconstruction3d.py` with these public types and functions:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math
import statistics
import numpy as np

from .motion import LANDMARK_COUNT, resample_timeline

FRAME_COUNT = 48
VIEW_YAW = {"front": 0.0, "side": math.pi / 2.0, "oblique": math.pi / 4.0}
CANONICAL_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "canonical_3d_models.json"


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


def _segment_lengths(image_xy: np.ndarray) -> dict[str, np.ndarray]:
    shoulder = (image_xy[:, 11] + image_xy[:, 12]) * 0.5
    pelvis = (image_xy[:, 23] + image_xy[:, 24]) * 0.5
    return {
        "torso": np.linalg.norm(shoulder - pelvis, axis=1),
        "shoulder_width": np.linalg.norm(image_xy[:, 11] - image_xy[:, 12], axis=1),
        "hip_width": np.linalg.norm(image_xy[:, 23] - image_xy[:, 24], axis=1),
        "thigh_l": np.linalg.norm(image_xy[:, 23] - image_xy[:, 25], axis=1),
        "thigh_r": np.linalg.norm(image_xy[:, 24] - image_xy[:, 26], axis=1),
        "shin_l": np.linalg.norm(image_xy[:, 25] - image_xy[:, 27], axis=1),
        "shin_r": np.linalg.norm(image_xy[:, 26] - image_xy[:, 28], axis=1),
    }


def compute_fixed_scale(image_xy, visibility, *, view, min_visibility=0.35):
    xy = np.asarray(image_xy, dtype=float)
    vis = np.asarray(visibility, dtype=float)
    if xy.ndim != 3 or xy.shape[1:] != (LANDMARK_COUNT, 2):
        raise ValueError("image_xy must have shape [frames,33,2]")
    measures = _segment_lengths(xy)
    index_groups = {
        "torso": (11, 12, 23, 24), "shoulder_width": (11, 12),
        "hip_width": (23, 24), "thigh_l": (23, 25),
        "thigh_r": (24, 26), "shin_l": (25, 27), "shin_r": (26, 28),
    }
    view_weight = 0.20 if view == "side" else 1.0
    weighted = []
    for name, values in measures.items():
        valid = np.all(vis[:, index_groups[name]] >= min_visibility, axis=1)
        clean = values[valid & np.isfinite(values) & (values > 1e-5)]
        if not clean.size:
            continue
        center = float(np.median(clean))
        mad = float(np.median(np.abs(clean - center)))
        if mad > 1e-9:
            clean = clean[np.abs(clean - center) <= 3.5 * 1.4826 * mad]
        weight = view_weight if "width" in name else 1.0
        weighted.extend([float(np.median(clean))] * max(1, round(weight * 5)))
    if not weighted:
        raise ValueError("No reliable body segments for clip scale")
    return float(np.median(weighted))


def normalize_clip_observation(clip_id, samples, *, view, yaw=None, quality=100.0, source_url=""):
    if len(samples) < 2:
        raise ValueError("At least two shot samples are required")
    image = np.asarray([sample["image_landmarks"] for sample in samples], dtype=float)
    xy, visibility = image[:, :, :2], image[:, :, 3]
    scale = compute_fixed_scale(xy, visibility, view=view)
    catch_pelvis = ((xy[0, 23] + xy[0, 24]) * 0.5).copy()
    pose_samples = []
    for sample, points, vis in zip(samples, xy, visibility):
        normalized = np.zeros((LANDMARK_COUNT, 4), dtype=float)
        normalized[:, 0] = (points[:, 0] - catch_pelvis[0]) / scale
        normalized[:, 1] = (catch_pelvis[1] - points[:, 1]) / scale
        normalized[:, 3] = np.clip(vis, 0.0, 1.0)
        pose_samples.append({**sample, "pose": normalized.tolist()})
    resampled = resample_timeline(pose_samples, frame_count=FRAME_COUNT)
    pose = np.asarray([frame["pose"] for frame in resampled], dtype=float)
    angles = np.asarray([[frame[k] for k in ("elbow", "shoulder", "hip", "knee")] for frame in resampled])
    return ClipObservation(
        clip_id=clip_id, view=view, yaw=float(VIEW_YAW[view] if yaw is None else yaw),
        quality=float(quality), uv=pose[:, :, :2], visibility=pose[:, :, 3],
        world_angles=angles, frame_scale=np.full(FRAME_COUNT, scale),
        phases=tuple(str(frame["phase"]) for frame in resampled), source_url=source_url,
    )
```

- [ ] **Step 5: Run the focused test and the existing motion tests**

Run:

```bash
python -m unittest tests.test_reconstruction3d tests.test_motion -v
```

Expected: fixed-scale test passes and all existing motion tests remain green.

- [ ] **Step 6: Commit Task 1**

```bash
git add app/reconstruction3d.py tests/test_reconstruction3d.py requirements-data.txt .github/workflows/ci.yml .github/workflows/quality.yml
git commit -m "Add fixed-scale Curry observations"
```

---

### Task 2: Robust yaw-aware multi-view initialization

**Files:**
- Modify: `app/reconstruction3d.py`
- Modify: `tests/test_reconstruction3d.py`

**Interfaces:**
- Consumes: `Sequence[ClipObservation]` from Task 1.
- Produces: `reproject_xyz()`, `filter_observations()`, `initialize_xyz()`, and `observation_dispersion()`.

- [ ] **Step 1: Add failing synthetic XYZ, projection, oblique, and outlier tests**

Add helpers that project a known `[48,33,3]` motion into `ClipObservation` objects, then add:

```python
def make_xyz():
    xyz = np.zeros((FRAME_COUNT, 33, 3), dtype=float)
    for frame in range(FRAME_COUNT):
        p = frame / (FRAME_COUNT - 1)
        xyz[frame, :, 0] = np.linspace(-0.35, 0.35, 33) + 0.02 * np.sin(np.pi * p)
        xyz[frame, :, 1] = np.linspace(1.65, 0.05, 33) + 0.14 * np.sin(np.pi * p)
        xyz[frame, :, 2] = np.linspace(-0.20, 0.25, 33) + 0.05 * p
    return xyz


def observation_from_xyz(xyz, clip_id, view, yaw, noise=0.0):
    uv = reproject_xyz(xyz, yaw)
    if noise:
        rng = np.random.default_rng(abs(hash(clip_id)) % (2**32))
        uv = uv + rng.normal(0.0, noise, uv.shape)
    return ClipObservation(
        clip_id, view, yaw, 95.0, uv, np.full((FRAME_COUNT, 33), 0.95),
        np.zeros((FRAME_COUNT, 4)), np.ones(FRAME_COUNT), tuple(["rise"] * FRAME_COUNT)
    )


class Reconstruction3DTests(unittest.TestCase):
    # keep the Task 1 test
    def test_front_side_reconstruct_known_xyz(self):
        expected = make_xyz()
        clips = [
            observation_from_xyz(expected, "f1", "front", 0.0),
            observation_from_xyz(expected, "s1", "side", np.pi / 2),
        ]
        actual, confidence = initialize_xyz(clips)
        np.testing.assert_allclose(actual, expected, atol=1e-8)
        self.assertGreater(float(confidence.mean()), 0.9)

    def test_reprojection_matches_each_camera(self):
        expected = make_xyz()
        for yaw in (0.0, np.pi / 4, np.pi / 2):
            uv = reproject_xyz(expected, yaw)
            self.assertEqual(uv.shape, (FRAME_COUNT, 33, 2))
            np.testing.assert_allclose(uv[:, :, 1], expected[:, :, 1], atol=1e-12)

    def test_one_severe_outlier_clip_does_not_move_median_motion(self):
        expected = make_xyz()
        clips = [
            observation_from_xyz(expected, "f1", "front", 0.0, 0.002),
            observation_from_xyz(expected, "f2", "front", 0.0, 0.002),
            observation_from_xyz(expected, "f3", "front", 0.0, 0.002),
            observation_from_xyz(expected, "s1", "side", np.pi / 2, 0.002),
            observation_from_xyz(expected, "s2", "side", np.pi / 2, 0.002),
            observation_from_xyz(expected, "s3", "side", np.pi / 2, 0.002),
        ]
        outlier = observation_from_xyz(expected, "bad", "front", 0.0)
        outlier = dataclasses.replace(outlier, uv=outlier.uv + 5.0)
        actual, _ = initialize_xyz(clips + [outlier])
        self.assertLess(float(np.sqrt(np.mean((actual - expected) ** 2))), 0.01)

    def test_oblique_observation_participates_in_least_squares(self):
        expected = make_xyz()
        clips = [
            observation_from_xyz(expected, "f", "front", 0.0, 0.003),
            observation_from_xyz(expected, "o", "oblique", np.pi / 4, 0.003),
            observation_from_xyz(expected, "s", "side", np.pi / 2, 0.003),
        ]
        actual, _ = initialize_xyz(clips)
        self.assertLess(float(np.sqrt(np.mean((actual - expected) ** 2))), 0.01)
```

- [ ] **Step 2: Run the new tests and verify missing-symbol failures**

Run:

```bash
python -m unittest tests.test_reconstruction3d -v
```

Expected: imports fail for `reproject_xyz` and `initialize_xyz`.

- [ ] **Step 3: Implement robust initialization**

Add to `app/reconstruction3d.py`:

```python
def reproject_xyz(xyz, yaw):
    points = np.asarray(xyz, dtype=float)
    u = points[:, :, 0] * math.cos(yaw) + points[:, :, 2] * math.sin(yaw)
    return np.stack((u, points[:, :, 1]), axis=-1)


def _weighted_median(values, weights):
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cutoff = 0.5 * float(weights.sum())
    return float(values[min(int(np.searchsorted(np.cumsum(weights), cutoff)), len(values) - 1)])


def _mad_mask(values, threshold=3.5):
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    if mad <= 1e-9:
        return np.abs(values - center) <= 1e-6
    return np.abs(values - center) <= threshold * 1.4826 * mad


def initialize_xyz(clips, *, min_visibility=0.35):
    if len({round(c.yaw, 3) for c in clips}) < 2:
        raise ValueError("At least two distinct camera yaw groups are required")
    xyz = np.zeros((FRAME_COUNT, LANDMARK_COUNT, 3), dtype=float)
    confidence = np.zeros((FRAME_COUNT, LANDMARK_COUNT), dtype=float)
    for frame in range(FRAME_COUNT):
        for landmark in range(LANDMARK_COUNT):
            grouped, vertical, vertical_weights = {}, [], []
            for clip in clips:
                visibility = float(clip.visibility[frame, landmark])
                if visibility < min_visibility:
                    continue
                weight = visibility * max(0.05, min(1.0, clip.quality / 100.0))
                key = round(float(clip.yaw), 3)
                grouped.setdefault(key, []).append((float(clip.uv[frame, landmark, 0]), weight))
                vertical.append(float(clip.uv[frame, landmark, 1]))
                vertical_weights.append(weight)
            if len(grouped) < 2:
                raise ValueError(f"Insufficient observations at frame={frame} landmark={landmark}")
            rows, horizontal, weights = [], [], []
            for group_yaw, observations in grouped.items():
                values = np.asarray([item[0] for item in observations], dtype=float)
                group_weights = np.asarray([item[1] for item in observations], dtype=float)
                keep = _mad_mask(values)
                rows.append((math.cos(group_yaw), math.sin(group_yaw)))
                horizontal.append(_weighted_median(values[keep], group_weights[keep]))
                weights.append(float(group_weights[keep].sum()))
            A = np.asarray(rows, dtype=float)
            u = np.asarray(horizontal, dtype=float)
            w = np.asarray(weights, dtype=float)
            if np.linalg.matrix_rank(A) < 2:
                raise ValueError("Camera yaw groups are not sufficiently separated")
            root_w = np.sqrt(w)[:, None]
            xz, *_ = np.linalg.lstsq(A * root_w, u * root_w[:, 0], rcond=None)
            vertical = np.asarray(vertical, dtype=float)
            vertical_weights = np.asarray(vertical_weights, dtype=float)
            keep_y = _mad_mask(vertical)
            y = _weighted_median(vertical[keep_y], vertical_weights[keep_y])
            xyz[frame, landmark] = (xz[0], y, xz[1])
            confidence[frame, landmark] = min(1.0, float(vertical_weights[keep_y].sum()) / 3.0)
    return xyz, confidence


def observation_dispersion(clips, xyz):
    residuals = np.stack([clip.uv - reproject_xyz(xyz, clip.yaw) for clip in clips])
    uv_mad = np.median(np.abs(residuals - np.median(residuals, axis=0)), axis=0)
    axis = np.zeros((FRAME_COUNT, LANDMARK_COUNT, 3), dtype=float)
    front = [i for i, clip in enumerate(clips) if abs(math.cos(clip.yaw)) >= 0.7]
    side = [i for i, clip in enumerate(clips) if abs(math.sin(clip.yaw)) >= 0.7]
    axis[:, :, 0] = np.median(np.abs(residuals[front, :, :, 0]), axis=0)
    axis[:, :, 2] = np.median(np.abs(residuals[side, :, :, 0]), axis=0)
    axis[:, :, 1] = uv_mad[:, :, 1]
    return axis
```

Import `dataclasses` in the test module because `ClipObservation` is frozen and outlier fixtures are created with `dataclasses.replace()`.

- [ ] **Step 4: Run reconstruction tests**

```bash
python -m unittest tests.test_reconstruction3d -v
```

Expected: all Task 1 and Task 2 tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add app/reconstruction3d.py tests/test_reconstruction3d.py
git commit -m "Add robust multi-view initialization"
```

---

### Task 3: SciPy sparse bundle adjustment and validation metrics

**Files:**
- Modify: `app/reconstruction3d.py`
- Modify: `tests/test_reconstruction3d.py`

**Interfaces:**
- Consumes: `ClipObservation[]`, initial XYZ, and confidence from Task 2.
- Produces: `OptimizationConfig`, `optimize_canonical_motion()`, `major_bone_lengths()`, and `validate_canonical_motion()`.

- [ ] **Step 1: Add failing kinematic and temporal tests**

Add tests using a kinematically valid synthetic pose whose elbows, wrists, hips, knees, and ankles move over 48 frames:

```python
from app.reconstruction3d import (
    OptimizationConfig,
    major_bone_lengths,
    optimize_canonical_motion,
    validate_canonical_motion,
)


def make_kinematic_xyz():
    xyz = np.zeros((FRAME_COUNT, 33, 3), dtype=float)
    for frame in range(FRAME_COUNT):
        p = frame / (FRAME_COUNT - 1)
        pelvis_y = 1.0 + 0.14 * np.sin(np.pi * p)
        for index in range(33):
            xyz[frame, index] = (0.0, pelvis_y + 0.25, 0.0)
        xyz[frame, 23] = (-0.14, pelvis_y, 0.0)
        xyz[frame, 24] = (0.14, pelvis_y, 0.0)
        xyz[frame, 11] = (-0.18, pelvis_y + 0.50, 0.0)
        xyz[frame, 12] = (0.18, pelvis_y + 0.50, 0.0)
        arm_angle = -1.15 + 1.0 * p
        for shoulder, elbow, wrist, sign in ((11, 13, 15, -1), (12, 14, 16, 1)):
            upper = np.array([sign * 0.10 * np.cos(arm_angle), 0.30 * np.sin(-arm_angle), 0.24 * p])
            upper *= 0.31 / np.linalg.norm(upper)
            xyz[frame, elbow] = xyz[frame, shoulder] + upper
            fore = np.array([sign * 0.06, 0.25, 0.10 + 0.06 * p])
            fore *= 0.27 / np.linalg.norm(fore)
            xyz[frame, wrist] = xyz[frame, elbow] + fore
        for hip, knee, ankle, sign in ((23, 25, 27, -1), (24, 26, 28, 1)):
            thigh = np.array([sign * 0.03, -0.44, 0.08 * (1.0 - p)])
            thigh *= 0.45 / np.linalg.norm(thigh)
            xyz[frame, knee] = xyz[frame, hip] + thigh
            shin = np.array([0.0, -0.43, -0.05 * (1.0 - p)])
            shin *= 0.44 / np.linalg.norm(shin)
            xyz[frame, ankle] = xyz[frame, knee] + shin
        xyz[frame, 29] = xyz[frame, 27] + (0.0, -0.03, -0.03)
        xyz[frame, 30] = xyz[frame, 28] + (0.0, -0.03, -0.03)
        xyz[frame, 31] = xyz[frame, 27] + (-0.02, -0.02, 0.18)
        xyz[frame, 32] = xyz[frame, 28] + (0.02, -0.02, 0.18)
    return xyz


def make_six_noisy_clips(xyz, noise):
    return [
        observation_from_xyz(xyz, f"front-{index}", "front", 0.0, noise)
        for index in range(3)
    ] + [
        observation_from_xyz(xyz, f"side-{index}", "side", np.pi / 2, noise)
        for index in range(3)
    ]


def bone_length_series(xyz, start, end):
    return np.linalg.norm(xyz[:, start] - xyz[:, end], axis=1)


def test_bundle_adjustment_reduces_bone_and_projection_error(self):
    expected = make_kinematic_xyz()
    clips = make_six_noisy_clips(expected, noise=0.012)
    initial, confidence = initialize_xyz(clips)
    optimized, report = optimize_canonical_motion(
        clips, initial, confidence, OptimizationConfig(max_nfev=80)
    )
    self.assertTrue(report.success, report.message)
    for start, end in ((11, 13), (13, 15), (12, 14), (14, 16),
                       (23, 25), (25, 27), (24, 26), (26, 28)):
        values = bone_length_series(optimized, start, end)
        self.assertLess(float(np.std(values) / np.mean(values)), 0.01)
    self.assertLess(report.projection_rmse_after, report.projection_rmse_before)


def test_bundle_adjustment_suppresses_single_frame_velocity_spike(self):
    expected = make_kinematic_xyz()
    clips = make_six_noisy_clips(expected, noise=0.004)
    damaged_uv = clips[0].uv.copy()
    damaged_uv[20, 16] += (0.8, 0.8)
    damaged = dataclasses.replace(clips[0], uv=damaged_uv)
    initial, confidence = initialize_xyz([damaged, *clips[1:]])
    optimized, _ = optimize_canonical_motion(
        [damaged, *clips[1:]], initial, confidence, OptimizationConfig(max_nfev=80)
    )
    speed = np.linalg.norm(np.diff(optimized[:, 16], axis=0), axis=1)
    self.assertLess(float(speed.max() / np.percentile(speed, 95)), 3.0)
```

- [ ] **Step 2: Run tests and verify missing optimizer failures**

```bash
python -m unittest tests.test_reconstruction3d -v
```

Expected: missing `OptimizationConfig` or `optimize_canonical_motion`.

- [ ] **Step 3: Implement player-specific bone estimation and sparse residuals**

Add constants and dataclasses:

```python
from dataclasses import dataclass

MAJOR_BONES = (
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (11, 23), (12, 24), (11, 12), (23, 24),
)


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


def major_bone_lengths(xyz, confidence):
    lengths = {}
    for start, end in MAJOR_BONES:
        values = np.linalg.norm(xyz[:, start] - xyz[:, end], axis=1)
        weight = np.minimum(confidence[:, start], confidence[:, end])
        values = values[weight >= 0.35]
        if not values.size:
            raise ValueError(f"No reliable frames for bone {start}-{end}")
        lengths[f"{start}-{end}"] = float(np.median(values))
    return lengths
```

Implement angle targets, residuals, and the exact sparse dependency pattern. The right-side triplets match Curry's shooting hand:

```python
ANGLE_TRIPLETS = (
    (12, 14, 16),  # elbow
    (14, 12, 24),  # shoulder
    (12, 24, 26),  # hip
    (24, 26, 28),  # knee
)


def _angle_degrees(a, vertex, c):
    left, right = a - vertex, c - vertex
    denom = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-9)
    cosine = float(np.clip(np.dot(left, right) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _world_angle_targets(clips):
    stack = np.asarray([clip.world_angles for clip in clips], dtype=float)
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    targets = np.nanmedian(np.where(finite, stack, np.nan), axis=0)
    weights = np.clip(counts / 3.0, 0.0, 1.0)
    targets[counts < 3] = 0.0
    weights[counts < 3] = 0.0
    return targets, weights


def _point_columns(frame, landmark):
    base = (frame * LANDMARK_COUNT + landmark) * 3
    return (base, base + 1, base + 2)


def build_jacobian_sparsity(clips):
    from scipy.sparse import lil_matrix
    projection_rows = len(clips) * FRAME_COUNT * LANDMARK_COUNT * 2
    bone_rows = len(MAJOR_BONES) * FRAME_COUNT
    temporal_rows = (FRAME_COUNT - 2) * LANDMARK_COUNT * 3
    anchor_rows = 3
    angle_rows = FRAME_COUNT * len(ANGLE_TRIPLETS)
    total_rows = projection_rows + bone_rows + temporal_rows + anchor_rows + angle_rows
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
            sparse[row, list(_point_columns(frame, start)) + list(_point_columns(frame, end))] = 1
            row += 1
    for frame in range(1, FRAME_COUNT - 1):
        for landmark in range(LANDMARK_COUNT):
            for axis in range(3):
                sparse[row, [
                    _point_columns(frame - 1, landmark)[axis],
                    _point_columns(frame, landmark)[axis],
                    _point_columns(frame + 1, landmark)[axis],
                ]] = 1
                row += 1
    for axis in range(3):
        sparse[row, [_point_columns(0, 23)[axis], _point_columns(0, 24)[axis]]] = 1
        row += 1
    for frame in range(FRAME_COUNT):
        for a, vertex, c in ANGLE_TRIPLETS:
            columns = list(_point_columns(frame, a)) + list(_point_columns(frame, vertex)) + list(_point_columns(frame, c))
            sparse[row, columns] = 1
            row += 1
    assert row == total_rows
    return sparse.tocsr()


def projection_rmse(clips, xyz):
    squared, weights = [], []
    for clip in clips:
        delta = reproject_xyz(xyz, clip.yaw) - clip.uv
        weight = np.clip(clip.visibility, 0.0, 1.0)[..., None]
        squared.append((delta * delta * weight).ravel())
        weights.append(np.broadcast_to(weight, delta.shape).ravel())
    return float(np.sqrt(np.sum(np.concatenate(squared)) / max(np.sum(np.concatenate(weights)), 1e-9)))


def optimize_canonical_motion(clips, initial, confidence, config=OptimizationConfig()):
    from scipy.optimize import least_squares

    initial = np.asarray(initial, dtype=float)
    lengths = major_bone_lengths(initial, confidence)
    catch_pelvis = (initial[0, 23] + initial[0, 24]) * 0.5
    angle_targets, angle_weights = _world_angle_targets(clips)

    def residual(flat):
        xyz = flat.reshape(FRAME_COUNT, LANDMARK_COUNT, 3)
        values = []
        for clip in clips:
            projected = reproject_xyz(xyz, clip.yaw)
            weight = np.sqrt(
                np.clip(clip.visibility, 0.0, 1.0)
                * max(0.05, min(1.0, clip.quality / 100.0))
            )
            values.extend((config.projection_weight * weight[..., None] * (projected - clip.uv)).ravel())
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
                current = _angle_degrees(xyz[frame, a], xyz[frame, vertex], xyz[frame, c])
                normalized = (current - angle_targets[frame, index]) / 90.0
                values.append(config.world_angle_weight * angle_weights[frame, index] * normalized)
        return np.asarray(values, dtype=float)

    sparsity = build_jacobian_sparsity(clips)
    before = projection_rmse(clips, initial)
    result = least_squares(
        residual, initial.ravel(), jac_sparsity=sparsity, loss="soft_l1",
        f_scale=config.f_scale, max_nfev=config.max_nfev, method="trf",
    )
    optimized = result.x.reshape(FRAME_COUNT, LANDMARK_COUNT, 3)
    return optimized, OptimizationReport(
        bool(result.success), str(result.message), float(result.cost), int(result.nfev),
        before, projection_rmse(clips, optimized),
    )
```

Do not add any MediaPipe world XYZ residual. The only world-derived residuals are the four robust shooting-side angles above.

- [ ] **Step 4: Implement validation metrics and gates**

```python
def projection_rmse_by_view(clips, xyz):
    grouped = {}
    for view in ("front", "side", "oblique"):
        selected = [clip for clip in clips if clip.view == view]
        if selected:
            grouped[view] = projection_rmse(selected, xyz)
    return grouped


def validate_canonical_motion(xyz, clips, report, confidence):
    points = np.asarray(xyz, dtype=float)
    reasons = []
    if points.shape != (FRAME_COUNT, LANDMARK_COUNT, 3) or not np.isfinite(points).all():
        reasons.append("invalid 48x33x3 shape or non-finite coordinate")
    view_counts = {view: sum(c.view == view for c in clips) for view in ("front", "side")}
    if view_counts["front"] < 3 or view_counts["side"] < 3:
        reasons.append("requires at least 3 front and 3 side clips")
    bone_cv = {}
    for start, end in MAJOR_BONES:
        values = np.linalg.norm(points[:, start] - points[:, end], axis=1)
        bone_cv[f"{start}-{end}"] = float(np.std(values) / max(np.mean(values), 1e-9))
    if max(bone_cv.values(), default=1.0) > 0.01:
        reasons.append("major bone coefficient of variation exceeds 1%")
    per_view_rmse = projection_rmse_by_view(clips, points)
    if any(per_view_rmse.get(view, float("inf")) > 0.08 for view in ("front", "side")):
        reasons.append("front or side projection RMSE exceeds 0.08")
    speed = np.linalg.norm(np.diff(points, axis=0), axis=2).ravel()
    spike_ratio = float(speed.max() / max(np.percentile(speed, 95), 1e-9))
    if spike_ratio > 3.0:
        reasons.append("temporal velocity spike ratio exceeds 3")
    phases = clips[0].phases
    release_index = next((index for index, phase in enumerate(phases) if phase == "release"), 36)
    targets, target_weights = _world_angle_targets(clips)
    angle_errors = {}
    for index, (a, vertex, c) in enumerate(ANGLE_TRIPLETS):
        measured = _angle_degrees(points[release_index, a], points[release_index, vertex], points[release_index, c])
        target = float(targets[release_index, index])
        if target_weights[release_index, index] > 0:
            angle_errors[("elbow", "shoulder", "hip", "knee")[index]] = abs(measured - target)
    release_angle_mae = float(np.mean(list(angle_errors.values()))) if len(angle_errors) == 4 else float("inf")
    if release_angle_mae > 8.0:
        reasons.append("release joint-angle MAE exceeds 8 degrees")
    paired = (
        ((11, 13), (12, 14)), ((13, 15), (14, 16)),
        ((23, 25), (24, 26)), ((25, 27), (26, 28)),
    )
    symmetry = {}
    for left, right in paired:
        left_length = float(np.median(np.linalg.norm(points[:, left[0]] - points[:, left[1]], axis=1)))
        right_length = float(np.median(np.linalg.norm(points[:, right[0]] - points[:, right[1]], axis=1)))
        key = f"{left[0]}-{left[1]}:{right[0]}-{right[1]}"
        symmetry[key] = abs(left_length - right_length) / max((left_length + right_length) * 0.5, 1e-9)
    if max(symmetry.values(), default=0.0) > 0.10:
        reasons.append("left/right paired-bone difference exceeds 10%")
    return {
        "passed": not reasons and report.success,
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
```

- [ ] **Step 5: Run reconstruction tests and compile the module**

```bash
python -m py_compile app/reconstruction3d.py
python -m unittest tests.test_reconstruction3d -v
```

Expected: all synthetic optimization and validation tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add app/reconstruction3d.py tests/test_reconstruction3d.py
git commit -m "Add sparse Curry bundle adjustment"
```

---

### Task 4: Versioned model serialization and skeleton adapter

**Files:**
- Modify: `app/reconstruction3d.py`
- Modify: `tests/test_reconstruction3d.py`

**Interfaces:**
- Consumes: optimized XYZ, dispersion, confidence, phases, provenance, validation, optimizer report.
- Produces: `build_model_entry()`, `write_validated_model()`, `load_canonical_player_skeleton()`, 33 semantic landmark names, and compatible bones.

- [ ] **Step 1: Add failing serialization and invalid-model fallback tests**

```python
import json
import tempfile
from pathlib import Path

from app.reconstruction3d import (
    build_model_entry,
    load_canonical_player_skeleton,
    write_validated_model,
)


def test_valid_model_round_trip_has_48_frames_and_33_landmarks(self):
    xyz = make_kinematic_xyz()
    entry = build_model_entry(
        player_key="stephen_curry", xyz=xyz,
        dispersion=np.zeros_like(xyz), confidence=np.ones((FRAME_COUNT, 33)),
        phases=tuple(["rise"] * 35 + ["release"] + ["follow_through"] * 12),
        clips=[], bone_lengths={}, optimizer={"success": True},
        validation={"passed": True, "reasons": []},
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "canonical.json"
        write_validated_model(path, entry)
        skeleton = load_canonical_player_skeleton("stephen_curry", path=path)
    self.assertEqual(len(skeleton["frames"]), 48)
    self.assertEqual(len(skeleton["frames"][0]["landmarks"]), 33)
    self.assertEqual(skeleton["quality_mode"], "multi_view_3d")


def test_invalid_model_returns_none_and_preserves_fallback_contract(self):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "canonical.json"
        path.write_text(json.dumps({"stephen_curry": {"validation": {"passed": False}}}))
        self.assertIsNone(load_canonical_player_skeleton("stephen_curry", path=path))
        self.assertIsNone(load_canonical_player_skeleton("devin_booker", path=path))
```

- [ ] **Step 2: Run the tests and verify missing serialization symbols**

```bash
python -m unittest tests.test_reconstruction3d -v
```

Expected: imports fail for the new model functions.

- [ ] **Step 3: Implement model schema and safe loader**

Define all 33 names in MediaPipe index order using renderer-compatible `_l` and `_r` suffixes:

```python
MP_LANDMARK_NAMES = (
    "nose", "eye_inner_l", "eye_l", "eye_outer_l", "eye_inner_r", "eye_r", "eye_outer_r",
    "ear_l", "ear_r", "mouth_l", "mouth_r", "shoulder_l", "shoulder_r",
    "elbow_l", "elbow_r", "wrist_l", "wrist_r", "pinky_l", "pinky_r",
    "index_l", "index_r", "thumb_l", "thumb_r", "hip_l", "hip_r",
    "knee_l", "knee_r", "ankle_l", "ankle_r", "heel_l", "heel_r",
    "foot_index_l", "foot_index_r",
)

MP_BONES = (
    ("nose", "eye_inner_l"), ("eye_inner_l", "eye_l"), ("eye_l", "eye_outer_l"),
    ("eye_outer_l", "ear_l"), ("nose", "eye_inner_r"), ("eye_inner_r", "eye_r"),
    ("eye_r", "eye_outer_r"), ("eye_outer_r", "ear_r"), ("mouth_l", "mouth_r"),
    ("shoulder_l", "shoulder_r"), ("shoulder_l", "elbow_l"), ("elbow_l", "wrist_l"),
    ("wrist_l", "pinky_l"), ("wrist_l", "index_l"), ("wrist_l", "thumb_l"),
    ("pinky_l", "index_l"), ("shoulder_r", "elbow_r"), ("elbow_r", "wrist_r"),
    ("wrist_r", "pinky_r"), ("wrist_r", "index_r"), ("wrist_r", "thumb_r"),
    ("pinky_r", "index_r"), ("shoulder_l", "hip_l"), ("shoulder_r", "hip_r"),
    ("hip_l", "hip_r"), ("hip_l", "knee_l"), ("knee_l", "ankle_l"),
    ("ankle_l", "heel_l"), ("heel_l", "foot_index_l"), ("ankle_l", "foot_index_l"),
    ("hip_r", "knee_r"), ("knee_r", "ankle_r"), ("ankle_r", "heel_r"),
    ("heel_r", "foot_index_r"), ("ankle_r", "foot_index_r"),
)
```

`build_model_entry()` must round coordinates to six decimals, store `model="curry_multiview_bundle_v1"`, `quality_mode="multi_view_3d"`, `units="body_scale"`, normalized progress, phases, XYZ, dispersion, confidence, bone lengths, source clips, optimizer configuration, and validation.

`write_validated_model(path, entry)` must raise `ValueError` unless `entry["validation"]["passed"] is True`, preserve unrelated existing entries, write to `path.with_suffix(".tmp")`, and atomically replace the target.

`load_canonical_player_skeleton()` must return `None` for missing files, missing players, failed validation, wrong shape, or non-finite coordinates. For a valid model, return the existing renderer schema:

```python
frames = []
duration = float(entry.get("duration", 1.0))
for index, coordinates in enumerate(entry["xyz"]):
    frames.append({
        "t": round(duration * index / (FRAME_COUNT - 1), 4),
        "phase": str(entry["phases"][index]),
        "landmarks": {
            name: [float(value) for value in coordinates[landmark_index]]
            for landmark_index, name in enumerate(MP_LANDMARK_NAMES)
        },
    })
skeleton = {
    "model": "curry_multiview_bundle_v1",
    "space": "canonical_3d",
    "source_space": "multi_view_2d_bundle_adjustment",
    "quality_mode": "multi_view_3d",
    "units": "body_scale",
    "hand": "right",
    "view": "merged",
    "duration": duration,
    "landmark_order": list(MP_LANDMARK_NAMES),
    "bones": [list(bone) for bone in MP_BONES],
    "frames": frames,
    "validation": entry["validation"],
}
```

Apply one global floor shift only when building the entry, using reliable catch-phase ankle/heel/foot-index Y values. Never shift individual frames.

- [ ] **Step 4: Run serialization tests and existing skeleton tests**

```bash
python -m unittest tests.test_reconstruction3d tests.test_skeleton -v
```

Expected: all pass; existing 18-joint fallback tests are unchanged.

- [ ] **Step 5: Commit Task 4**

```bash
git add app/reconstruction3d.py tests/test_reconstruction3d.py
git commit -m "Add validated Curry model format"
```

---

### Task 5: Optional raw observation capture and Curry-only builder

**Files:**
- Modify: `app/analyze.py`
- Modify: `scripts/youtube_profile.py`
- Create: `scripts/build_curry_canonical3d.py`
- Modify: `tests/test_reconstruction3d.py`
- Modify: `tests/test_motion.py`

**Interfaces:**
- Consumes: existing MediaPipe detections and shot-span logic.
- Produces: `ViewAnalysis.raw_timeline`, `analyze_view(..., capture_observations=False)`, and an offline Curry build CLI.

- [ ] **Step 1: Add failing tests for observation capture isolation**

Test the pure raw-timeline helper rather than running MediaPipe in unit tests:

```python
from app.angles import AngleSnapshot
from app.analyze import ViewAnalysis, _raw_timeline_from_shot
from app.shot_span import ShotSpan


def test_raw_observations_are_available_to_builder_but_not_api_payload(self):
    sequence = []
    for frame in range(5):
        snapshot = AngleSnapshot(
            elbow=120 + frame * 10, shoulder=90 + frame * 10,
            hip=145 + frame * 4, knee=125 + frame * 7,
            hand="right", space="3d",
        )
        image = [[0.5, 0.5 - frame * 0.01, 0.0, 0.95] for _ in range(33)]
        world = [[0.0, frame * 0.01, 0.0, 0.95] for _ in range(33)]
        sequence.append({
            "frame": frame, "snapshot": snapshot, "wrist_y": 0.5 - frame * 0.01,
            "image_landmarks": image, "world_landmarks": world,
        })
    span = ShotSpan(catch_index=0, dip_index=1, release_index=3, followthrough_index=4)
    timeline = _raw_timeline_from_shot(sequence, span, fps=30.0)
    self.assertGreaterEqual(len(timeline), 2)
    self.assertEqual(len(timeline[0]["image_landmarks"]), 33)
    view = ViewAnalysis(
        view="front", release_frame_index=10, frames_scanned=20, space="3d",
        hand="right",
        release_angles={"elbow": 150, "shoulder": 120, "hip": 157, "knee": 146},
        raw_timeline=timeline,
    )
    self.assertNotIn("raw_timeline", view.to_dict(include_skeleton=False))
```

- [ ] **Step 2: Run the focused test and verify signature/field failure**

```bash
python -m unittest tests.test_reconstruction3d -v
```

Expected: `ViewAnalysis` rejects `raw_timeline` or helper import fails.

- [ ] **Step 3: Add opt-in raw capture without changing default analysis**

In `ViewAnalysis`, add:

```python
raw_timeline: List[dict] = field(default_factory=list, repr=False)
```

At the start of `to_dict()`, remove it before returning:

```python
data.pop("raw_timeline", None)
```

Extend `analyze_view()` with `capture_observations: bool = False`. Keep the existing normalized `sequence` unchanged. When capture is enabled, store a parallel raw sequence containing frame index, `AngleSnapshot`, wrist Y, image landmark payload, and world landmark payload. After shot-span detection, `_raw_timeline_from_shot()` must slice catch-through-follow-through, assign the existing `phase_label()`, preserve raw image/world coordinates, and set `t=0` at catch.

Do not include raw observations in Flask responses, SQLite sessions, or `nba_player_models.json`.

Extend `scripts.youtube_profile.analyze_best()` with `capture_observations: bool = False`, pass it to the fine `analyze_view()` call, and include `meta["raw_timeline"] = view.raw_timeline` only when the flag is true. Existing callers receive the same metadata keys as before.

- [ ] **Step 4: Make certificate bypass explicit and opt-in**

Change the helper signature without changing existing callers:

```python
def download_clip(url: str, out_dir: Path, *, no_check_certificates: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "clip.%(ext)s")
    if "/shorts/" in url:
        video_id = url.rstrip("/").split("/")[-1].split("?")[0]
        url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--extractor-args", "youtube:player_client=web,android,ios",
        "-f", "bv*[height<=720]+ba/b[height<=720]/best[height<=720]/best",
        "-o", template, "--no-playlist", url,
    ]
    if no_check_certificates:
        cmd.insert(3, "--no-check-certificates")
    subprocess.run(cmd, check=True, capture_output=True)
    files = sorted(out_dir.glob("clip.*"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("Download produced no file")
    return files[0]
```

The default must remain `False`.

- [ ] **Step 5: Implement the Curry-only build CLI**

Create `scripts/build_curry_canonical3d.py` with:

```python
CURRY_KEY = "stephen_curry"
CURRY_NAME = "Stephen Curry"
DEFAULT_FRONT_URLS = (
    "https://www.youtube.com/watch?v=eETSPnYf85E",
    "https://www.youtube.com/shorts/VN1J6EgUQvg",
    "https://www.youtube.com/shorts/iQVNOyQES0A",
    "https://www.youtube.com/shorts/W3pFskLg04A",
)


def current_side_urls(data):
    clips = (((data.get(CURRY_NAME) or {}).get("views") or {}).get("side") or {}).get("clips") or []
    return tuple(dict.fromkeys(str(clip.get("youtube_url") or "") for clip in clips if clip.get("youtube_url")))
```

CLI arguments:

```text
--front-url URL        repeatable override/addition
--side-url URL         repeatable override/addition
--min-per-view 3       must not accept values below 3
--max-frames 400
--no-check-certificates
--output models/canonical_3d_models.json
--report models/curry_canonical3d_validation.json
```

For each source, download to a temporary directory, call `analyze_best(..., capture_observations=True)`, verify `meta["view"]` against the assigned view, pass `meta["raw_timeline"]` to `normalize_clip_observation()`, and record accepted/rejected provenance. Continue until three accepted clips exist for each required view; try remaining candidates when a clip is rejected.

Then call `initialize_xyz()`, `optimize_canonical_motion()`, `observation_dispersion()`, `validate_canonical_motion()`, and `build_model_entry()`. Always write the diagnostic report. Call `write_validated_model()` only when validation passes. Never modify `nba_player_models.json`.

- [ ] **Step 6: Run builder unit tests and all pre-existing unit tests**

```bash
python -m unittest tests.test_reconstruction3d tests.test_motion tests.test_skeleton tests.test_multiview3d -v
```

Expected: all pass without downloading videos.

- [ ] **Step 7: Commit Task 5**

```bash
git add app/analyze.py scripts/youtube_profile.py scripts/build_curry_canonical3d.py tests/test_reconstruction3d.py tests/test_motion.py
git commit -m "Add Curry canonical model builder"
```

---

### Task 6: Generate Curry model, validate, integrate API, and verify Render regressions

**Files:**
- Create: `models/canonical_3d_models.json`
- Create: `models/curry_canonical3d_validation.json`
- Modify: `app/server.py`
- Modify: `tests/test_app_smoke.py`
- Modify if and only if demonstrated necessary: `static/skeleton3d.js`
- Modify: `docs/superpowers/specs/2026-08-16-curry-canonical-3d-bundle-adjustment-design.md`

**Interfaces:**
- Consumes: validated model loader from Task 4 and builder from Task 5.
- Produces: live Curry skeleton endpoint preference with unchanged fallback behavior.

- [ ] **Step 1: Acquire the MediaPipe task model locally for offline generation**

If `models/pose_landmarker_full.task` is absent, download the official full pose landmarker asset without committing a duplicate:

```bash
curl -fL --retry 3 \
  -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
```

Verify its size is greater than 9 MB:

```bash
test "$(wc -c < models/pose_landmarker_full.task)" -gt 9000000
```

- [ ] **Step 2: Run the real Curry builder**

```bash
python scripts/build_curry_canonical3d.py --min-per-view 3 --no-check-certificates
```

Expected: at least three accepted front and three accepted side clips, a diagnostic report, and a validation-passed `stephen_curry` model. If fewer clips pass, add public Curry front/side URLs through the repeatable CLI flags and rerun; do not relax validation thresholds.

- [ ] **Step 3: Inspect and record quantitative gates before API integration**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('models/curry_canonical3d_validation.json').read_text())
validation = report['validation']
assert validation['passed'], validation['reasons']
assert validation['view_counts']['front'] >= 3
assert validation['view_counts']['side'] >= 3
assert max(validation['bone_cv'].values()) <= 0.01
assert validation['projection_rmse']['front'] <= 0.08
assert validation['projection_rmse']['side'] <= 0.08
assert validation['release_angle_mae_deg'] <= 8.0
print(json.dumps(validation, indent=2))
PY
```

Save the exact output for the final report.

- [ ] **Step 4: Write failing API preference and fallback tests**

Use `unittest.mock.patch` so tests do not depend on the committed model contents:

```python
@patch("app.server.load_canonical_player_skeleton")
def test_curry_prefers_valid_multiview_canonical_skeleton(self, load_model):
    load_model.return_value = {
        "model": "curry_multiview_bundle_v1", "space": "canonical_3d",
        "quality_mode": "multi_view_3d", "frames": [{"t": 0.0, "landmarks": {}}],
        "bones": [], "duration": 0.0,
    }
    res = self.client.get("/api/players/stephen_curry/skeleton")
    self.assertEqual(res.status_code, 200)
    self.assertEqual(res.get_json()["skeleton"]["quality_mode"], "multi_view_3d")


@patch("app.server.load_canonical_player_skeleton", return_value=None)
def test_non_curry_and_invalid_curry_model_use_existing_fallback(self, _load_model):
    curry = self.client.get("/api/players/stephen_curry/skeleton").get_json()["skeleton"]
    self.assertNotEqual(curry.get("model"), "curry_multiview_bundle_v1")
    other = self.client.get("/api/players/devin_booker/skeleton").get_json()["skeleton"]
    self.assertEqual(other["space"], "canonical_3d")
```

- [ ] **Step 5: Run API tests and verify preference failure**

```bash
python -m unittest tests.test_app_smoke -v
```

Expected: patch target is missing or Curry still returns the old model.

- [ ] **Step 6: Add the minimal endpoint preference**

Import the loader in `app/server.py`:

```python
from .reconstruction3d import load_canonical_player_skeleton
```

At the beginning of `player_skeleton()`:

```python
canonical = load_canonical_player_skeleton(player_key)
if canonical is not None:
    return jsonify({
        "player_key": player_key,
        "display_name": "Stephen Curry" if player_key == "stephen_curry" else player_key,
        "skeleton": canonical,
    })
```

Do not alter the existing DB lookup and fallback block below it.

- [ ] **Step 7: Verify renderer compatibility before editing JavaScript**

Validate that every configured bone endpoint exists in every frame and that the existing generic interpolation/project loop accepts the semantic landmark dictionary:

```bash
python - <<'PY'
from app.reconstruction3d import load_canonical_player_skeleton
s = load_canonical_player_skeleton('stephen_curry')
assert s and len(s['frames']) == 48
for frame in s['frames']:
    for a, b in s['bones']:
        assert a in frame['landmarks'] and b in frame['landmarks']
print('renderer schema ok')
PY
```

Leave `static/skeleton3d.js` unchanged when this passes. If it fails, change only the semantic side-detection logic and add a JavaScript rendered-HTML regression; do not replace the renderer.

- [ ] **Step 8: Run full regression, dataset validation, compile, and Render health checks**

```bash
python -m compileall -q app scripts
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/validate_motion_dataset.py --min-clips 3
python -c "from app.server import app; c=app.test_client(); r=c.get('/api/health'); assert r.status_code == 200 and r.get_json()['ok']"
docker build -t shooting-form-analysis:curry-3d .
docker run --rm -d --name shooting-form-curry-test -p 18080:10000 shooting-form-analysis:curry-3d
```

Poll `http://127.0.0.1:18080/api/health`, confirm HTTP 200, query the Curry skeleton endpoint, then stop the container:

```bash
curl -fsS http://127.0.0.1:18080/api/health
curl -fsS http://127.0.0.1:18080/api/players/stephen_curry/skeleton | python -m json.tool >/dev/null
docker stop shooting-form-curry-test
```

Expected: all tests pass, dataset validation passes, health is 200, Curry returns `quality_mode=multi_view_3d`, and Render runtime installs no SciPy because `requirements.txt` is unchanged.

- [ ] **Step 9: Commit Task 6**

```bash
git add models/canonical_3d_models.json models/curry_canonical3d_validation.json app/server.py tests/test_app_smoke.py docs/superpowers/specs/2026-08-16-curry-canonical-3d-bundle-adjustment-design.md
git commit -m "Serve validated Curry canonical 3D motion"
```

---

## Final handoff checklist

- Report the audited old data flow and root cause.
- List every added and modified file.
- State the front/side/yaw projection equations.
- Report fixed-scale and catch-origin behavior.
- Report phase anchors and set-point fallback.
- Report visibility, Hampel, MAD, and soft-L1 outlier handling.
- Report the exact bundle-adjustment objective and weights used.
- Report accepted/rejected Curry clips and the final 48x33x3 shape.
- Compare old and new bone variance, velocity spikes, projection errors, left/right consistency, and four release-angle errors.
- Report complete unit, dataset, compile, Docker, and health-check results.
- State that the Render runtime requirements, DB schema, matching, other players, and renderer are unchanged.
- State remaining limitations: orthographic camera approximation, inferred yaw, non-synchronized repeated shots, no ball trajectory, and Curry-only coverage.
