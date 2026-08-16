"""Tests for robust Curry multi-view canonical 3D reconstruction."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from app.angles import AngleSnapshot
from app.analyze import ViewAnalysis, _raw_timeline_from_shot
from app.reconstruction3d import (
    ClipObservation,
    FRAME_COUNT,
    OptimizationConfig,
    OptimizationReport,
    build_model_entry,
    compute_fixed_scale,
    filter_cross_clip_residuals,
    filter_observations,
    initialize_xyz,
    load_canonical_player_skeleton,
    optimize_canonical_motion,
    normalize_clip_observation,
    reproject_xyz,
    validate_canonical_motion,
    write_validated_model,
)
from app.shot_span import ShotSpan
from scripts.youtube_profile import download_clip


def synthetic_image_timeline(frame_count: int = 21, jump: float = 0.18) -> list[dict]:
    samples = []
    for frame in range(frame_count):
        progress = frame / (frame_count - 1)
        phase = (
            "catch"
            if progress < 0.22
            else "dip"
            if progress < 0.30
            else "rise"
            if progress < 0.74
            else "release"
            if progress < 0.78
            else "follow_through"
        )
        vertical = jump * max(0.0, 1.0 - abs(progress - 0.78) / 0.35)
        points = np.zeros((33, 4), dtype=float)
        points[:, 3] = 0.95
        points[:, :2] = (0.50, 0.45 - vertical)
        points[11, :2] = (0.42, 0.30 - vertical)
        points[12, :2] = (0.58, 0.30 - vertical)
        points[23, :2] = (0.46, 0.55 - vertical)
        points[24, :2] = (0.54, 0.55 - vertical)
        points[25, :2] = (0.46, 0.72 - vertical)
        points[26, :2] = (0.54, 0.72 - vertical)
        points[27, :2] = (0.46, 0.90 - vertical)
        points[28, :2] = (0.54, 0.90 - vertical)
        samples.append(
            {
                "t": frame / 30.0,
                "frame": frame,
                "phase": phase,
                "image_landmarks": points.tolist(),
                "world_landmarks": [],
                "elbow": 110 + 50 * progress,
                "shoulder": 90 + 55 * progress,
                "hip": 140 + 20 * progress,
                "knee": 125 + 35 * progress,
            }
        )
    return samples


def make_xyz() -> np.ndarray:
    xyz = np.zeros((FRAME_COUNT, 33, 3), dtype=float)
    for frame in range(FRAME_COUNT):
        progress = frame / (FRAME_COUNT - 1)
        xyz[frame, :, 0] = np.linspace(-0.35, 0.35, 33) + 0.02 * np.sin(
            np.pi * progress
        )
        xyz[frame, :, 1] = np.linspace(1.65, 0.05, 33) + 0.14 * np.sin(
            np.pi * progress
        )
        xyz[frame, :, 2] = np.linspace(-0.20, 0.25, 33) + 0.05 * progress
    return xyz


def observation_from_xyz(
    xyz: np.ndarray,
    clip_id: str,
    view: str,
    yaw: float,
    noise: float = 0.0,
) -> ClipObservation:
    uv = reproject_xyz(xyz, yaw)
    if noise:
        seed = sum((index + 1) * ord(char) for index, char in enumerate(clip_id))
        rng = np.random.default_rng(seed)
        uv = uv + rng.normal(0.0, noise, uv.shape)
    return ClipObservation(
        clip_id=clip_id,
        view=view,
        yaw=yaw,
        quality=95.0,
        uv=uv,
        visibility=np.full((FRAME_COUNT, 33), 0.95),
        world_angles=np.full((FRAME_COUNT, 4), np.nan),
        frame_scale=np.ones(FRAME_COUNT),
        phases=tuple(["rise"] * FRAME_COUNT),
    )


def make_kinematic_xyz() -> np.ndarray:
    xyz = np.zeros((FRAME_COUNT, 33, 3), dtype=float)
    for frame in range(FRAME_COUNT):
        progress = frame / (FRAME_COUNT - 1)
        pelvis_y = 1.0 + 0.14 * np.sin(np.pi * progress)
        for index in range(33):
            xyz[frame, index] = (0.0, pelvis_y + 0.25, 0.0)
        xyz[frame, 23] = (-0.14, pelvis_y, 0.0)
        xyz[frame, 24] = (0.14, pelvis_y, 0.0)
        xyz[frame, 11] = (-0.18, pelvis_y + 0.50, 0.0)
        xyz[frame, 12] = (0.18, pelvis_y + 0.50, 0.0)
        arm_angle = -1.15 + 1.0 * progress
        for shoulder, elbow, wrist, sign in (
            (11, 13, 15, -1),
            (12, 14, 16, 1),
        ):
            upper = np.array(
                [
                    sign * 0.10 * np.cos(arm_angle),
                    0.30 * np.sin(-arm_angle),
                    0.24 * progress,
                ]
            )
            upper *= 0.31 / np.linalg.norm(upper)
            xyz[frame, elbow] = xyz[frame, shoulder] + upper
            forearm = np.array(
                [sign * 0.06, 0.25, 0.10 + 0.06 * progress]
            )
            forearm *= 0.27 / np.linalg.norm(forearm)
            xyz[frame, wrist] = xyz[frame, elbow] + forearm
        for hip, knee, ankle, sign in (
            (23, 25, 27, -1),
            (24, 26, 28, 1),
        ):
            thigh = np.array(
                [sign * 0.03, -0.44, 0.08 * (1.0 - progress)]
            )
            thigh *= 0.45 / np.linalg.norm(thigh)
            xyz[frame, knee] = xyz[frame, hip] + thigh
            shin = np.array([0.0, -0.43, -0.05 * (1.0 - progress)])
            shin *= 0.44 / np.linalg.norm(shin)
            xyz[frame, ankle] = xyz[frame, knee] + shin
        xyz[frame, 29] = xyz[frame, 27] + (0.0, -0.03, -0.03)
        xyz[frame, 30] = xyz[frame, 28] + (0.0, -0.03, -0.03)
        xyz[frame, 31] = xyz[frame, 27] + (-0.02, -0.02, 0.18)
        xyz[frame, 32] = xyz[frame, 28] + (0.02, -0.02, 0.18)
    return xyz


def make_six_noisy_clips(xyz: np.ndarray, noise: float) -> list[ClipObservation]:
    return [
        observation_from_xyz(xyz, f"front-{index}", "front", 0.0, noise)
        for index in range(3)
    ] + [
        observation_from_xyz(
            xyz, f"side-{index}", "side", np.pi / 2, noise
        )
        for index in range(3)
    ]


def bone_length_series(xyz: np.ndarray, start: int, end: int) -> np.ndarray:
    return np.linalg.norm(xyz[:, start] - xyz[:, end], axis=1)


def project_with_legacy_camera(
    point: list[float], width: int = 640, height: int = 430
) -> tuple[float, float]:
    yaw = -0.55
    pitch = -0.12
    x, y, z = float(point[0]), float(point[1]) - 0.88, float(point[2])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rotated_x = x * cy + z * sy
    rotated_z = -x * sy + z * cy
    rotated_y = y * cp - rotated_z * sp
    depth = y * sp + rotated_z * cp
    perspective = 3.3 / max(1.8, 3.3 + depth)
    scale = min(width, height) * 0.46
    return (
        width / 2 + rotated_x * scale * perspective,
        height * 0.54 - rotated_y * scale * perspective,
    )


class Reconstruction3DTests(unittest.TestCase):
    def test_one_fixed_scale_preserves_pelvis_vertical_motion(self):
        samples = synthetic_image_timeline()
        image = np.asarray([sample["image_landmarks"] for sample in samples], dtype=float)

        scale = compute_fixed_scale(image[:, :, :2], image[:, :, 3], view="front")
        clip = normalize_clip_observation(
            "front-a",
            samples,
            view="front",
            yaw=0.0,
            quality=90.0,
        )

        self.assertGreater(scale, 0.1)
        self.assertEqual(clip.uv.shape, (FRAME_COUNT, 33, 2))
        self.assertEqual(len(set(np.round(clip.frame_scale, 8))), 1)
        pelvis_y = clip.uv[:, [23, 24], 1].mean(axis=1)
        self.assertGreater(float(np.ptp(pelvis_y)), 0.10)

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

        actual, _ = initialize_xyz([*clips, outlier])

        self.assertLess(float(np.sqrt(np.mean((actual - expected) ** 2))), 0.01)

    def test_temporal_hampel_filter_disables_single_frame_landmark_jump(self):
        expected = make_xyz()
        clip = observation_from_xyz(expected, "damaged", "front", 0.0)
        damaged_uv = clip.uv.copy()
        damaged_uv[20, 16] += (0.18, -0.14)
        damaged = dataclasses.replace(clip, uv=damaged_uv)

        filtered = filter_observations([damaged])[0]

        self.assertEqual(filtered.visibility[20, 16], 0.0)
        self.assertGreater(filtered.visibility[19, 16], 0.9)
        self.assertGreater(filtered.visibility[21, 16], 0.9)

    def test_missing_side_landmark_uses_temporal_seed_without_aborting(self):
        expected = make_kinematic_xyz()
        clips = make_six_noisy_clips(expected, noise=0.001)
        damaged = []
        for clip in clips:
            if clip.view != "side":
                damaged.append(clip)
                continue
            visibility = clip.visibility.copy()
            visibility[20, 16] = 0.0
            damaged.append(dataclasses.replace(clip, visibility=visibility))

        actual, confidence = initialize_xyz(damaged)

        self.assertTrue(np.isfinite(actual).all())
        self.assertLess(float(np.linalg.norm(actual[20, 16] - expected[20, 16])), 0.02)
        self.assertLess(confidence[20, 16], confidence[19, 16])

    def test_cross_clip_mad_removes_persistent_bad_clip_before_optimizer(self):
        expected = make_kinematic_xyz()
        clips = make_six_noisy_clips(expected, noise=0.002)
        bad_uv = clips[2].uv.copy()
        bad_uv[:, 11:17] += (0.45, -0.35)
        damaged = [*clips[:2], dataclasses.replace(clips[2], uv=bad_uv), *clips[3:]]
        initial, _ = initialize_xyz(damaged)

        filtered = filter_cross_clip_residuals(damaged, initial)
        seed, confidence = initialize_xyz(filtered)
        optimized, report = optimize_canonical_motion(
            filtered, seed, confidence, OptimizationConfig(max_nfev=80)
        )

        self.assertEqual(float(filtered[2].visibility[:, 11:17].max()), 0.0)
        self.assertTrue(report.success)
        self.assertLess(
            float(np.sqrt(np.mean((optimized - expected) ** 2))), 0.01
        )

    def test_validation_counts_only_clips_with_effective_post_filter_coverage(self):
        expected = make_kinematic_xyz()
        clips = make_six_noisy_clips(expected, noise=0.0)
        masked = dataclasses.replace(
            clips[2], visibility=np.zeros_like(clips[2].visibility)
        )
        report = OptimizationReport(True, "ok", 0.0, 1, 0.0, 0.0)

        validation = validate_canonical_motion(
            expected,
            [*clips[:2], masked, *clips[3:]],
            report,
            np.ones((FRAME_COUNT, 33)),
        )

        self.assertEqual(validation["view_counts"]["front"], 2)
        self.assertIn(
            "requires at least 3 front and 3 side clips",
            validation["reasons"],
        )

    def test_full_timeline_one_yaw_occlusion_uses_paired_bone_prior(self):
        expected = make_kinematic_xyz()
        clips = make_six_noisy_clips(expected, noise=0.002)
        damaged = []
        for clip in clips:
            if clip.view != "side":
                damaged.append(clip)
                continue
            visibility = clip.visibility.copy()
            visibility[:, 13] = 0.0
            damaged.append(dataclasses.replace(clip, visibility=visibility))

        initial, confidence = initialize_xyz(damaged)
        optimized, report = optimize_canonical_motion(
            damaged,
            initial,
            confidence,
            OptimizationConfig(max_nfev=80),
        )

        self.assertTrue(report.success, report.message)
        self.assertTrue(np.isfinite(optimized).all())

    def test_geometry_filter_rejects_systematically_collapsed_paired_limb(self):
        expected = make_kinematic_xyz()
        clip = observation_from_xyz(expected, "collapsed", "front", 0.0)
        collapsed_uv = clip.uv.copy()
        collapsed_uv[:, 13] = collapsed_uv[:, 11]
        collapsed = dataclasses.replace(clip, uv=collapsed_uv)

        filtered = filter_observations([collapsed])[0]

        self.assertEqual(float(filtered.visibility[:, 13].max()), 0.0)

    def test_oblique_observation_participates_in_least_squares(self):
        expected = make_xyz()
        clips = [
            observation_from_xyz(expected, "f", "front", 0.0, 0.003),
            observation_from_xyz(expected, "o", "oblique", np.pi / 4, 0.003),
            observation_from_xyz(expected, "s", "side", np.pi / 2, 0.003),
        ]

        actual, _ = initialize_xyz(clips)

        self.assertLess(float(np.sqrt(np.mean((actual - expected) ** 2))), 0.01)

    def test_bundle_adjustment_reduces_bone_and_projection_error(self):
        expected = make_kinematic_xyz()
        clips = make_six_noisy_clips(expected, noise=0.012)
        initial, confidence = initialize_xyz(clips)

        optimized, report = optimize_canonical_motion(
            clips,
            initial,
            confidence,
            OptimizationConfig(max_nfev=80),
        )

        self.assertTrue(report.success, report.message)
        for start, end in (
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
            (23, 25),
            (25, 27),
            (24, 26),
            (26, 28),
        ):
            values = bone_length_series(optimized, start, end)
            self.assertLess(float(np.std(values) / np.mean(values)), 0.01)
        self.assertLess(report.projection_rmse_after, report.projection_rmse_before)

    def test_bundle_adjustment_suppresses_single_frame_velocity_spike(self):
        expected = make_kinematic_xyz()
        clips = make_six_noisy_clips(expected, noise=0.004)
        damaged_uv = clips[0].uv.copy()
        damaged_uv[20, 16] += (0.8, 0.8)
        damaged = dataclasses.replace(clips[0], uv=damaged_uv)
        combined = [damaged, *clips[1:]]
        initial, confidence = initialize_xyz(combined)

        optimized, _ = optimize_canonical_motion(
            combined,
            initial,
            confidence,
            OptimizationConfig(max_nfev=80),
        )

        speed = np.linalg.norm(np.diff(optimized[:, 16], axis=0), axis=1)
        self.assertLess(float(speed.max() / np.percentile(speed, 95)), 3.0)

    def test_valid_model_round_trip_has_48_frames_and_33_landmarks(self):
        xyz = make_kinematic_xyz()
        entry = build_model_entry(
            player_key="stephen_curry",
            xyz=xyz,
            dispersion=np.zeros_like(xyz),
            confidence=np.ones((FRAME_COUNT, 33)),
            phases=tuple(
                ["rise"] * 35 + ["release"] + ["follow_through"] * 12
            ),
            clips=[],
            bone_lengths={},
            optimizer={"success": True},
            validation={"passed": True, "reasons": []},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "canonical.json"
            write_validated_model(path, entry)
            skeleton = load_canonical_player_skeleton(
                "stephen_curry", path=path
            )

        self.assertIsNotNone(skeleton)
        self.assertEqual(len(skeleton["frames"]), 48)
        self.assertEqual(len(skeleton["frames"][0]["landmarks"]), 33)
        self.assertEqual(skeleton["quality_mode"], "multi_view_3d")
        for frame in skeleton["frames"]:
            for start, end in skeleton["bones"]:
                self.assertIn(start, frame["landmarks"])
                self.assertIn(end, frame["landmarks"])

    def test_invalid_model_returns_none_and_preserves_fallback_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "canonical.json"
            path.write_text(
                json.dumps(
                    {"stephen_curry": {"validation": {"passed": False}}}
                ),
                encoding="utf-8",
            )

            self.assertIsNone(
                load_canonical_player_skeleton("stephen_curry", path=path)
            )
            self.assertIsNone(
                load_canonical_player_skeleton("devin_booker", path=path)
            )

    def test_malformed_model_variants_all_return_none(self):
        xyz = make_kinematic_xyz()
        valid = build_model_entry(
            player_key="stephen_curry",
            xyz=xyz,
            dispersion=np.zeros_like(xyz),
            confidence=np.ones((FRAME_COUNT, 33)),
            phases=tuple(["rise"] * FRAME_COUNT),
            clips=[],
            bone_lengths={},
            optimizer={"success": True},
            validation={"passed": True},
        )
        variants = [
            "not-an-entry",
            {"validation": True},
            {**valid, "model": "wrong_version"},
            {**valid, "player_key": "devin_booker"},
            {**valid, "phases": "x" * FRAME_COUNT},
            {**valid, "duration": -1.0},
            {**valid, "duration": float("nan")},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "canonical.json"
            for entry in variants:
                path.write_text(
                    json.dumps({"stephen_curry": entry}), encoding="utf-8"
                )
                self.assertIsNone(
                    load_canonical_player_skeleton("stephen_curry", path=path),
                    entry,
                )

    def test_loader_applies_one_global_renderer_fit_without_clipping(self):
        xyz = make_kinematic_xyz() * 3.5
        phases = tuple(["catch"] * 10 + ["rise"] * 38)
        entry = build_model_entry(
            player_key="stephen_curry",
            xyz=xyz,
            dispersion=np.zeros_like(xyz),
            confidence=np.ones((FRAME_COUNT, 33)),
            phases=phases,
            clips=[],
            bone_lengths={},
            optimizer={"success": True},
            validation={"passed": True},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "canonical.json"
            write_validated_model(path, entry)
            skeleton = load_canonical_player_skeleton(
                "stephen_curry", path=path
            )

        for width, height in ((640, 430), (320, 320)):
            projected = [
                project_with_legacy_camera(point, width, height)
                for frame in skeleton["frames"]
                for point in frame["landmarks"].values()
            ]
            self.assertTrue(all(0.0 <= x <= width for x, _ in projected))
            self.assertTrue(all(0.0 <= y <= height for _, y in projected))
        pelvis = np.asarray(
            [
                np.mean(
                    [
                        frame["landmarks"]["hip_l"][1],
                        frame["landmarks"]["hip_r"][1],
                    ]
                )
                for frame in skeleton["frames"]
            ]
        )
        self.assertGreater(float(np.ptp(pelvis)), 0.05)

    def test_raw_observations_are_available_to_builder_but_not_api_payload(self):
        sequence = []
        for frame in range(5):
            snapshot = AngleSnapshot(
                elbow=120 + frame * 10,
                shoulder=90 + frame * 10,
                hip=145 + frame * 4,
                knee=125 + frame * 7,
                hand="right",
                space="3d",
            )
            image = [
                [0.5, 0.5 - frame * 0.01, 0.0, 0.95]
                for _ in range(33)
            ]
            world = [
                [0.0, frame * 0.01, 0.0, 0.95] for _ in range(33)
            ]
            sequence.append(
                {
                    "frame": frame,
                    "snapshot": snapshot,
                    "wrist_y": 0.5 - frame * 0.01,
                    "image_landmarks": image,
                    "world_landmarks": world,
                }
            )
        span = ShotSpan(
            catch_index=0,
            dip_index=1,
            release_index=3,
            followthrough_index=4,
        )

        timeline = _raw_timeline_from_shot(sequence, span, fps=30.0)

        self.assertGreaterEqual(len(timeline), 2)
        self.assertEqual(len(timeline[0]["image_landmarks"]), 33)
        view = ViewAnalysis(
            view="front",
            release_frame_index=10,
            frames_scanned=20,
            space="3d",
            hand="right",
            release_angles={
                "elbow": 150,
                "shoulder": 120,
                "hip": 157,
                "knee": 146,
            },
            raw_timeline=timeline,
        )
        self.assertNotIn("raw_timeline", view.to_dict(include_skeleton=False))

    def test_download_rejects_html_disguised_as_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "clip"

            def fake_download(*_args, **_kwargs):
                output.mkdir(parents=True, exist_ok=True)
                (output / "clip.mp4").write_text(
                    "<html><body>Site Unavailable</body></html>",
                    encoding="utf-8",
                )

            with mock.patch(
                "scripts.youtube_profile.subprocess.run", side_effect=fake_download
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError, "valid video"
                ):
                    download_clip("https://example.test/video", output)


if __name__ == "__main__":
    unittest.main()
