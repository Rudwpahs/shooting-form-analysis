"""Video → 3D joint-angle profiles (degrees only)."""

from __future__ import annotations

import base64
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .angles import AngleSnapshot, angles_from_landmarks, angles_plausible, median_angles
from .multiview3d import merge_multiview_release
from .motion import normalize_landmarks, smooth_timeline, timeline_quality as evaluate_timeline_quality
from .pose import PoseCandidate, close_detector, create_pose_detector
from .shot_span import (
    ShotSpan,
    detect_shot_span,
    phase_label,
    shot_span_is_complete,
    summarize_phases,
    timeline_duration_is_plausible,
)
from .skeleton import build_skeleton_timeline, release_skeleton


VIEW_TAGS = ("front", "side", "oblique")


@dataclass
class PersonPreview:
    person_index: int
    bbox: Tuple[float, float, float, float]
    thumb_jpeg_b64: str


@dataclass
class ViewAnalysis:
    view: str
    release_frame_index: int
    frames_scanned: int
    space: str
    hand: str
    release_angles: Dict[str, float]
    sequence_angles: Dict[str, List[float]] = field(default_factory=dict)
    fps: float = 0.0
    timeline: List[dict] = field(default_factory=list)
    catch_frame_index: int = -1
    dip_frame_index: int = -1
    followthrough_frame_index: int = -1
    release_landmarks: List[dict] = field(default_factory=list)
    motion_quality: Dict[str, object] = field(default_factory=dict)
    raw_timeline: List[dict] = field(default_factory=list, repr=False)
    error: str = ""

    def to_dict(self, *, include_skeleton: bool = True) -> dict:
        data = asdict(self)
        data.pop("sequence_angles", None)
        data.pop("raw_timeline", None)
        # Landmarks are large; keep only a flag in API payloads.
        lms = data.pop("release_landmarks", None) or []
        data["has_release_landmarks"] = bool(lms)
        data["phases"] = {
            "catch": self.catch_frame_index,
            "dip": self.dip_frame_index,
            "release": self.release_frame_index,
            "follow_through": self.followthrough_frame_index,
        }
        data["phase_summary"] = summarize_phases(self.timeline)
        if include_skeleton:
            if self.timeline:
                data["skeleton"] = build_skeleton_timeline(
                    self.timeline,
                    hand=self.hand or "right",
                    view=self.view,
                    source_space=self.space or "3d",
                )
            elif self.release_angles:
                data["skeleton"] = release_skeleton(
                    self.release_angles,
                    hand=self.hand or "right",
                    view=self.view,
                    source_space=self.space or "3d",
                )
        return data


@dataclass
class SessionAnalysis:
    views: List[ViewAnalysis]
    person_index: int
    release_angles_merged: Dict[str, float]
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "person_index": self.person_index,
            "release_angles_merged": self.release_angles_merged,
            "views": [v.to_dict() for v in self.views],
            "error": self.error,
        }


def _resize(frame: np.ndarray, height: int = 720) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == height:
        return frame
    width = max(1, int(round(w * height / h)))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _encode_thumb(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> str:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = bbox
    pad = 0.05
    xa = max(0, int((x0 - pad) * w))
    ya = max(0, int((y0 - pad) * h))
    xb = min(w, int((x1 + pad) * w))
    yb = min(h, int((y1 + pad) * h))
    crop = frame[ya:yb, xa:xb] if xb > xa and yb > ya else frame
    crop = _resize(crop, 180)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _pick_pose(poses: List[PoseCandidate], person_index: int) -> Optional[PoseCandidate]:
    if not poses:
        return None
    for pose in poses:
        if pose.person_index == person_index:
            return pose
    return poses[0] if person_index == 0 else None


def _angles_for_pose(pose: PoseCandidate, hand: Optional[str]) -> Optional[AngleSnapshot]:
    """Single-view angles from MediaPipe world landmarks (monocular 3D estimate).

    True multi-view 3D is handled in analyze_session via triangulation when
    front/side/oblique clips are uploaded together.
    """
    if pose.world_landmarks:
        world = angles_from_landmarks(pose.world_landmarks, hand, space="3d")
        if world is not None and angles_plausible(world):
            return world
        if world is not None:
            return world
    return angles_from_landmarks(pose.image_landmarks, hand, space="2d")


def _landmarks_payload(landmarks) -> List[dict]:
    out = []
    for lm in landmarks or []:
        if hasattr(lm, "x") and hasattr(lm, "y"):
            out.append(
                {
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(getattr(lm, "z", 0.0)),
                    "visibility": float(getattr(lm, "visibility", 1.0)),
                }
            )
            continue
        out.append(
            {
                "x": float(lm[0]),
                "y": float(lm[1]),
                "z": float(lm[2]) if len(lm) > 2 else 0.0,
                "visibility": 1.0,
            }
        )
    return out


def _wrist_image_y(pose: PoseCandidate, hand: str) -> float:
    idx = 15 if hand == "left" else 16
    lms = pose.image_landmarks
    if idx >= len(lms):
        return float("inf")
    return float(lms[idx].y)


def list_people_in_video(
    video_path: Path,
    sample_time_sec: float = 0.4,
    num_poses: int = 3,
    render_height: int = 720,
) -> List[PersonPreview]:
    """Return detectable people near an early frame for user selection."""
    cap = cv2.VideoCapture(str(video_path))
    detector = None
    try:
        if not cap.isOpened():
            return []
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_idx = max(0, int(sample_time_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            return []
        frame = _resize(frame, render_height)
        detector = create_pose_detector(fps=fps, num_poses=num_poses)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        poses = detector.detect(rgb)
        return [
            PersonPreview(
                person_index=p.person_index,
                bbox=p.bbox,
                thumb_jpeg_b64=_encode_thumb(frame, p.bbox),
            )
            for p in poses
        ]
    finally:
        cap.release()
        if detector is not None:
            close_detector(detector)


def _timeline_from_shot(
    sequence: List[Tuple[int, AngleSnapshot, float, List[List[float]]]],
    span: ShotSpan,
    fps: float,
) -> List[dict]:
    """Angle samples from catch through follow-through. t=0 at catch."""
    if fps <= 0:
        fps = 30.0
    if not shot_span_is_complete(span):
        return []
    catch_frame = sequence[span.catch_index][0]
    ft_frame = sequence[span.followthrough_index][0]
    samples: List[dict] = []
    for seq_i, (frame_idx, snap, _wrist_y, pose) in enumerate(sequence):
        if frame_idx < catch_frame or frame_idx > ft_frame:
            continue
        angles = snap.as_dict()
        sample = {
                "t": round((frame_idx - catch_frame) / fps, 4),
                "frame": int(frame_idx),
                "phase": phase_label(seq_i, span),
                "elbow": round(float(angles["elbow"]), 2),
                "shoulder": round(float(angles["shoulder"]), 2),
                "hip": round(float(angles["hip"]), 2),
                "knee": round(float(angles["knee"]), 2),
            }
        if pose:
            sample["pose"] = pose
        samples.append(sample)
    return samples if timeline_duration_is_plausible(samples, fps=fps) else []


def _raw_timeline_from_shot(
    sequence: Sequence[dict], span: ShotSpan, fps: float
) -> List[dict]:
    """Keep builder-only image/world observations across the detected shot."""
    if fps <= 0:
        fps = 30.0
    if not sequence or not shot_span_is_complete(span):
        return []
    catch_frame = int(sequence[span.catch_index]["frame"])
    followthrough_frame = int(sequence[span.followthrough_index]["frame"])
    samples: List[dict] = []
    for sequence_index, item in enumerate(sequence):
        frame = int(item["frame"])
        if frame < catch_frame or frame > followthrough_frame:
            continue
        snapshot = item["snapshot"]
        angles = snapshot.as_dict()
        samples.append(
            {
                "t": round((frame - catch_frame) / fps, 4),
                "frame": frame,
                "phase": phase_label(sequence_index, span),
                "elbow": float(angles["elbow"]),
                "shoulder": float(angles["shoulder"]),
                "hip": float(angles["hip"]),
                "knee": float(angles["knee"]),
                "image_landmarks": item["image_landmarks"],
                "world_landmarks": item["world_landmarks"],
            }
        )
    return samples


def _landmark_arrays(landmarks) -> List[List[float]]:
    return [
        [
            float(item["x"]),
            float(item["y"]),
            float(item["z"]),
            float(item["visibility"]),
        ]
        for item in _landmarks_payload(landmarks)
    ]


def release_score(angles: dict) -> float:
    elbow = float(angles.get("elbow", 0) or 0)
    shoulder = float(angles.get("shoulder", 0) or 0)
    if elbow < 100 or shoulder < 70:
        return 0.0
    base = elbow * 1.2 + shoulder * 0.6
    knee = float(angles.get("knee", 0) or 0)
    hip = float(angles.get("hip", 0) or 0)
    penalty = 0.0
    if knee < 120 or knee > 175:
        penalty += 50
    if hip < 90 or hip > 175:
        penalty += 50
    return max(0.0, base - penalty)


def _select_release_candidate(
    candidates: Sequence[Tuple[float, int, AngleSnapshot, List[dict]]],
    fps: float,
) -> Tuple[float, int, AngleSnapshot, List[dict]]:
    """Prefer an extended-arm local wrist peak after a clear upward rise.

    The old implementation selected the globally highest wrist, which often
    picked a wave, block, or post-shot follow-through.  Ball/hand separation is
    the eventual gold standard; this temporal score is a safer pose-only
    fallback for ordinary YouTube footage.
    """
    if len(candidates) < 3:
        return min(candidates, key=lambda item: item[0])
    wrists = np.asarray([item[0] for item in candidates], dtype=np.float64)
    frames = [item[1] for item in candidates]
    lo, hi = float(np.min(wrists)), float(np.max(wrists))
    span = max(hi - lo, 1e-6)
    pre_frames = max(3, int(round(max(fps, 1.0) * 0.9)))
    post_frames = max(2, int(round(max(fps, 1.0) * 0.22)))
    best = None
    best_score = -float("inf")
    for index, item in enumerate(candidates):
        wrist, frame, snap, _landmarks = item
        pose_score = release_score(snap.as_dict())
        if pose_score <= 0:
            continue
        before = [
            wrists[j]
            for j in range(index)
            if frames[j] >= frame - pre_frames
        ]
        after = [
            wrists[j]
            for j in range(index + 1, len(candidates))
            if frames[j] <= frame + post_frames
        ]
        rise = max(before, default=wrist) - wrist
        height = (hi - wrist) / span
        hold = 0.0 if not after else max(0.0, 1.0 - abs(float(np.median(after)) - wrist) / 0.12)
        temporal_local_peak = (
            (index == 0 or wrist <= wrists[index - 1] + 0.01)
            and (index == len(candidates) - 1 or wrist <= wrists[index + 1] + 0.01)
        )
        score = 4.0 * rise + 0.9 * height + 0.5 * hold + pose_score / 300.0
        if temporal_local_peak:
            score += 0.35
        if score > best_score:
            best_score = score
            best = item
    return best or min(candidates, key=lambda item: item[0])


def analyze_view(
    video_path: Path,
    view: str = "side",
    person_index: int = 0,
    hand: Optional[str] = None,
    max_frames: int = 240,
    num_poses: int = 3,
    render_height: int = 720,
    keep_sequence: bool = True,
    after_release_sec: float = 1.0,
    frame_stride: int = 1,
    start_time_sec: float = 0.0,
    capture_observations: bool = False,
) -> ViewAnalysis:
    cap = cv2.VideoCapture(str(video_path))
    detector = None
    try:
        if not cap.isOpened():
            return ViewAnalysis(
                view=view,
                release_frame_index=-1,
                frames_scanned=0,
                space="",
                hand="",
                release_angles={},
                error="Could not open video.",
            )

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if start_time_sec > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(start_time_sec * fps))))
        frame_stride = max(1, int(frame_stride))
        detector = create_pose_detector(fps=fps / frame_stride, num_poses=num_poses)

        candidates: List[Tuple[float, int, AngleSnapshot, List[dict]]] = []
        sequence: List[Tuple[int, AngleSnapshot, float, List[List[float]]]] = []
        raw_sequence: List[dict] = []
        frames_scanned = 0
        after_frames = max(1, int(round(max(after_release_sec, 0.6) * fps)))
        hard_cap = max_frames + after_frames

        while frames_scanned < hard_cap:
            ok, frame = cap.read()
            if not ok:
                break
            current_frame = frames_scanned
            frames_scanned += 1
            if current_frame % frame_stride:
                continue
            frame = _resize(frame, render_height)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            poses = detector.detect(rgb)
            pose = _pick_pose(poses, person_index)
            if pose is not None:
                snap = _angles_for_pose(pose, hand)
                if snap is not None:
                    wrist_y = _wrist_image_y(pose, snap.hand)
                    landmark_payload = _landmarks_payload(pose.image_landmarks)
                    normalized_pose = normalize_landmarks(landmark_payload, hand=snap.hand)
                    candidates.append(
                        (wrist_y, current_frame, snap, landmark_payload)
                    )
                    if keep_sequence or capture_observations:
                        sequence.append((current_frame, snap, wrist_y, normalized_pose))
                    if capture_observations:
                        raw_sequence.append(
                            {
                                "frame": current_frame,
                                "snapshot": snap,
                                "wrist_y": wrist_y,
                                "image_landmarks": _landmark_arrays(
                                    pose.image_landmarks
                                ),
                                "world_landmarks": _landmark_arrays(
                                    pose.world_landmarks
                                ),
                            }
                        )
            if frames_scanned >= max_frames and candidates:
                _, rel_so_far, _, _ = min(candidates, key=lambda item: item[0])
                if frames_scanned > rel_so_far + after_frames:
                    break

        if not candidates:
            return ViewAnalysis(
                view=view,
                release_frame_index=-1,
                frames_scanned=frames_scanned,
                space="",
                hand="",
                release_angles={},
                fps=fps,
                error="No reliable pose for the selected person.",
            )

        _, release_idx, release_snap, release_lms = _select_release_candidate(candidates, fps)
        seq_dict: Dict[str, List[float]] = {}
        if keep_sequence and sequence:
            for key in ("elbow", "shoulder", "hip", "knee"):
                seq_dict[key] = [float(s.as_dict()[key]) for _, s, _, _ in sequence]

        wrist_series = [w for _, _, w, _ in sequence]
        seq_frames = [f for f, _, _, _ in sequence]
        try:
            release_seq_i = seq_frames.index(release_idx)
        except ValueError:
            release_seq_i = min(range(len(seq_frames)), key=lambda i: abs(seq_frames[i] - release_idx)) if seq_frames else 0
        span = (
            detect_shot_span(
                wrist_series,
                fps,
                release_index=release_seq_i,
                frame_indices=seq_frames,
            )
            if wrist_series
            else ShotSpan(0, 0, 0, 0)
        )
        timeline = _timeline_from_shot(sequence, span, fps) if sequence else []
        if not timeline and len(sequence) >= 8 and 1 < release_seq_i < len(sequence) - 2:
            release_frame = seq_frames[release_seq_i]
            catch_i = release_seq_i
            while catch_i > 0 and seq_frames[catch_i - 1] >= release_frame - int(round(1.0 * fps)):
                catch_i -= 1
            follow_i = release_seq_i
            while follow_i + 1 < len(sequence) and seq_frames[follow_i + 1] <= release_frame + int(round(0.5 * fps)):
                follow_i += 1
            if catch_i < release_seq_i < follow_i:
                dip_i = max(
                    range(catch_i, release_seq_i),
                    key=lambda index: wrist_series[index],
                )
                fallback = ShotSpan(catch_i, dip_i, release_seq_i, follow_i)
                fallback_timeline = _timeline_from_shot(sequence, fallback, fps)
                if fallback_timeline:
                    span = fallback
                    timeline = fallback_timeline
        if timeline:
            timeline = smooth_timeline(timeline, window=5)
        raw_timeline = (
            _raw_timeline_from_shot(raw_sequence, span, fps)
            if capture_observations and timeline
            else []
        )
        catch_frame = seq_frames[span.catch_index] if seq_frames else -1
        dip_frame = seq_frames[span.dip_index] if seq_frames else -1
        ft_frame = seq_frames[span.followthrough_index] if seq_frames else -1

        release_angles = release_snap.as_dict()
        smoothed_release = next((sample for sample in timeline if sample.get("phase") == "release"), None)
        if smoothed_release:
            release_angles = {
                key: float(smoothed_release[key])
                for key in ("elbow", "shoulder", "hip", "knee")
            }
        return ViewAnalysis(
            view=view,
            release_frame_index=release_idx,
            frames_scanned=frames_scanned,
            space=release_snap.space,
            hand=release_snap.hand,
            release_angles=release_angles,
            sequence_angles=seq_dict,
            fps=fps,
            timeline=timeline,
            catch_frame_index=catch_frame,
            dip_frame_index=dip_frame,
            followthrough_frame_index=ft_frame,
            release_landmarks=release_lms,
            motion_quality=evaluate_timeline_quality(timeline, fps=fps) if timeline else {},
            raw_timeline=raw_timeline,
        )
    except Exception as exc:
        return ViewAnalysis(
            view=view,
            release_frame_index=-1,
            frames_scanned=0,
            space="",
            hand="",
            release_angles={},
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        cap.release()
        if detector is not None:
            close_detector(detector)


def pick_best_person(
    video_path: Path,
    view: str = "side",
    hand: Optional[str] = None,
    max_frames: int = 180,
) -> int:
    """Choose the detected person whose release pose looks most like a jump shot."""
    best_i = 0
    best_s = -1.0
    for person_index in range(3):
        result = analyze_view(
            video_path,
            view=view,
            person_index=person_index,
            hand=hand,
            max_frames=max_frames,
            keep_sequence=False,
            after_release_sec=0.3,
        )
        if result.error or not result.release_angles:
            continue
        score = release_score(result.release_angles)
        if score > best_s:
            best_s = score
            best_i = person_index
    return best_i


def analyze_session(
    videos: Sequence[Tuple[Path, str]],
    person_index: Optional[int] = 0,
    hand: Optional[str] = None,
    max_frames: int = 240,
    auto_person: bool = False,
) -> SessionAnalysis:
    """Analyze 1–3 videos; merge release with multi-view 3D triangulation when possible."""
    if auto_person or person_index is None:
        first_path, first_view = videos[0]
        tag = first_view if first_view in VIEW_TAGS else "side"
        person_index = pick_best_person(
            first_path,
            view=tag,
            hand=hand,
            max_frames=min(max_frames, 200),
        )
    views: List[ViewAnalysis] = []
    for path, view in videos:
        tag = view if view in VIEW_TAGS else "side"
        views.append(
            analyze_view(
                path,
                view=tag,
                person_index=person_index,
                hand=hand,
                max_frames=max_frames,
                after_release_sec=0.7,
            )
        )

    # Prefer geometric multi-view 3D when ≥2 tagged cameras have release landmarks.
    landmark_views = {
        v.view: v.release_landmarks
        for v in views
        if v.release_landmarks and not v.error and v.view in VIEW_TAGS
    }
    mv = merge_multiview_release(landmark_views, hand=hand) if len(landmark_views) >= 2 else None
    if mv is not None:
        return SessionAnalysis(
            views=views,
            person_index=person_index,
            release_angles_merged=mv.as_dict(),
        )

    ok_snaps = [
        AngleSnapshot(
            elbow=v.release_angles["elbow"],
            shoulder=v.release_angles["shoulder"],
            hip=v.release_angles["hip"],
            knee=v.release_angles["knee"],
            hand=v.hand or "right",
            space=v.space or "3d",
        )
        for v in views
        if v.release_angles and not v.error
    ]
    merged = median_angles(ok_snaps)
    if merged is None:
        return SessionAnalysis(
            views=views,
            person_index=person_index,
            release_angles_merged={},
            error="No view produced usable angles.",
        )
    return SessionAnalysis(
        views=views,
        person_index=person_index,
        release_angles_merged=merged.as_dict(),
    )
