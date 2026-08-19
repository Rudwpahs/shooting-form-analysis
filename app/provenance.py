"""Approved-source contracts for player reference data.

A pose sequence can be geometrically smooth while still belonging to the wrong
person, a video game, or a non-shooting action.  This module makes those
semantic checks explicit.  A profile may be used for matching only after its
source clips carry complete, human-reviewed provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse

PROFILE_DRAFT = "draft"
PROFILE_UNVERIFIED = "unverified_legacy"
PROFILE_VERIFIED_2D = "verified_2d"
PROFILE_VERIFIED_3D = "verified_3d"
PROFILE_REJECTED = "rejected"

PROFILE_STATUSES = frozenset(
    {
        PROFILE_DRAFT,
        PROFILE_UNVERIFIED,
        PROFILE_VERIFIED_2D,
        PROFILE_VERIFIED_3D,
        PROFILE_REJECTED,
    }
)
MATCHABLE_STATUSES = frozenset({PROFILE_VERIFIED_2D, PROFILE_VERIFIED_3D})
THREED_STATUSES = frozenset({PROFILE_VERIFIED_3D})

REQUIRED_CLIP_FIELDS = (
    "clip_id",
    "player_key",
    "source_url",
    "footage_type",
    "identity_status",
    "shot_status",
    "review_status",
    "reviewer",
    "catch_frame",
    "release_frame",
    "followthrough_end_frame",
)


@dataclass(frozen=True)
class ClipProvenance:
    """A human-reviewed clip used to build a player reference profile."""

    clip_id: str
    player_key: str
    source_url: str
    footage_type: str
    identity_status: str
    shot_status: str
    review_status: str
    reviewer: str
    catch_frame: int
    release_frame: int
    followthrough_end_frame: int
    fps: float = 0.0
    view: str = "unknown"
    notes: str = ""
    license_status: str = "unknown"
    ball_visible_ratio: float | None = None
    occlusion_ratio: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any]) -> "ClipProvenance":
        return cls(
            clip_id=str(item.get("clip_id") or "").strip(),
            player_key=str(item.get("player_key") or "").strip(),
            source_url=str(item.get("source_url") or "").strip(),
            footage_type=str(item.get("footage_type") or "").strip().lower(),
            identity_status=str(item.get("identity_status") or "").strip().lower(),
            shot_status=str(item.get("shot_status") or "").strip().lower(),
            review_status=str(item.get("review_status") or "").strip().lower(),
            reviewer=str(item.get("reviewer") or "").strip(),
            catch_frame=_int_or_default(item.get("catch_frame"), -1),
            release_frame=_int_or_default(item.get("release_frame"), -1),
            followthrough_end_frame=_int_or_default(item.get("followthrough_end_frame"), -1),
            fps=_float_or_default(item.get("fps"), 0.0),
            view=str(item.get("view") or "unknown").strip().lower(),
            notes=str(item.get("notes") or "").strip(),
            license_status=str(item.get("license_status") or "unknown").strip().lower(),
            ball_visible_ratio=_optional_ratio(item.get("ball_visible_ratio")),
            occlusion_ratio=_optional_ratio(item.get("occlusion_ratio")),
            metadata=dict(item.get("metadata") or {}),
        )

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        for key in REQUIRED_CLIP_FIELDS:
            value = getattr(self, key)
            if value in ("", None) or (isinstance(value, int) and value < 0):
                errors.append(f"missing {key}")
        parsed = urlparse(self.source_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            errors.append("source_url must be an absolute http(s) URL")
        if self.footage_type != "real":
            errors.append("footage_type must be real")
        if self.identity_status != "verified":
            errors.append("identity_status must be verified")
        if self.shot_status != "verified":
            errors.append("shot_status must be verified")
        if self.review_status != "approved":
            errors.append("review_status must be approved")
        if not (0 <= self.catch_frame < self.release_frame < self.followthrough_end_frame):
            errors.append("shot phase frames must be ordered")
        if self.fps and self.fps < 24:
            errors.append("fps must be at least 24 when supplied")
        if self.ball_visible_ratio is not None and self.ball_visible_ratio < 0.6:
            errors.append("ball_visible_ratio below 0.60")
        if self.occlusion_ratio is not None and self.occlusion_ratio > 0.25:
            errors.append("occlusion_ratio above 0.25")
        return errors

    @property
    def approved(self) -> bool:
        return not self.validation_errors()

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "player_key": self.player_key,
            "source_url": self.source_url,
            "footage_type": self.footage_type,
            "identity_status": self.identity_status,
            "shot_status": self.shot_status,
            "review_status": self.review_status,
            "reviewer": self.reviewer,
            "catch_frame": self.catch_frame,
            "release_frame": self.release_frame,
            "followthrough_end_frame": self.followthrough_end_frame,
            "fps": self.fps,
            "view": self.view,
            "notes": self.notes,
            "license_status": self.license_status,
            "ball_visible_ratio": self.ball_visible_ratio,
            "occlusion_ratio": self.occlusion_ratio,
            "metadata": dict(self.metadata),
        }


def profile_status_from_clips(
    clips: Iterable[Mapping[str, Any] | ClipProvenance],
    *,
    player_key: str,
    canonical_3d_verified: bool = False,
    minimum_clips: int = 3,
) -> tuple[str, list[str]]:
    """Return a conservative publication status and explicit rejection reasons."""
    parsed: list[ClipProvenance] = []
    errors: list[str] = []
    for raw in clips:
        clip = raw if isinstance(raw, ClipProvenance) else ClipProvenance.from_mapping(raw)
        if clip.player_key != player_key:
            errors.append(f"{clip.clip_id or 'clip'}: player_key mismatch")
            continue
        clip_errors = clip.validation_errors()
        if clip_errors:
            errors.extend(f"{clip.clip_id or 'clip'}: {message}" for message in clip_errors)
            continue
        parsed.append(clip)

    if errors:
        return PROFILE_UNVERIFIED, errors
    if len(parsed) < minimum_clips:
        return PROFILE_UNVERIFIED, [f"requires at least {minimum_clips} approved clips"]
    unique_sources = {clip.source_url for clip in parsed}
    if len(unique_sources) < minimum_clips:
        return PROFILE_UNVERIFIED, ["requires independent approved source clips"]
    if canonical_3d_verified:
        return PROFILE_VERIFIED_3D, []
    return PROFILE_VERIFIED_2D, []


def is_matchable(status: str) -> bool:
    return str(status or "") in MATCHABLE_STATUSES


def is_verified_3d(status: str) -> bool:
    return str(status or "") in THREED_STATUSES


def normalize_profile_status(value: str | None) -> str:
    status = str(value or PROFILE_UNVERIFIED).strip().lower()
    return status if status in PROFILE_STATUSES else PROFILE_UNVERIFIED


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_ratio(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if 0.0 <= parsed <= 1.0 else None
