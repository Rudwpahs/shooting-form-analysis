"""Capture-session synchronization checks for multi-view basketball analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class SyncOffset:
    camera_id: str
    offset_frames: int
    confidence: float
    method: str

    @classmethod
    def from_mapping(cls, camera_id: str, data: Mapping[str, Any]) -> "SyncOffset":
        return cls(
            camera_id=str(camera_id),
            offset_frames=int(data.get("offset_frames", 0)),
            confidence=float(data.get("confidence", 0.0)),
            method=str(data.get("method") or "unknown"),
        )


def validate_sync_payload(
    payload: Mapping[str, Any] | None,
    *,
    required_camera_ids: set[str] | None = None,
    max_offset_frames: int = 2,
    min_confidence: float = 0.8,
) -> dict:
    """Require explicit synchronization evidence before geometrical 3D use."""
    if not payload:
        return {
            "status": "not_provided",
            "valid": False,
            "offsets": {},
            "reasons": ["synchronization payload was not provided"],
        }
    raw_offsets = payload.get("offsets") or {}
    if not isinstance(raw_offsets, Mapping):
        return {
            "status": "rejected",
            "valid": False,
            "offsets": {},
            "reasons": ["offsets must be an object"],
        }
    parsed = {}
    reasons: list[str] = []
    for camera_id, data in raw_offsets.items():
        try:
            item = SyncOffset.from_mapping(str(camera_id), data or {})
        except (TypeError, ValueError) as exc:
            reasons.append(f"{camera_id}: {exc}")
            continue
        parsed[item.camera_id] = {
            "offset_frames": item.offset_frames,
            "confidence": item.confidence,
            "method": item.method,
        }
        if abs(item.offset_frames) > max_offset_frames:
            reasons.append(f"{item.camera_id}: offset exceeds {max_offset_frames} frames")
        if item.confidence < min_confidence:
            reasons.append(f"{item.camera_id}: synchronization confidence below {min_confidence:.2f}")
        if item.method not in {"audio_clap", "led_flash", "timestamp", "manual_verified"}:
            reasons.append(f"{item.camera_id}: unsupported synchronization method")
    expected = required_camera_ids or set(parsed)
    missing = sorted(expected - set(parsed))
    if missing:
        reasons.append("missing synchronization offsets for " + ", ".join(missing))
    if len(parsed) < 2:
        reasons.append("requires offsets for at least two cameras")
    return {
        "status": "verified" if not reasons else "rejected",
        "valid": not reasons,
        "offsets": parsed,
        "max_offset_frames": max_offset_frames,
        "min_confidence": min_confidence,
        "reasons": reasons,
    }
