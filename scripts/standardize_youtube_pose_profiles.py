"""Standardize reviewed YouTube single-view pose profiles for the initial roster.

This never converts a camera-relative pose estimate into a calibrated metric 3D
model. It only attaches source-review evidence and clear model boundaries to
the existing player pose profiles.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODELS_PATH = ROOT / "models" / "nba_player_models.json"
REVIEWS_DIR = ROOT / "artifacts" / "youtube_visual_reviews"
OUTPUT_PATH = ROOT / "models" / "youtube_single_view_pose_profiles.json"

FIELDS = (
    "VERDICT",
    "PLAYER_MATCH",
    "REAL_SHOT",
    "BALL_HAND_RELEASE_VISIBLE",
    "FULL_BODY_VISIBLE",
    "CAMERA_VIEW",
    "POSE_EXTRACTION",
    "EVIDENCE",
)
REVIEW_CANDIDATE_OVERRIDES = {"anthony_edwards": 2}


def extract_fields(text: str) -> dict[str, str]:
    values = {}
    for field in FIELDS:
        matches = re.findall(rf"^{field}:\s*(.+)$", text, flags=re.MULTILINE)
        values[field.lower()] = matches[-1].strip() if matches else "UNAVAILABLE"
    return values


def chosen_clip(profile: dict) -> dict:
    clips = [
        clip
        for view in (profile.get("views") or {}).values()
        for clip in (view.get("clips") or [])
        if clip.get("youtube_url")
    ]
    return max(clips, key=lambda clip: (float((clip.get("quality") or {}).get("score") or 0), float(clip.get("metadata_score") or 0)))


def main() -> int:
    models = json.loads(MODELS_PATH.read_text(encoding="utf-8"))
    profiles: list[dict] = []
    for display_name, profile in models.items():
        meta = profile.setdefault("meta", {})
        player_key = str(meta.get("player_key") or display_name.lower().replace(" ", "_"))
        candidate_index = REVIEW_CANDIDATE_OVERRIDES.get(player_key, 0)
        suffix = "" if candidate_index == 0 else f"_candidate_{candidate_index}"
        review_file = REVIEWS_DIR / f"{player_key}{suffix}.md"
        evidence = extract_fields(review_file.read_text(encoding="utf-8")) if review_file.exists() else {}
        clips = sorted(
            [
                clip
                for view in (profile.get("views") or {}).values()
                for clip in (view.get("clips") or [])
                if clip.get("youtube_url")
            ],
            key=lambda clip: (float((clip.get("quality") or {}).get("score") or 0), float(clip.get("metadata_score") or 0)),
            reverse=True,
        )
        clip = clips[candidate_index]
        accepted = all(
            evidence.get(key) == expected
            for key, expected in (
                ("verdict", "ACCEPT"),
                ("player_match", "YES"),
                ("real_shot", "YES"),
                ("ball_hand_release_visible", "YES"),
                ("full_body_visible", "YES"),
            )
        ) and evidence.get("pose_extraction") in {"suitable", "limited"}
        model_status = "youtube_pose_candidate" if accepted else "youtube_pose_rejected"
        boundary = {
            "model_kind": "single_view_youtube_3d_pose_estimate",
            "coordinate_space": "camera_relative_pose_estimate",
            "calibration_status": "not_available",
            "metric_3d_status": "not_available",
            "publication_rule": "Do not label as calibrated, metric, or verified_3d.",
        }
        meta["model_kind"] = boundary["model_kind"]
        meta["model_status"] = model_status
        meta["model_boundary"] = boundary
        meta["youtube_review"] = {
            "source_url": clip["youtube_url"],
            "source_title": clip.get("title", ""),
            "review_file": str(review_file.relative_to(ROOT)),
            **evidence,
        }
        profiles.append(
            {
                "player_key": player_key,
                "display_name": display_name,
                "model_status": model_status,
                "sample_count": profile.get("sample_count", 0),
                "metrics": profile.get("metrics", {}),
                "source_url": clip["youtube_url"],
                "source_title": clip.get("title", ""),
                "review": evidence,
                "model_boundary": boundary,
            }
        )

    MODELS_PATH.write_text(json.dumps(models, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "schema_version": "youtube-pose-v1",
                "profile_count": len(profiles),
                "candidate_count": sum(item["model_status"] == "youtube_pose_candidate" for item in profiles),
                "rejected_count": sum(item["model_status"] == "youtube_pose_rejected" for item in profiles),
                "profiles": profiles,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"profiles": len(profiles), "candidates": sum(item["model_status"] == "youtube_pose_candidate" for item in profiles)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
