"""Deterministic quality gate for learned player motion profiles."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.angles import AngleSnapshot, angles_plausible  # noqa: E402
from app.motion import motion_distance, timeline_quality  # noqa: E402
from scripts.youtube_profile import release_angles_plausible  # noqa: E402

MODEL_PATH = ROOT / "models" / "nba_player_models.json"
CATALOG_PATH = ROOT / "models" / "youtube_candidate_catalog.json"
REPORT_PATH = ROOT / "models" / "data_quality_report.json"


def _valid_angles(angles: dict) -> bool:
    try:
        return angles_plausible(
            AngleSnapshot(
                elbow=float(angles["elbow"]), shoulder=float(angles["shoulder"]),
                hip=float(angles["hip"]), knee=float(angles["knee"]),
                hand="right", space="3d",
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def validate(min_clips: int = 5) -> dict:
    models = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8")) if CATALOG_PATH.exists() else {}
    failures = []
    warnings = []
    players = []
    seen_urls = {}
    prototypes = {}
    all_holdout_scores = []

    for name, entry in sorted(models.items()):
        meta = entry.get("meta") or {}
        primary_view = str(meta.get("primary_view") or "side")
        block = (entry.get("views") or {}).get(primary_view) or {}
        clips = block.get("clips") or []
        timeline = block.get("timeline") or {}
        samples = timeline.get("samples") or []
        source_count = int(timeline.get("source_count") or 0)
        row_failures = []
        if meta.get("profile_version") != "motion_v2":
            row_failures.append("legacy profile")
        if int(entry.get("sample_count") or 0) < min_clips:
            row_failures.append(f"fewer than {min_clips} accepted clips")
        if not _valid_angles(block.get("angles") or {}):
            row_failures.append("invalid release-angle profile")
        if len(samples) != 48 or any(len(sample.get("pose") or []) != 33 for sample in samples):
            row_failures.append("prototype is not 48 frames x 33 landmarks")
        if source_count < min(2, min_clips):
            row_failures.append("prototype was learned from fewer than 2 same-view clips")
        prototype_quality = timeline_quality(samples, fps=float(timeline.get("fps") or 30.0)) if samples else {"valid": False, "score": 0}
        if samples and not prototype_quality.get("valid"):
            row_failures.append("prototype continuity gate failed")
        bad_clips = 0
        holdout_scores = []
        for clip in clips:
            quality = clip.get("quality") or {}
            if not quality.get("valid") or float(quality.get("score") or 0) < 45:
                bad_clips += 1
            if not release_angles_plausible(clip.get("angles") or {}):
                row_failures.append("clip has implausible release angles")
            url = str(clip.get("youtube_url") or "")
            if url:
                previous = seen_urls.get(url)
                if previous and previous != name:
                    failures.append(f"duplicate source across players: {previous} / {name} / {url}")
                seen_urls[url] = name
            if clip.get("holdout_motion_score") is not None:
                score = float(clip["holdout_motion_score"])
                holdout_scores.append(score)
                all_holdout_scores.append(score)
        if bad_clips:
            row_failures.append(f"{bad_clips} clips failed stored quality gate")
        if len(clips) >= 3 and len(holdout_scores) != len(clips):
            row_failures.append("missing leave-one-video-out scores")
        if holdout_scores and statistics.median(holdout_scores) < 12.0:
            row_failures.append("leave-one-video-out median score below 12")
        if samples:
            prototypes[name] = samples
        failures.extend(f"{name}: {message}" for message in row_failures)
        players.append(
            {
                "display_name": name,
                "profile_version": meta.get("profile_version") or "legacy",
                "primary_view": primary_view,
                "accepted_clips": len(clips),
                "candidate_clips": len(catalog.get(name) or []),
                "prototype_frames": len(samples),
                "prototype_source_count": source_count,
                "prototype_quality": prototype_quality,
                "holdout_score_median": round(statistics.median(holdout_scores), 2) if holdout_scores else None,
                "failures": row_failures,
            }
        )

    pairwise = []
    names = sorted(prototypes)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            distance, coverage = motion_distance(prototypes[left_name], prototypes[right_name])
            pairwise.append(
                {"left": left_name, "right": right_name, "distance": round(distance, 5), "coverage": round(coverage, 3)}
            )
    pairwise.sort(key=lambda row: row["distance"])
    if pairwise and pairwise[0]["distance"] < 0.01:
        warnings.append("Two player prototypes are nearly identical; inspect closest_pairwise_profiles.")

    candidate_count = len({item.get("video_id") for items in catalog.values() for item in items if item.get("video_id")})
    if candidate_count < max(100, len(models) * min_clips * 2):
        failures.append("candidate catalog is too small for the requested acceptance target")

    report = {
        "schema_version": "motion_dataset_validation_v1",
        "passed": not failures,
        "criteria": {
            "minimum_accepted_clips_per_player": min_clips,
            "profile_version": "motion_v2",
            "prototype_shape": "48 frames x 33 landmarks",
            "minimum_clip_quality": 45,
            "minimum_same_view_prototype_sources": min(2, min_clips),
            "minimum_holdout_median_score": 12,
            "minimum_candidate_pool": max(100, len(models) * min_clips * 2),
        },
        "summary": {
            "players": len(models),
            "motion_v2_players": sum(row["profile_version"] == "motion_v2" for row in players),
            "accepted_clips": sum(row["accepted_clips"] for row in players),
            "unique_candidate_videos": candidate_count,
            "unique_accepted_videos": len(seen_urls),
            "holdout_score_median": round(statistics.median(all_holdout_scores), 2) if all_holdout_scores else None,
        },
        "players": players,
        "closest_pairwise_profiles": pairwise[:10],
        "warnings": warnings,
        "failures": failures,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-clips", type=int, default=5)
    args = parser.parse_args()
    report = validate(min_clips=max(1, args.min_clips))
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    if report["failures"]:
        print("\nFAILURES")
        for failure in report["failures"]:
            print(f"- {failure}")
        raise SystemExit(1)
    print(f"\nPASS -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
