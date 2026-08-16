"""Build a validated player canonical motion from independent views.

This is an offline data tool. It never modifies nba_player_models.json and
SciPy is deliberately not imported by the serving path.

The Curry CLI is retained for backwards compatibility; the reusable builder
is shared by the Paris 2024 USA batch.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction3d import (  # noqa: E402
    CANONICAL_MODEL_NAME,
    MIN_EFFECTIVE_CLIP_COVERAGE,
    OptimizationConfig,
    build_model_entry,
    filter_observations,
    filter_cross_clip_residuals,
    clip_observation_coverage,
    initialize_xyz,
    major_bone_lengths,
    normalize_clip_observation,
    observation_dispersion,
    optimize_canonical_motion,
    validate_canonical_motion,
    write_validated_model,
)
from scripts.youtube_profile import (  # noqa: E402
    analyze_best,
    download_clip,
    load_json,
    youtube_id,
)


CURRY_KEY = "stephen_curry"
CURRY_NAME = "Stephen Curry"
DEFAULT_FRONT_URLS = (
    "https://www.youtube.com/watch?v=eETSPnYf85E",
    "https://www.youtube.com/shorts/VN1J6EgUQvg",
    "https://www.youtube.com/shorts/iQVNOyQES0A",
    "https://www.youtube.com/shorts/W3pFskLg04A",
)


def player_identity_matches(player_name: str, title: str) -> bool:
    """Require the distinctive player surname in source metadata."""
    expected = str(player_name).casefold().replace("’", "'").split()
    candidate = str(title).casefold().replace("’", "'")
    if not expected or not candidate:
        return False
    surname = expected[-1]
    return surname in re.findall(r"[a-z0-9]+", candidate)


def current_view_urls(
    data: dict, player_name: str, view: str
) -> tuple[str, ...]:
    clips = (
        (((data.get(player_name) or {}).get("views") or {}).get(view) or {}).get(
            "clips"
        )
        or []
    )
    return tuple(
        dict.fromkeys(
            str(clip.get("youtube_url") or "")
            for clip in clips
            if clip.get("youtube_url")
            and player_identity_matches(
                player_name, str(clip.get("title") or "")
            )
        )
    )


def current_side_urls(data: dict) -> tuple[str, ...]:
    """Backwards-compatible Curry source helper."""
    return current_view_urls(data, CURRY_NAME, "side")


def _unique_urls(urls: Sequence[str]) -> tuple[str, ...]:
    unique: dict[str, str] = {}
    for url in urls:
        clean = str(url).strip()
        if not clean:
            continue
        unique.setdefault(youtube_id(clean), clean)
    return tuple(unique.values())


def _minimum_three(value: str) -> int:
    parsed = int(value)
    if parsed < 3:
        raise argparse.ArgumentTypeError("min-per-view must be at least 3")
    return parsed


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_player_canonical_model(
    *,
    player_key: str,
    player_name: str,
    front_urls: Sequence[str],
    side_urls: Sequence[str],
    min_per_view: int,
    max_frames: int,
    no_check_certificates: bool,
    output_path: Path,
    max_repeated_failures: int = 3,
) -> dict:
    if min_per_view < 3:
        raise ValueError("canonical reconstruction requires 3 clips per view")
    candidates = {
        "front": _unique_urls(front_urls),
        "side": _unique_urls(side_urls),
    }
    accepted = []
    rejected: list[dict] = []
    provenance: list[dict] = []
    repeated_failures: dict[tuple[str, str], int] = {}

    with tempfile.TemporaryDirectory(
        prefix=f"{player_key}_canonical3d_"
    ) as temporary:
        root = Path(temporary)
        for view in ("front", "side"):
            for source_index, url in enumerate(candidates[view]):
                record = {"view": view, "youtube_url": url}
                try:
                    clip_path = download_clip(
                        url,
                        root / view / f"{source_index}_{youtube_id(url)}",
                        no_check_certificates=no_check_certificates,
                    )
                    _angles, metadata = analyze_best(
                        clip_path,
                        view_tag=view,
                        hand="right",
                        max_frames=max_frames,
                        capture_observations=True,
                    )
                    detected_view = str(metadata.get("view") or "")
                    if detected_view != view:
                        raise ValueError(
                            f"assigned {view}, pose geometry classified {detected_view}"
                        )
                    raw_timeline = metadata.get("raw_timeline") or []
                    if len(raw_timeline) < 2:
                        raise ValueError("analysis produced no complete raw shot timeline")
                    quality = float((metadata.get("quality") or {}).get("score") or 0.0)
                    observation = normalize_clip_observation(
                        clip_id=youtube_id(url),
                        samples=raw_timeline,
                        view=view,
                        quality=quality,
                        source_url=url,
                    )
                    accepted.append(observation)
                    record.update(
                        {
                            "status": "accepted",
                            "detected_view": detected_view,
                            "quality": quality,
                            "raw_frames": len(raw_timeline),
                            "release_frame_index": metadata.get(
                                "release_frame_index"
                            ),
                        }
                    )
                    provenance.append(record)
                    print(
                        f"[{view}] accepted {url} quality={quality:.1f} "
                        f"frames={len(raw_timeline)}"
                    )
                except Exception as exc:
                    signature = f"{type(exc).__name__}: {exc}"
                    record.update(
                        {
                            "status": "rejected",
                            "reason": signature,
                        }
                    )
                    rejected.append(record)
                    provenance.append(record)
                    print(f"[{view}] rejected {url}: {record['reason']}")
                    key = (view, signature)
                    repeated_failures[key] = repeated_failures.get(key, 0) + 1
                    if repeated_failures[key] >= max(
                        1, int(max_repeated_failures)
                    ):
                        provenance.append(
                            {
                                "view": view,
                                "status": "aborted_repeated_failure",
                                "reason": signature,
                                "remaining_candidates": len(candidates[view])
                                - source_index
                                - 1,
                            }
                        )
                        print(
                            f"[{view}] aborting after repeated failure: "
                            f"{signature}"
                        )
                        break

    view_counts = {
        view: sum(observation.view == view for observation in accepted)
        for view in ("front", "side")
    }
    report: dict = {
        "player_key": player_key,
        "display_name": player_name,
        "model": CANONICAL_MODEL_NAME,
        "candidate_counts": {
            view: len(candidates[view]) for view in ("front", "side")
        },
        "accepted_counts": view_counts,
        "sources": provenance,
        "rejected_count": len(rejected),
        "output": str(output_path),
    }
    if any(view_counts[view] < min_per_view for view in ("front", "side")):
        report.update(
            {
                "passed": False,
                "reasons": [
                    f"requires {min_per_view} accepted front and side clips"
                ],
            }
        )
        return report

    accepted = filter_observations(accepted)
    initial, confidence = initialize_xyz(accepted)
    accepted = filter_cross_clip_residuals(accepted, initial)
    effective = []
    for observation in accepted:
        coverage = clip_observation_coverage(observation)
        record = next(
            (
                item
                for item in provenance
                if youtube_id(str(item.get("youtube_url") or ""))
                == observation.clip_id
            ),
            None,
        )
        if record is not None:
            record["post_filter_coverage"] = round(coverage, 4)
        if coverage >= MIN_EFFECTIVE_CLIP_COVERAGE:
            effective.append(observation)
        elif record is not None:
            record["status"] = "rejected_post_filter"
            record["reason"] = (
                f"coverage {coverage:.3f} below "
                f"{MIN_EFFECTIVE_CLIP_COVERAGE:.3f}"
            )
    accepted = effective
    view_counts = {
        view: sum(observation.view == view for observation in accepted)
        for view in ("front", "side")
    }
    report["accepted_counts"] = view_counts
    report["rejected_count"] = sum(
        item.get("status") != "accepted" for item in provenance
    )
    if any(view_counts[view] < min_per_view for view in ("front", "side")):
        report.update(
            {
                "passed": False,
                "reasons": [
                    f"requires {min_per_view} effective front and side clips"
                ],
            }
        )
        return report
    initial, confidence = initialize_xyz(accepted)
    config = OptimizationConfig()
    optimized, optimizer_report = optimize_canonical_motion(
        accepted, initial, confidence, config
    )
    validation = validate_canonical_motion(
        optimized, accepted, optimizer_report, confidence
    )
    dispersion = observation_dispersion(accepted, optimized)
    lengths = major_bone_lengths(optimized, confidence)
    phases = accepted[0].phases
    entry = build_model_entry(
        player_key=player_key,
        xyz=optimized,
        dispersion=dispersion,
        confidence=confidence,
        phases=phases,
        clips=accepted,
        bone_lengths=lengths,
        optimizer={"config": asdict(config), "report": asdict(optimizer_report)},
        validation=validation,
        duration=float(np.median([clip.duration for clip in accepted])),
        provenance=provenance,
    )
    report.update(
        {
            "passed": bool(validation["passed"]),
            "reasons": validation["reasons"],
            "optimizer": asdict(optimizer_report),
            "validation": validation,
        }
    )
    if validation["passed"]:
        write_validated_model(output_path, entry)
    return report


def build_curry_canonical_model(
    *,
    front_urls: Sequence[str],
    side_urls: Sequence[str],
    min_per_view: int,
    max_frames: int,
    no_check_certificates: bool,
    output_path: Path,
) -> dict:
    """Backwards-compatible Curry wrapper for existing automation."""
    return build_player_canonical_model(
        player_key=CURRY_KEY,
        player_name=CURRY_NAME,
        front_urls=front_urls,
        side_urls=side_urls,
        min_per_view=min_per_view,
        max_frames=max_frames,
        no_check_certificates=no_check_certificates,
        output_path=output_path,
    )


def _resolved_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Curry's canonical 3D motion from front and side clips"
    )
    parser.add_argument("--front-url", action="append", default=[])
    parser.add_argument("--side-url", action="append", default=[])
    parser.add_argument("--min-per-view", type=_minimum_three, default=3)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--no-check-certificates", action="store_true")
    parser.add_argument(
        "--output", default="models/canonical_3d_models.json"
    )
    parser.add_argument(
        "--report", default="models/curry_canonical3d_validation.json"
    )
    args = parser.parse_args()

    output_path = _resolved_path(args.output)
    report_path = _resolved_path(args.report)
    report: dict
    try:
        data = load_json()
        front_urls = _unique_urls([*args.front_url, *DEFAULT_FRONT_URLS])
        side_urls = _unique_urls([*args.side_url, *current_side_urls(data)])
        report = build_curry_canonical_model(
            front_urls=front_urls,
            side_urls=side_urls,
            min_per_view=args.min_per_view,
            max_frames=max(60, int(args.max_frames)),
            no_check_certificates=bool(args.no_check_certificates),
            output_path=output_path,
        )
    except Exception as exc:
        report = {
            "player_key": CURRY_KEY,
            "passed": False,
            "reasons": [f"{type(exc).__name__}: {exc}"],
            "output": str(output_path),
        }
    _write_report(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report.get("passed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
