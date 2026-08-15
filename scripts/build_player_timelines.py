"""Build validated catch-to-follow-through timelines for every roster player."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analyze import analyze_view, pick_best_person  # noqa: E402
from app.db import connect, ensure_seeded, upsert_timeline  # noqa: E402
from scripts.youtube_profile import (  # noqa: E402
    MODELS_JSON,
    download_clip,
    load_json,
    player_key_from_name,
    save_json,
    youtube_id,
)


def candidate_clips(entry: dict) -> list[tuple[str, dict]]:
    """Prefer side-view clips, then other scored clips, with duplicate videos removed."""
    views = entry.get("views") or {}
    out: list[tuple[str, dict]] = []
    seen: set[str] = set()

    def add(view_name: str, clip: dict) -> None:
        url = clip.get("youtube_url")
        if not url:
            return
        video_id = youtube_id(url)
        if video_id in seen:
            return
        seen.add(video_id)
        out.append((view_name, clip))

    for clip in (views.get("side") or {}).get("clips") or []:
        add("side", clip)
    for view_name in ("front", "merged", "oblique"):
        for clip in (views.get(view_name) or {}).get("clips") or []:
            add("side" if view_name == "merged" else view_name, clip)

    out.sort(
        key=lambda item: (
            item[0] == "side",
            float(item[1].get("score") or 0),
        ),
        reverse=True,
    )
    return out[:5]


def build_one(name: str, entry: dict, tmp: Path) -> dict | None:
    candidates = candidate_clips(entry)
    if not candidates:
        print("  SKIP no clips")
        return None

    for view_name, clip in candidates:
        url = str(clip["youtube_url"])
        print(f"  try [{view_name}] {url}")
        try:
            video_path = download_clip(url, tmp / youtube_id(url))
        except Exception as exc:
            print(f"    download fail: {exc}")
            continue

        person = int(clip.get("person_index") or 0)
        result = analyze_view(
            video_path,
            view=view_name,
            person_index=person,
            hand="right",
            max_frames=400,
            after_release_sec=0.7,
        )
        if result.error or not result.timeline:
            person = pick_best_person(video_path, view="side", hand="right", max_frames=220)
            result = analyze_view(
                video_path,
                view="side",
                person_index=person,
                hand="right",
                max_frames=400,
                after_release_sec=0.7,
            )
        if result.error or not result.timeline:
            print(f"    analyze fail: {result.error or 'timeline rejected'}")
            continue

        store_view = "side" if view_name == "merged" else view_name
        duration = float(result.timeline[-1]["t"])
        timeline = {
            "t0": "catch",
            "fps": round(float(result.fps), 3),
            "phases": {
                "catch": result.catch_frame_index,
                "dip": result.dip_frame_index,
                "release": result.release_frame_index,
                "follow_through": result.followthrough_frame_index,
            },
            "youtube_url": url,
            "person_index": person,
            "samples": result.timeline,
            "after_sec": duration,
        }
        print(
            f"    ok person={person} samples={len(result.timeline)} "
            f"duration={duration:.3f}s catch={result.catch_frame_index} "
            f"release={result.release_frame_index} ft={result.followthrough_frame_index}"
        )
        return {"view": store_view, "timeline": timeline}
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", action="append", default=[], help="Only these display names")
    parser.add_argument("--skip-existing", action="store_true", help="Skip players with a timeline")
    parser.add_argument("--include-curry", action="store_true", help="Also rebuild Stephen Curry")
    args = parser.parse_args()

    data = load_json()
    names = args.player or sorted(data)
    if not args.include_curry and not args.player:
        names = [name for name in names if name != "Stephen Curry"]

    if not args.skip_existing:
        for name in names:
            entry = data.get(name)
            if not isinstance(entry, dict):
                continue
            for block in (entry.get("views") or {}).values():
                if isinstance(block, dict):
                    block.pop("timeline", None)
        save_json(data)

    ok = 0
    failed = 0
    with tempfile.TemporaryDirectory(prefix="player_timelines_") as tmpdir:
        tmp = Path(tmpdir)
        for name in names:
            entry = data.get(name)
            print(f"\n=== {name}")
            if not isinstance(entry, dict):
                print("  SKIP missing profile")
                failed += 1
                continue

            views = entry.setdefault("views", {})
            has_timeline = any(
                isinstance((views.get(view) or {}).get("timeline"), dict)
                for view in views
            )
            if args.skip_existing and has_timeline:
                print("  skip existing timeline")
                ok += 1
                continue

            built = build_one(name, entry, tmp)
            if not built:
                print("  WARNING no valid timeline")
                failed += 1
                continue

            view_name = str(built["view"])
            timeline = built["timeline"]
            views.setdefault(view_name, {})["timeline"] = timeline
            save_json(data)

            key = str((entry.get("meta") or {}).get("player_key") or player_key_from_name(name))
            samples = timeline["samples"]
            conn = connect()
            try:
                upsert_timeline(
                    conn,
                    key,
                    view_name,
                    float(timeline["fps"]),
                    samples,
                    t0="catch",
                    after_sec=float(timeline["after_sec"]),
                    youtube_url=str(timeline["youtube_url"]),
                )
            finally:
                conn.close()
            ok += 1

    save_json(data)
    ensure_seeded()
    print(f"\nDone ok={ok} failed={failed} saved={MODELS_JSON}")


if __name__ == "__main__":
    main()
