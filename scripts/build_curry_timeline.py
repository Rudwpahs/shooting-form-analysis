"""Store Curry joint angles from catch through follow-through."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analyze import analyze_view  # noqa: E402
from app.db import connect, ensure_seeded, upsert_timeline  # noqa: E402
from scripts.youtube_profile import MODELS_JSON, download_clip, load_json  # noqa: E402

CURRY_NAME = "Stephen Curry"


def curry_clip_jobs(data: dict) -> list[dict]:
    views = ((data.get(CURRY_NAME) or {}).get("views") or {})
    jobs = []
    for view_name in ("side", "front"):
        block = views.get(view_name) or {}
        clips = block.get("clips") or []
        if not clips:
            continue
        best = max(clips, key=lambda c: float(c.get("score") or 0))
        url = best.get("youtube_url")
        if not url:
            continue
        jobs.append(
            {
                "view": view_name,
                "url": url,
                "person_index": int(best.get("person_index") or 0),
            }
        )
    return jobs


def main() -> None:
    data = load_json()
    jobs = curry_clip_jobs(data)
    if not jobs:
        raise SystemExit("No Curry clip URLs found in nba_player_models.json")

    entry = data.setdefault(CURRY_NAME, {})
    views = entry.setdefault("views", {})

    with tempfile.TemporaryDirectory(prefix="curry_timeline_") as tmpdir:
        tmp = Path(tmpdir)
        for job in jobs:
            view_name = job["view"]
            url = job["url"]
            print(f"[{view_name}] download {url}")
            clip = download_clip(url, tmp / view_name)
            result = analyze_view(
                clip,
                view=view_name,
                person_index=job["person_index"],
                hand="right",
                max_frames=400,
                after_release_sec=0.7,
            )
            if result.error or not result.timeline:
                print(f"  skip: {result.error or 'empty timeline'}")
                continue

            phases = {
                "catch": result.catch_frame_index,
                "dip": result.dip_frame_index,
                "release": result.release_frame_index,
                "follow_through": result.followthrough_frame_index,
            }
            duration = float(result.timeline[-1]["t"]) if result.timeline else 0.0
            timeline = {
                "t0": "catch",
                "fps": round(float(result.fps), 3),
                "phases": phases,
                "youtube_url": url,
                "person_index": job["person_index"],
                "samples": result.timeline,
            }
            block = views.setdefault(view_name, {})
            block["timeline"] = timeline
            print(
                f"  fps={result.fps:.2f} catch={phases['catch']} dip={phases['dip']} "
                f"release={phases['release']} follow_through={phases['follow_through']} "
                f"samples={len(result.timeline)} t=0..{duration:.3f}s"
            )

    MODELS_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")

    conn = connect()
    try:
        for job in jobs:
            view_name = job["view"]
            timeline = ((views.get(view_name) or {}).get("timeline") or {})
            samples = timeline.get("samples") or []
            if not samples:
                continue
            upsert_timeline(
                conn,
                "stephen_curry",
                view_name,
                float(timeline.get("fps") or 30),
                samples,
                t0="catch",
                after_sec=float((samples[-1] or {}).get("t") or 0),
                youtube_url=str(timeline.get("youtube_url") or ""),
            )
    finally:
        conn.close()

    ensure_seeded()
    print(f"Saved Curry timelines to {MODELS_JSON}")


if __name__ == "__main__":
    main()
