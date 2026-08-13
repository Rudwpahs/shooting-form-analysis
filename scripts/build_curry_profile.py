"""Build player angle profile from YouTube clip(s)."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.youtube_profile import (  # noqa: E402
    MODELS_JSON,
    analyze_best,
    download_clip,
    load_json,
    median_angles,
    persist_player_profile,
    release_score,
)

DEFAULT_URLS = [
    "https://www.youtube.com/watch?v=xIDwiuxU_8A",
    "https://www.youtube.com/shorts/iQVNOyQES0A",
]

CURRY_NAME = "Stephen Curry"
CURRY_KEY = "stephen_curry"


def persist_curry_view(
    view_tag: str,
    angles: dict,
    *,
    space: str,
    source: str,
    clips: List[dict],
) -> None:
    from scripts.youtube_profile import save_view_to_db

    data = load_json()
    entry = data.setdefault(CURRY_NAME, {"sample_count": 0, "metrics": {}, "meta": {}, "views": {}})
    views = entry.setdefault("views", {})
    views[view_tag] = {
        "angles": angles,
        "space": space,
        "source": source,
        "clips": clips,
    }
    entry["sample_count"] = sum(len(v.get("clips") or [1]) for v in views.values())
    if view_tag == "merged" or not entry.get("metrics"):
        entry["metrics"] = {
            "Elbow angle": angles["elbow"],
            "Shoulder angle": angles["shoulder"],
            "Hip angle": angles["hip"],
            "Knee angle": angles["knee"],
        }
        entry["meta"] = {
            "space": space,
            "hand": "right",
            "source": source,
            "primary_view": view_tag,
            "player_key": CURRY_KEY,
        }
    MODELS_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    save_view_to_db(CURRY_KEY, CURRY_NAME, angles, view=view_tag, space=space, source=source)


def main():
    parser = argparse.ArgumentParser(description="Build Curry profile from YouTube")
    parser.add_argument("--url", action="append", dest="urls", help="YouTube URL (repeatable)")
    parser.add_argument("--view", default="side", choices=["front", "side", "oblique", "merged"])
    parser.add_argument("--hand", default="right", choices=["right", "left"])
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--aggregate", default="median", choices=["median", "best"])
    args = parser.parse_args()

    urls = args.urls or DEFAULT_URLS
    collected: List[dict] = []
    clip_meta: List[dict] = []

    with tempfile.TemporaryDirectory(prefix="curry_build_") as tmp:
        for i, url in enumerate(urls):
            slug = url.split("/")[-1][:12].replace("?", "_")
            try:
                clip = download_clip(url, Path(tmp) / f"{i}_{slug}")
                print(f"Downloaded: {clip}")
                angles, meta = analyze_best(
                    clip,
                    view_tag=args.view,
                    hand=args.hand,
                    max_frames=args.max_frames,
                )
                score = release_score(angles)
                print(f"  score={score:.1f} person=#{meta['person_index']} frame={meta['release_frame_index']}")
                print(f"  angles={json.dumps(angles)}")
                collected.append(angles)
                clip_meta.append({"youtube_url": url, "score": score, "angles": angles, **meta})
            except Exception as exc:
                print(f"Skip {url}: {exc}")

    if not collected:
        raise SystemExit("No clips produced usable angles.")

    if args.aggregate == "best":
        best_i = max(range(len(clip_meta)), key=lambda i: clip_meta[i]["score"])
        final = collected[best_i]
        space = clip_meta[best_i].get("space") or "3d"
        clips_out = [clip_meta[best_i]]
    else:
        final = median_angles(collected)
        space = clip_meta[0].get("space") or "3d"
        clips_out = clip_meta

    print(f"\nFinal [{args.view}] ({args.aggregate}):", json.dumps(final, indent=2))
    persist_curry_view(
        args.view,
        final,
        space=space,
        source="youtube_self_measured",
        clips=clips_out,
    )
    print(f"Saved view '{args.view}' to {MODELS_JSON} and SQLite")


if __name__ == "__main__":
    main()
