"""Discover YouTube shooting clips for 2026 All-Star shooters and build angle profiles."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.youtube_profile import (  # noqa: E402
    MODELS_JSON,
    analyze_best,
    download_clip,
    persist_player_profile,
    release_score,
    search_player_clip_urls,
    youtube_id,
)

ALLSTAR_SHOOTERS = [
    {"display_name": "Devin Booker", "search_name": "Devin Booker"},
    {"display_name": "Kevin Durant", "search_name": "Kevin Durant"},
    {"display_name": "Donovan Mitchell", "search_name": "Donovan Mitchell"},
    {"display_name": "Anthony Edwards", "search_name": "Anthony Edwards"},
    {"display_name": "Tyrese Maxey", "search_name": "Tyrese Maxey"},
    {"display_name": "Luka Dončić", "search_name": "Luka Doncic"},
    {"display_name": "Jamal Murray", "search_name": "Jamal Murray"},
    {"display_name": "Jalen Brunson", "search_name": "Jalen Brunson"},
    {"display_name": "Jaylen Brown", "search_name": "Jaylen Brown"},
    {"display_name": "Kawhi Leonard", "search_name": "Kawhi Leonard"},
    {"display_name": "Norman Powell", "search_name": "Norman Powell"},
    {"display_name": "De'Aaron Fox", "search_name": "DeAaron Fox"},
    {"display_name": "Shai Gilgeous-Alexander", "search_name": "Shai Gilgeous-Alexander"},
    {"display_name": "LeBron James", "search_name": "LeBron James"},
    {"display_name": "Victor Wembanyama", "search_name": "Victor Wembanyama"},
]

TARGET_CLIPS = 5
VIEW_TAG = "side"


def collect_clips_for_player(
    search_name: str,
    tmp: Path,
    *,
    target: int = TARGET_CLIPS,
    max_attempts: int = 25,
) -> List[dict]:
    urls = search_player_clip_urls(search_name)
    print(f"  found {len(urls)} candidate URLs")
    records: List[dict] = []
    tried: set[str] = set()

    for url in urls:
        if len(records) >= target:
            break
        vid = youtube_id(url)
        if vid in tried:
            continue
        tried.add(vid)
        if len(tried) > max_attempts:
            break

        slug = vid or url.split("/")[-1][:12]
        print(f"  [{len(records)+1}/{target}] trying {url}")
        try:
            clip = download_clip(url, tmp / slug)
            angles, meta = analyze_best(clip, view_tag=VIEW_TAG, max_frames=240)
            score = release_score(angles)
            if score <= 0:
                print(f"    low score ({score:.1f}), skip")
                continue
            record = {
                "youtube_url": url,
                "score": score,
                "angles": angles,
                **meta,
            }
            records.append(record)
            print(f"    ok score={score:.1f} angles={json.dumps(angles)}")
        except Exception as exc:
            print(f"    skip: {exc}")

    return records


def process_player(entry: dict, tmp: Path, *, target: int) -> Optional[dict]:
    display_name = entry["display_name"]
    search_name = entry["search_name"]
    from scripts.youtube_profile import player_key_from_name

    player_key = player_key_from_name(display_name)
    print(f"\n{'='*60}\n{display_name} ({player_key})\n{'='*60}")

    records = collect_clips_for_player(search_name, tmp, target=target)
    if not records:
        print(f"  WARNING: no usable clips for {display_name}")
        return None

    merged = persist_player_profile(
        display_name,
        player_key,
        records,
        view_tag="merged",
        hand="right",
    )
    print(f"  saved {len(records)} clips, merged={json.dumps(merged)}")
    return {"display_name": display_name, "clips": len(records), "merged": merged}


def main():
    parser = argparse.ArgumentParser(description="Build All-Star shooter profiles from YouTube")
    parser.add_argument("--target", type=int, default=TARGET_CLIPS, help="Clips per player")
    parser.add_argument("--player", help="Process one display_name only")
    parser.add_argument("--skip-existing", action="store_true", help="Skip players already in JSON")
    args = parser.parse_args()

    roster = ALLSTAR_SHOOTERS
    if args.player:
        roster = [p for p in ALLSTAR_SHOOTERS if p["display_name"].lower() == args.player.lower()]
        if not roster:
            raise SystemExit(f"Unknown player: {args.player}")

    existing = set()
    if args.skip_existing and MODELS_JSON.exists():
        data = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
        existing = set(data.keys())

    summary = []
    with tempfile.TemporaryDirectory(prefix="allstar_discover_") as tmpdir:
        tmp = Path(tmpdir)
        for entry in roster:
            if args.skip_existing and entry["display_name"] in existing:
                print(f"\nSkip existing: {entry['display_name']}")
                continue
            result = process_player(entry, tmp, target=args.target)
            if result:
                summary.append(result)

    from app.db import ensure_seeded

    ensure_seeded()
    print(f"\n{'='*60}\nDone. {len(summary)} players updated.")
    print(f"Profile: {MODELS_JSON}")
    for row in summary:
        print(f"  - {row['display_name']}: {row['clips']} clips")


if __name__ == "__main__":
    main()
