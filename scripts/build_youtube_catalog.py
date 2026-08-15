"""Build a reviewable YouTube candidate catalog without downloading videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.discover_allstar_players import ALLSTAR_SHOOTERS  # noqa: E402
from scripts.youtube_profile import search_player_clip_candidates  # noqa: E402

CATALOG_PATH = ROOT / "models" / "youtube_candidate_catalog.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-query", type=int, default=12)
    parser.add_argument("--player", action="append", default=[])
    args = parser.parse_args()

    selected = {name.casefold() for name in args.player}
    roster = [
        player for player in ALLSTAR_SHOOTERS
        if not selected or player["display_name"].casefold() in selected
    ]
    if selected and not roster:
        raise SystemExit("No matching player names")

    existing = {}
    if CATALOG_PATH.exists():
        existing = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    total = 0
    for player in roster:
        name = player["display_name"]
        print(f"\n=== {name}")
        candidates = search_player_clip_candidates(
            player["search_name"], per_query=max(1, args.per_query)
        )
        existing[name] = candidates
        total += len(candidates)
        print(f"  accepted metadata candidates: {len(candidates)}")

    ordered = {key: existing[key] for key in sorted(existing)}
    CATALOG_PATH.write_text(
        json.dumps(ordered, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    unique = {
        item["video_id"]
        for items in ordered.values()
        for item in items
        if item.get("video_id")
    }
    print(f"\nSaved {total} refreshed rows / {len(unique)} unique videos -> {CATALOG_PATH}")


if __name__ == "__main__":
    main()
