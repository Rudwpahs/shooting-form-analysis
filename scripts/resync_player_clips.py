"""Re-analyze each player's best YouTube clip with the current pipeline and sync DB/JSON."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analyze import analyze_view, pick_best_person, release_score  # noqa: E402
from app.db import connect, ensure_seeded, upsert_player  # noqa: E402
from scripts.youtube_profile import (  # noqa: E402
    MODELS_JSON,
    download_clip,
    load_json,
    player_key_from_name,
    youtube_id,
)

PLAYERS = [
    "Stephen Curry",
    "Devin Booker",
    "Kevin Durant",
    "Donovan Mitchell",
    "Anthony Edwards",
    "Tyrese Maxey",
    "Luka Dončić",
    "Jamal Murray",
    "Jalen Brunson",
    "Jaylen Brown",
    "Kawhi Leonard",
    "Norman Powell",
    "De'Aaron Fox",
    "Shai Gilgeous-Alexander",
    "LeBron James",
    "Victor Wembanyama",
]


def ranked_clips(payload: dict) -> list[dict]:
    clips = []
    for view in (payload.get("views") or {}).values():
        for clip in (view or {}).get("clips") or []:
            url = clip.get("youtube_url")
            if url:
                clips.append(clip)
    clips.sort(key=lambda c: float(c.get("score") or 0), reverse=True)
    # unique by video id
    seen = set()
    out = []
    for clip in clips:
        vid = youtube_id(clip["youtube_url"])
        if vid in seen:
            continue
        seen.add(vid)
        out.append(clip)
    return out


def best_clip(payload: dict) -> dict | None:
    clips = ranked_clips(payload)
    return clips[0] if clips else None


def main() -> None:
    data = load_json()
    with tempfile.TemporaryDirectory(prefix="resync_") as tmp:
        tmp_path = Path(tmp)
        for name in PLAYERS:
            entry = data.get(name) or {}
            candidates = ranked_clips(entry)
            if not candidates:
                print(f"SKIP {name}: no clip")
                continue
            print(f"\n=== {name}")
            meta = None
            for clip in candidates[:5]:
                url = clip["youtube_url"]
                vid = youtube_id(url)
                print(f"  try {url}")
                try:
                    path = download_clip(url, tmp_path / f"{vid}_{name[:8]}")
                except Exception as exc:
                    print(f"    download fail: {exc}")
                    continue
                person = pick_best_person(path, view="side", hand="right", max_frames=220)
                result = analyze_view(
                    path,
                    view="side",
                    person_index=person,
                    hand="right",
                    max_frames=320,
                    after_release_sec=0.7,
                )
                if result.error or not result.release_angles:
                    print(f"    analyze fail: {result.error}")
                    continue
                score = release_score(result.release_angles)
                if score <= 0:
                    print(f"    low score {score}")
                    continue
                angles = result.release_angles
                meta = {
                    "youtube_url": url,
                    "score": score,
                    "angles": angles,
                    "person_index": person,
                    "release_frame_index": result.release_frame_index,
                    "space": result.space,
                    "hand": result.hand,
                    "view": "side",
                }
                print(
                    f"    ok person={person} frame={result.release_frame_index} "
                    f"score={score:.1f} e={angles['elbow']:.1f} s={angles['shoulder']:.1f}"
                )
                break
            if meta is None:
                print("  WARNING: no usable clip")
                continue
            url = meta["youtube_url"]
            vid = youtube_id(url)
            key = player_key_from_name(name)
            views = entry.setdefault("views", {})
            block = views.setdefault(
                "merged",
                {"angles": {}, "space": "3d", "source": "youtube_self_measured", "clips": []},
            )
            others = [c for c in (block.get("clips") or []) if youtube_id(c.get("youtube_url") or "") != vid]
            others.append(meta)
            others.sort(key=lambda c: float(c.get("score") or 0), reverse=True)
            block["clips"] = others[:5]
            best = meta
            block["angles"] = dict(best["angles"])
            block["space"] = best.get("space") or "3d"
            block["method"] = "best clip (resync current analyzer)"
            views["side"] = {
                "angles": dict(best["angles"]),
                "space": best.get("space") or "3d",
                "source": "youtube_self_measured",
                "clips": [best],
            }
            entry["metrics"] = {
                "Elbow angle": best["angles"]["elbow"],
                "Shoulder angle": best["angles"]["shoulder"],
                "Hip angle": best["angles"]["hip"],
                "Knee angle": best["angles"]["knee"],
            }
            entry["meta"] = {
                "space": best.get("space") or "3d",
                "hand": best.get("hand") or "right",
                "source": "youtube_self_measured",
                "primary_view": "side",
                "player_key": key,
            }
            entry["sample_count"] = len(block["clips"])
            data[name] = entry

            conn = connect()
            try:
                for view_name in ("merged", "side", f"clip:{vid}"):
                    upsert_player(
                        conn,
                        player_key=key,
                        display_name=name,
                        angles=best["angles"],
                        view=view_name,
                        hand=best.get("hand") or "right",
                        source="youtube_self_measured",
                        space=best.get("space") or "3d",
                    )
            finally:
                conn.close()

    MODELS_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    ensure_seeded()
    print(f"\nSaved {MODELS_JSON}")


if __name__ == "__main__":
    main()
