"""Auto-discover YouTube Curry clips and build front/side/merged angle profiles."""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.youtube_profile import (  # noqa: E402
    MODELS_JSON,
    analyze_best,
    download_clip,
    load_json,
    release_score,
    yt_search,
    youtube_id,
)

from scripts.build_curry_profile import persist_curry_view  # noqa: E402
CANDIDATES = {
    "side": [
        "https://www.youtube.com/shorts/x1kPe03mVpg",
        "https://www.youtube.com/shorts/iQVNOyQES0A",
        "https://www.youtube.com/shorts/W3pFskLg04A",
        "https://www.youtube.com/watch?v=xIDwiuxU_8A",
        "https://www.youtube.com/watch?v=BtA8hbcGgSE",
    ],
    "front": [
        "https://www.youtube.com/shorts/VN1J6EgUQvg",
        "https://www.youtube.com/shorts/iQVNOyQES0A",
        "https://www.youtube.com/shorts/W3pFskLg04A",
    ],
}

SEARCH_QUERIES = {
    "side": "stephen curry shooting form slow motion side view",
    "front": "stephen curry shooting form slow motion front view",
}


def try_clip(url: str, view_tag: str, tmp: Path, hand: str = "right") -> Optional[Tuple[float, dict, dict]]:
    slug = url.split("/")[-1][:16].replace("?", "_")
    try:
        clip = download_clip(url, tmp / slug)
        angles, meta = analyze_best(clip, view_tag=view_tag, hand=hand, max_frames=280)
        score = release_score(angles)
        return score, angles, {"youtube_url": url, "score": score, **meta}
    except Exception as exc:
        print(f"  skip {url}: {exc}")
        return None


def pick_best_for_view(
    view_tag: str,
    urls: List[str],
    tmp: Path,
    *,
    exclude_ids: Optional[set] = None,
) -> Optional[Tuple[dict, dict]]:
    best: Optional[Tuple[float, dict, dict]] = None
    seen = set()
    exclude_ids = exclude_ids or set()
    for url in urls:
        vid = youtube_id(url)
        if url in seen or vid in seen or vid in exclude_ids:
            continue
        seen.add(url)
        seen.add(vid)
        print(f"\n[{view_tag}] trying {url}")
        result = try_clip(url, view_tag, tmp)
        if result is None:
            continue
        score, angles, meta = result
        print(f"  score={score:.1f} angles={json.dumps(angles)}")
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, angles, meta)
    if best is None:
        return None
    _, angles, meta = best
    return angles, meta


def median_views(views: Dict[str, dict]) -> dict:
    keys = ("elbow", "shoulder", "hip", "knee")
    samples = [views[v]["angles"] for v in ("front", "side") if v in views and views[v].get("angles")]
    if not samples:
        return {}
    return {k: float(statistics.median([s[k] for s in samples])) for k in keys}


def main():
    hand = "right"
    with tempfile.TemporaryDirectory(prefix="curry_discover_") as tmpdir:
        tmp = Path(tmpdir)
        results: Dict[str, Tuple[dict, dict]] = {}

        for view_tag in ("side", "front"):
            urls = list(CANDIDATES[view_tag])
            urls.extend(yt_search(SEARCH_QUERIES[view_tag], limit=5))
            exclude = {youtube_id(u) for _, (a, m) in results.items() for u in [m.get("youtube_url", "")]}
            picked = pick_best_for_view(view_tag, urls, tmp, exclude_ids=exclude)
            if picked:
                results[view_tag] = picked
            else:
                print(f"WARNING: no usable clip for {view_tag}")

        if not results:
            raise SystemExit("Could not find any usable Curry clips.")

        for view_tag, (angles, meta) in results.items():
            persist_curry_view(
                view_tag,
                angles,
                space=str(meta.get("space") or "3d"),
                source="youtube_self_measured",
                clips=[meta],
            )
            print(f"\nSaved {view_tag}: {json.dumps(angles, indent=2)}")

        if "front" in results and "side" in results:
            data = load_json()
            entry = data.setdefault("Stephen Curry", {})
            views_block = entry.setdefault("views", {})
            merged = median_views(views_block)
            if merged:
                views_block["merged"] = {
                    "angles": merged,
                    "space": "3d",
                    "source": "youtube_self_measured",
                    "method": "median of front + side",
                }
                entry["metrics"] = {
                    "Elbow angle": merged["elbow"],
                    "Shoulder angle": merged["shoulder"],
                    "Hip angle": merged["hip"],
                    "Knee angle": merged["knee"],
                }
                entry["meta"] = {
                    "space": "3d",
                    "hand": hand,
                    "source": "youtube_self_measured",
                    "primary_view": "merged",
                }
                MODELS_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
                from app.db import connect, upsert_player

                conn = connect()
                try:
                    upsert_player(
                        conn,
                        player_key="stephen_curry",
                        display_name="Stephen Curry",
                        angles=merged,
                        view="merged",
                        hand=hand,
                        source="youtube_self_measured",
                        space="3d",
                    )
                finally:
                    conn.close()
                print(f"\nSaved merged: {json.dumps(merged, indent=2)}")

    from app.db import ensure_seeded

    ensure_seeded()
    print(f"\nDone. Profile: {MODELS_JSON}")


if __name__ == "__main__":
    main()
