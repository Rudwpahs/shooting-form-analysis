"""Build Stephen Curry angle profile from YouTube clip(s)."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODELS_JSON = ROOT / "models" / "nba_player_models.json"

DEFAULT_URLS = [
    "https://www.youtube.com/watch?v=xIDwiuxU_8A",
    "https://www.youtube.com/shorts/iQVNOyQES0A",
]


def download_clip(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "curry.%(ext)s")
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        "best[height<=720][ext=mp4]/best[height<=720]/best",
        "-o",
        template,
        "--no-playlist",
        url,
    ]
    subprocess.run(cmd, check=True)
    files = sorted(out_dir.glob("curry.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("Download produced no file")
    return files[0]


def release_score(angles: dict) -> float:
    elbow = float(angles.get("elbow", 0))
    shoulder = float(angles.get("shoulder", 0))
    if elbow < 100 or shoulder < 70:
        return 0.0
    return elbow * 1.2 + shoulder * 0.6


def analyze_best(
    path: Path,
    *,
    view_tag: str = "side",
    hand: str = "right",
    max_frames: int = 300,
) -> Tuple[dict, dict]:
    from app.analyze import analyze_view, list_people_in_video

    people = list_people_in_video(path, num_poses=3)
    candidates: List[Tuple[float, dict, dict]] = []
    indices = [p.person_index for p in people] or [0, 1, 2]

    for person_index in indices:
        view = analyze_view(
            path,
            view=view_tag,
            person_index=person_index,
            hand=hand,
            max_frames=max_frames,
            num_poses=3,
        )
        if view.release_angles and not view.error:
            score = release_score(view.release_angles)
            meta = {
                "person_index": person_index,
                "release_frame_index": view.release_frame_index,
                "space": view.space,
                "hand": view.hand,
                "view": view_tag,
            }
            candidates.append((score, view.release_angles, meta))

    if not candidates:
        raise RuntimeError("No usable pose/angles in clip")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def median_angles(samples: List[dict]) -> dict:
    keys = ("elbow", "shoulder", "hip", "knee")
    return {k: float(statistics.median([s[k] for s in samples])) for k in keys}


def load_json() -> dict:
    if not MODELS_JSON.exists():
        return {}
    return json.loads(MODELS_JSON.read_text(encoding="utf-8"))


def save_view_to_db(angles: dict, *, view: str, space: str, source: str, hand: str = "right") -> None:
    from app.db import connect, upsert_player

    conn = connect()
    try:
        upsert_player(
            conn,
            player_key="stephen_curry",
            display_name="Stephen Curry",
            angles=angles,
            view=view,
            hand=hand,
            source=source,
            space=space,
        )
    finally:
        conn.close()


def persist_curry_view(
    view_tag: str,
    angles: dict,
    *,
    space: str,
    source: str,
    clips: List[dict],
) -> None:
    data = load_json()
    entry = data.setdefault("Stephen Curry", {"sample_count": 0, "metrics": {}, "meta": {}, "views": {}})
    views: Dict[str, dict] = entry.setdefault("views", {})
    views[view_tag] = {
        "angles": angles,
        "space": space,
        "source": source,
        "clips": clips,
    }
    entry["sample_count"] = sum(len(v.get("clips") or [1]) for v in views.values())
    if view_tag == "merged" or "metrics" not in entry or not entry.get("metrics"):
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
        }
    MODELS_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    save_view_to_db(angles, view=view_tag, space=space, source=source)


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
                clip_meta.append({"youtube_url": url, "score": score, **meta})
            except Exception as exc:
                print(f"Skip {url}: {exc}")

    if not collected:
        raise SystemExit("No clips produced usable angles.")

    if args.aggregate == "best":
        best_i = max(range(len(clip_meta)), key=lambda i: clip_meta[i]["score"])
        final = collected[best_i]
        space = clip_meta[best_i].get("space") or "3d"
    else:
        final = median_angles(collected)
        space = clip_meta[0].get("space") or "3d"

    print(f"\nFinal [{args.view}] ({args.aggregate}):", json.dumps(final, indent=2))
    persist_curry_view(
        args.view,
        final,
        space=space,
        source="youtube_self_measured",
        clips=clip_meta,
    )
    print(f"Saved view '{args.view}' to {MODELS_JSON} and SQLite")


if __name__ == "__main__":
    main()
