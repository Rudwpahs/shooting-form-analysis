"""Build Stephen Curry angle profile from YouTube clip(s)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODELS_JSON = ROOT / "models" / "nba_player_models.json"

DEFAULT_URLS = [
    # Slow-motion side-ish clips; first working URL wins unless --all.
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
    """Higher = more release-like (extended elbow, open shoulder)."""
    elbow = float(angles.get("elbow", 0))
    shoulder = float(angles.get("shoulder", 0))
    if elbow < 120 or shoulder < 90:
        return 0.0
    return elbow * 1.2 + shoulder * 0.6


def analyze_best(path: Path, hand: str = "right", max_frames: int = 300) -> Tuple[dict, dict]:
    from app.analyze import analyze_view, list_people_in_video

    people = list_people_in_video(path, num_poses=3)
    candidates: List[Tuple[float, dict, dict]] = []
    indices = [p.person_index for p in people] or [0, 1, 2]

    for person_index in indices:
        view = analyze_view(
            path,
            view="side",
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
            }
            candidates.append((score, view.release_angles, meta))

    if not candidates:
        raise RuntimeError("No usable pose/angles in clip")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def save_profile(angles: dict, *, space: str, source: str, meta_extra: dict) -> None:
    data = {}
    if MODELS_JSON.exists():
        data = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    data["Stephen Curry"] = {
        "sample_count": 1,
        "metrics": {
            "Elbow angle": angles["elbow"],
            "Shoulder angle": angles["shoulder"],
            "Hip angle": angles["hip"],
            "Knee angle": angles["knee"],
        },
        "meta": {
            "space": space,
            "hand": "right",
            "source": source,
            **meta_extra,
        },
    }
    MODELS_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")

    from app.db import connect, upsert_player

    conn = connect()
    try:
        upsert_player(
            conn,
            player_key="stephen_curry",
            display_name="Stephen Curry",
            angles=angles,
            hand="right",
            source=source,
            space=space,
        )
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Build Curry profile from YouTube")
    parser.add_argument("--url", action="append", dest="urls", help="YouTube URL (repeatable)")
    parser.add_argument("--hand", default="right", choices=["right", "left"])
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()

    urls = args.urls or DEFAULT_URLS
    last_error = None

    with tempfile.TemporaryDirectory(prefix="curry_build_") as tmp:
        for url in urls:
            try:
                clip = download_clip(url, Path(tmp) / url.split("=")[-1][:8])
                print(f"Downloaded: {clip} from {url}")
                angles, meta = analyze_best(clip, hand=args.hand, max_frames=args.max_frames)
                score = release_score(angles)
                print(f"Score: {score:.1f} person #{meta['person_index']} frame {meta['release_frame_index']}")
                print("Angles:", json.dumps(angles, indent=2))
                if score < 150:
                    print("Warning: low release score — try another clip or upload side-view manually.")
                save_profile(
                    angles,
                    space=str(meta.get("space") or "3d"),
                    source="youtube_self_measured",
                    meta_extra={"youtube_url": url, **meta},
                )
                print(f"Saved to {MODELS_JSON} and local SQLite DB")
                return
            except Exception as exc:
                last_error = exc
                print(f"Skip {url}: {exc}")

    raise SystemExit(f"All URLs failed. Last error: {last_error}")


if __name__ == "__main__":
    main()
