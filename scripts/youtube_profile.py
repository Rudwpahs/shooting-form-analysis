"""Shared YouTube search, download, and player angle profile helpers."""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS_JSON = ROOT / "models" / "nba_player_models.json"


def player_key_from_name(name: str) -> str:
    s = name.lower().replace("'", "")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def youtube_id(url: str) -> str:
    url = url.strip()
    if "v=" in url:
        return url.split("v=")[1].split("&")[0][:11]
    return url.rstrip("/").split("/")[-1].split("?")[0][:11]


def yt_search(query: str, limit: int = 8) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        f"ytsearch{limit}:{query}",
        "--get-id",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=90)
        urls = []
        for line in out.stdout.splitlines():
            vid = line.strip()
            if len(vid) == 11:
                urls.append(f"https://www.youtube.com/watch?v={vid}")
            elif vid.startswith("http"):
                urls.append(vid)
        return urls
    except Exception as exc:
        print(f"  search failed '{query}': {exc}")
        return []


def download_clip(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    template = str(out_dir / "clip.%(ext)s")
    if "/shorts/" in url:
        vid = url.rstrip("/").split("/")[-1].split("?")[0]
        url = f"https://www.youtube.com/watch?v={vid}"
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
    subprocess.run(cmd, check=True, capture_output=True)
    files = sorted(out_dir.glob("clip.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("Download produced no file")
    return files[0]


def release_score(angles: dict) -> float:
    elbow = float(angles.get("elbow", 0))
    shoulder = float(angles.get("shoulder", 0))
    if elbow < 100 or shoulder < 70:
        return 0.0
    base = elbow * 1.2 + shoulder * 0.6
    knee = float(angles.get("knee", 0))
    hip = float(angles.get("hip", 0))
    penalty = 0.0
    if knee < 120 or knee > 175:
        penalty += 50
    if hip < 90 or hip > 175:
        penalty += 50
    return max(0.0, base - penalty)


def analyze_best(
    path: Path,
    *,
    view_tag: str = "side",
    hand: str = "right",
    max_frames: int = 260,
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


def save_view_to_db(
    player_key: str,
    display_name: str,
    angles: dict,
    *,
    view: str,
    space: str,
    source: str,
    hand: str = "right",
) -> None:
    from app.db import connect, upsert_player

    conn = connect()
    try:
        upsert_player(
            conn,
            player_key=player_key,
            display_name=display_name,
            angles=angles,
            view=view,
            hand=hand,
            source=source,
            space=space,
        )
    finally:
        conn.close()


def persist_player_profile(
    display_name: str,
    player_key: str,
    clip_records: List[dict],
    *,
    view_tag: str = "merged",
    hand: str = "right",
    source: str = "youtube_self_measured",
) -> dict:
    """Save median angles from multiple analyzed clips."""
    if not clip_records:
        raise ValueError("clip_records is empty")

    angles_samples = [r["angles"] for r in clip_records]
    merged = median_angles(angles_samples)
    space = str(clip_records[0].get("space") or "3d")

    data = load_json()
    entry = data.setdefault(
        display_name,
        {"sample_count": 0, "metrics": {}, "meta": {}, "views": {}},
    )
    views: Dict[str, dict] = entry.setdefault("views", {})
    views[view_tag] = {
        "angles": merged,
        "space": space,
        "source": source,
        "method": f"median of {len(clip_records)} clips",
        "clips": clip_records,
    }
    entry["sample_count"] = len(clip_records)
    entry["metrics"] = {
        "Elbow angle": merged["elbow"],
        "Shoulder angle": merged["shoulder"],
        "Hip angle": merged["hip"],
        "Knee angle": merged["knee"],
    }
    entry["meta"] = {
        "space": space,
        "hand": hand,
        "source": source,
        "primary_view": view_tag,
        "player_key": player_key,
    }
    MODELS_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    save_view_to_db(
        player_key,
        display_name,
        merged,
        view=view_tag,
        space=space,
        source=source,
        hand=hand,
    )
    return merged


def search_player_clip_urls(search_name: str, *, per_query: int = 6) -> List[str]:
    suffixes = [
        "shooting form slow motion",
        "shooting breakdown",
        "shooting form analyze",
        "jump shot breakdown",
        "shooting mechanics",
        "jump shot slow motion",
        "form shooting",
    ]
    seen: set[str] = set()
    urls: List[str] = []
    for suffix in suffixes:
        for url in yt_search(f"{search_name} {suffix}", limit=per_query):
            vid = youtube_id(url)
            if vid in seen:
                continue
            seen.add(vid)
            urls.append(url)
    return urls
