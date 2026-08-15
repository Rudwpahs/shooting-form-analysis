"""Shared YouTube search, download, and player angle profile helpers."""

from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
import sys
import unicodedata
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


def yt_search_metadata(query: str, limit: int = 12) -> List[dict]:
    """Search without downloading and return reproducible public metadata."""
    try:
        from yt_dlp import YoutubeDL

        options = {
            "extract_flat": True,
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "playlistend": limit,
        }
        with YoutubeDL(options) as ydl:
            result = ydl.extract_info(f"ytsearch{limit}:{query}", download=False) or {}
        rows = []
        for entry in result.get("entries") or []:
            video_id = str(entry.get("id") or "")
            if len(video_id) != 11:
                continue
            rows.append(
                {
                    "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                    "video_id": video_id,
                    "title": str(entry.get("title") or ""),
                    "duration": float(entry.get("duration") or 0.0),
                    "uploader": str(entry.get("uploader") or entry.get("channel") or ""),
                    "query": query,
                }
            )
        return rows
    except Exception as exc:
        print(f"  metadata search failed '{query}': {exc}")
        return []


def _ascii_words(value: str) -> List[str]:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    return re.findall(r"[a-z0-9]+", value)


def candidate_metadata_score(search_name: str, item: dict) -> float:
    """Conservative title/duration filter for shooting-form source videos."""
    title = str(item.get("title") or "")
    words = set(_ascii_words(title))
    name_words = [word for word in _ascii_words(search_name) if len(word) > 2]
    if name_words and not any(word in words for word in name_words[-2:]):
        return 0.0
    lowered = " ".join(_ascii_words(title))
    strong = (
        "shooting form",
        "jump shot",
        "jumpshot",
        "slow motion",
        "shot mechanics",
        "shooting mechanics",
        "shooting breakdown",
        "free throw",
        "shooting workout",
        "pregame shooting",
    )
    weak = ("shooting", "jumper", "workout", "practice", "form")
    rejected = (
        "top 10",
        "top 25",
        "career plays",
        "career highlights",
        "highlights",
        "mixtape",
        "best plays",
        "shut down",
        "defense",
        "reaction",
        "nba 2k",
        "game recap",
    )
    strong_hits = sum(term in lowered for term in strong)
    weak_hits = sum(term in lowered for term in weak)
    if strong_hits == 0 and weak_hits < 2:
        return 0.0
    if any(term in lowered for term in rejected) and strong_hits == 0:
        return 0.0
    duration = float(item.get("duration") or 0.0)
    if duration and not 8.0 <= duration <= 1200.0:
        return 0.0
    score = 50.0 + strong_hits * 18.0 + weak_hits * 4.0
    if "slow motion" in lowered or "slowmo" in lowered:
        score += 18.0
    if "side view" in lowered or "side angle" in lowered:
        score += 12.0
    if "60fps" in lowered or "120fps" in lowered:
        score += 8.0
    if duration and duration <= 240:
        score += 8.0
    return score


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
        "--extractor-args",
        "youtube:player_client=web,android,ios",
        "-f",
        "bv*[height<=720]+ba/b[height<=720]/best[height<=720]/best",
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


def release_angles_plausible(angles: dict) -> bool:
    """Stricter gate for the detected ball-release instant.

    The generic angle validator intentionally accepts deep bends from any shot
    phase.  A release frame, however, should already have an extended lower
    body.  This catches the most common YouTube tracking error: selecting the
    dip/catch or a bystander's motion as release.
    """
    try:
        elbow = float(angles["elbow"])
        shoulder = float(angles["shoulder"])
        hip = float(angles["hip"])
        knee = float(angles["knee"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        85.0 <= elbow <= 179.0
        and 55.0 <= shoulder <= 175.0
        and 100.0 <= hip <= 179.0
        and 100.0 <= knee <= 179.0
    )


def analyze_best(
    path: Path,
    *,
    view_tag: str = "side",
    hand: str = "right",
    max_frames: int = 260,
) -> Tuple[dict, dict]:
    import cv2

    from app.analyze import analyze_view, list_people_in_video
    from app.motion import estimate_view

    people = list_people_in_video(path, num_poses=3)
    candidates: List[Tuple[float, dict, dict]] = []
    diagnostics: List[str] = []
    if people:
        # Curated form/workout clips normally frame the shooter as the dominant
        # subject.  Re-running the full video once per bystander is both slow
        # and a common source of wrong-player profiles.
        ranked_people = sorted(
            people,
            key=lambda person: (
                (person.bbox[2] - person.bbox[0]) * (person.bbox[3] - person.bbox[1]),
                -abs((person.bbox[0] + person.bbox[2]) / 2.0 - 0.5),
            ),
            reverse=True,
        )
        indices = [person.person_index for person in ranked_people[:2]]
    else:
        indices = [0]

    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    coarse_frames = min(total_frames or int(fps * 60), int(fps * 60))

    for person_index in indices:
        coarse = analyze_view(
            path,
            view=view_tag,
            person_index=person_index,
            hand=hand,
            max_frames=max(60, coarse_frames),
            num_poses=3,
            render_height=360,
            keep_sequence=False,
            after_release_sec=0.3,
            frame_stride=6,
        )
        start_time_sec = max(0.0, coarse.release_frame_index / max(fps, 1.0) - 2.0)
        view = analyze_view(
            path,
            view=view_tag,
            person_index=person_index,
            hand=hand,
            max_frames=min(max_frames, 210),
            num_poses=3,
            render_height=576,
            frame_stride=1,
            start_time_sec=start_time_sec,
        )
        if view.release_angles and not view.error:
            if not release_angles_plausible(view.release_angles):
                diagnostics.append(f"person {person_index}: implausible release angles")
                continue
            quality = view.motion_quality or {}
            if not quality.get("valid"):
                diagnostics.append(
                    f"person {person_index}: quality={quality.get('score', 0)} "
                    f"visibility={quality.get('mean_visibility', 0)} "
                    f"gap={quality.get('max_frame_gap', 0)}"
                )
                continue
            detected_view = estimate_view(view.timeline) if view.timeline else view_tag
            score = release_score(view.release_angles) + float(quality.get("score") or 0.0)
            meta = {
                "person_index": person_index,
                "release_frame_index": view.release_frame_index,
                "source_start_time_sec": round(start_time_sec, 3),
                "space": view.space,
                "hand": view.hand,
                "view": detected_view,
                "timeline": view.timeline,
                "quality": quality,
            }
            candidates.append((score, view.release_angles, meta))
            if float(quality.get("score") or 0.0) >= 75.0:
                break
        else:
            diagnostics.append(f"person {person_index}: {view.error or 'no release angles'}")

    if not candidates:
        raise RuntimeError("No usable pose/angles in clip; " + "; ".join(diagnostics))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def median_angles(samples: List[dict]) -> dict:
    keys = ("elbow", "shoulder", "hip", "knee")
    return {k: float(statistics.median([s[k] for s in samples])) for k in keys}


def load_json() -> dict:
    if not MODELS_JSON.exists():
        return {}
    return json.loads(MODELS_JSON.read_text(encoding="utf-8"))


def save_json(data: dict, path: Path = MODELS_JSON) -> None:
    """Write deterministic JSON without platform-dependent doubled CR bytes."""
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


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
    """Learn phase-aligned view profiles from quality-controlled clips."""
    if not clip_records:
        raise ValueError("clip_records is empty")
    from app.angles import AngleSnapshot, angles_plausible
    from app.db import connect, upsert_player, upsert_timeline
    from app.motion import build_motion_prototype, motion_distance, motion_similarity_score

    valid_records = []
    for record in clip_records:
        angles = record.get("angles") or {}
        try:
            snap = AngleSnapshot(
                elbow=float(angles["elbow"]), shoulder=float(angles["shoulder"]),
                hip=float(angles["hip"]), knee=float(angles["knee"]),
                hand=hand, space=str(record.get("space") or "3d"),
            )
        except (KeyError, TypeError, ValueError):
            continue
        if (
            angles_plausible(snap)
            and release_angles_plausible(angles)
            and (record.get("quality") or {}).get("valid")
        ):
            valid_records.append(record)
    if not valid_records:
        raise ValueError("No quality-controlled clip records")

    grouped: Dict[str, List[dict]] = {}
    for record in valid_records:
        detected_view = str(record.get("view") or "side")
        if detected_view not in ("front", "side", "oblique"):
            detected_view = "side"
        grouped.setdefault(detected_view, []).append(record)
    primary_view = max(grouped, key=lambda key: (len(grouped[key]), key == "side"))
    primary_records = grouped[primary_view]
    merged = median_angles([record["angles"] for record in primary_records])
    space = str(primary_records[0].get("space") or "3d")

    data = load_json()
    entry = data.setdefault(
        display_name,
        {"sample_count": 0, "metrics": {}, "meta": {}, "views": {}},
    )
    views: Dict[str, dict] = {}
    for detected_view, records in sorted(grouped.items()):
        angles = median_angles([record["angles"] for record in records])
        prototype = build_motion_prototype([record.get("timeline") or [] for record in records])
        compact_clips = []
        for record_index, record in enumerate(records):
            compact = {key: value for key, value in record.items() if key != "timeline"}
            if len(records) >= 3:
                holdout = build_motion_prototype(
                    [other.get("timeline") or [] for index, other in enumerate(records) if index != record_index]
                )
                if holdout.get("samples"):
                    distance, coverage = motion_distance(record.get("timeline") or [], holdout["samples"])
                    if math.isfinite(distance):
                        compact["holdout_motion_distance"] = round(float(distance), 5)
                        compact["holdout_motion_score"] = round(
                            motion_similarity_score(distance, coverage), 2
                        )
                        compact["holdout_landmark_coverage"] = round(float(coverage), 3)
            compact_clips.append(compact)
        views[detected_view] = {
            "angles": angles,
            "space": str(records[0].get("space") or "3d"),
            "source": source,
            "method": f"quality-controlled median of {len(records)} clips",
            "clips": compact_clips,
        }
        if prototype["samples"]:
            views[detected_view]["timeline"] = {
                "t0": "catch",
                "fps": 30.0,
                "phases": {"catch": 0, "dip": 12, "release": 36, "follow_through": 47},
                "youtube_url": "",
                "person_index": 0,
                "samples": prototype["samples"],
                "after_sec": float(prototype["samples"][-1]["t"]),
                "source_count": prototype["source_count"],
                "quality": prototype["quality"],
                "method": prototype["method"],
            }
    # Merged is an alias of the dominant verified view, not a median of mixed cameras.
    views["merged"] = json.loads(json.dumps(views[primary_view]))
    views["merged"]["source_view"] = primary_view
    entry["views"] = views
    entry["sample_count"] = len(valid_records)
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
        "primary_view": primary_view,
        "player_key": player_key,
        "profile_version": "motion_v2",
    }
    save_json(data)

    conn = connect()
    try:
        conn.execute("DELETE FROM player_timelines WHERE player_key = ?", (player_key,))
        conn.execute("DELETE FROM player_angles WHERE player_key = ?", (player_key,))
        conn.commit()
        for stored_view, block in views.items():
            upsert_player(
                conn, player_key, display_name, block["angles"], view=stored_view,
                hand=hand, source=source, space=str(block.get("space") or space),
            )
            timeline = block.get("timeline") or {}
            if timeline.get("samples"):
                upsert_timeline(
                    conn, player_key, stored_view, float(timeline.get("fps") or 30.0),
                    timeline["samples"], t0="catch",
                    after_sec=float(timeline.get("after_sec") or 1.0),
                    youtube_url="",
                )
    finally:
        conn.close()
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


def search_player_clip_candidates(search_name: str, *, per_query: int = 12) -> List[dict]:
    suffixes = [
        "shooting form slow motion",
        "jump shot slow motion side view",
        "jump shot 60fps",
        "shooting mechanics breakdown",
        "shooting workout close up",
        "pregame shooting workout",
        "free throw slow motion",
        "shooting form side angle",
        "practice shooting slow motion",
    ]
    by_id: Dict[str, dict] = {}
    for suffix in suffixes:
        query = f"{search_name} {suffix}"
        for item in yt_search_metadata(query, limit=per_query):
            score = candidate_metadata_score(search_name, item)
            if score <= 0:
                continue
            item["metadata_score"] = score
            video_id = str(item["video_id"])
            previous = by_id.get(video_id)
            if previous is None or score > float(previous.get("metadata_score") or 0.0):
                by_id[video_id] = item
    return sorted(by_id.values(), key=lambda item: (-float(item["metadata_score"]), item["video_id"]))
