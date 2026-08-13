"""Upload several stored player clips through /api/analyze and print match ranks."""

from __future__ import annotations

import json
import sys
import tempfile
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.youtube_profile import download_clip, load_json, youtube_id  # noqa: E402

API = "http://127.0.0.1:7860/api/analyze"
PLAYERS = [
    "Devin Booker",
    "Kevin Durant",
    "Anthony Edwards",
    "Jamal Murray",
    "LeBron James",
    "Shai Gilgeous-Alexander",
]


def best_clip(payload: dict) -> dict | None:
    """Prefer the side-view primary clip (current analyzer), then highest score."""
    side_clips = ((payload.get("views") or {}).get("side") or {}).get("clips") or []
    for clip in side_clips:
        ang = clip.get("angles") or {}
        if all(float(ang.get(k, 0) or 0) > 0 for k in ("elbow", "shoulder", "hip", "knee")):
            if float(clip.get("score") or 0) > 0:
                return clip
    clips = []
    for view in (payload.get("views") or {}).values():
        for clip in (view or {}).get("clips") or []:
            ang = clip.get("angles") or {}
            if all(float(ang.get(k, 0) or 0) > 0 for k in ("elbow", "shoulder", "hip", "knee")):
                if float(clip.get("score") or 0) > 0:
                    clips.append(clip)
    if not clips:
        return None
    return max(clips, key=lambda c: float(c.get("score") or 0))


def post_clip(path: Path, person_index: int = 0) -> dict:
    boundary = uuid.uuid4().hex
    data = path.read_bytes()
    parts = []

    def add(name: str, value, filename=None, content_type=None):
        disp = f'Content-Disposition: form-data; name="{name}"'
        if filename:
            disp += f'; filename="{filename}"'
            content_type = content_type or "video/mp4"
        header = [f"--{boundary}", disp]
        if content_type:
            header.append(f"Content-Type: {content_type}")
        header.append("")
        body = value if isinstance(value, bytes) else str(value).encode()
        parts.append(("\r\n".join(header) + "\r\n").encode() + body + b"\r\n")

    add("video_side", data, filename=path.name, content_type="video/mp4")
    add("auto_person", "1")
    add("hand", "right")
    add("lang", "ko")
    add("max_frames", "280")
    raw = b"".join(parts) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(API, data=raw, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    with urllib.request.urlopen(req, timeout=300) as res:
        return json.loads(res.read().decode())


def main() -> None:
    models = load_json()
    rows = []
    with tempfile.TemporaryDirectory(prefix="match_probe_") as tmp:
        tmp_path = Path(tmp)
        for name in PLAYERS:
            clip = best_clip(models.get(name) or {})
            if not clip:
                print(f"SKIP {name}: no clip angles")
                continue
            url = clip["youtube_url"]
            print(f"\n=== {name} {url}")
            local_map = {
                "Anthony Edwards": ROOT / "uploads" / "Edwards" / "clip.mp4",
                "Jamal Murray": ROOT / "uploads" / "Murray" / "clip.mp4",
                "Shai Gilgeous-Alexander": ROOT / "uploads" / "SGA" / "clip.mp4",
            }
            try:
                local = local_map.get(name)
                if local and local.exists() and youtube_id(url) in {
                    "Mx4H3OWD0KM",
                    "w-ks1Skt1Mk",
                }:
                    video = local
                    print(f"  using local {video}")
                else:
                    video = download_clip(url, tmp_path / youtube_id(url))
                payload = post_clip(video)
                print(f"  analyzed_person={payload.get('analysis', {}).get('person_index')}")
            except Exception as exc:
                print(f"  FAIL {exc}")
                rows.append((name, None, str(exc)))
                continue
            matches = payload.get("matches") or []
            top = matches[0] if matches else {}
            names = [m.get("display_name") for m in matches[:3]]
            ok = top.get("display_name") == name
            print(
                f"  top={top.get('display_name')} view={top.get('matched_view')} "
                f"score={top.get('score')} rmse={top.get('distance_deg')} ok={ok}"
            )
            print(f"  top3={names}")
            rows.append((name, ok, top.get("display_name")))

    print("\n==== SUMMARY")
    hits = 0
    for name, ok, top in rows:
        mark = "PASS" if ok else "FAIL"
        if ok:
            hits += 1
        print(f"  {mark:4} expected={name} got={top}")
    print(f"{hits}/{len(rows)} top-1 correct")


if __name__ == "__main__":
    main()
