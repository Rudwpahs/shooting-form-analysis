"""Review one high-quality YouTube shooting candidate per roster player.

The script delegates visual assessment to ``manus-analyze-video`` and preserves
the raw review per player. It produces candidate evidence only; it never marks
a player as a calibrated or commercially verified 3D model.
"""

from __future__ import annotations

import json
import subprocess
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models" / "nba_player_models.json"
OUTPUT_DIR = ROOT / "artifacts" / "youtube_visual_reviews"


def score(clip: dict) -> tuple[float, float]:
    quality = clip.get("quality") or {}
    return (float(quality.get("score") or 0.0), float(clip.get("metadata_score") or 0.0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--player", help="Review only this display name")
    parser.add_argument("--candidate-index", type=int, default=0, help="Ranked candidate index to review")
    args = parser.parse_args()
    data = json.loads(MODELS.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    for display_name, profile in data.items():
        if args.player and display_name != args.player:
            continue
        player_key = str((profile.get("meta") or {}).get("player_key") or display_name.lower().replace(" ", "_"))
        clips = [
            clip
            for view in (profile.get("views") or {}).values()
            for clip in (view.get("clips") or [])
            if clip.get("youtube_url")
        ]
        if not clips:
            index.append({"player": display_name, "player_key": player_key, "status": "no_source"})
            continue
        ranked = sorted(clips, key=score, reverse=True)
        if args.candidate_index >= len(ranked):
            index.append({"player": display_name, "player_key": player_key, "status": "candidate_index_unavailable"})
            continue
        chosen = ranked[args.candidate_index]
        url = str(chosen["youtube_url"])
        prompt = f"""Review this YouTube video as a candidate source for a single-view basketball shooting pose profile for {display_name}. Return these exact fields on separate lines: VERDICT: ACCEPT, REJECT, or UNCERTAIN; PLAYER_MATCH: YES, NO, or UNCERTAIN; REAL_SHOT: YES, NO, or UNCERTAIN; BALL_HAND_RELEASE_VISIBLE: YES, NO, or UNCERTAIN; FULL_BODY_VISIBLE: YES, NO, or UNCERTAIN; CAMERA_VIEW: single, multiple_unsynchronized, or uncertain; POSE_EXTRACTION: suitable, limited, or unsuitable; EVIDENCE: a concise visual basis. Do not infer camera calibration or commercial rights."""
        output_file = OUTPUT_DIR / f"{player_key}_candidate_{args.candidate_index}.md"
        result = subprocess.run(
            ["manus-analyze-video", url, prompt],
            capture_output=True,
            text=True,
            timeout=180,
        )
        output_file.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        index.append(
            {
                "player": display_name,
                "player_key": player_key,
                "source_url": url,
                "source_title": chosen.get("title", ""),
                "legacy_pose_quality": chosen.get("quality", {}),
                "return_code": result.returncode,
                "review_file": str(output_file.relative_to(ROOT)),
            }
        )
        print(f"{player_key}: return_code={result.returncode}")
    (OUTPUT_DIR / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
