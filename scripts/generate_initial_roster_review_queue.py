"""Create a non-approving provenance review queue from legacy player clips.

This script intentionally does not infer player identity, shot events, licensing,
or camera calibration. Every output row remains pending until a human reviewer
fills the required evidence fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("models/nba_player_models.json"))
    parser.add_argument("--output", type=Path, default=Path("data/initial_roster_review_queue.jsonl"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = json.loads(args.models.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for display_name, profile in models.items():
        player_key = profile.get("meta", {}).get("player_key", display_name.lower().replace(" ", "_"))
        seen: set[str] = set()
        serial = 0
        for view_name, view in profile.get("views", {}).items():
            for clip in view.get("clips", []):
                source_url = str(clip.get("youtube_url") or "").strip()
                if not source_url or source_url in seen:
                    continue
                seen.add(source_url)
                serial += 1
                rows.append(
                    {
                        "queue_id": f"{player_key}-{serial:02d}",
                        "player_key": player_key,
                        "player_display_name": display_name,
                        "source_url": source_url,
                        "source_title": clip.get("title", ""),
                        "legacy_view": view_name,
                        "legacy_release_frame": clip.get("release_frame_index"),
                        "legacy_pose_quality": clip.get("quality", {}),
                        "review_state": "pending",
                        "required_reviewer_fields": [
                            "identity_status=verified",
                            "shot_status=verified",
                            "footage_type=real",
                            "license_status",
                            "catch_frame",
                            "release_frame",
                            "followthrough_end_frame",
                            "ball_visible_ratio>=0.60",
                            "occlusion_ratio<=0.25",
                            "camera_calibration_reference",
                        ],
                        "review_notes": "Legacy metadata only. Do not import or reconstruct until reviewed.",
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "players": len(models), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
