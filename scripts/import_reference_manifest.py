"""Import human-reviewed player reference clips from a JSONL manifest.

This script never downloads video or guesses player identity.  It records only
reviewed provenance supplied by an operator, rejects invalid records, and
recomputes every affected player's publication status.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from app.db import DEFAULT_DB, connect, refresh_player_verification, upsert_reference_clip
from app.provenance import ClipProvenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSONL file containing reviewed clip records")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite profile database path")
    parser.add_argument("--strict", action="store_true", help="Stop after the first invalid manifest row")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    conn = connect(args.db)
    affected: set[str] = set()
    imported = 0
    invalid = 0
    try:
        for line_number, line in enumerate(args.manifest.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("line must be a JSON object")
                clip = ClipProvenance.from_mapping(raw)
                player = conn.execute(
                    "SELECT 1 FROM players WHERE player_key = ?", (clip.player_key,)
                ).fetchone()
                if player is None:
                    raise ValueError(f"unknown player_key: {clip.player_key}")
                errors = upsert_reference_clip(conn, clip)
                if errors:
                    raise ValueError("; ".join(errors))
                affected.add(clip.player_key)
                imported += 1
            except (ValueError, json.JSONDecodeError) as exc:
                invalid += 1
                print(f"line {line_number}: rejected — {exc}", file=sys.stderr)
                if args.strict:
                    return 1

        statuses = defaultdict(list)
        for player_key in sorted(affected):
            status, reasons = refresh_player_verification(conn, player_key)
            statuses[status].append(player_key)
            reason_text = "" if not reasons else " | " + "; ".join(reasons)
            print(f"{player_key}: {status}{reason_text}")
    finally:
        conn.close()

    print(json.dumps({"imported": imported, "invalid": invalid, "affected_players": len(affected)}, indent=2))
    return 0 if invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
