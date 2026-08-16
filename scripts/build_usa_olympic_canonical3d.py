"""Build canonical 3D models for rostered Paris 2024 USA players.

Only players that are also present in ``nba_player_models.json`` are included.
Each player is published independently, so one failed reconstruction cannot
replace another player's validated model or disturb the legacy API fallback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_curry_canonical3d import (  # noqa: E402
    DEFAULT_FRONT_URLS,
    _minimum_three,
    _unique_urls,
    _write_report,
    build_player_canonical_model,
    current_view_urls,
    player_identity_matches,
)
from scripts.youtube_profile import (  # noqa: E402
    load_json,
    yt_search_metadata,
)


USA_OLYMPIC_PLAYERS = (
    ("stephen_curry", "Stephen Curry"),
    ("devin_booker", "Devin Booker"),
    ("kevin_durant", "Kevin Durant"),
    ("anthony_edwards", "Anthony Edwards"),
    ("lebron_james", "LeBron James"),
)
PLAYER_NAMES = dict(USA_OLYMPIC_PLAYERS)
CATALOG_PATH = ROOT / "models" / "youtube_candidate_catalog.json"


def catalog_urls_for_view(
    catalog: Mapping[str, object], player_name: str, view: str
) -> tuple[str, ...]:
    """Return identity-checked catalog sources explicitly tagged by view."""
    markers = {
        "front": ("front view", "front angle"),
        "side": ("side view", "side angle"),
    }
    if view not in markers:
        raise ValueError(f"unsupported catalog view: {view}")
    urls = []
    records = catalog.get(player_name) or []
    if not isinstance(records, list):
        return ()
    for record in records:
        if not isinstance(record, Mapping):
            continue
        title = str(record.get("title") or "")
        query = str(record.get("query") or "").casefold()
        url = str(record.get("youtube_url") or "")
        if (
            url
            and player_identity_matches(player_name, title)
            and any(marker in query for marker in markers[view])
        ):
            urls.append(url)
    return _unique_urls(urls)


def catalog_urls_for_player(
    catalog: Mapping[str, object], player_name: str, *, limit: int = 12
) -> tuple[str, ...]:
    """Return identity-checked candidates for geometry-based view detection."""
    records = catalog.get(player_name) or []
    if not isinstance(records, list):
        return ()
    urls = [
        str(record.get("youtube_url") or "")
        for record in records
        if isinstance(record, Mapping)
        and player_identity_matches(
            player_name, str(record.get("title") or "")
        )
    ]
    return _unique_urls(urls)[: max(0, int(limit))]


def discover_urls_for_view(
    player_name: str,
    view: str,
    *,
    limit: int,
    no_check_certificates: bool = False,
) -> tuple[str, ...]:
    """Discover identity-checked metadata candidates without downloading video."""
    if limit <= 0:
        return ()
    query = f"{player_name} shooting form slow motion {view} view"
    records = yt_search_metadata(
        query,
        limit=limit,
        no_check_certificates=no_check_certificates,
    )
    return _unique_urls(
        [
            str(record.get("youtube_url") or "")
            for record in records
            if player_identity_matches(
                player_name, str(record.get("title") or "")
            )
        ]
    )


def _parse_overrides(values: Sequence[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for value in values:
        player_key, separator, url = str(value).partition("=")
        if not separator or player_key not in PLAYER_NAMES or not url.strip():
            raise argparse.ArgumentTypeError(
                "URL overrides must use supported_player_key=https://..."
            )
        out.setdefault(player_key, []).append(url.strip())
    return out


def _load_catalog(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Paris 2024 USA canonical 3D player models"
    )
    parser.add_argument(
        "--player",
        action="append",
        choices=tuple(PLAYER_NAMES),
        default=[],
        help="Limit the batch to one or more supported player keys",
    )
    parser.add_argument(
        "--front-url",
        action="append",
        default=[],
        metavar="PLAYER_KEY=URL",
    )
    parser.add_argument(
        "--side-url",
        action="append",
        default=[],
        metavar="PLAYER_KEY=URL",
    )
    parser.add_argument("--discover-limit", type=int, default=8)
    parser.add_argument("--min-per-view", type=_minimum_three, default=3)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--max-repeated-failures", type=int, default=3)
    parser.add_argument("--no-check-certificates", action="store_true")
    parser.add_argument(
        "--output", default="models/canonical_3d_models.json"
    )
    parser.add_argument(
        "--report", default="models/usa_olympic_canonical3d_validation.json"
    )
    args = parser.parse_args()

    selected = tuple(args.player) or tuple(PLAYER_NAMES)
    front_overrides = _parse_overrides(args.front_url)
    side_overrides = _parse_overrides(args.side_url)
    profiles = load_json()
    catalog = _load_catalog(CATALOG_PATH)
    output_path = ROOT / args.output
    report_path = ROOT / args.report
    reports: dict[str, dict] = {}

    for player_key in selected:
        player_name = PLAYER_NAMES[player_key]
        print(f"\n=== {player_name}")
        front_urls = [
            *front_overrides.get(player_key, []),
            *catalog_urls_for_view(catalog, player_name, "front"),
            *catalog_urls_for_player(catalog, player_name, limit=8),
            *discover_urls_for_view(
                player_name,
                "front",
                limit=max(0, args.discover_limit),
                no_check_certificates=bool(args.no_check_certificates),
            ),
        ]
        if player_key == "stephen_curry":
            front_urls.extend(DEFAULT_FRONT_URLS)
        side_urls = [
            *side_overrides.get(player_key, []),
            *current_view_urls(profiles, player_name, "side"),
            *catalog_urls_for_view(catalog, player_name, "side"),
            *discover_urls_for_view(
                player_name,
                "side",
                limit=max(0, args.discover_limit),
                no_check_certificates=bool(args.no_check_certificates),
            ),
        ]
        try:
            reports[player_key] = build_player_canonical_model(
                player_key=player_key,
                player_name=player_name,
                front_urls=_unique_urls(front_urls),
                side_urls=_unique_urls(side_urls),
                min_per_view=args.min_per_view,
                max_frames=max(60, int(args.max_frames)),
                no_check_certificates=bool(args.no_check_certificates),
                output_path=output_path,
                max_repeated_failures=max(
                    1, int(args.max_repeated_failures)
                ),
            )
        except Exception as exc:
            reports[player_key] = {
                "player_key": player_key,
                "display_name": player_name,
                "passed": False,
                "reasons": [f"{type(exc).__name__}: {exc}"],
                "output": str(output_path),
            }

    aggregate = {
        "roster": [
            {"player_key": key, "display_name": PLAYER_NAMES[key]}
            for key in selected
        ],
        "passed_count": sum(
            report.get("passed") is True for report in reports.values()
        ),
        "failed_count": sum(
            report.get("passed") is not True for report in reports.values()
        ),
        "players": reports,
        "output": str(output_path),
    }
    _write_report(report_path, aggregate)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    if aggregate["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
