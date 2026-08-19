"""SQLite store for angle-only player profiles and analysis sessions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .provenance import (
    ClipProvenance,
    is_matchable,
    normalize_profile_status,
    profile_status_from_clips,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "shooting_angles.db"
LEGACY_JSON = ROOT / "models" / "nba_player_models.json"


SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    player_key TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    hand TEXT DEFAULT 'right',
    source TEXT DEFAULT 'self_measured',
    notes TEXT DEFAULT '',
    verification_status TEXT DEFAULT 'unverified_legacy',
    verification_reasons_json TEXT DEFAULT '[]',
    provenance_json TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS player_angles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_key TEXT NOT NULL,
    view TEXT NOT NULL DEFAULT 'merged',
    elbow REAL NOT NULL,
    shoulder REAL NOT NULL,
    hip REAL NOT NULL,
    knee REAL NOT NULL,
    space TEXT DEFAULT '3d',
    UNIQUE(player_key, view),
    FOREIGN KEY(player_key) REFERENCES players(player_key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_index INTEGER DEFAULT 0,
    hand TEXT,
    release_angles_json TEXT NOT NULL,
    views_json TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS player_timelines (
    player_key TEXT NOT NULL,
    view TEXT NOT NULL,
    fps REAL NOT NULL,
    t0 TEXT DEFAULT 'release',
    after_sec REAL DEFAULT 1.0,
    samples_json TEXT NOT NULL,
    youtube_url TEXT DEFAULT '',
    UNIQUE(player_key, view),
    FOREIGN KEY(player_key) REFERENCES players(player_key) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS player_reference_clips (
    clip_id TEXT PRIMARY KEY,
    player_key TEXT NOT NULL,
    source_url TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    approval_status TEXT NOT NULL DEFAULT 'draft',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(player_key) REFERENCES players(player_key) ON DELETE CASCADE
);
"""


def connect(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate_players_table(conn)
    return conn


def _migrate_players_table(conn: sqlite3.Connection) -> None:
    """Add verification columns for databases created before provenance support."""
    existing = {str(row["name"]) for row in conn.execute("PRAGMA table_info(players)")}
    additions = {
        "verification_status": "TEXT DEFAULT 'unverified_legacy'",
        "verification_reasons_json": "TEXT DEFAULT '[]'",
        "provenance_json": "TEXT DEFAULT '{}'",
        "updated_at": "TEXT DEFAULT CURRENT_TIMESTAMP",
    }
    for column, definition in additions.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE players ADD COLUMN {column} {definition}")
    conn.commit()


def upsert_player(
    conn: sqlite3.Connection,
    player_key: str,
    display_name: str,
    angles: Dict[str, float],
    *,
    view: str = "merged",
    hand: str = "right",
    source: str = "self_measured",
    space: str = "3d",
    verification_status: Optional[str] = None,
    verification_reasons: Optional[List[str]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> None:
    status = normalize_profile_status(verification_status) if verification_status else None
    reasons_json = json.dumps(verification_reasons or []) if verification_reasons is not None else None
    provenance_json = json.dumps(provenance or {}) if provenance is not None else None
    conn.execute(
        """
        INSERT INTO players(
            player_key, display_name, hand, source, verification_status,
            verification_reasons_json, provenance_json, updated_at
        )
        VALUES (?, ?, ?, ?, COALESCE(?, 'unverified_legacy'), COALESCE(?, '[]'), COALESCE(?, '{}'), CURRENT_TIMESTAMP)
        ON CONFLICT(player_key) DO UPDATE SET
            display_name=excluded.display_name,
            hand=excluded.hand,
            source=excluded.source,
            verification_status=COALESCE(?, players.verification_status),
            verification_reasons_json=COALESCE(?, players.verification_reasons_json),
            provenance_json=COALESCE(?, players.provenance_json),
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            player_key, display_name, hand, source, status, reasons_json, provenance_json,
            status, reasons_json, provenance_json,
        ),
    )
    conn.execute(
        """
        INSERT INTO player_angles(player_key, view, elbow, shoulder, hip, knee, space)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(player_key, view) DO UPDATE SET
            elbow=excluded.elbow,
            shoulder=excluded.shoulder,
            hip=excluded.hip,
            knee=excluded.knee,
            space=excluded.space
        """,
        (
            player_key,
            view,
            float(angles["elbow"]),
            float(angles["shoulder"]),
            float(angles["hip"]),
            float(angles["knee"]),
            space,
        ),
    )
    conn.commit()


def upsert_timeline(
    conn: sqlite3.Connection,
    player_key: str,
    view: str,
    fps: float,
    samples: List[Dict[str, Any]],
    *,
    t0: str = "release",
    after_sec: float = 1.0,
    youtube_url: str = "",
) -> None:
    conn.execute(
        """
        INSERT INTO player_timelines(player_key, view, fps, t0, after_sec, samples_json, youtube_url)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(player_key, view) DO UPDATE SET
            fps=excluded.fps,
            t0=excluded.t0,
            after_sec=excluded.after_sec,
            samples_json=excluded.samples_json,
            youtube_url=excluded.youtube_url
        """,
        (player_key, view, float(fps), t0, float(after_sec), json.dumps(samples), youtube_url),
    )
    conn.commit()


def upsert_reference_clip(conn: sqlite3.Connection, clip: ClipProvenance) -> list[str]:
    """Store a reviewed source clip and return contract violations, if any."""
    errors = clip.validation_errors()
    approval = "approved" if not errors else "rejected"
    conn.execute(
        """
        INSERT INTO player_reference_clips(clip_id, player_key, source_url, provenance_json, approval_status, updated_at)
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(clip_id) DO UPDATE SET
            player_key=excluded.player_key,
            source_url=excluded.source_url,
            provenance_json=excluded.provenance_json,
            approval_status=excluded.approval_status,
            updated_at=CURRENT_TIMESTAMP
        """,
        (clip.clip_id, clip.player_key, clip.source_url, json.dumps(clip.to_dict()), approval),
    )
    conn.commit()
    return errors


def set_player_verification(
    conn: sqlite3.Connection,
    player_key: str,
    status: str,
    *,
    reasons: Optional[List[str]] = None,
    provenance: Optional[Dict[str, Any]] = None,
) -> None:
    conn.execute(
        """
        UPDATE players
        SET verification_status = ?,
            verification_reasons_json = ?,
            provenance_json = COALESCE(?, provenance_json),
            updated_at = CURRENT_TIMESTAMP
        WHERE player_key = ?
        """,
        (
            normalize_profile_status(status),
            json.dumps(reasons or []),
            json.dumps(provenance) if provenance is not None else None,
            player_key,
        ),
    )
    conn.commit()


def refresh_player_verification(
    conn: sqlite3.Connection,
    player_key: str,
    *,
    canonical_3d_verified: bool = False,
) -> tuple[str, List[str]]:
    """Recompute a profile status from the persisted clip provenance contract."""
    rows = conn.execute(
        "SELECT provenance_json FROM player_reference_clips WHERE player_key = ? ORDER BY clip_id",
        (player_key,),
    ).fetchall()
    clips: List[Dict[str, Any]] = []
    for row in rows:
        decoded = _decode_json_object(row["provenance_json"])
        if decoded:
            clips.append(decoded)
    status, reasons = profile_status_from_clips(
        clips,
        player_key=player_key,
        canonical_3d_verified=canonical_3d_verified,
    )
    set_player_verification(
        conn,
        player_key,
        status,
        reasons=reasons,
        provenance={
            "approved_clip_count": sum(
                1 for clip in clips if not ClipProvenance.from_mapping(clip).validation_errors()
            ),
            "reference_clip_count": len(clips),
            "verification_version": "provenance_v1",
        },
    )
    return status, reasons


def list_player_catalog(
    conn: sqlite3.Connection,
    view: str = "merged",
    *,
    verified_only: bool = False,
) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT p.player_key, p.display_name, p.hand, p.source,
               p.verification_status, p.verification_reasons_json, p.provenance_json,
               a.elbow, a.shoulder, a.hip, a.knee, a.space, a.view
        FROM players p
        JOIN player_angles a ON a.player_key = p.player_key
        WHERE a.view = ? AND a.space = '3d' AND p.source != 'archetype'
        ORDER BY p.display_name
        """,
        (view,),
    ).fetchall()
    catalog = []
    for row in rows:
        catalog.append(
            {
                "player_key": row["player_key"],
                "display_name": row["display_name"],
                "hand": row["hand"],
                "source": row["source"],
                "verification_status": str(row["verification_status"] or "unverified_legacy"),
                "verification_reasons": _decode_json_list(row["verification_reasons_json"]),
                "provenance": _decode_json_object(row["provenance_json"]),
                "space": row["space"],
                "view": row["view"],
                "angles": {
                    "elbow": float(row["elbow"]),
                    "shoulder": float(row["shoulder"]),
                    "hip": float(row["hip"]),
                    "knee": float(row["knee"]),
                },
            }
        )
    return [row for row in catalog if not verified_only or is_matchable(row["verification_status"])]


def list_player_angle_rows(
    conn: sqlite3.Connection,
    *,
    verified_only: bool = False,
) -> List[Dict[str, Any]]:
    """Every player_key + view angle row for view-aware matching."""
    rows = conn.execute(
        """
        SELECT p.player_key, p.display_name, p.hand, p.source,
               p.verification_status, p.verification_reasons_json, p.provenance_json,
               a.elbow, a.shoulder, a.hip, a.knee, a.space, a.view
        FROM players p
        JOIN player_angles a ON a.player_key = p.player_key
        WHERE a.space = '3d' AND p.source != 'archetype'
        ORDER BY p.display_name, a.view
        """
    ).fetchall()
    timeline_rows = conn.execute(
        "SELECT player_key, view, samples_json FROM player_timelines"
    ).fetchall()
    timelines: Dict[tuple, List[Dict[str, Any]]] = {}
    for timeline_row in timeline_rows:
        try:
            decoded = json.loads(timeline_row["samples_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = []
        if isinstance(decoded, list):
            timelines[(str(timeline_row["player_key"]), str(timeline_row["view"]))] = decoded

    catalog = []
    for row in rows:
        catalog.append(
            {
                "player_key": row["player_key"],
                "display_name": row["display_name"],
                "hand": row["hand"],
                "source": row["source"],
                "verification_status": str(row["verification_status"] or "unverified_legacy"),
                "verification_reasons": _decode_json_list(row["verification_reasons_json"]),
                "provenance": _decode_json_object(row["provenance_json"]),
                "space": row["space"],
                "view": row["view"],
                "angles": {
                    "elbow": float(row["elbow"]),
                    "shoulder": float(row["shoulder"]),
                    "hip": float(row["hip"]),
                    "knee": float(row["knee"]),
                },
                "timeline": timelines.get((str(row["player_key"]), str(row["view"])), []),
            }
        )
    return [row for row in catalog if not verified_only or is_matchable(row["verification_status"])]


def get_player_skeleton_source(conn: sqlite3.Connection, player_key: str) -> Optional[Dict[str, Any]]:
    """Return one player's preferred 3D angles and motion timeline."""
    profile = conn.execute(
        """
        SELECT p.player_key, p.display_name, p.hand, p.source,
               p.verification_status, p.verification_reasons_json, p.provenance_json,
               a.elbow, a.shoulder, a.hip, a.knee, a.space, a.view
        FROM players p
        JOIN player_angles a ON a.player_key = p.player_key
        WHERE p.player_key = ? AND a.space = '3d' AND p.source != 'archetype'
        ORDER BY CASE WHEN a.view = 'merged' THEN 0 WHEN a.view = 'side' THEN 1 ELSE 2 END,
                 a.view
        LIMIT 1
        """,
        (player_key,),
    ).fetchone()
    if profile is None:
        return None

    timeline = conn.execute(
        """
        SELECT view, fps, t0, after_sec, samples_json, youtube_url
        FROM player_timelines
        WHERE player_key = ?
        ORDER BY CASE WHEN view = 'side' THEN 0 WHEN view = 'merged' THEN 1 ELSE 2 END,
                 view
        LIMIT 1
        """,
        (player_key,),
    ).fetchone()
    samples: List[Dict[str, Any]] = []
    timeline_view = str(profile["view"])
    fps = 30.0
    if timeline is not None:
        try:
            decoded = json.loads(timeline["samples_json"])
            if isinstance(decoded, list):
                samples = decoded
        except (TypeError, ValueError, json.JSONDecodeError):
            samples = []
        timeline_view = str(timeline["view"])
        fps = float(timeline["fps"] or 30.0)

    return {
        "player_key": str(profile["player_key"]),
        "display_name": str(profile["display_name"]),
        "hand": str(profile["hand"] or "right"),
        "source": str(profile["source"] or ""),
        "verification_status": str(profile["verification_status"] or "unverified_legacy"),
        "verification_reasons": _decode_json_list(profile["verification_reasons_json"]),
        "provenance": _decode_json_object(profile["provenance_json"]),
        "space": str(profile["space"] or "3d"),
        "view": timeline_view,
        "fps": fps,
        "angles": {
            "elbow": float(profile["elbow"]),
            "shoulder": float(profile["shoulder"]),
            "hip": float(profile["hip"]),
            "knee": float(profile["knee"]),
        },
        "samples": samples,
    }


def _decode_json_list(value: Any) -> List[Any]:
    try:
        decoded = json.loads(value or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        decoded = []
    return decoded if isinstance(decoded, list) else []


def _decode_json_object(value: Any) -> Dict[str, Any]:
    try:
        decoded = json.loads(value or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        decoded = {}
    return decoded if isinstance(decoded, dict) else {}


def save_session(
    conn: sqlite3.Connection,
    *,
    person_index: int,
    hand: Optional[str],
    release_angles: Dict[str, float],
    views: List[dict],
) -> int:
    cur = conn.execute(
        """
        INSERT INTO sessions(person_index, hand, release_angles_json, views_json)
        VALUES (?, ?, ?, ?)
        """,
        (person_index, hand, json.dumps(release_angles), json.dumps(views)),
    )
    conn.commit()
    return int(cur.lastrowid)


def seed_from_legacy_json(conn: sqlite3.Connection, path: Path = LEGACY_JSON) -> int:
    """Import existing nba_player_models.json angle fields (degrees only)."""
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return 0
    count = 0
    for name, payload in data.items():
        metrics = (payload or {}).get("metrics") or {}
        angles = {
            "elbow": float(metrics.get("Elbow angle", metrics.get("elbow", 0))),
            "shoulder": float(metrics.get("Shoulder angle", metrics.get("shoulder", 0))),
            "hip": float(metrics.get("Hip angle", metrics.get("hip", 0))),
            "knee": float(metrics.get("Knee angle", metrics.get("knee", 0))),
        }
        if not all(angles.values()):
            continue
        key = name.lower().replace(" ", "_")
        upsert_player(
            conn,
            player_key=key,
            display_name=name,
            angles=angles,
            view="merged",
            source="legacy_json_self_measured",
            space="2d",
        )
        count += 1
    return count


def ensure_seeded(db_path: Path = DEFAULT_DB) -> Path:
    conn = connect(db_path)
    try:
        n = conn.execute("SELECT COUNT(*) AS c FROM player_angles").fetchone()["c"]
        if n == 0:
            seed_from_legacy_json(conn)
            upsert_player(
                conn,
                player_key="pro_baseline",
                display_name="Pro Baseline (archetype)",
                angles={"elbow": 170.0, "shoulder": 145.0, "hip": 168.0, "knee": 165.0},
                source="archetype",
                space="3d",
            )
        # Sync all player profiles from models/nba_player_models.json
        if LEGACY_JSON.exists():
            data = json.loads(LEGACY_JSON.read_text(encoding="utf-8"))
            for name, payload in data.items():
                if not isinstance(payload, dict):
                    continue
                meta = payload.get("meta") or {}
                hand = str(meta.get("hand") or "right")
                source = str(meta.get("source") or "legacy_json_self_measured")
                player_key = str(meta.get("player_key") or name.lower().replace(" ", "_"))

                views_block = payload.get("views") or {}
                if views_block:
                    for view_name, view_payload in views_block.items():
                        ang = (view_payload or {}).get("angles") or {}
                        if not all(k in ang for k in ("elbow", "shoulder", "hip", "knee")):
                            continue
                        upsert_player(
                            conn,
                            player_key=player_key,
                            display_name=name,
                            angles={k: float(ang[k]) for k in ("elbow", "shoulder", "hip", "knee")},
                            view=str(view_name),
                            hand=hand,
                            source=str((view_payload or {}).get("source") or source),
                            space=str((view_payload or {}).get("space") or "3d"),
                        )
                        for clip in (view_payload or {}).get("clips") or []:
                            clip_ang = (clip or {}).get("angles") or {}
                            if not all(k in clip_ang for k in ("elbow", "shoulder", "hip", "knee")):
                                continue
                            if not all(float(clip_ang[k]) > 0 for k in ("elbow", "shoulder", "hip", "knee")):
                                continue
                            url = str((clip or {}).get("youtube_url") or "")
                            vid = url.split("v=")[-1][:11] if "v=" in url else url.rstrip("/").split("/")[-1][:11]
                            if len(vid) < 8:
                                continue
                            upsert_player(
                                conn,
                                player_key=player_key,
                                display_name=name,
                                angles={k: float(clip_ang[k]) for k in ("elbow", "shoulder", "hip", "knee")},
                                view=f"clip:{vid}",
                                hand=hand,
                                source=str((clip or {}).get("source") or source),
                                space=str((clip or {}).get("space") or "3d"),
                            )
                        timeline = (view_payload or {}).get("timeline") or {}
                        samples = timeline.get("samples") if isinstance(timeline, dict) else None
                        if samples:
                            upsert_timeline(
                                conn,
                                player_key,
                                str(view_name),
                                float(timeline.get("fps") or 30),
                                samples,
                                t0=str(timeline.get("t0") or "release"),
                                after_sec=float(timeline.get("after_sec") or 1.0),
                                youtube_url=str(timeline.get("youtube_url") or ""),
                            )
                else:
                    metrics = payload.get("metrics") or {}
                    angles = {
                        "elbow": float(metrics.get("Elbow angle", 0)),
                        "shoulder": float(metrics.get("Shoulder angle", 0)),
                        "hip": float(metrics.get("Hip angle", 0)),
                        "knee": float(metrics.get("Knee angle", 0)),
                    }
                    if all(angles.values()):
                        upsert_player(
                            conn,
                            player_key=player_key,
                            display_name=name,
                            angles=angles,
                            view="merged",
                            hand=hand,
                            source=source,
                            space=str(meta.get("space") or "3d"),
                        )
    finally:
        conn.close()
    return db_path
