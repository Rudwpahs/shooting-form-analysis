"""Angle-only similarity matching (no length / height features)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from .angles import JOINT_KEYS, AngleSnapshot, angle_distance, angles_plausible, normalize_angle_dict, similarity_score


DEFAULT_WEIGHTS = {
    # Shoulder is most sensitive to monocular depth error — downweight it.
    "elbow": 1.5,
    "shoulder": 0.7,
    "hip": 1.1,
    "knee": 1.1,
}


@dataclass
class MatchResult:
    player_key: str
    display_name: str
    distance_deg: float
    score: float
    reference_angles: Dict[str, float]
    deltas_deg: Dict[str, float]
    matched_view: str = "merged"
    matched_space: str = "3d"

    def to_dict(self) -> dict:
        return {
            "player_key": self.player_key,
            "display_name": self.display_name,
            "distance_deg": round(self.distance_deg, 3),
            "score": round(self.score, 2),
            "matched_view": self.matched_view,
            "matched_space": self.matched_space,
            "reference_angles": {k: round(v, 2) for k, v in self.reference_angles.items()},
            "deltas_deg": {k: round(v, 2) for k, v in self.deltas_deg.items()},
        }


def _row_angles(row: Mapping) -> Dict[str, float]:
    return normalize_angle_dict(row.get("angles") or row)


def _angles_ok(angles: Mapping[str, float]) -> bool:
    if len(angles) < 4:
        return False
    snap = AngleSnapshot(
        elbow=float(angles["elbow"]),
        shoulder=float(angles["shoulder"]),
        hip=float(angles["hip"]),
        knee=float(angles["knee"]),
        hand="right",
        space="3d",
    )
    return angles_plausible(snap)


def _space_family(space: object) -> str:
    value = str(space or "3d").lower()
    if value.startswith("3d"):
        return "3d"
    if value.startswith("2d"):
        return "2d"
    return value


def _pick_refs(player_rows: Sequence[Mapping], query_view: str, query_space: str) -> List[Mapping]:
    """Use the same camera view when it exists, and always include clip samples.

    If the player has no matching camera view, use every stored sample
    (merged + clips + other views) so a single clip can still match.
    """
    query_view = str(query_view or "merged")
    query_space = _space_family(query_space)
    compatible = [
        row
        for row in player_rows
        if _space_family(row.get("space")) == query_space
    ]
    if not compatible:
        return []
    same = [r for r in compatible if str(r.get("view") or "") == query_view]
    clips = [r for r in compatible if str(r.get("view") or "").startswith("clip:")]
    if same:
        return same + [c for c in clips if c not in same]
    return compatible


def match_angles(
    query: Mapping[str, float],
    catalog: Sequence[Mapping],
    *,
    top_k: int = 5,
    weights: Optional[Mapping[str, float]] = None,
    query_view: str = "merged",
    query_space: str = "3d",
) -> List[MatchResult]:
    """catalog items: {player_key, display_name, angles, view?} — degrees only."""
    return match_views(
        [{"view": query_view, "space": query_space, "angles": query}],
        catalog,
        top_k=top_k,
        weights=weights,
    )


def match_views(
    query_views: Sequence[Mapping],
    catalog: Sequence[Mapping],
    *,
    top_k: int = 5,
    weights: Optional[Mapping[str, float]] = None,
) -> List[MatchResult]:
    """Match each uploaded camera view to the same view on a player when it exists.

    Players without that view fall back to their merged (or any) profile.
    Multi-view uploads average the per-view distances.
    """
    queries = []
    for item in query_views:
        angles = normalize_angle_dict(item.get("angles") or item)
        if len(angles) < 2:
            continue
        queries.append(
            {
                "view": str(item.get("view") or "merged"),
                "space": _space_family(item.get("space")),
                "angles": angles,
            }
        )
    if not queries:
        return []

    grouped: Dict[str, List[Mapping]] = {}
    for row in catalog:
        key = str(row.get("player_key") or row.get("key") or row.get("display_name") or "")
        if not key:
            continue
        grouped.setdefault(key, []).append(row)

    w = weights or DEFAULT_WEIGHTS
    results: List[MatchResult] = []
    for player_key, rows in grouped.items():
        dists: List[float] = []
        used_views: List[str] = []
        used_spaces: List[str] = []
        best_ref: Optional[Dict[str, float]] = None
        best_query: Optional[Dict[str, float]] = None
        display_name = str(rows[0].get("display_name") or player_key)
        for query in queries:
            refs = _pick_refs(rows, query["view"], query["space"])
            if not refs:
                continue
            scored = []
            for ref_row in refs:
                ref = _row_angles(ref_row)
                if len(ref) < 2 or not _angles_ok(ref):
                    continue
                scored.append(
                    (
                        angle_distance(query["angles"], ref, weights=w),
                        ref,
                        str(ref_row.get("view") or "merged"),
                        _space_family(ref_row.get("space")),
                    )
                )
            if not scored:
                continue
            dist, ref, view_name, space_name = min(scored, key=lambda item: item[0])
            dists.append(dist)
            used_views.append(view_name)
            used_spaces.append(space_name)
            best_ref = ref
            best_query = query["angles"]
        if not dists or best_ref is None or best_query is None:
            continue
        dist = float(sum(dists) / len(dists))
        results.append(
            MatchResult(
                player_key=player_key,
                display_name=display_name,
                distance_deg=dist,
                score=similarity_score(dist),
                reference_angles=best_ref,
                deltas_deg={
                    k: float(best_query.get(k, 0.0) - best_ref.get(k, 0.0))
                    for k in JOINT_KEYS
                    if k in best_query and k in best_ref
                },
                matched_view="+".join(used_views),
                matched_space="+".join(used_spaces),
            )
        )

    results.sort(key=lambda m: m.distance_deg)
    return results[: max(1, top_k)]


def match_player(
    query_views: Sequence[Mapping],
    catalog: Sequence[Mapping],
    player_key: str,
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> Optional[MatchResult]:
    """Return a comparison for one explicitly selected player."""
    selected_rows = [
        row
        for row in catalog
        if str(row.get("player_key") or row.get("key") or "") == str(player_key)
    ]
    results = match_views(query_views, selected_rows, top_k=1, weights=weights)
    return results[0] if results else None
