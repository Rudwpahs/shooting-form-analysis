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

    def to_dict(self) -> dict:
        return {
            "player_key": self.player_key,
            "display_name": self.display_name,
            "distance_deg": round(self.distance_deg, 3),
            "score": round(self.score, 2),
            "matched_view": self.matched_view,
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


def _pick_refs(player_rows: Sequence[Mapping], query_view: str) -> List[Mapping]:
    """Use the same camera view when it exists, and always include clip samples.

    If the player has no matching camera view, use every stored sample
    (merged + clips + other views) so a single clip can still match.
    """
    query_view = str(query_view or "merged")
    same = [r for r in player_rows if str(r.get("view") or "") == query_view]
    clips = [r for r in player_rows if str(r.get("view") or "").startswith("clip:")]
    if same:
        return same + [c for c in clips if c not in same]
    return list(player_rows)


def match_angles(
    query: Mapping[str, float],
    catalog: Sequence[Mapping],
    *,
    top_k: int = 5,
    weights: Optional[Mapping[str, float]] = None,
    query_view: str = "merged",
) -> List[MatchResult]:
    """catalog items: {player_key, display_name, angles, view?} — degrees only."""
    return match_views(
        [{"view": query_view, "angles": query}],
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
        queries.append({"view": str(item.get("view") or "merged"), "angles": angles})
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
        best_ref: Optional[Dict[str, float]] = None
        best_query: Optional[Dict[str, float]] = None
        display_name = str(rows[0].get("display_name") or player_key)
        for query in queries:
            refs = _pick_refs(rows, query["view"])
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
                    )
                )
            if not scored:
                continue
            dist, ref, view_name = min(scored, key=lambda item: item[0])
            dists.append(dist)
            used_views.append(view_name)
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
            )
        )

    results.sort(key=lambda m: m.distance_deg)
    return results[: max(1, top_k)]
