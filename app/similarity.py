"""Angle-only similarity matching (no length / height features)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

from .angles import JOINT_KEYS, angle_distance, normalize_angle_dict, similarity_score


DEFAULT_WEIGHTS = {
    "elbow": 1.4,
    "shoulder": 1.2,
    "hip": 1.0,
    "knee": 1.0,
}


@dataclass
class MatchResult:
    player_key: str
    display_name: str
    distance_deg: float
    score: float
    reference_angles: Dict[str, float]
    deltas_deg: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "player_key": self.player_key,
            "display_name": self.display_name,
            "distance_deg": round(self.distance_deg, 3),
            "score": round(self.score, 2),
            "reference_angles": {k: round(v, 2) for k, v in self.reference_angles.items()},
            "deltas_deg": {k: round(v, 2) for k, v in self.deltas_deg.items()},
        }


def match_angles(
    query: Mapping[str, float],
    catalog: Sequence[Mapping],
    *,
    top_k: int = 5,
    weights: Optional[Mapping[str, float]] = None,
) -> List[MatchResult]:
    """catalog items: {player_key, display_name, angles: {elbow,...}} — degrees only."""
    q = normalize_angle_dict(query)
    if len(q) < 2:
        return []

    w = weights or DEFAULT_WEIGHTS
    results: List[MatchResult] = []
    for row in catalog:
        ref = normalize_angle_dict(row.get("angles") or row)
        if len(ref) < 2:
            continue
        dist = angle_distance(q, ref, weights=w)
        deltas = {
            k: float(q.get(k, 0.0) - ref.get(k, 0.0))
            for k in JOINT_KEYS
            if k in q and k in ref
        }
        results.append(
            MatchResult(
                player_key=str(row.get("player_key") or row.get("key") or row.get("display_name")),
                display_name=str(row.get("display_name") or row.get("player_key") or "unknown"),
                distance_deg=dist,
                score=similarity_score(dist),
                reference_angles=ref,
                deltas_deg=deltas,
            )
        )

    results.sort(key=lambda m: m.distance_deg)
    return results[: max(1, top_k)]
