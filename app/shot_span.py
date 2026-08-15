"""Detect catch → dip → release → follow-through from wrist height over time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


MAX_CATCH_TO_RELEASE_SEC = 1.5
MAX_FOLLOWTHROUGH_SEC = 0.55
MIN_TIMELINE_DURATION_SEC = 0.40
MAX_TIMELINE_DURATION_SEC = MAX_CATCH_TO_RELEASE_SEC + MAX_FOLLOWTHROUGH_SEC


@dataclass(frozen=True)
class ShotSpan:
    catch_index: int
    dip_index: int
    release_index: int
    followthrough_index: int


def _smooth(values: Sequence[float], win: int = 5) -> List[float]:
    if win < 1 or len(values) <= 1:
        return [float(v) for v in values]
    half = win // 2
    out: List[float] = []
    n = len(values)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out.append(sum(float(values[j]) for j in range(a, b)) / (b - a))
    return out


def detect_shot_span(
    wrist_y: Sequence[float],
    fps: float,
    release_index: Optional[int] = None,
    frame_indices: Optional[Sequence[int]] = None,
) -> ShotSpan:
    """wrist_y is image-normalized (smaller = higher in the frame).

    Catch = start of the dip after gathering the ball.
    Release = highest wrist.
    Follow-through = arm begins to drop, with a minimum hold after release.
    """
    n = len(wrist_y)
    if n == 0:
        return ShotSpan(0, 0, 0, 0)
    fps = fps if fps > 1 else 30.0
    if frame_indices is None:
        frames = list(range(n))
    else:
        if len(frame_indices) != n:
            raise ValueError("frame_indices must have the same length as wrist_y")
        frames = [int(frame) for frame in frame_indices]
        if any(right < left for left, right in zip(frames, frames[1:])):
            raise ValueError("frame_indices must be in nondecreasing order")
    y = _smooth(wrist_y)

    if release_index is None:
        release_i = min(range(n), key=lambda i: y[i])
    else:
        release_i = max(0, min(n - 1, int(release_index)))

    release_frame = frames[release_i]
    pre = max(1, int(round(MAX_CATCH_TO_RELEASE_SEC * fps)))
    start = release_i
    while start > 0 and frames[start - 1] >= release_frame - pre:
        start -= 1
    dip_i = start + max(range(release_i - start + 1), key=lambda k: y[start + k])

    catch_pre = max(1, int(round(1.0 * fps)))
    c0 = dip_i
    while c0 > start and frames[c0 - 1] >= frames[dip_i] - catch_pre:
        c0 -= 1
    dip_y = y[dip_i]
    catch_i = c0
    found_hold = False
    for i in range(dip_i - 1, c0, -1):
        if i <= 0 or i >= n - 1:
            continue
        higher_than_dip = (dip_y - y[i]) >= 0.025
        local_min = y[i] <= y[i - 1] and y[i] <= y[i + 1]
        if higher_than_dip and local_min:
            catch_i = i
            found_hold = True
            break
    if not found_hold:
        lookahead_frames = max(1, int(round(0.12 * fps)))
        for i in range(c0, dip_i):
            j = i
            while j + 1 <= dip_i and frames[j + 1] <= frames[i] + lookahead_frames:
                j += 1
            if j > i and y[j] - y[i] >= 0.02:
                catch_i = i
                break

    rise = max(0.0, dip_y - y[release_i])
    drop_thresh = y[release_i] + max(0.035, 0.32 * rise)
    min_ft = max(1, int(round(0.22 * fps)))
    max_ft = max(min_ft, int(round(MAX_FOLLOWTHROUGH_SEC * fps)))
    ft_i = release_i
    for i in range(release_i + 1, n):
        elapsed = frames[i] - release_frame
        if elapsed > max_ft:
            break
        ft_i = i
        if elapsed >= min_ft and y[i] >= drop_thresh:
            ft_i = i
            break

    if catch_i > dip_i:
        catch_i = dip_i
    if dip_i > release_i:
        dip_i = release_i
    if ft_i < release_i:
        ft_i = release_i
    return ShotSpan(catch_i, dip_i, release_i, ft_i)


def timeline_duration_is_plausible(samples: Sequence[dict], fps: float = 30.0) -> bool:
    """Reject incomplete or frame-gap-inflated catch-to-follow-through timelines."""
    if len(samples) < 2:
        return False
    try:
        duration = float(samples[-1]["t"]) - float(samples[0]["t"])
    except (KeyError, TypeError, ValueError):
        return False
    frame_tolerance = 2.0 / max(float(fps), 1.0)
    return MIN_TIMELINE_DURATION_SEC <= duration <= MAX_TIMELINE_DURATION_SEC + frame_tolerance


def shot_span_is_complete(span: ShotSpan) -> bool:
    """A stored timeline must contain motion on both sides of release."""
    return (
        span.catch_index <= span.dip_index < span.release_index
        and span.catch_index < span.release_index < span.followthrough_index
    )


PHASE_ORDER = ("catch", "dip", "rise", "release", "follow_through")


def phase_label(index: int, span: ShotSpan) -> str:
    if index < span.dip_index:
        return "catch"
    if index == span.dip_index:
        return "dip"
    if index < span.release_index:
        return "rise"
    if index == span.release_index:
        return "release"
    return "follow_through"


def summarize_phases(samples: List[dict]) -> List[dict]:
    """One row per phase with representative angles and time span."""
    buckets = {name: [] for name in PHASE_ORDER}
    for sample in samples:
        phase = str(sample.get("phase") or "")
        if phase in buckets:
            buckets[phase].append(sample)
    rows: List[dict] = []
    for phase in PHASE_ORDER:
        arr = buckets[phase]
        if not arr:
            rows.append(
                {
                    "phase": phase,
                    "count": 0,
                    "t_start": None,
                    "t_end": None,
                    "frame": None,
                    "angles": None,
                }
            )
            continue
        if phase == "catch":
            pick = arr[0]
        elif phase == "follow_through":
            pick = arr[-1]
        else:
            pick = arr[len(arr) // 2]
        rows.append(
            {
                "phase": phase,
                "count": len(arr),
                "t_start": float(arr[0]["t"]),
                "t_end": float(arr[-1]["t"]),
                "frame": pick.get("frame"),
                "angles": {
                    "elbow": float(pick["elbow"]),
                    "shoulder": float(pick["shoulder"]),
                    "hip": float(pick["hip"]),
                    "knee": float(pick["knee"]),
                },
            }
        )
    return rows
