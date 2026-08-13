"""Detect catch → dip → release → follow-through from wrist height over time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


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
    y = _smooth(wrist_y)

    if release_index is None:
        release_i = min(range(n), key=lambda i: y[i])
    else:
        release_i = max(0, min(n - 1, int(release_index)))

    pre = max(1, int(round(1.5 * fps)))
    start = max(0, release_i - pre)
    dip_i = start + max(range(release_i - start + 1), key=lambda k: y[start + k])

    catch_pre = max(1, int(round(1.0 * fps)))
    c0 = max(0, dip_i - catch_pre)
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
        for i in range(c0, dip_i):
            if y[min(i + 3, dip_i)] - y[i] >= 0.02:
                catch_i = i
                break

    rise = max(0.0, dip_y - y[release_i])
    drop_thresh = y[release_i] + max(0.035, 0.32 * rise)
    min_ft = max(1, int(round(0.22 * fps)))
    max_ft = max(min_ft, int(round(0.55 * fps)))
    ft_i = min(n - 1, release_i + max_ft)
    for i in range(release_i + min_ft, min(n, release_i + max_ft + 1)):
        if y[i] >= drop_thresh:
            ft_i = i
            break

    if catch_i > dip_i:
        catch_i = dip_i
    if dip_i > release_i:
        dip_i = release_i
    if ft_i < release_i:
        ft_i = release_i
    return ShotSpan(catch_i, dip_i, release_i, ft_i)


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
