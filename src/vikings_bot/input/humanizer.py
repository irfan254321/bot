"""Human-like input timing & smooth S-curve mouse motion helpers."""

from __future__ import annotations

import random


def ease_mouse_deltas(
    dx: int,
    dy: int = 0,
    steps: int = 6,
    add_micro_arc: bool = True,
    rng: random.Random | None = None,
) -> list[tuple[int, int]]:
    """Split a relative mouse move (dx, dy) into smooth cubic S-curve steps.

    - Guarantees exact net displacement: sum(sx) == dx and sum(sy) == dy (zero pitch drift).
    - Adds a tiny temporary vertical micro-arc during horizontal turns so turns look like a real wrist.
    """
    steps = max(1, int(steps))
    if steps == 1:
        return [(int(dx), int(dy))]

    r = rng or random
    arc_amp = 0.0
    if add_micro_arc and abs(dx) >= 24 and steps >= 4:
        # Subtle 1..3 px vertical bow mid-turn that returns to exact net dy at t=1.0
        arc_amp = r.uniform(-2.6, 2.6)

    out: list[tuple[int, int]] = []
    sent_x = 0
    sent_y = 0
    for i in range(1, steps + 1):
        t = i / float(steps)
        # Cubic smoothstep: slow ease-in -> natural cruise -> smooth settle
        s = t * t * (3.0 - 2.0 * t)
        # Parabolic arc: 0 at t=0 and t=1, peaks at t=0.5 -> zero net vertical pitch drift!
        arc = arc_amp * (4.0 * t * (1.0 - t)) if i < steps else 0.0
        tx = int(round(dx * s)) if i < steps else int(dx)
        ty = int(round(dy * s + arc)) if i < steps else int(dy)
        out.append((tx - sent_x, ty - sent_y))
        sent_x = tx
        sent_y = ty
    return out


class Humanizer:
    """Bounded natural timing & aim variance so automated actions look organic without hurting speed."""

    def __init__(self, enabled: bool = True, seed: int | None = None) -> None:
        self.enabled = bool(enabled)
        self._rng = random.Random(seed) if seed is not None else random.Random()

    def jitter(
        self,
        base: float,
        pct: float = 0.18,
        lo: float | None = None,
        hi: float | None = None,
    ) -> float:
        """Triangular-distributed jitter centered around base (clusters naturally like human fingers)."""
        if not self.enabled or base <= 0.0 or pct <= 0.0:
            return base
        span = base * pct
        val = self._rng.triangular(base - span, base + span, base)
        if lo is not None:
            val = max(lo, val)
        if hi is not None:
            val = min(hi, val)
        return val

    def jitter_int(self, base: int, pct: float = 0.20, min_abs: int = 1) -> int:
        if not self.enabled or base == 0 or pct <= 0.0:
            return base
        sign = 1 if base > 0 else -1
        mag = abs(base)
        val = int(round(self.jitter(float(mag), pct=pct, lo=float(min_abs))))
        return sign * max(min_abs, val)

    def aim_offset_px(self, max_px: float = 10.0) -> float:
        if not self.enabled or max_px <= 0.0:
            return 0.0
        return self._rng.triangular(-max_px, max_px, 0.0)

    def reaction_pause(self, min_sec: float = 0.12, max_sec: float = 0.28) -> float:
        if not self.enabled or max_sec <= 0.0:
            return 0.0
        mid = (min_sec + max_sec) * 0.45
        return self._rng.triangular(min_sec, max_sec, mid)
