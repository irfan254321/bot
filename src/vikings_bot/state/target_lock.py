"""Strict single-target lock with spatial IoU>=0.30 / <85px hysteresis for Mouse Mode 1-click combat."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vikings_bot.state.models import MobTarget


def bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union between two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter) / float(area_a + area_b - inter)


@dataclass
class TargetLockState:
    lock_id: int
    mob: MobTarget
    locked_at: float
    last_seen_ts: float
    saw_hp_bar: bool = False
    min_hp_ratio: float = 1.0
    contested_ks: bool = False


class StrictTargetLock:
    """Maintains commitment to the 1x-clicked Gold Rustler without ever switching mid-fight."""

    def __init__(
        self,
        lock_iou_threshold: float = 0.30,
        lock_dist_threshold_px: float = 85.0,
        lost_grace_sec: float = 1.35,
    ) -> None:
        self.lock_iou_threshold = float(lock_iou_threshold)
        self.lock_dist_threshold_px = float(lock_dist_threshold_px)
        self.lost_grace_sec = float(lost_grace_sec)
        self._next_id: int = 1
        self.state: TargetLockState | None = None

    @property
    def active(self) -> bool:
        return self.state is not None

    @property
    def locked_mob(self) -> MobTarget | None:
        return None if self.state is None else self.state.mob

    @property
    def lock_id(self) -> int | None:
        return None if self.state is None else self.state.lock_id

    def lock(self, mob: MobTarget, now: float) -> int:
        lid = self._next_id
        self._next_id += 1
        self.state = TargetLockState(
            lock_id=lid,
            mob=mob,
            locked_at=now,
            last_seen_ts=now,
            saw_hp_bar=bool(mob.has_hp_bar),
            min_hp_ratio=float(mob.hp_ratio_est) if mob.hp_ratio_est is not None else 1.0,
            contested_ks=False,
        )
        return lid

    def clear(self) -> None:
        self.state = None

    def absent_sec(self, now: float) -> float:
        if self.state is None:
            return 0.0
        return max(0.0, now - self.state.last_seen_ts)

    def _matches(self, cur: MobTarget, cand: MobTarget, scale_x: float) -> tuple[bool, float]:
        iou = bbox_iou(cur.body_bbox, cand.body_bbox)
        dist = math.hypot(cand.click_x - cur.click_x, cand.click_y - cur.click_y)
        dist_lim = self.lock_dist_threshold_px * max(0.55, scale_x)
        ok = (iou >= self.lock_iou_threshold) or (dist < dist_lim)
        # Prefer higher IoU and smaller displacement
        score = dist - (iou * 120.0)
        return (ok, score)

    def update(
        self,
        mobs: tuple[MobTarget, ...],
        now: float,
        frame_width: int,
        contested_mobs: tuple[MobTarget, ...] = (),
    ) -> MobTarget | None:
        """Update locked target ONLY from spatially matched detection; detect KS if moved into contested."""
        if self.state is None:
            return None
        scale_x = max(0.45, frame_width / 1920.0)
        cur = self.state.mob

        # Check if our locked target was just flagged as contested (another player KS'd it)
        for cm in contested_mobs:
            ok_c, _ = self._matches(cur, cm, scale_x)
            if ok_c:
                self.state.contested_ks = True
                return None

        best_mob: MobTarget | None = None
        best_score = 1e9
        for cand in mobs:
            ok, sc = self._matches(cur, cand, scale_x)
            if not ok:
                continue
            # Prefer engaged candidate when tracking an active fight
            if cand.has_hp_bar:
                sc -= 25.0
            if sc < best_score:
                best_score = sc
                best_mob = cand

        if best_mob is not None:
            self.state.mob = best_mob
            self.state.last_seen_ts = now
            if best_mob.has_hp_bar:
                self.state.saw_hp_bar = True
                if best_mob.hp_ratio_est is not None:
                    self.state.min_hp_ratio = min(self.state.min_hp_ratio, float(best_mob.hp_ratio_est))
            return best_mob
        return None

    def is_target_dead_or_lost(self, now: float) -> bool:
        if self.state is None:
            return False
        return self.absent_sec(now) >= self.lost_grace_sec
