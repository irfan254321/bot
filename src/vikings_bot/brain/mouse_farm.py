"""Mouse-Mode 1-Click Farming Controller (Gold Rustler engage, Gold Bag loot, MMB camera sweep)."""

from __future__ import annotations

import math
from typing import Callable

from vikings_bot.config import BotConfig
from vikings_bot.input.actions import (
    ActionIntent,
    ClickScreenIntent,
    DragCameraIntent,
    TapKeyIntent,
)
from vikings_bot.input.humanizer import Humanizer
from vikings_bot.state.models import LootDrop, MobTarget, WorldState
from vikings_bot.state.tracker import WorldStateTracker


class MouseFarmController:
    """Pure Mouse-Mode farming actions: 1x click mob torso, wait until dead, tap E / click Gold Bag, MMB sweep."""

    def __init__(self, config: BotConfig, tracker: WorldStateTracker) -> None:
        self.config = config
        self.tracker = tracker
        c = config.controls
        f = config.farming
        self.player_center_norm = config.player_center_norm()

        self.attack_btn: str = str(c.get("attack_button", "left"))
        self.pickup_key: str = str(c.get("pickup_key", "e"))
        self.camera_btn: str = str(c.get("camera_rotate_button", "middle"))
        self.humanizer = Humanizer(enabled=bool(c.get("humanize", True)))

        self.mob_max_engage_dist_px: float = float(f.get("mob_max_engage_dist_px", 560.0))
        self.mob_min_y_ratio: float = float(f.get("mob_min_y_ratio", 0.14))
        self.loot_e_close_dist_px: float = float(f.get("loot_e_close_dist_px", 85.0))
        self.loot_e_interval_sec: float = float(f.get("loot_e_interval_sec", 0.14))
        self.loot_click_interval_sec: float = float(f.get("loot_click_interval_sec", 0.42))
        self.loot_max_local_dist_px: float = float(f.get("loot_max_local_dist_px", 360.0))
        self.camera_sweep_drag_px: int = int(f.get("camera_sweep_drag_px", 185))
        self.camera_sweep_interval_sec: float = float(f.get("camera_sweep_interval_sec", 0.85))

        self._sweep_dir: int = 1
        self._sweep_dir_chosen: bool = False

    def reset_sweep_direction(self) -> None:
        self._sweep_dir_chosen = False

    def is_point_blocked_by_player(
        self,
        x: float,
        y: float,
        world: WorldState,
        pad_x: float = 28.0,
        pad_y: float = 20.0,
    ) -> bool:
        """Pre-Click Anti-PK Guard: return True if (x, y) overlaps another player's bounding box."""
        boxes = world.snapshot.other_player_boxes
        pts = world.snapshot.other_players
        if not boxes and not pts:
            return False
        sx = world.snapshot.frame_width / 1920.0
        sy = world.snapshot.frame_height / 1080.0
        px_pad = pad_x * sx
        py_pad = pad_y * sy
        for ox1, oy1, ox2, oy2 in boxes:
            if (ox1 - px_pad) <= x <= (ox2 + px_pad) and (oy1 - py_pad) <= y <= (oy2 + py_pad):
                return True
        if not boxes:
            pt_r = 68.0 * sx
            for opx, opy in pts:
                if math.hypot(x - opx, y - opy) <= pt_r:
                    return True
        return False

    def pick_best_gold_rustler(
        self,
        world: WorldState,
        *,
        allow_wounded: bool = False,
        is_ignored: Callable[[MobTarget], bool] | None = None,
    ) -> MobTarget | None:
        """Choose the nearest safe, un-contested Gold Rustler within engage bounds."""
        fw = max(1, world.snapshot.frame_width)
        fh = max(1, world.snapshot.frame_height)
        scale_x = 1.0 if fw <= 1100 else (fw / 1920.0)
        max_dist = self.mob_max_engage_dist_px * scale_x
        min_y = self.mob_min_y_ratio * fh
        rear_y_cut = 0.72 * fh
        rear_fallback: MobTarget | None = None

        for mob in world.mobs:
            if mob.contested:
                continue
            if not allow_wounded and mob.has_hp_bar:
                continue
            if is_ignored is not None and is_ignored(mob):
                continue
            if self.is_point_blocked_by_player(mob.click_x, mob.click_y, world):
                continue
            _mx, my, _mw, _mh = mob.nameplate_bbox
            if my < min_y:
                continue
            if mob.dist_to_player_px > max_dist:
                continue
            if float(mob.click_y) > rear_y_cut:
                if rear_fallback is None:
                    rear_fallback = mob
                continue
            return mob
        return rear_fallback

    def build_engage_click(self, mob: MobTarget) -> list[ActionIntent]:
        """Mouse Mode 1-Click Engage: click left ONCE on upper-torso hitbox (zero follow-up clicks!)."""
        return [
            ClickScreenIntent(
                x=int(mob.click_x),
                y=int(mob.click_y),
                button=self.attack_btn,
                reason=f"mouse_engage_once({mob.click_x},{mob.click_y})",
                no_jitter=True,
            )
        ]

    def select_local_loot(self, world: WorldState) -> list[LootDrop]:
        """Return safe Gold Bag drops within local pickup radius, keeping last_kill_pos priority first."""
        fw = max(1, world.snapshot.frame_width)
        scale_x = fw / 1920.0
        lim = self.loot_max_local_dist_px * max(0.60, scale_x)
        out: list[LootDrop] = []
        for bag in world.loot:
            if bag.dist_to_player_px > lim and not bag.near_last_kill:
                continue
            if self.is_point_blocked_by_player(bag.click_x, bag.click_y, world, pad_x=22.0, pad_y=16.0):
                continue
            out.append(bag)
        return out

    def build_loot_step(
        self,
        world: WorldState,
        now: float,
    ) -> tuple[list[ActionIntent], LootDrop | None, str]:
        """Execute one Mouse Mode loot action: tap E if close (<=85px), else click Gold Bag 1x.

        Returns `(intents, target_drop, action_kind)` where `action_kind` is `'tap_e' | 'click_bag' | 'wait'`.
        """
        local_bags = self.select_local_loot(world)
        if not local_bags:
            if self.tracker.is_ready("loot_e_cleanup", now):
                self.tracker.set_cooldown("loot_e_cleanup", now, self.loot_e_interval_sec)
                return ([TapKeyIntent(key=self.pickup_key, reason="loot_cleanup_e")], None, "tap_e")
            return ([], None, "wait")

        bag = local_bags[0]
        fw = max(1, world.snapshot.frame_width)
        scale_x = max(0.55, fw / 1920.0)
        close_r = self.loot_e_close_dist_px * scale_x

        if bag.dist_to_player_px <= close_r:
            if not self.tracker.is_ready("loot_tap_e", now):
                return ([], bag, "wait")
            cd = self.humanizer.jitter(self.loot_e_interval_sec, pct=0.15, lo=0.09)
            self.tracker.set_cooldown("loot_tap_e", now, cd)
            return (
                [TapKeyIntent(key=self.pickup_key, reason=f"loot_e_close(d={bag.dist_to_player_px:.0f})")],
                bag,
                "tap_e",
            )

        if not self.tracker.is_ready("loot_click_bag", now):
            return ([], bag, "wait")
        cd = self.humanizer.jitter(self.loot_click_interval_sec, pct=0.15, lo=0.25)
        self.tracker.set_cooldown("loot_click_bag", now, cd)
        return (
            [
                ClickScreenIntent(
                    x=int(bag.click_x),
                    y=int(bag.click_y),
                    button="left",
                    reason=f"loot_click_bag({bag.click_x},{bag.click_y})",
                    no_jitter=True,
                )
            ],
            bag,
            "click_bag",
        )

    def build_mmb_camera_sweep(self, world: WorldState, now: float) -> list[ActionIntent]:
        """Rotate camera horizontally via Middle Mouse Button drag (strict dy=0, zero pitch drift)."""
        if not self.tracker.is_ready("camera_sweep", now):
            return []
        sweep_mag = abs(self.camera_sweep_drag_px)
        if not self._sweep_dir_chosen:
            self._sweep_dir_chosen = True
            if world.snapshot.minimap.red_blips and world.heading_source == "delta":
                nearest = world.snapshot.minimap.red_blips[0]
                rel = (nearest.world_angle_deg - world.camera_heading_deg + 180.0) % 360.0 - 180.0
                if 24.0 <= abs(rel) <= 135.0:
                    self._sweep_dir = -1 if rel > 0 else 1

        dx = self.humanizer.jitter_int(self._sweep_dir * sweep_mag, pct=0.12, min_abs=110)
        cd = self.humanizer.jitter(self.camera_sweep_interval_sec, pct=0.15, lo=0.65)
        self.tracker.set_cooldown("camera_sweep", now, cd)
        est_yaw = -dx / 2.5
        return [
            DragCameraIntent(
                dx=dx,
                dy=0,
                button=self.camera_btn,
                reason="search_mmb_camera_sweep",
                est_yaw_deg=est_yaw,
                start_x=world.snapshot.frame_width // 2,
                start_y=world.snapshot.frame_height // 2,
            )
        ]
