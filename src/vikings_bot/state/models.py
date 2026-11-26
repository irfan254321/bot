"""Typed immutable models shared across Perception, State, Brain, and Runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class BotState(str, Enum):
    PAUSED = "PAUSED"
    ENSURE_MOUSE_MODE = "ENSURE_MOUSE_MODE"
    DEAD_RESPAWN = "DEAD_RESPAWN"
    REPAIR_RECALL = "REPAIR_RECALL"
    REPAIR_AT_NPC = "REPAIR_AT_NPC"
    TELEPORT_TO_SPOT = "TELEPORT_TO_SPOT"
    TRAVEL_TO_SPOT = "TRAVEL_TO_SPOT"
    NAVIGATE_TO_SPOT = "NAVIGATE_TO_SPOT"
    SEARCH_MOB = "SEARCH_MOB"
    COMBAT_WAIT = "COMBAT_WAIT"
    LOOT = "LOOT"
    AT_SPOT_IDLE = "AT_SPOT_IDLE"
    RECOVER_STUCK = "RECOVER_STUCK"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class BotMode(str, Enum):
    FULL = "full"
    NAVIGATE = "navigate"
    COMBAT = "combat"
    LOOT = "loot"


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def distance_to(self, other: Point2D) -> float:
        return math.hypot(other.x - self.x, other.y - self.y)


@dataclass(frozen=True)
class CoordReading:
    x: int
    y: int
    confidence: float
    raw_text: str = ""

    @property
    def point(self) -> Point2D:
        return Point2D(float(self.x), float(self.y))


@dataclass(frozen=True)
class MinimapBlip:
    dx_px: float
    dy_px: float
    world_angle_deg: float  # +X East = 0, +Y North = 90
    dist_px: float


@dataclass(frozen=True)
class MinimapReading:
    red_blips: tuple[MinimapBlip, ...] = ()
    heading_deg: float | None = None
    heading_confidence: float = 0.0


@dataclass(frozen=True)
class MobTarget:
    """Detected monster target with separated upper-torso click point and ground foot point."""

    nameplate_bbox: tuple[int, int, int, int]  # (x, y, w, h)
    body_bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    click_x: int
    click_y: int
    foot_x: int
    foot_y: int
    dist_to_player_px: float
    has_hp_bar: bool = False
    hp_ratio_est: float | None = None
    confidence: float = 1.0
    contested: bool = False
    contest_reason: str = ""


@dataclass(frozen=True)
class LootDrop:
    """Detected Gold Bag drop on the ground."""

    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    click_x: int
    click_y: int
    dist_to_player_px: float
    score: float = 1.0
    near_last_kill: bool = False


@dataclass(frozen=True)
class PerceptionSnapshot:
    timestamp: float
    frame_width: int
    frame_height: int
    hp_ratio: float
    mp_ratio: float
    coords: CoordReading | None
    minimap: MinimapReading
    mobs: tuple[MobTarget, ...] = ()
    contested_mobs: tuple[MobTarget, ...] = ()
    other_players: tuple[tuple[int, int], ...] = ()
    other_player_boxes: tuple[tuple[int, int, int, int], ...] = ()
    loot: tuple[LootDrop, ...] = ()
    dead_corpses: tuple[tuple[int, int, int, int], ...] = ()
    pickup_log_changed: bool = False
    ea_value: int | None = None
    addon_keep_sign_bbox: tuple[int, int, int, int] | None = None
    teleport_menu_bbox: tuple[int, int, int, int] | None = None
    npc_talk_dialog_bbox: tuple[int, int, int, int] | None = None
    merchant_sign_bbox: tuple[int, int, int, int] | None = None
    trade_menu_bbox: tuple[int, int, int, int] | None = None
    repair_all_bbox: tuple[int, int, int, int] | None = None
    repair_yes_bbox: tuple[int, int, int, int] | None = None
    inventory_tab_bbox: tuple[int, int, int, int] | None = None
    respawn_dialog_bbox: tuple[int, int, int, int] | None = None
    detector_backend: str = "OpenCV-Gold"
    interface_mode: str = "unknown"  # "keyboard_banner" | "mouse_banner" | "mouse_cursor" | "no_cursor" | "unknown"
    detected_control_mode: str | None = None  # "mouse" | "keyboard" | None
    camera_pitch_state: str = "ok"
    sky_ratio: float | None = None
    horizon_y: float | None = None
    playfield_luma: float | None = None
    combat_target_hp_visible: bool = False
    combat_target_hp_ratio: float | None = None
    hit_damage_visible: bool = False
    local_hit_damage_visible: bool = False
    external_damage_points: tuple[tuple[int, int], ...] = ()
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def respawn_dialog_visible(self) -> bool:
        return self.respawn_dialog_bbox is not None

    @property
    def engaged_mob(self) -> MobTarget | None:
        for m in self.mobs:
            if m.has_hp_bar:
                return m
        return None

    @property
    def gold_bags(self) -> tuple[LootDrop, ...]:
        return self.loot

    @property
    def target_hp_visible(self) -> bool:
        return self.combat_target_hp_visible or (self.engaged_mob is not None)


@dataclass
class WorldState:
    snapshot: PerceptionSnapshot
    coords_smoothed: Point2D | None
    camera_heading_deg: float
    heading_source: str  # "delta" | "minimap" | "default"
    is_stuck: bool
    consecutive_stuck: int
    zero_hp_streak: int
    coords_age_sec: float
    waypoint_index: int
    locked_mob: MobTarget | None = None
    locked_mob_id: int | None = None
    locked_mob_visible: bool = False
    locked_mob_lost: bool = False
    locked_mob_contested_ks: bool = False
    target_hp_visible: bool = False
    last_kill_pos: tuple[int, int] | None = None
    last_kill_ts: float | None = None
    last_kill_until: float = 0.0
    pending_loot: bool = False
    is_dead: bool = False
    last_nonzero_hp_ratio: float = 1.0
    hp_dropped_below_35_recently: bool = False
    respawn_dialog_visible: bool = False
    travel_waypoint_index: int = 0
    last_action_summary: str = ""

    @property
    def hp_ratio(self) -> float:
        return self.snapshot.hp_ratio

    @property
    def mp_ratio(self) -> float:
        return self.snapshot.mp_ratio

    @property
    def mobs(self) -> tuple[MobTarget, ...]:
        return self.snapshot.mobs

    @property
    def loot(self) -> tuple[LootDrop, ...]:
        return self.snapshot.loot

    @property
    def engaged_mob(self) -> MobTarget | None:
        return self.snapshot.engaged_mob

    @property
    def playfield_luma(self) -> float | None:
        return self.snapshot.playfield_luma

    @property
    def sky_ratio(self) -> float | None:
        return self.snapshot.sky_ratio

    @property
    def horizon_y(self) -> float | None:
        return self.snapshot.horizon_y

    @property
    def hit_damage_visible(self) -> bool:
        return self.snapshot.local_hit_damage_visible or self.snapshot.hit_damage_visible

    @property
    def local_hit_damage_visible(self) -> bool:
        return self.snapshot.local_hit_damage_visible or self.snapshot.hit_damage_visible

    @property
    def external_damage_points(self) -> tuple[tuple[int, int], ...]:
        return self.snapshot.external_damage_points
