"""High-level interruptible ActionIntents and ActionExecutor."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Union

from vikings_bot.input.driver import DryRunInputDriver, InputDriver
from vikings_bot.state.tracker import WorldStateTracker


@dataclass(frozen=True)
class TapKeyIntent:
    key: str
    reason: str = ""
    hold_sec: float = 0.04


@dataclass(frozen=True)
class HoldKeyIntent:
    key: str
    hold: bool
    reason: str = ""
    move_rel_angle_deg: float | None = None


@dataclass(frozen=True)
class ClickScreenIntent:
    x: int
    y: int
    button: str = "left"
    reason: str = ""
    ground_rel_angle_deg: float | None = None
    no_jitter: bool = False
    hold_sec: float = 0.04


@dataclass(frozen=True)
class MoveMouseRelIntent:
    dx: int
    dy: int = 0
    reason: str = ""
    est_yaw_deg: float = 0.0
    steps: int = 6
    step_delay: float = 0.005


@dataclass(frozen=True)
class DragCameraIntent:
    dx: int
    dy: int = 0
    button: str = "middle"
    reason: str = ""
    est_yaw_deg: float = 0.0
    start_x: int | None = None
    start_y: int | None = None


@dataclass(frozen=True)
class TypeChatCommandIntent:
    command: str
    reason: str = ""


@dataclass(frozen=True)
class NoOpIntent:
    reason: str = ""


ActionIntent = Union[
    TapKeyIntent,
    HoldKeyIntent,
    ClickScreenIntent,
    MoveMouseRelIntent,
    DragCameraIntent,
    TypeChatCommandIntent,
    NoOpIntent,
]


class ActionExecutor:
    """Executes ActionIntents via InputDriver, tracks held movement keys, and updates WorldStateTracker."""

    def __init__(
        self,
        driver: InputDriver,
        tracker: WorldStateTracker,
        click_jitter_px: int = 3,
    ) -> None:
        self.driver = driver
        self.tracker = tracker
        self.click_jitter_px = max(0, int(click_jitter_px))
        self._active_holds: dict[str, float] = {}

    def execute_many(
        self,
        intents: list[ActionIntent],
        now: float,
        origin_x: int = 0,
        origin_y: int = 0,
    ) -> str:
        summaries: list[str] = []
        is_dry = isinstance(self.driver, DryRunInputDriver)

        for intent in intents:
            if isinstance(intent, NoOpIntent):
                if intent.reason:
                    summaries.append(intent.reason)
                continue

            if isinstance(intent, TapKeyIntent):
                self.driver.tap_key(intent.key, hold_sec=intent.hold_sec)
                summaries.append(f"key:{intent.key}({intent.reason})")

            elif isinstance(intent, HoldKeyIntent):
                k = intent.key.lower()
                if intent.hold:
                    last_pulse = self._active_holds.get(k)
                    if last_pulse is None or (now - last_pulse) >= 0.30:
                        self.driver.key_down(k)
                        self._active_holds[k] = now
                    if intent.move_rel_angle_deg is not None:
                        self.tracker.notify_ground_click(now, intent.move_rel_angle_deg)
                    summaries.append(f"hold:{k}({intent.reason})")
                else:
                    if k in self._active_holds:
                        self.driver.key_up(k)
                        self._active_holds.pop(k, None)
                        summaries.append(f"release:{k}({intent.reason})")

            elif isinstance(intent, ClickScreenIntent):
                use_jitter = (not is_dry) and (not intent.no_jitter) and (self.click_jitter_px > 0)
                jx = random.randint(-self.click_jitter_px, self.click_jitter_px) if use_jitter else 0
                jy = random.randint(-self.click_jitter_px, self.click_jitter_px) if use_jitter else 0
                tx = origin_x + intent.x + jx
                ty = origin_y + intent.y + jy
                self.driver.mouse_click(tx, ty, button=intent.button, hold_sec=intent.hold_sec)
                if intent.ground_rel_angle_deg is not None:
                    self.tracker.notify_ground_click(now, intent.ground_rel_angle_deg)
                summaries.append(f"click:{intent.button}@({intent.x},{intent.y})({intent.reason})")

            elif isinstance(intent, MoveMouseRelIntent):
                self.driver.mouse_move_rel(
                    dx=intent.dx,
                    dy=intent.dy,
                    steps=intent.steps,
                    step_delay=intent.step_delay,
                    non_blocking=True,
                )
                self.tracker.notify_camera_mouse_dx(intent.dx)
                if abs(intent.est_yaw_deg) > 0.1:
                    self.tracker.notify_camera_rotated(intent.est_yaw_deg)
                summaries.append(f"cam_rel:dx={intent.dx}({intent.reason})")

            elif isinstance(intent, DragCameraIntent):
                sx = None if intent.start_x is None else origin_x + intent.start_x
                sy = None if intent.start_y is None else origin_y + intent.start_y
                self.driver.mouse_drag_rel(
                    dx=intent.dx,
                    dy=intent.dy,
                    button=intent.button,
                    start_x=sx,
                    start_y=sy,
                )
                self.tracker.notify_camera_drag(intent.dx)
                if abs(intent.est_yaw_deg) > 0.1:
                    self.tracker.notify_camera_rotated(intent.est_yaw_deg)
                summaries.append(f"drag_cam({intent.button}):dx={intent.dx}({intent.reason})")

            elif isinstance(intent, TypeChatCommandIntent):
                self.driver.send_chat_command(intent.command)
                summaries.append(f"chat:{intent.command}({intent.reason})")

        summary = "; ".join(summaries)
        if summary:
            self.tracker.last_action_summary = summary
        return summary

    def release_all(self) -> None:
        self._active_holds.clear()
        self.driver.release_all()
        self.tracker.notify_movement_stopped()
