"""OpenCV debug overlay renderer for offline replay, snapshots, and dual-monitor monitoring."""

from __future__ import annotations

import math

import cv2
import numpy as np

from vikings_bot.capture.screen import FramePacket
from vikings_bot.config import BotConfig, Waypoint, roi_to_pixels
from vikings_bot.state.models import BotState, WorldState


class DebugOverlay:
    """Renders HUD ROIs, Gold Rustler targets, Gold Bags, corpses, waypoint vector, and FSM banner."""

    def __init__(self, config: BotConfig, window_name: str = "Vikings Bot v2 Debug Overlay") -> None:
        self.config = config
        self.window_name = window_name
        self.scale = float(config.safety.get("overlay_scale", 0.5))
        self._window_open = False

    def render(
        self,
        frame: FramePacket,
        world: WorldState,
        fsm_state: BotState,
        waypoint: Waypoint | None = None,
        fps: float = 0.0,
    ) -> np.ndarray:
        canvas = frame.image_bgr.copy()
        w, h = frame.width, frame.height

        roi_colors = {
            "minimap": (255, 220, 80),
            "coords": (255, 255, 255),
            "hp_bar": (60, 60, 255),
            "mp_bar": (255, 110, 60),
            "playfield": (90, 90, 90),
            "loot_area": (0, 165, 255),
        }
        for name, color in roi_colors.items():
            if name in self.config.rois:
                x1, y1, x2, y2 = roi_to_pixels(self.config.roi(name), w, h)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)

        px = int(round(w * self.config.player_center_norm()[0]))
        py = int(round(h * self.config.player_center_norm()[1]))
        cv2.circle(canvas, (px, py), 6, (255, 255, 0), 2)

        if fsm_state not in (BotState.COMBAT_WAIT, BotState.LOOT, BotState.RETURN_TO_CENTER):
            for idx, mob in enumerate(world.mobs):
                bx1, by1, bx2, by2 = mob.body_bbox
                is_locked = world.locked_mob is not None and mob.nameplate_bbox == world.locked_mob.nameplate_bbox
                color = (0, 255, 255) if is_locked else ((0, 255, 0) if mob.has_hp_bar else (255, 100, 255))
                thickness = 3 if is_locked else 2
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, thickness)
                cv2.drawMarker(canvas, (mob.click_x, mob.click_y), color, cv2.MARKER_CROSS, 18, thickness)
                hp_s = f" {mob.hp_ratio_est:.0%}" if mob.hp_ratio_est is not None else ""
                if mob.class_name:
                    base = f"{mob.class_name} {mob.confidence:.2f}"
                    lbl = f"LOCK#{world.locked_mob_id} {base}{hp_s}" if is_locked else f"{base} {mob.dist_to_player_px:.0f}px{hp_s}"
                else:
                    lbl = f"LOCK#{world.locked_mob_id}{hp_s}" if is_locked else f"GR#{idx + 1} {mob.dist_to_player_px:.0f}px{hp_s}"
                cv2.putText(canvas, lbl, (bx1, max(16, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

            for cm in world.snapshot.contested_mobs:
                bx1, by1, bx2, by2 = cm.body_bbox
                cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (60, 146, 251), 1)
                cm_lbl = (
                    f"{cm.class_name} {cm.confidence:.2f} {cm.contest_reason or 'SKIP [KS]'}"
                    if cm.class_name
                    else (cm.contest_reason or "SKIP [KS]")
                )
                cv2.putText(
                    canvas,
                    cm_lbl,
                    (bx1, max(16, by1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (60, 146, 251),
                    1,
                )

        if fsm_state not in (BotState.COMBAT_WAIT, BotState.RETURN_TO_CENTER):
            for idx, pb in enumerate(world.snapshot.other_player_boxes):
                cv2.rectangle(canvas, (pb[0], pb[1]), (pb[2], pb[3]), (60, 60, 255), 2)
                if idx < len(world.snapshot.other_player_confs):
                    pl_lbl = f"player {world.snapshot.other_player_confs[idx]:.2f}"
                    cv2.putText(canvas, pl_lbl, (pb[0], max(16, pb[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 120, 255), 1)

            for idx, (cx1, cy1, cx2, cy2) in enumerate(world.snapshot.dead_corpses):
                cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (240, 120, 200), 2)
                cp_lbl = (
                    f"dead_rustler {world.snapshot.dead_corpse_confs[idx]:.2f}"
                    if idx < len(world.snapshot.dead_corpse_confs)
                    else "DEAD / MAYAT"
                )
                cv2.putText(canvas, cp_lbl, (cx1, max(16, cy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 160, 220), 1)

            for idx, bag in enumerate(world.loot):
                lx, ly, lw, lh = bag.bbox
                cv2.rectangle(canvas, (lx, ly), (lx + lw, ly + lh), (0, 215, 255), 2)
                cv2.circle(canvas, (bag.click_x, bag.click_y), 4, (0, 255, 255), -1)
                if idx < 16:
                    b_lbl = f"{bag.class_name} {bag.score:.2f}" if bag.class_name else f"G{idx + 1}"
                    cv2.putText(canvas, b_lbl, (lx, max(14, ly - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

        if waypoint is not None and world.coords_smoothed is not None:
            dx_w = waypoint.x - world.coords_smoothed.x
            dy_w = waypoint.y - world.coords_smoothed.y
            dist = math.hypot(dx_w, dy_w)
            bearing = math.degrees(math.atan2(dy_w, dx_w))
            rel = (bearing - world.camera_heading_deg + 180.0) % 360.0 - 180.0
            rad = math.radians(rel)
            vx = int(round(px - math.sin(rad) * 140))
            vy = int(round(py - math.cos(rad) * 110))
            cv2.arrowedLine(canvas, (px, py), (vx, vy), (255, 200, 0), 3, tipLength=0.25)
            cv2.putText(
                canvas,
                f"WP:{waypoint.name} ({waypoint.x:.0f},{waypoint.y:.0f}) d={dist:.1f}",
                (px - 120, py + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 220, 80),
                2,
            )

        cv2.rectangle(canvas, (180, 6), (1380, 56), (15, 15, 20), -1)
        coord_txt = (
            f"({world.coords_smoothed.x:.0f}, {world.coords_smoothed.y:.0f})"
            if world.coords_smoothed is not None
            else "(?, ?)"
        )
        banner = (
            f"STATE: {fsm_state.value} | HP: {world.hp_ratio:.0%} MP: {world.mp_ratio:.0%} | "
            f"POS: {coord_txt} | MOBS: {len(world.mobs)} GOLD: {len(world.loot)} | {fps:.1f} FPS"
        )
        cv2.putText(canvas, banner, (192, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2)
        if world.last_action_summary:
            cv2.putText(
                canvas,
                f"ACT: {world.last_action_summary[:95]}",
                (192, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (120, 240, 180),
                1,
            )
        return canvas

    def show(self, annotated_bgr: np.ndarray, wait_ms: int = 1) -> int:
        if self.scale != 1.0:
            w = max(1, int(round(annotated_bgr.shape[1] * self.scale)))
            h = max(1, int(round(annotated_bgr.shape[0] * self.scale)))
            disp = cv2.resize(annotated_bgr, (w, h), interpolation=cv2.INTER_AREA)
        else:
            disp = annotated_bgr
        cv2.imshow(self.window_name, disp)
        self._window_open = True
        return cv2.waitKey(wait_ms) & 0xFF

    def close(self) -> None:
        if self._window_open:
            cv2.destroyWindow(self.window_name)
            self._window_open = False
