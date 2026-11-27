"""Win32 Transparent & Click-Through real-time debug overlay for Vikings Bot v2 (Mouse Mode + AHK + Pure OpenCV)."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from vikings_bot.config import BotConfig
    from vikings_bot.state.models import BotState, WorldState

logger = logging.getLogger(__name__)

TRANSPARENT_COLOR_HEX = "#010203"
TRANSPARENT_COLOR_BGR = 0x00030201  # Win32 COLORREF: 0x00BBGGRR for RGB(1, 2, 3)

GWL_EXSTYLE = -20
GWLP_HWNDPARENT = -8
GA_ROOT = 2

WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_LAYERED = 0x00080000
WS_EX_NOACTIVATE = 0x08000000

CLICKTHROUGH_EXSTYLE_MASK = (
    WS_EX_LAYERED
    | WS_EX_TRANSPARENT
    | WS_EX_TOPMOST
    | WS_EX_NOACTIVATE
    | WS_EX_TOOLWINDOW
)

LWA_COLORKEY = 0x00000001
HWND_TOPMOST = -1
SW_HIDE = 0
SW_SHOWNOACTIVATE = 4

SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
SWP_SHOWWINDOW = 0x0040
SWP_NOOWNERZORDER = 0x0200

WDA_EXCLUDEFROMCAPTURE = 0x00000011


def _get_user32() -> Any | None:
    if sys.platform != "win32":
        return None
    try:
        return ctypes.windll.user32
    except Exception:
        return None


def _resolve_root_hwnd(user32: Any, hwnd: int) -> int:
    if not hwnd:
        return 0
    try:
        root_hwnd = int(user32.GetAncestor(wintypes.HWND(hwnd), GA_ROOT))
        if root_hwnd:
            return root_hwnd
    except Exception:
        pass
    return int(hwnd)


def apply_clickthrough_styles(
    hwnd: int,
    *,
    color_key_bgr: int = TRANSPARENT_COLOR_BGR,
    exclude_from_capture: bool = True,
    detach_owner: bool = False,
    user32_override: Any | None = None,
) -> dict[str, int | bool]:
    """Apply Win32 layered, click-through, topmost, non-activating & capture-exclude flags."""
    u32 = user32_override if user32_override is not None else _get_user32()
    if u32 is None or not hwnd:
        return {"hwnd": 0, "ex_style": 0, "applied": False, "capture_excluded": False}

    root_hwnd = _resolve_root_hwnd(u32, hwnd)
    if not root_hwnd:
        return {"hwnd": 0, "ex_style": 0, "applied": False, "capture_excluded": False}

    if detach_owner:
        try:
            set_long_ptr = getattr(u32, "SetWindowLongPtrW", None) or getattr(u32, "SetWindowLongW", None)
            if set_long_ptr is not None:
                set_long_ptr(wintypes.HWND(root_hwnd), GWLP_HWNDPARENT, 0)
        except Exception:
            pass

    cur_style = int(u32.GetWindowLongW(wintypes.HWND(root_hwnd), GWL_EXSTYLE))
    new_style = cur_style | CLICKTHROUGH_EXSTYLE_MASK
    u32.SetWindowLongW(wintypes.HWND(root_hwnd), GWL_EXSTYLE, new_style)

    try:
        u32.SetLayeredWindowAttributes(
            wintypes.HWND(root_hwnd),
            wintypes.COLORREF(color_key_bgr),
            wintypes.BYTE(255),
            wintypes.DWORD(LWA_COLORKEY),
        )
    except Exception:
        pass

    try:
        u32.SetWindowPos(
            wintypes.HWND(root_hwnd),
            wintypes.HWND(HWND_TOPMOST),
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_FRAMECHANGED | SWP_NOOWNERZORDER,
        )
    except Exception:
        pass

    capture_excluded = False
    if exclude_from_capture:
        try:
            res = u32.SetWindowDisplayAffinity(
                wintypes.HWND(root_hwnd),
                wintypes.DWORD(WDA_EXCLUDEFROMCAPTURE),
            )
            capture_excluded = bool(res)
        except Exception:
            capture_excluded = False

    return {
        "hwnd": root_hwnd,
        "ex_style": new_style,
        "applied": True,
        "capture_excluded": capture_excluded,
    }


def position_overlay_window(
    hwnd: int,
    left: int,
    top: int,
    width: int,
    height: int,
    *,
    visible: bool = True,
    user32_override: Any | None = None,
) -> bool:
    """Move/resize & show/hide overlay via Win32 SetWindowPos without ever activating it."""
    u32 = user32_override if user32_override is not None else _get_user32()
    if u32 is None or not hwnd:
        return False
    root_hwnd = _resolve_root_hwnd(u32, hwnd)
    if not root_hwnd:
        return False
    try:
        if not visible:
            u32.ShowWindow(wintypes.HWND(root_hwnd), SW_HIDE)
            return True
        flags = SWP_NOACTIVATE | SWP_NOOWNERZORDER | SWP_SHOWWINDOW
        u32.SetWindowPos(
            wintypes.HWND(root_hwnd),
            wintypes.HWND(HWND_TOPMOST),
            int(left),
            int(top),
            max(1, int(width)),
            max(1, int(height)),
            flags,
        )
        u32.ShowWindow(wintypes.HWND(root_hwnd), SW_SHOWNOACTIVATE)
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class OverlayMob:
    bbox: tuple[int, int, int, int]
    click_x: int
    click_y: int
    conf: float
    dist_px: float
    safe: bool = True
    locked: bool = False
    hp_ratio: float | None = None
    label: str = ""
    class_name: str = ""


@dataclass(frozen=True)
class OverlayPlayer:
    bbox: tuple[int, int, int, int]
    cx: int
    cy: int
    ks_radius_px: float = 75.0
    conf: float | None = None
    class_name: str = "player"


@dataclass(frozen=True)
class OverlayLoot:
    bbox: tuple[int, int, int, int]
    click_x: int
    click_y: int
    score: float = 1.0
    index: int = 1
    class_name: str = ""


@dataclass(frozen=True)
class OverlayCorpse:
    bbox: tuple[int, int, int, int]
    label: str = "DEAD / MAYAT"
    conf: float | None = None
    class_name: str = ""


@dataclass(frozen=True)
class OverlayFrameData:
    """Thread-safe immutable frame snapshot rendered by TransparentDebugOverlay."""

    game_rect: tuple[int, int, int, int]
    frame_width: int
    frame_height: int
    phase: str
    action: str
    loop_hz: float
    vision_ms: float
    target_hp_pct: float | None = None
    hp_pct: float = 100.0
    mp_pct: float = 100.0
    coords: tuple[int, int] | None = None
    backend: str = "OpenCV-Gold"
    input_mode: str = "AHK-VM"
    travel_desc: str = ""
    deaths_count: int = 0
    repairs_count: int = 0
    repair_rem_min: float | None = None
    mobs: tuple[OverlayMob, ...] = ()
    locked_mob: OverlayMob | None = None
    players: tuple[OverlayPlayer, ...] = ()
    loot: tuple[OverlayLoot, ...] = ()
    corpses: tuple[OverlayCorpse, ...] = ()
    self_char_box: tuple[int, int, int, int] | None = None
    self_core_box: tuple[int, int, int, int] | None = None
    player_center: tuple[int, int] | None = None
    overlay_hotkey: str = "F6"
    game_focused: bool = True
    timestamp: float = 0.0


def compute_self_exclusion_boxes(
    frame_width: int,
    frame_height: int,
    player_center_norm: tuple[float, float] = (0.50, 0.52),
    hud_fixed_pixel_ui: bool = False,
) -> tuple[tuple[int, int], tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Return (player_center, self_char_box, self_core_box) in frame pixels."""
    fw = max(1, int(frame_width))
    fh = max(1, int(frame_height))
    sx = fw / 1920.0
    sy = fh / 1080.0
    px = int(round(fw * player_center_norm[0]))
    py = int(round(fh * player_center_norm[1]))

    tol_x = max(46.0, 56.0 * sx) if hud_fixed_pixel_ui else max(36.0, 52.0 * sx)
    tol_up = max(60.0, 72.0 * sy) if hud_fixed_pixel_ui else max(48.0, 72.0 * sy)
    tol_dn = max(88.0, 95.0 * sy) if hud_fixed_pixel_ui else max(56.0, 72.0 * sy)
    self_char_box = (
        max(0, int(round(px - tol_x))),
        max(0, int(round(py - tol_up))),
        min(fw, int(round(px + tol_x))),
        min(fh, int(round(py + tol_dn))),
    )

    core_x = max(24.0, 34.0 * sx)
    core_up = max(30.0, 42.0 * sy)
    core_dn = max(36.0, 50.0 * sy)
    self_core_box = (
        max(0, int(round(px - core_x))),
        max(0, int(round(py - core_up))),
        min(fw, int(round(px + core_x))),
        min(fh, int(round(py + core_dn))),
    )
    return ((px, py), self_char_box, self_core_box)


def build_overlay_frame_data(
    *,
    game_rect: tuple[int, int, int, int],
    world: WorldState | None,
    fsm_state: BotState,
    config: BotConfig,
    loop_hz: float,
    vision_ms: float = 0.0,
    action_summary: str = "",
    backend: str = "",
    input_mode: str = "",
    game_focused: bool = True,
    travel_desc: str = "",
    deaths_count: int = 0,
    repairs_count: int = 0,
    repair_rem_min: float | None = None,
    hp_ratio_override: float | None = None,
    mp_ratio_override: float | None = None,
    coords_override: tuple[int, int] | None = None,
) -> OverlayFrameData:
    """Build an immutable OverlayFrameData from live WorldState & BotConfig."""
    left, top, gw, gh = game_rect
    fw = int(world.snapshot.frame_width) if world is not None else max(1, int(gw))
    fh = int(world.snapshot.frame_height) if world is not None else max(1, int(gh))
    scale_x = fw / 1920.0
    ks_radius = float(config.perception.get("ks_melee_radius_px", 75.0)) * scale_x
    hud_fixed = bool(config.perception.get("hud_fixed_pixel_ui", False))
    p_center, self_char_box, self_core_box = compute_self_exclusion_boxes(
        fw, fh, player_center_norm=config.player_center_norm(), hud_fixed_pixel_ui=hud_fixed
    )
    hotkey_lbl = str(config.safety.get("overlay_toggle", "F6")).upper()
    in_mode = input_mode or (
        "AHK-BG"
        if str(config.controls.get("ahk_mode", "vm_foreground")).lower() == "host_background"
        else "AHK-VM"
    )

    if world is None:
        return OverlayFrameData(
            game_rect=(int(left), int(top), int(gw), int(gh)),
            frame_width=fw,
            frame_height=fh,
            phase=fsm_state.value,
            action=action_summary or "IDLE",
            loop_hz=float(loop_hz),
            vision_ms=float(vision_ms),
            hp_pct=float(hp_ratio_override if hp_ratio_override is not None else 1.0) * 100.0,
            mp_pct=float(mp_ratio_override if mp_ratio_override is not None else 1.0) * 100.0,
            coords=coords_override,
            backend=backend or "OpenCV-Gold",
            input_mode=in_mode,
            travel_desc=travel_desc,
            deaths_count=deaths_count,
            repairs_count=repairs_count,
            repair_rem_min=repair_rem_min,
            self_char_box=self_char_box,
            self_core_box=self_core_box,
            player_center=p_center,
            overlay_hotkey=hotkey_lbl,
            game_focused=game_focused,
            timestamp=time.monotonic(),
        )

    snap = world.snapshot
    mobs_out: list[OverlayMob] = []
    locked_ov: OverlayMob | None = None
    st_val = getattr(fsm_state, "value", str(fsm_state))
    in_combat_wait = st_val in ("COMBAT_WAIT", "RETURN_TO_CENTER")
    in_loot = st_val == "LOOT"
    # Hide LOCK & MOB boxes during COMBAT_WAIT and LOOT!
    if not in_combat_wait and not in_loot:
        for mob in world.mobs:
            is_locked = world.locked_mob is not None and (
                mob.nameplate_bbox == world.locked_mob.nameplate_bbox
                or mob.body_bbox == world.locked_mob.body_bbox
            )
            om = OverlayMob(
                bbox=mob.body_bbox,
                click_x=int(mob.click_x),
                click_y=int(mob.click_y),
                conf=float(mob.confidence),
                dist_px=float(mob.dist_to_player_px),
                safe=True,
                locked=is_locked,
                hp_ratio=mob.hp_ratio_est,
                label=f"LOCK#{world.locked_mob_id}" if is_locked and world.locked_mob_id is not None else "",
                class_name=mob.class_name,
            )
            mobs_out.append(om)
            if is_locked:
                locked_ov = om

        if world.locked_mob is not None and locked_ov is None:
            lm = world.locked_mob
            locked_ov = OverlayMob(
                bbox=lm.body_bbox,
                click_x=int(lm.click_x),
                click_y=int(lm.click_y),
                conf=float(lm.confidence),
                dist_px=float(lm.dist_to_player_px),
                safe=True,
                locked=True,
                hp_ratio=lm.hp_ratio_est,
                label=f"LOCK#{world.locked_mob_id}" if world.locked_mob_id is not None else "LOCK",
                class_name=lm.class_name,
            )

        for cm in snap.contested_mobs:
            mobs_out.append(
                OverlayMob(
                    bbox=cm.body_bbox,
                    click_x=int(cm.click_x),
                    click_y=int(cm.click_y),
                    conf=float(cm.confidence),
                    dist_px=float(cm.dist_to_player_px),
                    safe=False,
                    locked=False,
                    hp_ratio=cm.hp_ratio_est,
                    label=cm.contest_reason or "SKIP [KS]",
                    class_name=cm.class_name,
                )
            )

    players_out = (
        ()
        if in_combat_wait
        else tuple(
            OverlayPlayer(
                bbox=(int(bx1), int(by1), int(bx2), int(by2)),
                cx=(int(bx1) + int(bx2)) // 2,
                cy=int(by1) + int(round(0.35 * max(1, int(by2) - int(by1)))),
                ks_radius_px=ks_radius,
                conf=snap.other_player_confs[i] if i < len(snap.other_player_confs) else None,
                class_name="player",
            )
            for i, (bx1, by1, bx2, by2) in enumerate(snap.other_player_boxes)
        )
    )

    # During COMBAT_WAIT draw ZERO GOLD/corpse boxes; during LOOT show close Gold Bags only!
    loot_out = (
        ()
        if in_combat_wait
        else tuple(
            OverlayLoot(
                bbox=(int(bag.bbox[0]), int(bag.bbox[1]), int(bag.bbox[0] + bag.bbox[2]), int(bag.bbox[1] + bag.bbox[3])),
                click_x=int(bag.click_x),
                click_y=int(bag.click_y),
                score=float(bag.score),
                index=idx,
                class_name=bag.class_name,
            )
            for idx, bag in enumerate(world.loot, start=1)
        )
    )

    corpses_out = ()

    target_hp_pct: float | None = None
    if locked_ov is not None and locked_ov.hp_ratio is not None:
        target_hp_pct = max(0.0, min(100.0, float(locked_ov.hp_ratio) * 100.0))
    elif snap.combat_target_hp_ratio is not None:
        target_hp_pct = max(0.0, min(100.0, float(snap.combat_target_hp_ratio) * 100.0))
    elif snap.engaged_mob is not None and snap.engaged_mob.hp_ratio_est is not None:
        target_hp_pct = max(0.0, min(100.0, float(snap.engaged_mob.hp_ratio_est) * 100.0))

    coords_val = (
        (int(round(world.coords_smoothed.x)), int(round(world.coords_smoothed.y)))
        if world.coords_smoothed is not None
        else coords_override
    )
    vis_ms = float(vision_ms) if vision_ms > 0.0 else float(snap.timings_ms.get("total", 0.0))
    act_str = action_summary or world.last_action_summary or "-"

    return OverlayFrameData(
        game_rect=(int(left), int(top), int(gw), int(gh)),
        frame_width=fw,
        frame_height=fh,
        phase=fsm_state.value,
        action=act_str,
        loop_hz=float(loop_hz),
        vision_ms=vis_ms,
        target_hp_pct=target_hp_pct,
        hp_pct=float(world.hp_ratio) * 100.0,
        mp_pct=float(world.mp_ratio) * 100.0,
        coords=coords_val,
        backend=backend or snap.detector_backend,
        input_mode=in_mode,
        travel_desc=travel_desc,
        deaths_count=deaths_count,
        repairs_count=repairs_count,
        repair_rem_min=repair_rem_min,
        mobs=tuple(mobs_out),
        locked_mob=locked_ov,
        players=players_out,
        loot=loot_out,
        corpses=corpses_out,
        self_char_box=self_char_box,
        self_core_box=self_core_box,
        player_center=p_center,
        overlay_hotkey=hotkey_lbl,
        game_focused=game_focused,
        timestamp=float(snap.timestamp),
    )


def draw_overlay_scene(canvas: Any, data: OverlayFrameData) -> int:
    """Render Gold Rustler detections, Anti-KS players, Gold Bags, corpses, self boxes & HUD banner."""
    canvas.delete("all")
    drawn = 0

    _, _, gw, gh = data.game_rect
    cw = max(1, int(gw))
    ch = max(1, int(gh))
    sx = cw / max(1.0, float(data.frame_width))
    sy = ch / max(1.0, float(data.frame_height))

    def _x(v: float) -> int:
        return int(round(v * sx))

    def _y(v: float) -> int:
        return int(round(v * sy))

    def _text_tag(x: int, y: int, text: str, fg: str, bg: str = "#0B1120", font_sz: int = 8) -> None:
        nonlocal drawn
        est_w = max(36, int(len(text) * (font_sz * 0.68)) + 8)
        est_h = font_sz + 6
        tx = max(4, min(cw - est_w - 4, x))
        ty = max(4, min(ch - est_h - 4, y))
        canvas.create_rectangle(tx, ty, tx + est_w, ty + est_h, fill=bg, outline="")
        canvas.create_text(
            tx + 4,
            ty + est_h // 2,
            text=text,
            fill=fg,
            anchor="w",
            font=("Consolas", font_sz, "bold"),
        )
        drawn += 2

    if data.self_char_box is not None:
        x1, y1, x2, y2 = data.self_char_box
        canvas.create_rectangle(_x(x1), _y(y1), _x(x2), _y(y2), outline="#64748B", width=1, dash=(4, 4))
        canvas.create_text(
            _x(x1) + 3, max(6, _y(y1) - 8), text="SELF", fill="#94A3B8", anchor="w", font=("Consolas", 8)
        )
        drawn += 2

    if data.self_core_box is not None:
        x1, y1, x2, y2 = data.self_core_box
        canvas.create_rectangle(_x(x1), _y(y1), _x(x2), _y(y2), outline="#38BDF8", width=1, dash=(2, 3))
        drawn += 1

    if data.player_center is not None:
        pcx, pcy = _x(data.player_center[0]), _y(data.player_center[1])
        canvas.create_line(pcx - 6, pcy, pcx + 6, pcy, fill="#94A3B8", width=1)
        canvas.create_line(pcx, pcy - 6, pcx, pcy + 6, fill="#94A3B8", width=1)
        drawn += 2

    for pl in data.players:
        px1, py1, px2, py2 = pl.bbox
        rx = max(10, int(round(pl.ks_radius_px * sx)))
        ry = max(10, int(round(pl.ks_radius_px * sy)))
        cx, cy = _x(pl.cx), _y(pl.cy)
        canvas.create_oval(cx - rx, cy - ry, cx + rx, cy + ry, outline="#FF3B3B", width=2, dash=(6, 4))
        canvas.create_rectangle(_x(px1), _y(py1), _x(px2), _y(py2), outline="#FF3B3B", width=2)
        pl_lbl = (
            f"{pl.class_name} {pl.conf:.2f} (R={int(round(pl.ks_radius_px))}px)"
            if pl.conf is not None
            else f"PLAYER (R={int(round(pl.ks_radius_px))}px)"
        )
        _text_tag(_x(px1), _y(py1) - 16, pl_lbl, "#FCA5A5", bg="#450A0A")
        drawn += 2

    for cp in data.corpses:
        cx1, cy1, cx2, cy2 = cp.bbox
        canvas.create_rectangle(_x(cx1), _y(cy1), _x(cx2), _y(cy2), outline="#C084FC", width=2, dash=(5, 3))
        _text_tag(_x(cx1), _y(cy1) - 16, f"{cp.label} [SKIP]", "#E9D5FF", bg="#3B0764")
        drawn += 1

    for bag in data.loot:
        lx1, ly1, lx2, ly2 = bag.bbox
        canvas.create_rectangle(_x(lx1), _y(ly1), _x(lx2), _y(ly2), outline="#FFD700", width=2)
        bcx, bcy = _x(bag.click_x), _y(bag.click_y)
        canvas.create_oval(bcx - 3, bcy - 3, bcx + 3, bcy + 3, fill="#FEF08A", outline="#CA8A04", width=1)
        if bag.index <= 16:
            bag_lbl = (
                f"{bag.class_name} {bag.score:.2f}"
                if bag.class_name
                else f"GOLD#{bag.index}"
            )
            _text_tag(_x(lx1), _y(ly1) - 15, bag_lbl, "#FEF08A", bg="#422006")
        drawn += 2

    safe_count = 0
    skip_count = 0
    for mob in data.mobs:
        if mob.safe:
            safe_count += 1
        else:
            skip_count += 1
        if mob.locked:
            continue
        mx1, my1, mx2, my2 = mob.bbox
        if mob.safe:
            hp_s = f" HP:{mob.hp_ratio:.0%}" if mob.hp_ratio is not None else ""
            if mob.class_name:
                lbl = f"{mob.class_name} {mob.conf:.2f} {int(round(mob.dist_px))}px{hp_s}"
            else:
                lbl = f"GOLD {int(round(mob.dist_px))}px{hp_s}"
            canvas.create_rectangle(_x(mx1), _y(my1), _x(mx2), _y(my2), outline="#00FF66", width=2)
            _text_tag(_x(mx1), _y(my1) - 16, lbl, "#86EFAC", bg="#052E16")
        else:
            base_lbl = mob.label or "SKIP [KS]"
            lbl = f"{mob.class_name} {mob.conf:.2f} {base_lbl}" if mob.class_name else base_lbl
            canvas.create_rectangle(_x(mx1), _y(my1), _x(mx2), _y(my2), outline="#FB923C", width=2, dash=(4, 3))
            _text_tag(_x(mx1), _y(my1) - 16, lbl, "#FED7AA", bg="#431407")
        drawn += 1

    if data.locked_mob is not None:
        lm = data.locked_mob
        lx1, ly1, lx2, ly2 = lm.bbox
        canvas.create_rectangle(_x(lx1), _y(ly1), _x(lx2), _y(ly2), outline="#00FFFF", width=3)
        ax, ay = _x(lm.click_x), _y(lm.click_y)
        canvas.create_line(ax - 12, ay, ax + 12, ay, fill="#FF2D2D", width=3)
        canvas.create_line(ax, ay - 12, ax, ay + 12, fill="#FF2D2D", width=3)
        canvas.create_oval(ax - 3, ay - 3, ax + 3, ay + 3, fill="#FACC15", outline="#FF2D2D", width=1)
        hp_s = f" HP:{lm.hp_ratio:.0%}" if lm.hp_ratio is not None else ""
        head = lm.label or "LOCK"
        cls_s = f" {lm.class_name} {lm.conf:.2f}" if lm.class_name else ""
        _text_tag(_x(lx1), _y(ly1) - 18, f"{head}{cls_s} {int(round(lm.dist_px))}px{hp_s}", "#67E8F9", bg="#083344", font_sz=9)
        drawn += 4

    # Keep debug banner compact at top (y=2..42) so it NEVER overlaps RYL's 'INTERFACE TYPE: Mouse / Keyboard' banner (y~55..115)
    bx1 = max(124, int(round(cw * 0.068)))
    by1 = 2
    banner_w = min(cw - bx1 - 12, 790)
    if banner_w < 360:
        bx1 = 8
        banner_w = max(240, min(cw - 16, 790))
    bx2 = bx1 + banner_w
    by2 = by1 + 40

    phase_color = {
        "COMBAT_WAIT": "#F87171",
        "LOOT": "#FACC15",
        "SEARCH_MOB": "#38BDF8",
        "DEAD_RESPAWN": "#EF4444",
        "REPAIR_RECALL": "#F59E0B",
        "REPAIR_AT_NPC": "#34D399",
        "TELEPORT_TO_SPOT": "#38BDF8",
        "TRAVEL_TO_SPOT": "#22D3EE",
        "NAVIGATE_TO_SPOT": "#C084FC",
        "PAUSED": "#94A3B8",
    }.get(data.phase.upper(), "#34D399")

    thp_txt = f"{data.target_hp_pct:.0f}%" if data.target_hp_pct is not None else "--"
    coord_txt = f"({data.coords[0]},{data.coords[1]})" if data.coords is not None else "(-,-)"
    act_clip = (data.action or "-")[:42]
    trv_tag = f" ({data.travel_desc})" if data.travel_desc else ""
    rep_txt = (
        " REP:NOW!"
        if (data.repair_rem_min is not None and data.repair_rem_min <= 0.01)
        else (f" REP:{data.repair_rem_min:.0f}m" if data.repair_rem_min is not None else "")
    )
    line1 = f"PHASE: {data.phase}{trv_tag}  |  MODE: {data.input_mode}  |  ACT: {act_clip}"
    line2 = (
        f"HZ: {data.loop_hz:.1f}  |  VIS: {data.vision_ms:.1f}ms  |  THP: {thp_txt}  |  "
        f"MOB: {safe_count}(skip:{skip_count}) GOLD: {len(data.loot)}  |  POS: {coord_txt}{rep_txt}  |  "
        f"D:{data.deaths_count} R:{data.repairs_count}  |  [F7]:Rep [{data.overlay_hotkey}]:OV"
    )

    canvas.create_rectangle(bx1, by1, bx2, by2, fill="#0B1120", outline="#22D3EE", width=1)
    canvas.create_rectangle(bx1, by1, bx1 + 4, by2, fill=phase_color, outline="")
    canvas.create_text(bx1 + 12, by1 + 11, text=line1, fill=phase_color, anchor="w", font=("Consolas", 9, "bold"))
    canvas.create_text(bx1 + 12, by1 + 28, text=line2, fill="#E2E8F0", anchor="w", font=("Consolas", 8, "bold"))
    drawn += 4
    return drawn


class TransparentDebugOverlay:
    """Real-time Win32 transparent & click-through overlay anchored over the RYL game window."""

    def __init__(
        self,
        *,
        master: Any | None = None,
        enabled: bool = True,
        refresh_ms: int = 33,
        exclude_from_capture: bool = True,
        on_toggle: Callable[[bool], None] | None = None,
    ) -> None:
        self._master = master
        self._enabled = bool(enabled)
        self._refresh_ms = max(16, int(refresh_ms))
        self._exclude_from_capture = bool(exclude_from_capture)
        self._on_toggle = on_toggle

        self._lock = threading.Lock()
        self._latest_data: OverlayFrameData | None = None
        self._dirty: bool = False
        self._hidden_until_frame: bool = False
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()

        self._thread: threading.Thread | None = None
        self._win: Any | None = None
        self._canvas: Any | None = None
        self._hwnd: int = 0
        self._last_rect: tuple[int, int, int, int] | None = None
        self._is_shown: bool = False
        self._after_id: str | None = None
        self._started: bool = False
        self._style_info: dict[str, int | bool] = {}

    @property
    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def style_info(self) -> dict[str, int | bool]:
        return dict(self._style_info)

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._stop_event.clear()
        if self._master is not None:
            self._init_attached_window()
        else:
            self._thread = threading.Thread(
                target=self._run_standalone_loop,
                name="VikingsDebugOverlay",
                daemon=True,
            )
            self._thread.start()
            self._ready_event.wait(timeout=2.0)

    def update(self, data: OverlayFrameData) -> None:
        with self._lock:
            self._latest_data = data
            self._hidden_until_frame = False
            self._dirty = True

    def clear_and_hide(self) -> None:
        with self._lock:
            self._latest_data = None
            self._hidden_until_frame = True
            self._dirty = True

    def set_enabled(self, enabled: bool, *, notify: bool = False) -> bool:
        val = bool(enabled)
        changed = False
        with self._lock:
            if self._enabled != val:
                self._enabled = val
                self._dirty = True
                changed = True
        if changed:
            logger.info("Transparent Debug Overlay: %s", "ON" if val else "OFF")
            if notify and self._on_toggle is not None:
                try:
                    self._on_toggle(val)
                except Exception:
                    pass
        return val

    def toggle(self) -> bool:
        with self._lock:
            new_val = not self._enabled
            self._enabled = new_val
            self._dirty = True
        logger.info("Transparent Debug Overlay toggled: %s", "ON" if new_val else "OFF")
        if self._on_toggle is not None:
            try:
                self._on_toggle(new_val)
            except Exception:
                pass
        return new_val

    def stop(self) -> None:
        self._stop_event.set()
        if self._master is not None:
            try:
                self._master.after(0, self._destroy_tk_window)
            except Exception:
                self._destroy_tk_window()
        elif self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._started = False

    def _create_window_objects(self, tk_mod: Any, is_toplevel: bool) -> None:
        win = tk_mod.Toplevel(self._master) if is_toplevel else tk_mod.Tk()
        win.withdraw()
        win.overrideredirect(True)
        win.configure(bg=TRANSPARENT_COLOR_HEX)
        try:
            win.wm_attributes("-topmost", True)
            win.wm_attributes("-transparentcolor", TRANSPARENT_COLOR_HEX)
        except Exception:
            pass

        canvas = tk_mod.Canvas(win, bg=TRANSPARENT_COLOR_HEX, highlightthickness=0, bd=0)
        canvas.pack(fill=tk_mod.BOTH, expand=True)
        win.update_idletasks()

        hwnd = int(win.winfo_id())
        self._style_info = apply_clickthrough_styles(
            hwnd,
            color_key_bgr=TRANSPARENT_COLOR_BGR,
            exclude_from_capture=self._exclude_from_capture,
            detach_owner=is_toplevel,
        )
        self._hwnd = int(self._style_info.get("hwnd") or hwnd)
        self._win = win
        self._canvas = canvas
        self._is_shown = False
        self._last_rect = None

    def _init_attached_window(self) -> None:
        try:
            import tkinter as tk

            self._create_window_objects(tk, is_toplevel=True)
            self._ready_event.set()
            self._schedule_next_tick()
        except Exception as exc:
            logger.warning("Failed to initialize attached TransparentDebugOverlay: %s", exc)
            self._ready_event.set()

    def _run_standalone_loop(self) -> None:
        try:
            import tkinter as tk

            self._create_window_objects(tk, is_toplevel=False)
            self._ready_event.set()
            self._schedule_next_tick()
            assert self._win is not None
            self._win.mainloop()
        except Exception as exc:
            logger.warning("Standalone TransparentDebugOverlay exited: %s", exc)
            self._ready_event.set()
        finally:
            self._win = None
            self._canvas = None

    def _schedule_next_tick(self) -> None:
        if self._stop_event.is_set() or self._win is None:
            return
        try:
            self._after_id = self._win.after(self._refresh_ms, self._on_timer_tick)
        except Exception:
            pass

    def _hide_window(self) -> None:
        if self._win is None:
            return
        if self._is_shown:
            if self._canvas is not None:
                try:
                    self._canvas.delete("all")
                except Exception:
                    pass
            try:
                self._win.withdraw()
            except Exception:
                pass
            position_overlay_window(self._hwnd, 0, 0, 1, 1, visible=False)
            self._is_shown = False

    def _on_timer_tick(self) -> None:
        if self._stop_event.is_set():
            self._destroy_tk_window()
            return

        with self._lock:
            enabled = self._enabled
            data = self._latest_data
            hidden_until = self._hidden_until_frame
            dirty = self._dirty
            self._dirty = False

        if not enabled or hidden_until or data is None:
            self._hide_window()
            self._schedule_next_tick()
            return

        left, top, width, height = data.game_rect
        if width < 200 or height < 200:
            self._hide_window()
            self._schedule_next_tick()
            return

        if self._win is not None and self._canvas is not None:
            rect = (int(left), int(top), int(width), int(height))
            if (not self._is_shown) or (rect != self._last_rect):
                try:
                    self._win.geometry(f"{rect[2]}x{rect[3]}+{rect[0]}+{rect[1]}")
                    self._canvas.configure(width=rect[2], height=rect[3])
                except Exception:
                    pass
                if not self._is_shown:
                    apply_clickthrough_styles(
                        self._hwnd or int(self._win.winfo_id()),
                        color_key_bgr=TRANSPARENT_COLOR_BGR,
                        exclude_from_capture=self._exclude_from_capture,
                        detach_owner=(self._master is not None),
                    )
                    try:
                        self._win.deiconify()
                        self._win.update_idletasks()
                    except Exception:
                        pass
                    self._style_info = apply_clickthrough_styles(
                        int(self._win.winfo_id()),
                        color_key_bgr=TRANSPARENT_COLOR_BGR,
                        exclude_from_capture=self._exclude_from_capture,
                        detach_owner=(self._master is not None),
                    )
                    self._hwnd = int(self._style_info.get("hwnd") or self._hwnd or int(self._win.winfo_id()))
                position_overlay_window(self._hwnd, rect[0], rect[1], rect[2], rect[3], visible=True)
                self._last_rect = rect
                self._is_shown = True
                dirty = True

            if dirty:
                try:
                    draw_overlay_scene(self._canvas, data)
                except Exception as exc:
                    logger.debug("TransparentDebugOverlay draw error: %s", exc)

        self._schedule_next_tick()

    def _destroy_tk_window(self) -> None:
        if self._win is None:
            return
        try:
            if self._after_id is not None:
                self._win.after_cancel(self._after_id)
                self._after_id = None
        except Exception:
            pass
        try:
            self._win.destroy()
        except Exception:
            pass
        self._win = None
        self._canvas = None
        self._is_shown = False
