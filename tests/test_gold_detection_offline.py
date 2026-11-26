"""Offline Pure OpenCV calibration & verification for Gold Rustler, Anti-KS, Gold Bag & Mayat rejection."""

from __future__ import annotations

from pathlib import Path
import time

import cv2
import numpy as np
import pytest

from vikings_bot.capture.screen import FramePacket
from vikings_bot.config import load_config
from vikings_bot.perception.gold_detector import GoldRustlerDetector
from vikings_bot.perception.hud_mask import build_playfield_hud_mask, get_static_hud_rects
from vikings_bot.perception.pipeline import PerceptionPipeline

SCREENSHOTS_DIR = Path(r"C:\Program Files (x86)\Vikings Online\ScreenShots")
UPLOADED_DIR = Path(r"C:\Users\Desto\.gemini\antigravity\brain\04b1a6fb-5f21-4214-944b-a647347f4420\.user_uploaded")


@pytest.fixture(scope="module")
def cfg():
    return load_config("config/default.yaml")


@pytest.fixture(scope="module")
def pipe(cfg):
    return PerceptionPipeline(cfg)


def _load_frame(filename: str) -> FramePacket:
    path = SCREENSHOTS_DIR / filename
    if not path.exists():
        pytest.skip(f"Screenshot not available: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None
    return FramePacket(image_bgr=img, timestamp=1000.0, source_name=filename)


def test_hud_masking_and_foot_point_vs_torso_click_projection():
    # Static HUD masks cover minimap, party, skill bar, HP/MP, chat for both 1080p and 1024x768 VM
    for fw, fh, fixed in ((1920, 1080, False), (1024, 768, True)):
        rects = get_static_hud_rects(fw, fh, hud_fixed_pixel_ui=fixed)
        assert len(rects) >= 5
        mask = build_playfield_hud_mask(fw, fh, (0, 0, fw, fh), hud_fixed_pixel_ui=fixed)
        assert mask[10, 10] == 0  # minimap masked
        assert mask[fh - 20, fw - 20] == 0  # chat masked
        assert mask[fh // 2, fw // 2] == 255  # playfield center open
        if fixed and fw == 1024 and fh == 768:
            # Chat tabs ('Public | Shout | Trade | Party | Guild') at x>=442, y>=518 must be 100% masked!
            assert mask[558, 530] == 0
            assert mask[518, 442] == 0
            # Left skill bar, bottom-left HP/MP panel, and right Ak'Kan/Human faction indicator
            assert mask[500, 20] == 0
            assert mask[700, 120] == 0
            assert mask[390, 980] == 0

    # Upper-torso click point is at y1 + 0.35*(y2-y1), foot point is at (cx, y2)
    body, (cx, cy), (fx, fy) = GoldRustlerDetector.project_body_from_nameplate(
        (920, 380, 80, 12), 1920, 1080, click_y_top_ratio=0.35
    )
    x1, y1, x2, y2 = body
    assert cx == fx == 960
    assert fy == y2
    assert cy == int(round(y1 + 0.35 * (y2 - y1)))
    assert y1 < cy < fy


def test_gold_rustler_detection_hp_bar_and_player_exclusion(pipe: PerceptionPipeline):
    # Frame (1): Gold Rustlers present + friendly player Nana-Gucci on left (must NOT be targeted!)
    f1 = _load_frame("Vikings Online_Mortega-_2026_11_22_04_06(1).jpg")
    mobs1, _cont1, _pl1, _pb1 = pipe.mobs.detect_with_context(f1)
    assert len(mobs1) >= 4, f"Expected >=4 Gold Rustlers in frame (1), got {len(mobs1)}"
    for m in mobs1:
        mx, my, _, _ = m.nameplate_bbox
        assert not (200 <= mx <= 330 and 430 <= my <= 510), f"False positive on Nana-Gucci: {m}"

    # Frame (30): multiple free Gold Rustlers
    f30 = _load_frame("Vikings Online_Mortega-_2026_11_22_04_09(30).jpg")
    mobs30 = pipe.mobs.detect(f30)
    assert len(mobs30) >= 3, f"Expected >=3 Gold Rustlers in frame (30), got {len(mobs30)}"

    # Frames (20) & (25): engaged Gold Rustler with active green HP bar
    for engaged_file in (
        "Vikings Online_Mortega-_2026_11_22_04_07(20).jpg",
        "Vikings Online_Mortega-_2026_11_22_04_07(25).jpg",
    ):
        fe = _load_frame(engaged_file)
        mobs_e = pipe.mobs.detect(fe)
        assert any(m.has_hp_bar for m in mobs_e), f"Expected engaged HP bar in {engaged_file}: {mobs_e}"


def test_anti_ks_combat_status_without_blind_150px_discard(cfg):
    """Verify untouched mob near another player (110px away) stays SAFE, while damaged mob next to player is SKIPPED."""
    det = GoldRustlerDetector(cfg)
    hsv_bg = np.zeros((1080, 1920, 3), dtype=np.uint8)
    hsv_bg[:, :] = (28, 175, 65)  # grass background
    bgr = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)

    def _paint_text_bar(img_bgr: np.ndarray, x: int, y: int, w: int, h: int, hsv_color: tuple[int, int, int]) -> None:
        patch = np.full((h, w, 3), hsv_color, dtype=np.uint8)
        # Simulate crisp character stroke gaps (50% density so fill_ratio < 0.78)
        patch[:, ::2] = (28, 175, 45)
        img_bgr[y : y + h, x : x + w] = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)

    # Other player nameplate at x=500..590, y=410..422 (green text, outside self zone)
    _paint_text_bar(bgr, 500, 410, 90, 12, (62, 185, 220))

    # Mob A (UNTOUCHED, no HP bar) standing 115px to the right of player (x=660..736, y=410..422) -> MUST STAY SAFE!
    _paint_text_bar(bgr, 660, 410, 76, 12, (162, 95, 225))

    # Mob B (DAMAGED HP ~55%) standing right beside player at x=545..620, y=428..440 -> MUST BE CONTESTED (SKIP)!
    _paint_text_bar(bgr, 545, 428, 76, 12, (162, 95, 225))
    # Dark HP bar track + half-filled bright green bar below Mob B
    bgr[443:448, 550:618] = (20, 20, 20)
    hp_g = cv2.cvtColor(np.full((4, 36, 3), (62, 220, 225), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    bgr[444:448, 550:586] = hp_g

    safe, contested, pl_pts, pl_boxes = det.detect_with_context(FramePacket(image_bgr=bgr, timestamp=1.0))
    assert len(pl_pts) >= 1 and len(pl_boxes) >= 1
    assert any(640 <= m.nameplate_bbox[0] <= 700 for m in safe), f"Untouched mob near player was falsely discarded: safe={safe}, cont={contested}"
    assert any(530 <= m.nameplate_bbox[0] <= 575 for m in contested), f"Damaged mob next to player should be contested: safe={safe}, cont={contested}"


def test_gold_bag_detection_precision_and_corpse_rejection(pipe: PerceptionPipeline):
    # Real dropped Gold Bags in frames (15) & (16)
    for loot_file in (
        "Vikings Online_Mortega-_2026_11_22_04_07(15).jpg",
        "Vikings Online_Mortega-_2026_11_22_04_07(16).jpg",
    ):
        fl = _load_frame(loot_file)
        snap = pipe.analyze(fl)
        assert len(snap.loot) >= 4, f"Expected >=4 gold bags in {loot_file}, got {len(snap.loot)}"

    # Empty ground frames (1), (6), (30) -> 0 false-positive gold bags
    for empty_file in (
        "Vikings Online_Mortega-_2026_11_22_04_06(1).jpg",
        "Vikings Online_Mortega-_2026_11_22_04_06(6).jpg",
        "Vikings Online_Mortega-_2026_11_22_04_09(30).jpg",
    ):
        fn = _load_frame(empty_file)
        snap_e = pipe.analyze(fn)
        assert len(snap_e.loot) == 0, f"Unexpected false-positive loot in {empty_file}: {snap_e.loot}"

    # Synthetic prone dead corpse ('mayat') with golden belt buckle + 1 genuine Gold Bag
    hsv_bg = np.zeros((1080, 1920, 3), dtype=np.uint8)
    hsv_bg[:, :] = (28, 185, 63)
    bgr = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)

    corpse_hsv = cv2.cvtColor(bgr[480:508, 700:784], cv2.COLOR_BGR2HSV)
    corpse_hsv[8:22, 0:22] = (28, 80, 18)
    corpse_hsv[6:22, 22:46] = (25, 35, 175)
    corpse_hsv[4:22, 46:72] = (88, 155, 85)
    corpse_hsv[6:18, 72:84] = (105, 110, 95)
    corpse_hsv[10:15, 43:48] = (16, 220, 210)
    corpse_hsv[15:23, 40:50] = (20, 90, 22)
    bgr[480:508, 700:784] = cv2.cvtColor(corpse_hsv, cv2.COLOR_HSV2BGR)

    bag_hsv = cv2.cvtColor(bgr[555:573, 1114:1128], cv2.COLOR_BGR2HSV)
    bag_hsv[4:18, 1:13] = (18, 120, 22)
    bag_hsv[1:6, 4:10] = (16, 230, 220)
    bgr[555:573, 1114:1128] = cv2.cvtColor(bag_hsv, cv2.COLOR_HSV2BGR)

    snap_s = pipe.analyze(FramePacket(image_bgr=bgr, timestamp=10.0))
    assert len(snap_s.dead_corpses) == 1
    assert len(snap_s.loot) == 1
    assert abs(snap_s.loot[0].click_x - 1121) <= 12


def test_pure_opencv_vm_cpu_latency_budget(cfg):
    """Ensure 1024x768 VM frame runs Pure OpenCV perception rapidly on CPU (no GPU/PyTorch)."""
    cfg_vm = load_config("config/default.yaml")
    cfg_vm.perception["hud_fixed_pixel_ui"] = True
    pipe_vm = PerceptionPipeline(cfg_vm)

    img = np.full((768, 1024, 3), (38, 82, 46), dtype=np.uint8)
    pkt = FramePacket(image_bgr=img, timestamp=1.0)
    # Warm up HUD mask cache
    pipe_vm.analyze(pkt)

    t0 = time.perf_counter()
    runs = 10
    for i in range(runs):
        pipe_vm.analyze(FramePacket(image_bgr=img, timestamp=2.0 + i * 0.05))
    avg_ms = ((time.perf_counter() - t0) * 1000.0) / runs
    assert avg_ms < 25.0, f"Pure OpenCV VM 1024x768 average latency too slow: {avg_ms:.1f}ms"


def test_dusk_gold_rustler_nameplate_and_hp_bar_below_loot_guard():
    """Verify dusk/night Gold Rustler nameplate detection & HP-bar below-strip (+110px) loot guard on 1024x768."""
    cfg_vm = load_config("config/default.yaml")
    cfg_vm.perception["hud_fixed_pixel_ui"] = True
    pipe_vm = PerceptionPipeline(cfg_vm)

    # Brownish dusk playfield background (1024x768)
    hsv_bg = np.zeros((768, 1024, 3), dtype=np.uint8)
    hsv_bg[:, :] = (18, 85, 58)
    bgr = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)

    # 1. Dimmed dusk/night magenta Gold Rustler nameplate at (340, 260, 58, 9) with low S=26, V=94
    np_patch = np.full((9, 58, 3), (144, 26, 94), dtype=np.uint8)
    np_patch[:, ::2] = (18, 85, 45)
    bgr[260:269, 340:398] = cv2.cvtColor(np_patch, cv2.COLOR_HSV2BGR)
    mobs = pipe_vm.mobs.detect(FramePacket(image_bgr=bgr, timestamp=1.0))
    assert any(320 <= m.nameplate_bbox[0] <= 360 and 250 <= m.nameplate_bbox[1] <= 275 for m in mobs), (
        f"Dusk/night Gold Rustler nameplate missed: {mobs}"
    )

    # 2. HP-bar below-strip loot guard: green HP bar & red HP bar with golden gleam +55px underneath (no mob bbox passed!)
    bgr2 = cv2.cvtColor(hsv_bg, cv2.COLOR_HSV2BGR)
    # Green HP bar at (260, 240, 52, 4) + golden chest gleam at (280, 295)
    bgr2[240:244, 260:312] = cv2.cvtColor(np.full((4, 52, 3), (62, 220, 220), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    bgr2[295:309, 276:288] = cv2.cvtColor(np.full((14, 12, 3), (18, 120, 22), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    bgr2[291:296, 279:285] = cv2.cvtColor(np.full((5, 6, 3), (16, 230, 220), dtype=np.uint8), cv2.COLOR_HSV2BGR)

    # Red HP bar at (620, 240, 52, 4) + golden chest gleam at (640, 300)
    bgr2[240:244, 620:672] = cv2.cvtColor(np.full((4, 52, 3), (2, 220, 220), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    bgr2[300:314, 636:648] = cv2.cvtColor(np.full((14, 12, 3), (18, 120, 22), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    bgr2[296:301, 639:645] = cv2.cvtColor(np.full((5, 6, 3), (16, 230, 220), dtype=np.uint8), cv2.COLOR_HSV2BGR)

    # Genuine isolated Gold Bag far from any HP bar at (420, 360) (1024x768 scale: 4x5 sparkle + dark pouch)
    bgr2[364:376, 416:426] = cv2.cvtColor(np.full((12, 10, 3), (18, 120, 22), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    bgr2[360:364, 419:424] = cv2.cvtColor(np.full((4, 5, 3), (16, 230, 220), dtype=np.uint8), cv2.COLOR_HSV2BGR)

    drops, _ = pipe_vm.loot.detect(FramePacket(image_bgr=bgr2, timestamp=2.0), mobs=())
    assert len(drops) == 1, f"Expected only the 1 isolated Gold Bag (both HP-bar gleams masked), got {drops}"
    assert abs(drops[0].click_x - 421) <= 12 and abs(drops[0].click_y - 369) <= 14


def test_vm_1024_raw_night_screenshot_regression():
    """Regression test on real VMware 1024x768 night screenshot (vm_1024_raw.jpg): POS, HP/MP, 3 mobs, 0 FP loot."""
    fixture = Path(__file__).parent / "fixtures" / "vm_1024_raw.jpg"
    assert fixture.exists(), f"Missing regression fixture: {fixture}"
    img = cv2.imread(str(fixture), cv2.IMREAD_COLOR)
    assert img is not None and img.shape == (768, 1024, 3)

    cfg_vm = load_config("config/default.yaml")
    cfg_vm.perception["hud_fixed_pixel_ui"] = True
    pipe_vm = PerceptionPipeline(cfg_vm)

    snap = pipe_vm.analyze(FramePacket(image_bgr=img, timestamp=1.0, source_name="vm_1024_raw.jpg"))
    assert snap.coords is not None, "Failed to read minimap coordinates on vm_1024_raw.jpg"
    assert (snap.coords.x, snap.coords.y) == (600, 1813), f"Expected POS (600, 1813), got {snap.coords}"
    assert snap.hp_ratio > 0.90, f"Expected HP > 90% on 5700/5700 bar, got {snap.hp_ratio:.2%}"
    assert snap.mp_ratio > 0.90, f"Expected MP > 90% on 6200/6200 bar, got {snap.mp_ratio:.2%}"
    assert len(snap.mobs) >= 3, f"Expected >= 3 night Gold Rustlers on vm_1024_raw.jpg, got {len(snap.mobs)}: {snap.mobs}"
    assert len(snap.loot) == 0, f"Expected 0 false-positive Gold Bags on vm_1024_raw.jpg, got {snap.loot}"
    for m in snap.mobs:
        assert (m.body_bbox[3] - m.body_bbox[1]) <= 135, f"Body bbox too tall on 1024x768: {m.body_bbox}"

    # Verify fill_ratio rejects solid horizon smudge (>0.68), 14x3 MORPH_CLOSE merges split 'Gold  Rustler',
    # and foreground body_bbox height is capped <= 135px on 1024x768
    hsv_syn = np.zeros((768, 1024, 3), dtype=np.uint8)
    hsv_syn[:, :] = (28, 120, 62)
    bgr_syn = cv2.cvtColor(hsv_syn, cv2.COLOR_HSV2BGR)
    # Solid reddish-magenta hill smudge (fill_ratio ~ 1.0 > 0.68 -> must be rejected!)
    bgr_syn[200:210, 220:265] = cv2.cvtColor(np.full((10, 45, 3), (160, 110, 140), dtype=np.uint8), cv2.COLOR_HSV2BGR)
    # Split 'Gold' + 10px gap + 'Rustler' (50% stroke density -> must merge into 1 single mob!)
    word_a = np.full((10, 24, 3), (160, 110, 190), dtype=np.uint8)
    word_a[:, ::2] = (28, 120, 50)
    word_b = np.full((10, 32, 3), (160, 110, 190), dtype=np.uint8)
    word_b[:, ::2] = (28, 120, 50)
    bgr_syn[480:490, 200:224] = cv2.cvtColor(word_a, cv2.COLOR_HSV2BGR)
    bgr_syn[480:490, 234:266] = cv2.cvtColor(word_b, cv2.COLOR_HSV2BGR)

    mobs_syn = pipe_vm.mobs.detect(FramePacket(image_bgr=bgr_syn, timestamp=2.0))
    assert len(mobs_syn) == 1, f"Expected 1 merged mob (and 0 solid hill smudge), got {mobs_syn}"
    assert mobs_syn[0].nameplate_bbox[2] >= 55
    assert (mobs_syn[0].body_bbox[3] - mobs_syn[0].body_bbox[1]) <= 135

    # Verify SELF zone (x=445..580, y=320..450 at 1024x768) is NEVER flagged as DEAD / MAYAT [SKIP]
    bgr_self = cv2.cvtColor(hsv_syn, cv2.COLOR_HSV2BGR)
    self_corpse = cv2.cvtColor(bgr_self[370:398, 470:554], cv2.COLOR_BGR2HSV)
    self_corpse[8:22, 0:22] = (28, 80, 18)
    self_corpse[6:22, 22:46] = (25, 35, 175)
    self_corpse[4:22, 46:72] = (88, 155, 85)
    self_corpse[6:18, 72:84] = (105, 110, 95)
    bgr_self[370:398, 470:554] = cv2.cvtColor(self_corpse, cv2.COLOR_HSV2BGR)
    pipe_vm.loot.detect(FramePacket(image_bgr=bgr_self, timestamp=3.0))
    assert pipe_vm.loot.last_corpse_boxes == (), f"SELF zone falsely marked as DEAD/MAYAT: {pipe_vm.loot.last_corpse_boxes}"
