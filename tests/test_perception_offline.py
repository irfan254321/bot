"""Offline HUD/Coord/Minimap verification against real Vikings Online screenshots."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from vikings_bot.capture.screen import FramePacket
from vikings_bot.config import load_config, roi_to_pixels
from vikings_bot.perception.pipeline import PerceptionPipeline

SCREENSHOTS_DIR = Path(r"C:\Program Files (x86)\Vikings Online\ScreenShots")
UPLOADED_DIR = Path(r"C:\Users\Desto\.gemini\antigravity\brain\04b1a6fb-5f21-4214-944b-a647347f4420\.user_uploaded")


@pytest.fixture(scope="module")
def config():
    return load_config("config/default.yaml")


@pytest.fixture(scope="module")
def pipeline(config):
    return PerceptionPipeline(config)


def _load_frame(filename: str) -> FramePacket:
    path = SCREENSHOTS_DIR / filename
    if not path.exists():
        pytest.skip(f"Screenshot not available: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None
    return FramePacket(image_bgr=img, timestamp=1000.0, source_name=filename)


@pytest.mark.parametrize(
    "filename,expected_x,expected_y",
    [
        ("Vikings Online_Mortega-_2026_11_22_04_06(1).jpg", 617, 1822),
        ("Vikings Online_Mortega-_2026_11_22_04_06(6).jpg", 601, 1846),
        ("Vikings Online_Mortega-_2026_11_22_04_06(11).jpg", 599, 1846),
        ("Vikings Online_Mortega-_2026_11_22_04_07(15).jpg", 603, 1844),
        ("Vikings Online_Mortega-_2026_11_22_04_07(16).jpg", 603, 1844),
        ("Vikings Online_Mortega-_2026_11_22_04_07(20).jpg", 618, 1836),
        ("Vikings Online_Mortega-_2026_11_22_04_07(25).jpg", 618, 1836),
        ("Vikings Online_Mortega-_2026_11_22_04_09(30).jpg", 615, 1835),
        ("Vikings Online_Baddies_2026_10_30_06_49(1).jpg", 1104, 1102),
        ("Vikings Online_Baddies_2026_10_30_06_59(120).jpg", 1142, 1540),
        ("Vikings Online_Baddies_2026_10_30_07_05(236).jpg", 995, 1125),
    ],
)
def test_coord_reader_exact_screenshots(pipeline: PerceptionPipeline, filename: str, expected_x: int, expected_y: int):
    frame = _load_frame(filename)
    reading = pipeline.coords.read(frame)
    assert reading is not None, f"Failed to parse coords on {filename}"
    assert (reading.x, reading.y) == (expected_x, expected_y), f"Got {reading} on {filename}"


def test_coord_reader_user_uploaded_crops(pipeline: PerceptionPipeline):
    samples = [
        ("media_1795316879757_eda578b9.png", 623, 1839),
        ("media_1795316888833_a9469fbb.png", 620, 1819),
    ]
    for fname, ex, ey in samples:
        fpath = UPLOADED_DIR / fname
        if not fpath.exists():
            continue
        crop = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        assert crop is not None
        reading = pipeline.coords.read_crop(crop)
        assert reading is not None, f"Failed to parse uploaded crop {fname}"
        assert (reading.x, reading.y) == (ex, ey), f"Got {reading} for {fname}"


def test_hud_bars_full_and_synthetic_partial(config, pipeline: PerceptionPipeline):
    frame = _load_frame("Vikings Online_Mortega-_2026_11_22_04_09(30).jpg")
    hp, mp = pipeline.hud_bars.read(frame)
    assert hp >= 0.95
    assert mp >= 0.95

    img_half = frame.image_bgr.copy()
    x1, y1, x2, y2 = roi_to_pixels(config.roi("hp_bar"), frame.width, frame.height)
    mid = (x1 + x2) // 2
    img_half[y1:y2, mid:x2] = (15, 15, 15)
    hp_half = pipeline.hud_bars.read_hp_ratio(FramePacket(image_bgr=img_half, timestamp=1.0))
    assert 0.40 <= hp_half <= 0.60

    img_zero = frame.image_bgr.copy()
    img_zero[y1:y2, x1:x2] = (15, 15, 15)
    hp_zero = pipeline.hud_bars.read_hp_ratio(FramePacket(image_bgr=img_zero, timestamp=1.0))
    assert hp_zero <= 0.02


def test_all_30_mortega_frames_coord_parse_rate(pipeline: PerceptionPipeline):
    if not SCREENSHOTS_DIR.exists():
        pytest.skip("ScreenShots directory missing")
    files = sorted(SCREENSHOTS_DIR.glob("*Mortega-*.jpg"))
    if not files:
        pytest.skip("No Mortega- screenshots found")
    parsed = 0
    for fpath in files:
        img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        snap = pipeline.analyze(FramePacket(image_bgr=img, timestamp=1.0, source_name=fpath.name))
        if snap.coords is not None and 300 <= snap.coords.x <= 2500 and 500 <= snap.coords.y <= 3200:
            parsed += 1
    rate = parsed / len(files)
    assert rate >= 0.90, f"Coord parse rate across {len(files)} Mortega frames too low: {parsed}/{len(files)} ({rate:.1%})"


def test_vm_1024x768_fixed_pixel_hud_bars_coords_and_minimap():
    """Verify VMware 1024x768 fixed-pixel HUD reads 3-4 digit coords & partial HP/MP (never POS (11,14) or HP 0%)."""
    import numpy as np

    cfg_vm = load_config("config/default.yaml")
    cfg_vm.perception["hud_fixed_pixel_ui"] = True
    pipe_vm = PerceptionPipeline(cfg_vm)

    # 1. Compose a realistic 1024x768 RYL frame with 1:1 native corner widgets from a real 1080p capture
    src_1080 = _load_frame("Vikings Online_Mortega-_2026_11_22_04_06(1).jpg").image_bgr
    vm = np.full((768, 1024, 3), (38, 82, 46), dtype=np.uint8)
    # Copy top-left 1:1 minimap + coord bar (0..130, 0..150)
    vm[0:150, 0:130] = src_1080[0:150, 0:130]
    # Copy bottom-left 1:1 HP/MP/EA panel (anchored to bottom edge: last 100 rows, 240 cols)
    vm[768 - 100 : 768, 0:240] = src_1080[1080 - 100 : 1080, 0:240]

    # Simulate HP at 5099/5700 (~89.5%) by darkening the rightmost 8px of the 80px HP bar (x=18..98 -> 90..98)
    vm[768 - 69 : 768 - 57, 90:99] = (15, 15, 15)

    snap = pipe_vm.analyze(FramePacket(image_bgr=vm, timestamp=1.0, source_name="vm_1024_real_hud"))
    assert snap.coords is not None
    assert (snap.coords.x, snap.coords.y) == (617, 1822), f"Misread 1024x768 coords: {snap.coords}"
    assert 0.80 <= snap.hp_ratio <= 0.95, f"Misread 1024x768 ~89.5% HP bar: {snap.hp_ratio}"
    assert snap.mp_ratio >= 0.95, f"Misread 1024x768 full MP bar: {snap.mp_ratio}"
    assert snap.ea_value == 94, f"Misread 1024x768 bottom-left EA value: {snap.ea_value}"
    assert len(snap.minimap.red_blips) >= 1

    # 2. Simulate VMware window border vertical shift (+10px down) on bottom-left HP bar -> adaptive fallback still works!
    vm_shift = vm.copy()
    vm_shift[768 - 100 : 768, 0:240] = (18, 18, 18)
    vm_shift[768 - 80 : 768, 0:240] = src_1080[1080 - 90 : 1080 - 10, 0:240]
    vm_shift[768 - 59 : 768 - 47, 90:99] = (15, 15, 15)
    hp_shifted = pipe_vm.hud_bars.read_hp_ratio(FramePacket(image_bgr=vm_shift, timestamp=2.0))
    assert 0.78 <= hp_shifted <= 0.96, f"Failed shifted 1024x768 HP fallback: {hp_shifted}"

    # 3. Genuine 0% HP (empty dark bar) still reads <= 0.02
    vm_dead = vm.copy()
    vm_dead[768 - 72 : 768 - 54, 14:102] = (15, 15, 15)
    assert pipe_vm.hud_bars.read_hp_ratio(FramePacket(image_bgr=vm_dead, timestamp=3.0)) <= 0.02

    # 4. Synthetic 1024x768 coord strip rendering exact '610,1841' at y=120..140 while y=85..100 has bright noise
    vm_610 = np.full((768, 1024, 3), (30, 50, 30), dtype=np.uint8)
    # Paint bright 2-digit-like white clutter inside old proportional ROI y=85..100 (must NOT yield POS (11,14)!)
    vm_610[88:96, 12:15] = (235, 235, 235)
    vm_610[88:96, 18:21] = (235, 235, 235)
    vm_610[88:96, 30:33] = (235, 235, 235)
    vm_610[88:96, 36:41] = (235, 235, 235)
    # Render real bitmap templates for '610,1841' on black coord strip at y=120..140, x=4..114
    vm_610[120:140, 4:114] = (10, 10, 10)
    cur_x = 18
    for ch in "610,1841":
        tmpl = pipe_vm.coords.templates[ch]
        u8 = np.clip(tmpl * 255.0, 0, 255).astype(np.uint8)
        ys_t, xs_t = np.where(u8 > 50)
        gh = int(ys_t.max() - ys_t.min() + 1)
        gw = int(xs_t.max() - xs_t.min() + 1)
        patch = u8[ys_t.min() : ys_t.max() + 1, xs_t.min() : xs_t.max() + 1]
        y_top = 133 - gh if ch != "," else 135 - gh
        for c in range(3):
            vm_610[y_top : y_top + gh, cur_x : cur_x + gw, c] = np.maximum(
                vm_610[y_top : y_top + gh, cur_x : cur_x + gw, c], patch
            )
        cur_x += gw + (2 if ch != "," else 3)

    r_610 = pipe_vm.coords.read(FramePacket(image_bgr=vm_610, timestamp=4.0))
    assert r_610 is not None and (r_610.x, r_610.y) == (610, 1841), f"Expected (610, 1841), got {r_610}"
