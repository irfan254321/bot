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
