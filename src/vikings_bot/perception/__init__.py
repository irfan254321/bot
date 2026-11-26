"""Pure OpenCV visual perception pipeline: HUD, Coords, Minimap, UI templates, Gold Rustler & Gold Bag."""

from vikings_bot.perception.coords import CoordReader, calibrate_digit_templates
from vikings_bot.perception.gold_detector import GoldRustlerDetector
from vikings_bot.perception.gold_loot_detector import GoldBagDetector
from vikings_bot.perception.hud_bars import HudBarReader
from vikings_bot.perception.hud_mask import build_playfield_hud_mask, compute_self_zone, get_static_hud_rects
from vikings_bot.perception.minimap import MinimapAnalyzer
from vikings_bot.perception.pipeline import PerceptionPipeline
from vikings_bot.perception.ui_matcher import UiTemplateMatcher

__all__ = [
    "CoordReader",
    "GoldBagDetector",
    "GoldRustlerDetector",
    "HudBarReader",
    "MinimapAnalyzer",
    "PerceptionPipeline",
    "UiTemplateMatcher",
    "build_playfield_hud_mask",
    "calibrate_digit_templates",
    "compute_self_zone",
    "get_static_hud_rects",
]
