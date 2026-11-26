"""Tests for Humanizer S-curve mouse easing, timing jitter, and Gold/Gems/Metal spot profiles."""

from __future__ import annotations

import random

from vikings_bot.input.humanizer import Humanizer, ease_mouse_deltas
from vikings_bot.targets import build_config_for_target, load_target_profiles


def test_ease_mouse_deltas_exact_net_displacement_and_zero_pitch_drift():
    rng = random.Random(42)
    for dx in (120, -185, 52, -14, 0):
        deltas = ease_mouse_deltas(dx=dx, dy=0, steps=10, add_micro_arc=True, rng=rng)
        assert len(deltas) == 10
        assert sum(sx for sx, _ in deltas) == dx
        assert sum(sy for _, sy in deltas) == 0
        if abs(dx) >= 100:
            mid_mag = abs(deltas[4][0]) + abs(deltas[5][0])
            edge_mag = abs(deltas[0][0]) + abs(deltas[-1][0])
            assert mid_mag > edge_mag


def test_humanizer_bounded_jitter_and_disable_mode():
    h_on = Humanizer(enabled=True, seed=7)
    samples = [h_on.jitter(0.10, pct=0.20, lo=0.07, hi=0.13) for _ in range(40)]
    assert all(0.07 <= s <= 0.13 for s in samples)
    assert len(set(round(s, 4) for s in samples)) > 10

    h_off = Humanizer(enabled=False, seed=7)
    assert h_off.jitter(0.10, pct=0.20) == 0.10
    assert h_off.reaction_pause(0.12, 0.25) == 0.0


def test_target_profiles_load_gold_gems_metal():
    profiles = load_target_profiles("config/targets.yaml")
    ids = [p.id for p in profiles]
    assert ids == ["gold", "gems", "metal"]
    gold, gems, metal = profiles
    assert gold.is_ready is True
    assert gems.is_ready is True
    assert metal.is_ready is False

    cfg_gold = build_config_for_target(gold, config_path="config/default.yaml", control_mode="keyboard", humanize=True)
    assert cfg_gold.controls["mode"] == "keyboard"
    assert cfg_gold.controls["humanize"] is True
    assert cfg_gold.active_route.name == "gold_rustler_spot"

    cfg_gems = build_config_for_target(gems, config_path="config/default.yaml", control_mode="keyboard", humanize=True)
    assert cfg_gems.active_route.name == "gem_rustler_spot"
    assert cfg_gems.farming.get("fallback_route_name") == "gem_rustler_fallback_spot"
