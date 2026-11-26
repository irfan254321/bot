"""Target monster profile loader for the multi-farm launcher (Gold / Gems / Metal)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from vikings_bot.config import BotConfig, load_config, resolve_bundled_config


@dataclass(frozen=True)
class TargetProfile:
    id: str
    display_name: str
    mob_name: str
    status: str  # "ready" | "coming_soon"
    route_name: str
    accent_color: str
    subtitle: str
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        return self.status.lower() == "ready"


DEFAULT_PROFILES: list[TargetProfile] = [
    TargetProfile(
        id="gold",
        display_name="GOLD",
        mob_name="Gold Rustler",
        status="ready",
        route_name="gold_rustler_spot",
        accent_color="#F5B731",
        subtitle="Gold Rustler • ±20 Spot • Space+LMB Drill • E Auto-Loot",
        description="Farming otomatis Gold Rustler dengan target-lock, jump-drill cepat, dan ambil semua tas gold.",
    ),
    TargetProfile(
        id="gems",
        display_name="GEMS",
        mob_name="Gems Monster",
        status="coming_soon",
        route_name="gold_rustler_spot",
        accent_color="#38BDF8",
        subtitle="Next Version • Profile Slot Ready",
        description="Auto-farm monster drop Gems (coming soon setelah Gold selesai).",
    ),
    TargetProfile(
        id="metal",
        display_name="METAL",
        mob_name="Metal Monster",
        status="coming_soon",
        route_name="gold_rustler_spot",
        accent_color="#94A3B8",
        subtitle="Next Version • Profile Slot Ready",
        description="Auto-farm monster drop Metal (coming soon setelah Gold selesai).",
    ),
]


def load_target_profiles(targets_path: str | Path = "config/targets.yaml") -> list[TargetProfile]:
    path = Path(targets_path)
    if not path.exists():
        path = resolve_bundled_config(targets_path)
    if not path.exists():
        return list(DEFAULT_PROFILES)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("targets") or []
    profiles: list[TargetProfile] = []
    for item in items:
        profiles.append(
            TargetProfile(
                id=str(item.get("id", "")).lower(),
                display_name=str(item.get("display_name", item.get("id", "TARGET"))).upper(),
                mob_name=str(item.get("mob_name", "Monster")),
                status=str(item.get("status", "coming_soon")),
                route_name=str(item.get("route_name", "gold_rustler_spot")),
                accent_color=str(item.get("accent_color", "#F5B731")),
                subtitle=str(item.get("subtitle", "")),
                description=str(item.get("description", "")),
                overrides=dict(item.get("overrides") or {}),
            )
        )
    return profiles or list(DEFAULT_PROFILES)


def build_config_for_target(
    profile: TargetProfile,
    config_path: str | Path = "config/default.yaml",
    control_mode: str | None = None,
    humanize: bool | None = None,
) -> BotConfig:
    """Load BotConfig customized for a specific target profile (Gold / Gems / Metal)."""
    cfg = load_config(config_path, route_override=profile.route_name)
    for section, values in (profile.overrides or {}).items():
        if isinstance(values, dict) and hasattr(cfg, section):
            getattr(cfg, section).update(values)
    if control_mode:
        cm = str(control_mode).strip().lower()
        if cm == "auto":
            cfg.controls["control_mode"] = "auto"
        else:
            cfg.controls["control_mode"] = cm
            cfg.controls["mode"] = cm
    if humanize is not None:
        cfg.controls["humanize"] = bool(humanize)
    return cfg
