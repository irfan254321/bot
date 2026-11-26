"""Configuration loading and typed helpers for Vikings Bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import yaml

NormROI = tuple[float, float, float, float]


def is_frozen_bundle() -> bool:
    """Return True when running inside a PyInstaller bundled executable."""
    return bool(getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"))


def get_bundle_dir(fallback: Path | None = None) -> Path:
    """Return the read-only bundled assets/config root (sys._MEIPASS when frozen)."""
    if is_frozen_bundle():
        return Path(str(getattr(sys, "_MEIPASS"))).resolve()
    if fallback is not None:
        return Path(fallback).resolve()
    return Path(__file__).resolve().parents[2]


def get_runtime_dir(fallback: Path | None = None) -> Path:
    """Return the writable runtime root (folder containing the .exe when frozen)."""
    if is_frozen_bundle():
        return Path(sys.executable).resolve().parent
    if fallback is not None:
        return Path(fallback).resolve()
    return Path.cwd().resolve()


def resolve_bundled_config(rel_path: str | Path) -> Path:
    """Resolve a config file checking writable runtime_dir first (user override), then bundle_dir."""
    p = Path(rel_path)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    if is_frozen_bundle():
        rt = (get_runtime_dir() / p).resolve()
        if rt.exists():
            return rt
        bd = (get_bundle_dir() / p).resolve()
        if bd.exists():
            return bd
    return (get_bundle_dir() / p).resolve()


@dataclass(frozen=True)
class Waypoint:
    name: str
    x: float
    y: float


@dataclass(frozen=True)
class FarmingRoute:
    name: str
    farm_radius: float
    arrival_radius: float
    loop: bool
    waypoints: list[Waypoint]
    travel_waypoints: list[Waypoint] = field(default_factory=list)


@dataclass(frozen=True)
class SkillConfig:
    key: str
    cooldown_sec: float = 3.0


@dataclass
class BotConfig:
    raw: dict[str, Any]
    base_dir: Path
    routes: dict[str, FarmingRoute] = field(default_factory=dict)
    active_route_name: str = "gold_rustler_spot"
    runtime_dir: Path | None = None

    @property
    def window(self) -> dict[str, Any]:
        return self.raw["window"]

    @property
    def rois(self) -> dict[str, Any]:
        return self.raw["rois"]

    @property
    def perception(self) -> dict[str, Any]:
        return self.raw["perception"]

    @property
    def controls(self) -> dict[str, Any]:
        return self.raw["controls"]

    @property
    def farming(self) -> dict[str, Any]:
        return self.raw["farming"]

    @property
    def safety(self) -> dict[str, Any]:
        return self.raw["safety"]

    @property
    def respawn_travel(self) -> dict[str, Any]:
        return self.raw.setdefault("respawn_travel", {})

    @property
    def repair(self) -> dict[str, Any]:
        return self.raw.setdefault("repair", {})

    @property
    def active_route(self) -> FarmingRoute:
        if self.active_route_name not in self.routes:
            raise KeyError(f"Unknown route: {self.active_route_name}")
        return self.routes[self.active_route_name]

    @property
    def skills(self) -> list[SkillConfig]:
        items = self.controls.get("skills") or []
        return [
            SkillConfig(key=str(item["key"]), cooldown_sec=float(item.get("cooldown_sec", 3.0)))
            for item in items
        ]

    def resolve_path(self, rel_or_abs: str | Path) -> Path:
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        # Writable outputs (logs, debug snapshots) always live beside the .exe / runtime_dir, never inside _MEIPASS
        if p.parts and p.parts[0].lower() == "logs":
            root = self.runtime_dir if self.runtime_dir is not None else self.base_dir
            return (root / p).resolve()
        # Allow optional per-VM overrides placed beside the .exe (e.g. config/waypoints.yaml)
        if self.runtime_dir is not None and self.runtime_dir != self.base_dir:
            override_p = (self.runtime_dir / p).resolve()
            if override_p.exists():
                return override_p
        return (self.base_dir / p).resolve()

    def roi(self, name: str) -> NormROI:
        vals = self.rois[name]
        return (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))

    def player_center_norm(self) -> tuple[float, float]:
        vals = self.rois["player_center"]
        return (float(vals[0]), float(vals[1]))


def roi_to_pixels(roi: NormROI, width: int, height: int) -> tuple[int, int, int, int]:
    """Convert normalized (x1, y1, x2, y2) to clamped integer pixel bounds."""
    x1 = max(0, min(width - 1, int(round(roi[0] * width))))
    y1 = max(0, min(height - 1, int(round(roi[1] * height))))
    x2 = max(x1 + 1, min(width, int(round(roi[2] * width))))
    y2 = max(y1 + 1, min(height, int(round(roi[3] * height))))
    return (x1, y1, x2, y2)


def load_waypoints(path: Path) -> tuple[str, dict[str, FarmingRoute]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    active = str(data.get("active_route", "gold_rustler_spot"))
    routes: dict[str, FarmingRoute] = {}
    for route_name, rdata in (data.get("routes") or {}).items():
        wps = [
            Waypoint(name=str(w.get("name", f"wp_{i}")), x=float(w["x"]), y=float(w["y"]))
            for i, w in enumerate(rdata.get("waypoints") or [])
        ]
        twps = [
            Waypoint(name=str(w.get("name", f"twp_{i}")), x=float(w["x"]), y=float(w["y"]))
            for i, w in enumerate(rdata.get("travel_waypoints") or [])
        ]
        routes[str(route_name)] = FarmingRoute(
            name=str(route_name),
            farm_radius=float(rdata.get("farm_radius", 35.0)),
            arrival_radius=float(rdata.get("arrival_radius", 6.0)),
            loop=bool(rdata.get("loop", True)),
            waypoints=wps,
            travel_waypoints=twps,
        )
    return active, routes


def load_config(
    config_path: str | Path = "config/default.yaml",
    waypoints_override: str | Path | None = None,
    route_override: str | None = None,
) -> BotConfig:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        cfg_path = resolve_bundled_config(config_path)
    else:
        cfg_path = cfg_path.resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    # Project base dir is parent of config/ if inside config/, else parent of file
    base_dir = cfg_path.parent.parent if cfg_path.parent.name == "config" else cfg_path.parent
    rt_dir = get_runtime_dir(base_dir) if is_frozen_bundle() else base_dir
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    wp_rel = waypoints_override or raw.get("farming", {}).get("waypoints_file", "config/waypoints.yaml")
    wp_path = Path(wp_rel)
    if not wp_path.is_absolute():
        rt_wp = (rt_dir / wp_path).resolve()
        bd_wp = (base_dir / wp_path).resolve()
        wp_path = rt_wp if rt_wp.exists() else bd_wp

    active_route = "gold_rustler_spot"
    routes: dict[str, FarmingRoute] = {}
    if wp_path.exists():
        active_route, routes = load_waypoints(wp_path)
    if route_override:
        active_route = route_override

    return BotConfig(
        raw=raw,
        base_dir=base_dir,
        routes=routes,
        active_route_name=active_route,
        runtime_dir=rt_dir,
    )
