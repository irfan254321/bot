"""Lightweight live session delay & state profiler."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vikings_bot.state.models import BotState, WorldState

logger = logging.getLogger(__name__)


class SessionTracer:
    """Records state durations, idle/delay streaks, and delay frames into logs/latest_trace.jsonl."""

    def __init__(
        self,
        trace_path: Path,
        frames_dir: Path,
        idle_warn_sec: float = 0.25,
        frame_capture_sec: float = 0.85,
        max_frames: int = 15,
    ) -> None:
        self.trace_path = trace_path
        self.frames_dir = frames_dir
        self.idle_warn_sec = idle_warn_sec
        self.frame_capture_sec = frame_capture_sec
        self.max_frames = max_frames

        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        for old in self.frames_dir.glob("delay_*.jpg"):
            try:
                old.unlink(missing_ok=True)
            except Exception:
                pass
        self._fp = self.trace_path.open("w", encoding="utf-8", buffering=1)

        self._t0: float | None = None
        self._prev_state: BotState = BotState.PAUSED
        self._state_since: float = 0.0
        self._idle_since: float | None = None
        self._idle_captured: bool = False
        self._frames_saved: int = 0
        self._last_hb_t: float = -10.0

        self.ticks: int = 0
        self.sum_analyze_ms: float = 0.0
        self.sum_exec_ms: float = 0.0
        self.max_exec_ms: float = 0.0
        self.state_time_sec: dict[str, float] = defaultdict(float)
        self.idle_events: list[dict[str, Any]] = []
        self._last_tick_ts: float | None = None

    def _write(self, obj: dict[str, Any]) -> None:
        try:
            self._fp.write(json.dumps(obj, separators=(",", ":")) + "\n")
            self._fp.flush()
        except Exception:
            pass

    def record_tick(
        self,
        now: float,
        state: BotState,
        world: WorldState,
        intents_count: int,
        action_summary: str,
        analyze_ms: float,
        exec_ms: float,
        active_cooldowns: list[str],
        annotated_fn: Any | None = None,
    ) -> None:
        if self._t0 is None:
            self._t0 = now
            self._state_since = now
            self._last_tick_ts = now
        dt = max(0.0, now - (self._last_tick_ts or now))
        self._last_tick_ts = now
        rel_t = round(now - self._t0, 3)

        self.ticks += 1
        self.sum_analyze_ms += analyze_ms
        self.sum_exec_ms += exec_ms
        if exec_ms > self.max_exec_ms:
            self.max_exec_ms = exec_ms

        if state not in (BotState.PAUSED, BotState.EMERGENCY_STOP):
            self.state_time_sec[state.value] += dt

        if state != self._prev_state:
            dur = round(max(0.0, now - self._state_since), 3)
            self._write(
                {
                    "t": rel_t,
                    "ev": "state",
                    "from": self._prev_state.value,
                    "to": state.value,
                    "dur_s": dur,
                    "pos": None
                    if world.coords_smoothed is None
                    else [round(world.coords_smoothed.x, 1), round(world.coords_smoothed.y, 1)],
                }
            )
            self._prev_state = state
            self._state_since = now
            self._idle_since = None
            self._idle_captured = False

        if state in (BotState.PAUSED, BotState.EMERGENCY_STOP, BotState.AT_SPOT_IDLE):
            self._idle_since = None
            self._idle_captured = False
        else:
            is_idle_tick = intents_count == 0 or (
                action_summary != "" and all(p.startswith("release:") for p in action_summary.split("; "))
            )
            if is_idle_tick:
                if self._idle_since is None:
                    self._idle_since = now
                    self._idle_captured = False
                idle_dur = now - self._idle_since
                if (
                    idle_dur >= self.frame_capture_sec
                    and not self._idle_captured
                    and self._frames_saved < self.max_frames
                    and annotated_fn is not None
                ):
                    self._idle_captured = True
                    self._frames_saved += 1
                    try:
                        img: np.ndarray = annotated_fn()
                        fname = f"delay_{self._frames_saved:02d}_{state.value}_t{int(rel_t)}s.jpg"
                        cv2.imwrite(str(self.frames_dir / fname), img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    except Exception:
                        pass
            else:
                if self._idle_since is not None:
                    idle_dur = now - self._idle_since
                    if idle_dur >= self.idle_warn_sec:
                        ev = {
                            "t": rel_t,
                            "ev": "delay",
                            "state": state.value,
                            "dur_s": round(idle_dur, 3),
                            "cds": active_cooldowns,
                            "next_act": action_summary[:80],
                        }
                        self.idle_events.append(ev)
                        self._write(ev)
                self._idle_since = None
                self._idle_captured = False

        if exec_ms >= 80.0:
            self._write(
                {
                    "t": rel_t,
                    "ev": "slow_exec",
                    "state": state.value,
                    "exec_ms": round(exec_ms, 1),
                    "analyze_ms": round(analyze_ms, 1),
                    "act": action_summary[:80],
                }
            )

        if (rel_t - self._last_hb_t) >= 1.0:
            self._last_hb_t = rel_t
            self._write(
                {
                    "t": rel_t,
                    "ev": "tick",
                    "state": state.value,
                    "an_ms": round(analyze_ms, 1),
                    "ex_ms": round(exec_ms, 1),
                    "pos": None
                    if world.coords_smoothed is None
                    else [round(world.coords_smoothed.x, 1), round(world.coords_smoothed.y, 1)],
                    "act": action_summary[:70],
                }
            )

    def summarize(self) -> str:
        if self.ticks == 0:
            return "TRACE: 0 ticks recorded."
        avg_an = self.sum_analyze_ms / self.ticks
        avg_ex = self.sum_exec_ms / self.ticks
        by_state: dict[str, list[float]] = defaultdict(list)
        for ev in self.idle_events:
            by_state[str(ev["state"])].append(float(ev["dur_s"]))
        delay_parts = [
            f"{st}:{len(durs)}x(avg={sum(durs)/len(durs):.2f}s,max={max(durs):.2f}s)"
            for st, durs in sorted(by_state.items(), key=lambda kv: -sum(kv[1]))[:4]
        ]
        delay_str = ", ".join(delay_parts) if delay_parts else "none"
        return (
            f"TRACE SUMMARY | ticks={self.ticks} | avg_analyze={avg_an:.1f}ms avg_exec={avg_ex:.1f}ms "
            f"max_exec={self.max_exec_ms:.1f}ms | delays(>={self.idle_warn_sec:.2f}s): {delay_str} | "
            f"delay_frames={self._frames_saved} -> {self.trace_path}"
        )

    def close(self) -> str:
        summary = self.summarize()
        self._write({"ev": "summary", "text": summary})
        try:
            self._fp.flush()
            self._fp.close()
        except Exception:
            pass
        return summary
