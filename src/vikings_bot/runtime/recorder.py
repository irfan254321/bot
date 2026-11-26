"""Snapshot & JSON state recorder for debugging stuck, black-frame, or hotkey captures."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vikings_bot.capture.screen import FramePacket
from vikings_bot.config import BotConfig
from vikings_bot.state.models import BotState, WorldState

logger = logging.getLogger(__name__)


class SnapshotRecorder:
    """Writes raw frame, optional annotated frame, and structured JSON state with auto-rotation."""

    def __init__(self, config: BotConfig) -> None:
        self.out_dir = config.resolve_path(config.safety.get("snapshots_dir", "logs/snapshots"))
        self.max_snapshots = max(5, int(config.safety.get("max_snapshots", 40)))
        self._lock = threading.Lock()

    def _prune_old_snapshots(self) -> None:
        try:
            json_files = sorted(self.out_dir.glob("*_state.json"), key=lambda p: p.stat().st_mtime)
            excess = len(json_files) - self.max_snapshots
            for jp in json_files[: max(0, excess)]:
                prefix = jp.name[: -len("_state.json")]
                for suffix in ("_state.json", "_raw.jpg", "_overlay.jpg"):
                    fp = self.out_dir / f"{prefix}{suffix}"
                    try:
                        fp.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception:
            pass

    def save_async(
        self,
        tag: str,
        frame: FramePacket,
        world: WorldState,
        fsm_state: BotState,
        annotated_bgr: np.ndarray | None = None,
    ) -> None:
        frame_copy = FramePacket(
            image_bgr=frame.image_bgr.copy(),
            timestamp=frame.timestamp,
            origin_x=frame.origin_x,
            origin_y=frame.origin_y,
            source_name=frame.source_name,
        )
        ann_copy = None if annotated_bgr is None else annotated_bgr.copy()
        threading.Thread(
            target=self.save,
            args=(tag, frame_copy, world, fsm_state, ann_copy),
            name=f"Snapshot-{tag}",
            daemon=True,
        ).start()

    def save(
        self,
        tag: str,
        frame: FramePacket,
        world: WorldState,
        fsm_state: BotState,
        annotated_bgr: np.ndarray | None = None,
    ) -> Path:
        with self._lock:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            prefix = f"{stamp}_{tag}"

            raw_path = self.out_dir / f"{prefix}_raw.jpg"
            cv2.imwrite(str(raw_path), frame.image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            if annotated_bgr is not None:
                ann_path = self.out_dir / f"{prefix}_overlay.jpg"
                cv2.imwrite(str(ann_path), annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            payload: dict[str, Any] = {
                "tag": tag,
                "timestamp": frame.timestamp,
                "source_name": frame.source_name,
                "resolution": [frame.width, frame.height],
                "fsm_state": fsm_state.value,
                "hp_ratio": round(world.hp_ratio, 4),
                "mp_ratio": round(world.mp_ratio, 4),
                "coords_raw": None
                if world.snapshot.coords is None
                else {
                    "x": world.snapshot.coords.x,
                    "y": world.snapshot.coords.y,
                    "confidence": round(world.snapshot.coords.confidence, 4),
                    "raw_text": world.snapshot.coords.raw_text,
                },
                "coords_smoothed": None
                if world.coords_smoothed is None
                else [round(world.coords_smoothed.x, 2), round(world.coords_smoothed.y, 2)],
                "camera_heading_deg": round(world.camera_heading_deg, 2),
                "heading_source": world.heading_source,
                "is_stuck": world.is_stuck,
                "consecutive_stuck": world.consecutive_stuck,
                "timings_ms": {k: round(v, 2) for k, v in world.snapshot.timings_ms.items()},
                "last_action_summary": world.last_action_summary,
            }
            json_path = self.out_dir / f"{prefix}_state.json"
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._prune_old_snapshots()
            logger.info("Saved debug snapshot (%s) -> %s", tag, json_path)
            return json_path
