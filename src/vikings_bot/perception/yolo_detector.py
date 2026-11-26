"""CPU-only YOLO11n ONNX Runtime detector for Vikings Bot v2 (4 classes: gold_rustler, gold_bag, player, dead_rustler)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from vikings_bot.config import BotConfig
from vikings_bot.state.models import LootDrop, MobTarget

logger = logging.getLogger(__name__)

CLS_GOLD_RUSTLER = 0
CLS_GOLD_BAG = 1
CLS_PLAYER = 2
CLS_DEAD_RUSTLER = 3

YOLO_CLASS_NAMES: dict[int, str] = {
    CLS_GOLD_RUSTLER: "gold_rustler",
    CLS_GOLD_BAG: "gold_bag",
    CLS_PLAYER: "player",
    CLS_DEAD_RUSTLER: "dead_rustler",
}


@dataclass(frozen=True)
class YoloRawDetection:
    """Single post-NMS, spatially-filtered YOLO11n detection in original frame coordinates."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)

    @property
    def cx(self) -> int:
        return (self.bbox[0] + self.bbox[2]) // 2

    @property
    def cy(self) -> int:
        return (self.bbox[1] + self.bbox[3]) // 2


@dataclass(frozen=True)
class YoloDetectionResult:
    """Structured perception output mapped from YOLO11n detections."""

    mobs: tuple[MobTarget, ...] = ()
    contested_mobs: tuple[MobTarget, ...] = ()
    loot: tuple[LootDrop, ...] = ()
    other_players: tuple[tuple[int, int], ...] = ()
    other_player_boxes: tuple[tuple[int, int, int, int], ...] = ()
    other_player_confs: tuple[float, ...] = ()
    dead_corpses: tuple[tuple[int, int, int, int], ...] = ()
    dead_corpse_confs: tuple[float, ...] = ()
    raw_detections: tuple[YoloRawDetection, ...] = ()


def letterbox_bgr(
    image_bgr: np.ndarray,
    new_shape: int = 640,
    pad_color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, float, float]:
    """Letterbox BGR image to (1, 3, new_shape, new_shape) RGB float32 in [0, 1].

    Returns:
        (blob_nchw, scale_ratio, pad_left, pad_top)
    """
    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        empty = np.zeros((1, 3, new_shape, new_shape), dtype=np.float32)
        return (empty, 1.0, 0.0, 0.0)

    r = min(float(new_shape) / float(h), float(new_shape) / float(w))
    new_w = max(1, int(round(w * r)))
    new_h = max(1, int(round(h * r)))

    dw = (float(new_shape) - float(new_w)) / 2.0
    dh = (float(new_shape) - float(new_h)) / 2.0

    if (w, h) != (new_w, new_h):
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image_bgr

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=pad_color,
    )
    if padded.shape[0] != new_shape or padded.shape[1] != new_shape:
        padded = cv2.resize(padded, (new_shape, new_shape), interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) * (1.0 / 255.0)
    blob = np.ascontiguousarray(chw[np.newaxis, ...])
    return (blob, r, float(left), float(top))


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Pure NumPy greedy NMS for [N, 4] xyxy boxes."""
    if boxes.shape[0] == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0.0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def is_in_self_box(cx: float, cy: float, frame_w: int, frame_h: int) -> bool:
    """Return True if (cx, cy) lies inside SELF exclusion box (x=472..552, y=350..432 at 1024x768, scaled)."""
    sx = float(frame_w) / 1024.0
    sy = float(frame_h) / 768.0
    return (472.0 * sx <= cx <= 552.0 * sx) and (350.0 * sy <= cy <= 432.0 * sy)


def is_in_hud_or_chat(cx: float, cy: float, frame_w: int, frame_h: int) -> bool:
    """Return True if (cx, cy) lies inside RYL HUD or chat regions (scaled from 1024x768)."""
    sx = float(frame_w) / 1024.0
    sy = float(frame_h) / 768.0
    # Bottom-right chat log + tabs (x>=442, y>=518 at 1024x768)
    if cx >= 442.0 * sx and cy >= 518.0 * sy:
        return True
    # Top-left minimap + coordinates
    if cx <= 120.0 * sx and cy <= 148.0 * sy:
        return True
    # Upper-left party status window
    if cx <= 115.0 * sx and 148.0 * sy <= cy <= 235.0 * sy:
        return True
    # Left vertical skill bar
    if cx <= 44.0 * sx and 338.0 * sy <= cy <= 678.0 * sy:
        return True
    # Bottom-left HP/MP/EA panel
    if cx <= 238.0 * sx and cy >= 676.0 * sy:
        return True
    # Right faction counter (Ak'Kan / Human)
    if cx >= 945.0 * sx and 370.0 * sy <= cy <= 410.0 * sy:
        return True
    return False


class Yolo11nOnnxDetector:
    """Lightweight CPU-only YOLO11n ONNX Runtime detector (imgsz=640, 4 classes, intra_op_num_threads=2)."""

    def __init__(
        self,
        config: BotConfig,
        *,
        session_override: Any | None = None,
    ) -> None:
        p = config.perception
        self.model_path: str = str(p.get("yolo_model_path", "models/yolo11n_ryl.onnx"))
        self.imgsz: int = max(160, int(p.get("yolo_imgsz", 640)))
        self.conf_threshold: float = float(p.get("yolo_conf_threshold", 0.45))
        self.gold_bag_conf_threshold: float = float(p.get("gold_bag_conf_threshold", 0.32))
        self.iou_threshold: float = float(p.get("yolo_iou_threshold", 0.45))
        self.intra_op_threads: int = max(1, int(p.get("yolo_intra_op_threads", 2)))
        self.anti_ks_enabled: bool = bool(p.get("anti_ks_enabled", True))
        self.anti_ks_use_hsv_damage: bool = bool(p.get("anti_ks_use_hsv_damage", False))
        self.anti_ks_player_min_conf: float = float(p.get("anti_ks_player_min_conf", 0.50))
        self.anti_ks_player_radius_px: float = float(p.get("anti_ks_player_radius_px", 240.0))
        self.anti_ks_damage_zone_radius_px: float = float(
            p.get("anti_ks_damage_zone_radius_px", self.anti_ks_player_radius_px)
        )
        self.click_y_top_ratio: float = float(p.get("click_y_top_ratio", 0.42))

        self._session: Any | None = None
        self._input_name: str = "images"
        self._output_name: str = "output0"
        self._load_error: str | None = None

        if session_override is not None:
            self._session = session_override
            self._resolve_io_names(session_override)
        else:
            self._load_onnx_session()

    def _resolve_io_names(self, session: Any) -> None:
        try:
            ins = session.get_inputs()
            if ins:
                self._input_name = str(ins[0].name)
        except Exception:
            self._input_name = "images"
        try:
            outs = session.get_outputs()
            if outs:
                self._output_name = str(outs[0].name)
        except Exception:
            self._output_name = "output0"

    def _load_onnx_session(self) -> None:
        model_file = Path(self.model_path)
        if not model_file.is_file():
            self._load_error = f"ONNX model not found: {model_file}"
            logger.warning("Yolo11nOnnxDetector disabled (%s); using fallback detector", self._load_error)
            return
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except Exception as exc:
            self._load_error = f"onnxruntime import failed: {exc}"
            logger.warning("Yolo11nOnnxDetector disabled (%s); using fallback detector", self._load_error)
            return

        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = self.intra_op_threads
            opts.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(model_file),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._resolve_io_names(self._session)
            logger.info(
                "Yolo11nOnnxDetector loaded %s (imgsz=%d, conf=%.2f, iou=%.2f, threads=%d, provider=CPUExecutionProvider)",
                model_file,
                self.imgsz,
                self.conf_threshold,
                self.iou_threshold,
                self.intra_op_threads,
            )
        except Exception as exc:
            self._session = None
            self._load_error = f"ONNX session init failed: {exc}"
            logger.warning("Yolo11nOnnxDetector disabled (%s); using fallback detector", self._load_error)

    @property
    def is_available(self) -> bool:
        return self._session is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def infer_raw(self, image_bgr: np.ndarray) -> list[YoloRawDetection]:
        """Run YOLO11n ONNX inference, decode (1, 8, 8400) + NMS, un-letterbox, and filter SELF + HUD/chat."""
        if self._session is None or image_bgr.size == 0:
            return []

        fh, fw = image_bgr.shape[:2]
        blob, ratio, pad_x, pad_y = letterbox_bgr(image_bgr, new_shape=self.imgsz)
        outputs = self._session.run([self._output_name], {self._input_name: blob})
        if not outputs:
            return []

        raw = np.asarray(outputs[0], dtype=np.float32)
        if raw.ndim == 3:
            raw = raw[0]
        if raw.ndim != 2:
            return []

        # Standard YOLO11n export is (8, 8400) = [cx, cy, w, h, cls0, cls1, cls2, cls3]
        if raw.shape[0] == 8 and raw.shape[1] != 8:
            preds = raw.T  # (8400, 8)
        elif raw.shape[1] == 8:
            preds = raw  # (8400, 8)
        elif raw.shape[0] < raw.shape[1] and raw.shape[0] >= 5:
            preds = raw.T
        else:
            preds = raw

        if preds.shape[1] < 5:
            return []

        cls_scores = preds[:, 4 : 4 + len(YOLO_CLASS_NAMES)]
        class_ids = np.argmax(cls_scores, axis=1)
        confs = cls_scores[np.arange(preds.shape[0]), class_ids]

        cls_thresholds = np.array(
            [
                self.conf_threshold,  # 0: gold_rustler
                self.gold_bag_conf_threshold,  # 1: gold_bag (lower threshold for small ground drops)
                self.conf_threshold,  # 2: player
                self.conf_threshold,  # 3: dead_rustler
            ],
            dtype=np.float32,
        )
        valid = confs >= cls_thresholds[class_ids]
        if not np.any(valid):
            return []

        preds = preds[valid]
        class_ids = class_ids[valid]
        confs = confs[valid]

        cx = preds[:, 0]
        cy = preds[:, 1]
        bw = preds[:, 2]
        bh = preds[:, 3]

        # Convert letterbox xywh -> original frame xyxy
        inv_r = 1.0 / max(1e-6, ratio)
        x1 = np.clip((cx - bw * 0.5 - pad_x) * inv_r, 0.0, float(fw))
        y1 = np.clip((cy - bh * 0.5 - pad_y) * inv_r, 0.0, float(fh))
        x2 = np.clip((cx + bw * 0.5 - pad_x) * inv_r, 0.0, float(fw))
        y2 = np.clip((cy + bh * 0.5 - pad_y) * inv_r, 0.0, float(fh))
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        detections: list[YoloRawDetection] = []
        for cid in range(len(YOLO_CLASS_NAMES)):
            cls_mask = class_ids == cid
            if not np.any(cls_mask):
                continue
            c_boxes = boxes_xyxy[cls_mask]
            c_scores = confs[cls_mask]
            keep = _nms_xyxy(c_boxes, c_scores, self.iou_threshold)
            cname = YOLO_CLASS_NAMES[cid]
            for k in keep:
                bx1, by1, bx2, by2 = c_boxes[k]
                if (bx2 - bx1) < 2.0 or (by2 - by1) < 2.0:
                    continue
                bcx = 0.5 * (bx1 + bx2)
                bcy = 0.5 * (by1 + by2)
                # Mandatory spatial exclusions: SELF box and static HUD / chat regions
                if is_in_self_box(bcx, bcy, fw, fh):
                    continue
                if is_in_hud_or_chat(bcx, bcy, fw, fh):
                    continue
                detections.append(
                    YoloRawDetection(
                        class_id=cid,
                        class_name=cname,
                        confidence=float(c_scores[k]),
                        bbox=(int(round(bx1)), int(round(by1)), int(round(bx2)), int(round(by2))),
                    )
                )
        return detections

    def detect(
        self,
        image_bgr: np.ndarray,
        *,
        include_mobs: bool = True,
        include_loot: bool = True,
        external_combat_zones: Sequence[tuple[float, float]] = (),
        loot_radius_limit_px: float | None = None,
    ) -> YoloDetectionResult:
        """Run YOLO11n inference and map detections to MobTarget, LootDrop, Players, and Dead Corpses."""
        if self._session is None or image_bgr.size == 0:
            return YoloDetectionResult()

        fh, fw = image_bgr.shape[:2]
        raw_dets = self.infer_raw(image_bgr)

        scale_1024 = float(fw) / 1024.0
        center_x = float(fw) * 0.5
        center_y = float(fh) * 0.5
        ks_radius = max(self.anti_ks_player_radius_px, self.anti_ks_damage_zone_radius_px) * scale_1024
        player_ks_r = self.anti_ks_player_radius_px * scale_1024

        other_players: list[tuple[int, int]] = []
        other_player_boxes: list[tuple[int, int, int, int]] = []
        other_player_confs: list[float] = []
        ks_players: list[tuple[int, int]] = []
        dead_corpses: list[tuple[int, int, int, int]] = []
        dead_corpse_confs: list[float] = []

        for det in raw_dets:
            if det.class_id == CLS_PLAYER:
                other_players.append((det.cx, det.cy))
                other_player_boxes.append(det.bbox)
                other_player_confs.append(det.confidence)
                if det.confidence >= self.anti_ks_player_min_conf:
                    ks_players.append((det.cx, det.cy))
            elif det.class_id == CLS_DEAD_RUSTLER:
                # Ignored as attack target; kept strictly for debug overlay visualization
                dead_corpses.append(det.bbox)
                dead_corpse_confs.append(det.confidence)

        def _ks_reason(px: float, py: float) -> str:
            if not self.anti_ks_enabled:
                return ""
            if any(math.hypot(px - ox, py - oy) <= player_ks_r for ox, oy in ks_players):
                return "SKIP [KS:PLAYER]"
            if self.anti_ks_use_hsv_damage and any(
                math.hypot(px - zx, py - zy) <= ks_radius for zx, zy in external_combat_zones
            ):
                return "SKIP [KS:DMG]"
            return ""

        mobs: list[MobTarget] = []
        contested_mobs: list[MobTarget] = []
        loot: list[LootDrop] = []

        if include_mobs:
            for det in raw_dets:
                if det.class_id != CLS_GOLD_RUSTLER:
                    continue
                x1, y1, x2, y2 = det.bbox
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                click_x = (x1 + x2) // 2
                click_y = int(round(y1 + bh * self.click_y_top_ratio))
                foot_x = click_x
                foot_y = y2
                dist = math.hypot(float(click_x) - center_x, float(click_y) - center_y)
                np_box = (x1, max(0, y1 - 14), bw, 14)

                reason = _ks_reason(float(det.cx), float(det.cy))
                target = MobTarget(
                    nameplate_bbox=np_box,
                    body_bbox=det.bbox,
                    click_x=click_x,
                    click_y=click_y,
                    foot_x=foot_x,
                    foot_y=foot_y,
                    dist_to_player_px=dist,
                    has_hp_bar=False,
                    hp_ratio_est=None,
                    confidence=det.confidence,
                    contested=bool(reason),
                    contest_reason=reason,
                    class_name=det.class_name,
                )
                if reason:
                    contested_mobs.append(target)
                else:
                    mobs.append(target)

        if include_loot:
            rad_lim = (loot_radius_limit_px * scale_1024) if loot_radius_limit_px is not None else None
            for det in raw_dets:
                if det.class_id != CLS_GOLD_BAG:
                    continue
                x1, y1, x2, y2 = det.bbox
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                cx = det.cx
                cy = det.cy
                dist = math.hypot(float(cx) - center_x, float(cy) - center_y)
                if rad_lim is not None and dist > rad_lim:
                    continue
                if _ks_reason(float(cx), float(cy)):
                    continue
                loot.append(
                    LootDrop(
                        bbox=(x1, y1, bw, bh),
                        click_x=cx,
                        click_y=cy,
                        dist_to_player_px=dist,
                        score=det.confidence,
                        class_name=det.class_name,
                    )
                )

        mobs.sort(key=lambda m: m.dist_to_player_px)
        contested_mobs.sort(key=lambda m: m.dist_to_player_px)
        loot.sort(key=lambda b: b.dist_to_player_px)

        return YoloDetectionResult(
            mobs=tuple(mobs),
            contested_mobs=tuple(contested_mobs),
            loot=tuple(loot),
            other_players=tuple(other_players),
            other_player_boxes=tuple(other_player_boxes),
            other_player_confs=tuple(other_player_confs),
            dead_corpses=tuple(dead_corpses),
            dead_corpse_confs=tuple(dead_corpse_confs),
            raw_detections=tuple(raw_dets),
        )
