"""CPU-only YOLO11n ONNX Runtime detector for Vikings Bot v2 (3 classes: gold_rustler, gold_bag, player)."""

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

YOLO_CLASS_NAMES: dict[int, str] = {
    CLS_GOLD_RUSTLER: "gold_rustler",
    CLS_GOLD_BAG: "gold_bag",
    CLS_PLAYER: "player",
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
    """Structured perception output mapped from 3-class YOLO11n detections."""

    mobs: tuple[MobTarget, ...] = ()
    contested_mobs: tuple[MobTarget, ...] = ()
    loot: tuple[LootDrop, ...] = ()
    other_players: tuple[tuple[int, int], ...] = ()
    other_player_boxes: tuple[tuple[int, int, int, int], ...] = ()
    other_player_confs: tuple[float, ...] = ()
    raw_detections: tuple[YoloRawDetection, ...] = ()


def letterbox_bgr(
    image_bgr: np.ndarray,
    new_shape: int | tuple[int, int] = 416,
    pad_color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, float, float]:
    """Letterbox BGR image to (1, 3, target_h, target_w) RGB float32 in [0, 1].

    Returns:
        (blob_nchw, scale_ratio, pad_left, pad_top)
    """
    if isinstance(new_shape, tuple):
        target_h, target_w = int(new_shape[0]), int(new_shape[1])
    else:
        target_h = target_w = int(new_shape)

    h, w = image_bgr.shape[:2]
    if h <= 0 or w <= 0:
        empty = np.zeros((1, 3, target_h, target_w), dtype=np.float32)
        return (empty, 1.0, 0.0, 0.0)

    r = min(float(target_h) / float(h), float(target_w) / float(w))
    new_w = max(1, int(round(w * r)))
    new_h = max(1, int(round(h * r)))

    dw = (float(target_w) - float(new_w)) / 2.0
    dh = (float(target_h) - float(new_h)) / 2.0

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
    if padded.shape[0] != target_h or padded.shape[1] != target_w:
        padded = cv2.resize(padded, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

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
    """Return True if (cx, cy) lies inside RYL HUD, right faction HUD (x>=935), or chat regions (scaled from 1024x768)."""
    sx = float(frame_w) / 1024.0
    sy = float(frame_h) / 768.0
    # Right HUD (Ak'Kan / Human counter) & extreme right edge freeze guard (x>=935 at 1024x768)
    if cx >= 935.0 * sx:
        return True
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
    return False


def is_warm_gold_drop_roi(image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    """Lightweight HSV color filter rejecting white/silver/gray drops; accepts warm red/orange/golden-yellow pouches."""
    fh, fw = image_bgr.shape[:2]
    x1 = max(0, min(fw - 1, int(bbox[0])))
    y1 = max(0, min(fh - 1, int(bbox[1])))
    x2 = max(x1 + 1, min(fw, int(bbox[2])))
    y2 = max(y1 + 1, min(fh, int(bbox[3])))
    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    # Warm red / orange / golden-yellow pixels (gold bags & dark-brown/orange pouches)
    warm = ((h <= 36) | (h >= 168)) & (s >= 70) & (v >= 65)
    # Bright low-saturation white/silver pixels (white junk items / armor drops)
    white_silver = (s <= 55) & (v >= 155)
    warm_cnt = int(warm.sum())
    white_cnt = int(white_silver.sum())
    total = max(1, roi.shape[0] * roi.shape[1])
    if warm_cnt < 3 and (warm_cnt / float(total)) < 0.03:
        return False
    if white_cnt > warm_cnt * 3 and (white_cnt / float(total)) >= 0.25:
        return False
    return True


class Yolo11nOnnxDetector:
    """Lightweight CPU-only 3-class YOLO11n ONNX Runtime detector (gold_rustler, gold_bag, player; intra_op_num_threads=2)."""

    def __init__(
        self,
        config: BotConfig,
        *,
        session_override: Any | None = None,
    ) -> None:
        p = config.perception
        self.model_path: str = str(p.get("yolo_model_path", "models/yolo11n_ryl.onnx"))
        self.imgsz: int = max(160, int(p.get("yolo_imgsz", 416)))
        self.imgsz_h: int = self.imgsz
        self.imgsz_w: int = self.imgsz
        self.conf_threshold: float = float(p.get("yolo_conf_threshold", 0.45))
        self.player_conf_threshold: float = float(
            p.get("player_conf_threshold", p.get("anti_ks_player_min_conf", 0.65))
        )
        self.gold_bag_conf_threshold: float = float(p.get("gold_bag_conf_threshold", 0.20))
        self.loot_tiled_gold_bag_conf_threshold: float = float(
            p.get("loot_tiled_gold_bag_conf_threshold", self.gold_bag_conf_threshold)
        )
        self.loot_bag_max_size_px: float = float(p.get("loot_bag_max_size_px", 65.0))
        self.loot_color_filter_enabled: bool = bool(p.get("loot_color_filter_enabled", True))
        self.iou_threshold: float = float(p.get("yolo_iou_threshold", 0.45))
        self.intra_op_threads: int = max(1, int(p.get("yolo_intra_op_threads", 2)))
        self.anti_ks_enabled: bool = bool(p.get("anti_ks_enabled", True))
        self.anti_ks_use_hsv_damage: bool = bool(p.get("anti_ks_use_hsv_damage", False))
        self.anti_ks_player_min_conf: float = float(
            p.get("anti_ks_player_min_conf", self.player_conf_threshold)
        )
        self.anti_ks_player_radius_px: float = float(p.get("anti_ks_player_radius_px", 240.0))
        self.anti_ks_player_self_ignore_radius_px: float = float(
            p.get("anti_ks_player_self_ignore_radius_px", 90.0)
        )
        self.anti_ks_player_confirm_frames: int = max(1, int(p.get("anti_ks_player_confirm_frames", 2)))
        self.anti_ks_damage_zone_radius_px: float = float(
            p.get("anti_ks_damage_zone_radius_px", self.anti_ks_player_radius_px)
        )
        # Click lower-chest / torso of gold_rustler bbox: y1 + 0.68 * (y2 - y1)
        self.click_y_top_ratio: float = float(p.get("yolo_click_y_ratio", 0.68))

        self._player_consec_frames: int = 0
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
                shape = getattr(ins[0], "shape", None)
                if shape is not None and len(shape) >= 4:
                    h_dim, w_dim = shape[2], shape[3]
                    if isinstance(h_dim, int) and h_dim >= 160 and isinstance(w_dim, int) and w_dim >= 160:
                        self.imgsz_h = int(h_dim)
                        self.imgsz_w = int(w_dim)
                        self.imgsz = max(self.imgsz_h, self.imgsz_w)
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
                "Yolo11nOnnxDetector loaded %s (imgsz=%dx%d, classes=3, conf=%.2f, player_conf=%.2f, bag_conf=%.2f, threads=%d)",
                model_file,
                self.imgsz_w,
                self.imgsz_h,
                self.conf_threshold,
                self.player_conf_threshold,
                self.gold_bag_conf_threshold,
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

    def _infer_crop_candidates(
        self,
        crop_bgr: np.ndarray,
        *,
        offset_x: int = 0,
        offset_y: int = 0,
        frame_w: int | None = None,
        frame_h: int | None = None,
        gold_bag_conf_override: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run 3-class ONNX on a single crop and return pre-NMS (boxes_xyxy, scores, class_ids) in full-frame pixels."""
        empty_b = np.zeros((0, 4), dtype=np.float32)
        empty_s = np.zeros((0,), dtype=np.float32)
        empty_c = np.zeros((0,), dtype=np.int64)
        if self._session is None or crop_bgr.size == 0:
            return (empty_b, empty_s, empty_c)

        ch, cw = crop_bgr.shape[:2]
        fw = cw if frame_w is None else frame_w
        fh = ch if frame_h is None else frame_h
        blob, ratio, pad_x, pad_y = letterbox_bgr(crop_bgr, new_shape=(self.imgsz_h, self.imgsz_w))
        outputs = self._session.run([self._output_name], {self._input_name: blob})
        if not outputs:
            return (empty_b, empty_s, empty_c)

        raw = np.asarray(outputs[0], dtype=np.float32)
        if raw.ndim == 3:
            raw = raw[0]
        if raw.ndim != 2:
            return (empty_b, empty_s, empty_c)

        # Standard 3-class YOLO11n export is (7, N) = [cx, cy, w, h, cls0, cls1, cls2];
        # also tolerates legacy (8, N) by reading only the 3 active class channels 4..6.
        if raw.shape[0] < raw.shape[1] and raw.shape[0] >= 7:
            preds = raw.T
        else:
            preds = raw

        n_cls = len(YOLO_CLASS_NAMES)
        if preds.shape[1] < 4 + n_cls:
            return (empty_b, empty_s, empty_c)

        cls_scores = preds[:, 4 : 4 + n_cls]
        class_ids = np.argmax(cls_scores, axis=1)
        confs = cls_scores[np.arange(preds.shape[0]), class_ids]

        gb_th = self.gold_bag_conf_threshold if gold_bag_conf_override is None else float(gold_bag_conf_override)
        cls_thresholds = np.array(
            [
                self.conf_threshold,  # 0: gold_rustler (0.45)
                gb_th,  # 1: gold_bag (0.20)
                self.player_conf_threshold,  # 2: player (0.65)
            ],
            dtype=np.float32,
        )
        valid = confs >= cls_thresholds[class_ids]
        if not np.any(valid):
            return (empty_b, empty_s, empty_c)

        preds = preds[valid]
        class_ids = class_ids[valid]
        confs = confs[valid]

        cx = preds[:, 0]
        cy = preds[:, 1]
        bw = preds[:, 2]
        bh = preds[:, 3]

        inv_r = 1.0 / max(1e-6, ratio)
        x1 = np.clip((cx - bw * 0.5 - pad_x) * inv_r + float(offset_x), 0.0, float(fw))
        y1 = np.clip((cy - bh * 0.5 - pad_y) * inv_r + float(offset_y), 0.0, float(fh))
        x2 = np.clip((cx + bw * 0.5 - pad_x) * inv_r + float(offset_x), 0.0, float(fw))
        y2 = np.clip((cy + bh * 0.5 - pad_y) * inv_r + float(offset_y), 0.0, float(fh))
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        return (boxes_xyxy, confs, class_ids)

    def _finalize_nms_and_spatial_filter(
        self,
        boxes_xyxy: np.ndarray,
        confs: np.ndarray,
        class_ids: np.ndarray,
        fw: int,
        fh: int,
    ) -> list[YoloRawDetection]:
        """Apply class-wise NMS (iou_threshold=0.45) and mandatory SELF, HUD/chat, and player <90px exclusions."""
        if boxes_xyxy.shape[0] == 0:
            return []
        sx = float(fw) / 1024.0
        self_cx = float(fw) * 0.5
        self_cy = float(fh) * 0.5
        player_self_r = self.anti_ks_player_self_ignore_radius_px * sx

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
                if is_in_self_box(bcx, bcy, fw, fh):
                    continue
                if is_in_hud_or_chat(bcx, bcy, fw, fh):
                    continue
                # Ignore class 2 'player' if center is < 90px from SELF center (512, 384)
                if cid == CLS_PLAYER and math.hypot(bcx - self_cx, bcy - self_cy) < player_self_r:
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

    def infer_raw(self, image_bgr: np.ndarray) -> list[YoloRawDetection]:
        """Run full-frame 3-class YOLO11n ONNX inference, decode + NMS, un-letterbox, and filter SELF + HUD/chat."""
        if self._session is None or image_bgr.size == 0:
            return []
        fh, fw = image_bgr.shape[:2]
        boxes, confs, cids = self._infer_crop_candidates(image_bgr, offset_x=0, offset_y=0, frame_w=fw, frame_h=fh)
        return self._finalize_nms_and_spatial_filter(boxes, confs, cids, fw, fh)

    def detect_loot_tiled(
        self,
        image_bgr: np.ndarray,
        *,
        external_combat_zones: Sequence[tuple[float, float]] = (),
        loot_radius_limit_px: float | None = None,
    ) -> YoloDetectionResult:
        """Tiled 3-crop high-res (512x512 overlapping) 3-class YOLO11n inference exclusively for BotState.LOOT.

        Filters out oversized boxes (>65x65px at 1024x768) and white/silver item drops via lightweight HSV check.
        """
        if self._session is None or image_bgr.size == 0:
            return YoloDetectionResult()

        fh, fw = image_bgr.shape[:2]
        sx = float(fw) / 1024.0
        sy = float(fh) / 768.0
        ty1 = max(0, int(round(140.0 * sy)))
        ty2 = min(fh, int(round(652.0 * sy)))
        tiles = (
            (0, ty1, min(fw, int(round(512.0 * sx))), ty2),
            (max(0, int(round(256.0 * sx))), ty1, min(fw, int(round(768.0 * sx))), ty2),
            (max(0, int(round(512.0 * sx))), ty1, fw, ty2),
        )

        all_boxes: list[np.ndarray] = []
        all_confs: list[np.ndarray] = []
        all_cids: list[np.ndarray] = []
        for x1, y1, x2, y2 in tiles:
            if (x2 - x1) < 16 or (y2 - y1) < 16:
                continue
            crop = image_bgr[y1:y2, x1:x2]
            b, s, c = self._infer_crop_candidates(
                crop,
                offset_x=x1,
                offset_y=y1,
                frame_w=fw,
                frame_h=fh,
                gold_bag_conf_override=self.loot_tiled_gold_bag_conf_threshold,
            )
            if b.shape[0] > 0:
                all_boxes.append(b)
                all_confs.append(s)
                all_cids.append(c)

        if all_boxes:
            merged_boxes = np.concatenate(all_boxes, axis=0)
            merged_confs = np.concatenate(all_confs, axis=0)
            merged_cids = np.concatenate(all_cids, axis=0)
            raw_dets = self._finalize_nms_and_spatial_filter(merged_boxes, merged_confs, merged_cids, fw, fh)
        else:
            raw_dets = []

        return self._build_detection_result(
            raw_dets,
            fw,
            fh,
            image_bgr=image_bgr,
            include_mobs=False,
            include_loot=True,
            external_combat_zones=external_combat_zones,
            loot_radius_limit_px=loot_radius_limit_px,
            enforce_loot_size_and_color=True,
        )

    def detect(
        self,
        image_bgr: np.ndarray,
        *,
        include_mobs: bool = True,
        include_loot: bool = True,
        external_combat_zones: Sequence[tuple[float, float]] = (),
        loot_radius_limit_px: float | None = None,
    ) -> YoloDetectionResult:
        """Run full-frame 3-class YOLO11n inference and map detections to MobTarget, LootDrop, and Players."""
        if self._session is None or image_bgr.size == 0:
            return YoloDetectionResult()
        fh, fw = image_bgr.shape[:2]
        raw_dets = self.infer_raw(image_bgr)
        return self._build_detection_result(
            raw_dets,
            fw,
            fh,
            image_bgr=image_bgr,
            include_mobs=include_mobs,
            include_loot=include_loot,
            external_combat_zones=external_combat_zones,
            loot_radius_limit_px=loot_radius_limit_px,
            enforce_loot_size_and_color=False,
        )

    def _build_detection_result(
        self,
        raw_dets: list[YoloRawDetection],
        fw: int,
        fh: int,
        *,
        image_bgr: np.ndarray | None = None,
        include_mobs: bool = True,
        include_loot: bool = True,
        external_combat_zones: Sequence[tuple[float, float]] = (),
        loot_radius_limit_px: float | None = None,
        enforce_loot_size_and_color: bool = False,
    ) -> YoloDetectionResult:
        scale_1024 = float(fw) / 1024.0
        scale_y = float(fh) / 768.0
        center_x = float(fw) * 0.5
        center_y = float(fh) * 0.5
        ks_radius = max(self.anti_ks_player_radius_px, self.anti_ks_damage_zone_radius_px) * scale_1024
        player_ks_r = self.anti_ks_player_radius_px * scale_1024
        max_bag_w = self.loot_bag_max_size_px * scale_1024
        max_bag_h = self.loot_bag_max_size_px * scale_y

        raw_players: list[tuple[int, int]] = []
        other_player_boxes: list[tuple[int, int, int, int]] = []
        other_player_confs: list[float] = []

        for det in raw_dets:
            if det.class_id == CLS_PLAYER:
                other_player_boxes.append(det.bbox)
                other_player_confs.append(det.confidence)
                if det.confidence >= self.anti_ks_player_min_conf:
                    raw_players.append((det.cx, det.cy))

        # Require >= 2 consecutive frames with a valid player before activating Anti-KS block
        if raw_players:
            self._player_consec_frames += 1
        else:
            self._player_consec_frames = 0
        player_confirmed = self._player_consec_frames >= self.anti_ks_player_confirm_frames
        ks_players = raw_players if player_confirmed else []

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
                if enforce_loot_size_and_color:
                    if float(bw) > max_bag_w or float(bh) > max_bag_h:
                        continue
                    if (
                        self.loot_color_filter_enabled
                        and image_bgr is not None
                        and not is_warm_gold_drop_roi(image_bgr, det.bbox)
                    ):
                        continue
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
            other_players=tuple(ks_players),
            other_player_boxes=tuple(other_player_boxes),
            other_player_confs=tuple(other_player_confs),
            raw_detections=tuple(raw_dets),
        )
