"""Sub-millisecond bitmap digit template matcher for Vikings Online (X, Y) HUD coordinates."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from vikings_bot.capture.screen import FramePacket
from vikings_bot.config import BotConfig
from vikings_bot.state.models import CoordReading

logger = logging.getLogger(__name__)

CANVAS_W = 10
CANVAS_H = 14
LABEL_TO_FILE = {
    "0": "0.png",
    "1": "1.png",
    "2": "2.png",
    "3": "3.png",
    "4": "4.png",
    "5": "5.png",
    "6": "6.png",
    "7": "7.png",
    "8": "8.png",
    "9": "9.png",
    ",": "comma.png",
}
FILE_TO_LABEL = {v: k for k, v in LABEL_TO_FILE.items()}


def _extract_white_mask(crop_bgr: np.ndarray, min_val: int = 130) -> np.ndarray:
    """Isolate crisp white coordinate text while rejecting the gold minimap border."""
    b = crop_bgr[:, :, 0]
    g = crop_bgr[:, :, 1]
    r = crop_bgr[:, :, 2]
    # Gold border has very low blue channel; white HUD text has high B, G, R
    min_rgb = np.minimum(np.minimum(b, g), r)
    max_rgb = np.maximum(np.maximum(b, g), r)
    mask = (min_rgb >= min_val) & ((max_rgb.astype(np.int16) - min_rgb.astype(np.int16)) <= 55)
    return mask.astype(np.uint8) * 255


def _normalize_glyph(glyph_gray: np.ndarray, anchor_bottom: bool = False) -> np.ndarray:
    """Place glyph onto a fixed CANVAS_W x CANVAS_H canvas without distorting aspect ratio."""
    h, w = glyph_gray.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)

    # Scale down only if larger than canvas (e.g. higher DPI cropped image)
    scale = min(1.0, (CANVAS_W - 2) / max(1, w), (CANVAS_H - 2) / max(1, h))
    if scale < 1.0:
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        glyph_gray = cv2.resize(glyph_gray, (nw, nh), interpolation=cv2.INTER_AREA)
        h, w = glyph_gray.shape[:2]

    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    x0 = (CANVAS_W - w) // 2
    y0 = (CANVAS_H - h - 1) if anchor_bottom else (CANVAS_H - h) // 2
    y0 = max(0, min(CANVAS_H - h, y0))
    canvas[y0 : y0 + h, x0 : x0 + w] = glyph_gray.astype(np.float32) / 255.0
    return canvas


def segment_glyphs(crop_bgr: np.ndarray, _min_val: int = 130) -> list[tuple[int, int, int, int, np.ndarray]]:
    """Segment coordinate text into left-to-right (x, y, w, h, normalized_canvas) glyphs."""
    mask = _extract_white_mask(crop_bgr, min_val=_min_val)
    # Dim VMware SVGA / software-rendered HUD text can sit around RGB 95..128; if standard threshold
    # finds too few bright white pixels, retry once with a lower white threshold without affecting bright 1080p!
    if _min_val >= 130:
        primary_boxes_px = int((mask > 0).sum())
        min_needed_px = 45 if crop_bgr.shape[1] >= 70 else 18
        if primary_boxes_px < min_needed_px:
            res_hi = segment_glyphs(crop_bgr, _min_val=129)
            if len(res_hi) >= (5 if crop_bgr.shape[1] >= 70 else 2):
                return res_hi
            res_lo = segment_glyphs(crop_bgr, _min_val=92)
            return res_lo if len(res_lo) > len(res_hi) else res_hi
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # Suppress non-white pixels so JPEG halos don't pollute template correlation
    clean_gray = np.where(mask > 0, gray, 0).astype(np.uint8)

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    raw_boxes: list[tuple[int, int, int, int]] = []
    h_img, w_img = mask.shape[:2]

    for idx in range(1, num_labels):
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        # Reject border-touching noise or huge blobs (like zoom '-' icon on the right)
        if x <= 1 or (x + w) >= w_img - 1:
            continue
        if area < 3 or h < 2 or h > int(h_img * 0.85) or w > 24:
            continue
        # Reject the zoom button '-' circle if ROI includes it near far right
        if w > 9 and h <= 3:
            continue
        raw_boxes.append((x, y, w, h))

    if not raw_boxes:
        return []

    # Keep boxes aligned to the dominant text baseline (median bottom y)
    bottoms = [y + h for (_x, y, _w, h) in raw_boxes if h >= 6]
    if not bottoms:
        return []
    baseline_y = float(np.median(bottoms))
    filtered: list[tuple[int, int, int, int]] = []
    for x, y, w, h in raw_boxes:
        # Commas extend slightly below baseline (+1..+3 px) and are short (h 2..5)
        if abs((y + h) - baseline_y) <= 4.5:
            filtered.append((x, y, w, h))

    filtered.sort(key=lambda b: b[0])

    # Split fused wide digits recursively if JPEG/VMware AA bridges a 1px gap
    final_boxes: list[tuple[int, int, int, int]] = []
    queue_boxes: list[tuple[int, int, int, int]] = list(filtered)
    while queue_boxes:
        x, y, w, h = queue_boxes.pop(0)
        if w >= 10 and h >= 6:
            sub = mask[y : y + h, x : x + w]
            col_sums = sub.sum(axis=0)
            mid_lo = max(2, w // 3)
            mid_hi = min(w - 2, (2 * w) // 3 + 1)
            split_rel = mid_lo + int(np.argmin(col_sums[mid_lo:mid_hi]))
            if split_rel >= 2 and (w - split_rel) >= 2:
                queue_boxes.insert(0, (x + split_rel, y, w - split_rel, h))
                queue_boxes.insert(0, (x, y, split_rel, h))
                continue
        final_boxes.append((x, y, w, h))

    glyphs: list[tuple[int, int, int, int, np.ndarray]] = []
    for x, y, w, h in final_boxes:
        patch = clean_gray[y : y + h, x : x + w]
        is_short = h <= 5
        canvas = _normalize_glyph(patch, anchor_bottom=is_short)
        glyphs.append((x, y, w, h, canvas))
    return glyphs


def calibrate_digit_templates(
    labeled_crops: list[tuple[np.ndarray, str]],
    output_dir: str | Path,
) -> dict[str, int]:
    """Build averaged canonical bitmap templates for '0'..'9' and ',' from labeled crops."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[np.ndarray]] = {}

    for crop_bgr, expected_text in labeled_crops:
        expected_chars = [ch for ch in expected_text if ch in LABEL_TO_FILE]
        glyphs = segment_glyphs(crop_bgr)
        if len(glyphs) != len(expected_chars):
            logger.warning(
                "Skipping calibration sample '%s': segmented %d glyphs, expected %d",
                expected_text,
                len(glyphs),
                len(expected_chars),
            )
            continue
        for ch, (_x, _y, _w, _h, canvas) in zip(expected_chars, glyphs):
            buckets.setdefault(ch, []).append(canvas)

    counts: dict[str, int] = {}
    for ch, samples in buckets.items():
        avg = np.mean(np.stack(samples, axis=0), axis=0)
        img_u8 = np.clip(avg * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out / LABEL_TO_FILE[ch]), img_u8)
        counts[ch] = len(samples)
    return counts


class CoordReader:
    """Reads integer world coordinates (X, Y) under the Vikings Online minimap."""

    def __init__(self, config: BotConfig) -> None:
        self.roi = config.roi("coords")
        self.min_confidence = float(config.perception.get("coord_min_confidence", 0.58))
        self.hud_fixed_pixel_ui = bool(config.perception.get("hud_fixed_pixel_ui", False))
        self.templates_dir = config.resolve_path(
            config.perception.get("digit_templates_dir", "assets/digit_templates")
        )
        self.templates: dict[str, np.ndarray] = {}
        self._digit_labels: list[str] = []
        self._tmpl_norm: np.ndarray | None = None
        self._tmpl_bin: np.ndarray | None = None
        self._tmpl_bin_sum: np.ndarray | None = None
        self.load_templates()

    def load_templates(self) -> None:
        self.templates.clear()
        self._digit_labels = []
        self._tmpl_norm = None
        self._tmpl_bin = None
        self._tmpl_bin_sum = None
        if not self.templates_dir.exists():
            return
        for fname, label in FILE_TO_LABEL.items():
            fpath = self.templates_dir / fname
            if not fpath.exists():
                continue
            img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (CANVAS_H, CANVAS_W):
                img = cv2.resize(img, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_AREA)
            self.templates[label] = img.astype(np.float32) / 255.0

        # Pre-stack non-comma digit templates for vectorized 1-call similarity matching (~60x faster)
        norm_rows: list[np.ndarray] = []
        bin_rows: list[np.ndarray] = []
        for label, tmpl in self.templates.items():
            if label == ",":
                continue
            flat = tmpl.reshape(-1)
            n = float(np.linalg.norm(flat))
            self._digit_labels.append(label)
            norm_rows.append(flat / n if n >= 1e-6 else np.zeros_like(flat))
            bin_rows.append((flat > 0.35).astype(np.float32))
        if norm_rows:
            self._tmpl_norm = np.stack(norm_rows, axis=0)  # (K, 140)
            self._tmpl_bin = np.stack(bin_rows, axis=0)    # (K, 140)
            self._tmpl_bin_sum = self._tmpl_bin.sum(axis=1)  # (K,)

    @staticmethod
    def _similarity(a: np.ndarray, b: np.ndarray) -> float:
        # Cosine similarity + soft IoU blend works robustly on tiny 10x14 bitmap fonts
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        cos = float(np.sum(a * b) / (norm_a * norm_b))
        bin_a = a > 0.35
        bin_b = b > 0.35
        union = int(np.logical_or(bin_a, bin_b).sum())
        iou = float(np.logical_and(bin_a, bin_b).sum()) / union if union > 0 else 0.0
        return 0.65 * cos + 0.35 * iou

    def _match_glyph(self, h: int, w: int, canvas: np.ndarray) -> tuple[str, float]:
        # Commas are distinctly short & narrow near baseline
        if h <= 5 and w <= 4 and "," in self.templates:
            return (",", max(0.85, self._similarity(canvas, self.templates[","])))

        if self._tmpl_norm is not None and self._tmpl_bin is not None and self._tmpl_bin_sum is not None:
            # Build all 5 +-1px shifts matching np.roll(np.roll(canvas, dy, 0), dx, 1) in one vectorized pass
            s0 = canvas
            s_l = np.empty_like(canvas)
            s_l[:, :-1] = canvas[:, 1:]
            s_l[:, -1] = canvas[:, 0]
            s_r = np.empty_like(canvas)
            s_r[:, 1:] = canvas[:, :-1]
            s_r[:, 0] = canvas[:, -1]
            s_u = np.empty_like(canvas)
            s_u[:-1, :] = canvas[1:, :]
            s_u[-1, :] = canvas[0, :]
            s_d = np.empty_like(canvas)
            s_d[1:, :] = canvas[:-1, :]
            s_d[0, :] = canvas[-1, :]

            shifts = np.stack((s0, s_l, s_r, s_u, s_d), axis=0).reshape(5, -1)
            norm_s = float(np.linalg.norm(shifts[0]))
            if norm_s < 1e-6:
                return ("?", 0.0)
            cos = (shifts / norm_s) @ self._tmpl_norm.T  # (5, K)
            s_bin = (shifts > 0.35).astype(np.float32)
            inter = s_bin @ self._tmpl_bin.T  # (5, K)
            s_bin_sum = float(s_bin[0].sum())
            union = s_bin_sum + self._tmpl_bin_sum[None, :] - inter
            iou = np.where(union > 0.0, inter / np.maximum(union, 1.0), 0.0)
            scores = 0.65 * cos + 0.35 * iou
            best_per_digit = scores.max(axis=0)
            best_idx = int(np.argmax(best_per_digit))
            return (self._digit_labels[best_idx], float(best_per_digit[best_idx]))

        best_label = "?"
        best_score = -1.0
        for label, tmpl in self.templates.items():
            if label == ",":
                continue
            score = self._similarity(canvas, tmpl)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                shifted = np.roll(np.roll(canvas, dy, axis=0), dx, axis=1)
                s = self._similarity(shifted, tmpl)
                if s > score:
                    score = s
            if score > best_score:
                best_score = score
                best_label = label
        return (best_label, best_score)

    def _read_crop_with_threshold(self, crop_bgr: np.ndarray, min_val: int) -> CoordReading | None:
        glyphs = segment_glyphs(crop_bgr, _min_val=min_val)
        if len(glyphs) < 3:
            return None

        chars: list[str] = []
        scores: list[float] = []
        prev_right: int | None = None
        n_glyphs = len(glyphs)
        for gi, (x, _y, w, h, canvas) in enumerate(glyphs):
            label, score = self._match_glyph(h, w, canvas)
            # Ignore stray leading/trailing tiny comma-sized specks at the ends of the bar
            if label == "," and (gi == 0 or gi == n_glyphs - 1):
                continue
            # If there's a noticeable horizontal gap and no comma was recognized, insert comma
            if prev_right is not None and (x - prev_right) >= 5 and "," not in chars and label != ",":
                chars.append(",")
                scores.append(0.85)
            chars.append(label)
            scores.append(score)
            prev_right = x + w

        raw = "".join(chars)
        if raw.count(",") != 1:
            return None
        left_str, right_str = raw.split(",", 1)
        if not (left_str.isdigit() and right_str.isdigit()):
            return None
        if not (1 <= len(left_str) <= 5 and 1 <= len(right_str) <= 5):
            return None

        conf = float(min(scores)) if scores else 0.0
        if conf < self.min_confidence:
            return None
        return CoordReading(x=int(left_str), y=int(right_str), confidence=conf, raw_text=raw)

    def read_crop(self, crop_bgr: np.ndarray) -> CoordReading | None:
        if not self.templates:
            return None
        # Try crisp 1080p threshold first, then gracefully step down for dimmer VMware 1024x768 HUD text
        for min_val in (130, 110, 88):
            res = self._read_crop_with_threshold(crop_bgr, min_val=min_val)
            if res is not None:
                return res
        return None

    @staticmethod
    def _is_plausible_world_coord(reading: CoordReading | None) -> bool:
        """Reject bogus 1-2 digit minimap noise (e.g. '11,14'); RYL world coords are 3-4 digits."""
        if reading is None:
            return False
        if not (50 <= reading.x <= 4096 and 50 <= reading.y <= 4096):
            return False
        return (reading.x >= 100) or (reading.y >= 100)

    def read(self, frame: FramePacket) -> CoordReading | None:
        fh, fw = frame.height, frame.width
        if not self.hud_fixed_pixel_ui:
            crop, _ = frame.crop_norm(self.roi)
            return self.read_crop(crop)

        # VMware 1024x768 fixed-pixel UI: RYL renders the minimap & coord strip at native 1:1 pixels.
        # Primary box x=4..114 covers full 4-digit city coords ('1066, 2627', '1094, 2644', '1409, 1773') without clipping!
        for fx1, fy1, fx2, fy2 in (
            (4, 120, min(fw, 114), min(fh, 144)),
            (5, 122, min(fw, 96), min(fh, 143)),
            (
                max(0, int(round(self.roi[0] * 1920.0)) - 2),
                max(0, int(round(self.roi[1] * 1080.0)) - 2),
                min(fw, max(44, int(round(self.roi[2] * 1920.0)) + 4)),
                min(fh, max(136, int(round(self.roi[3] * 1080.0)) + 2)),
            ),
        ):
            if (fy2 - fy1) >= 12 and (fx2 - fx1) >= 40:
                res_fixed = self.read_crop(frame.image_bgr[fy1:fy2, fx1:fx2])
                if self._is_plausible_world_coord(res_fixed):
                    return res_fixed

        # Adaptive dark coordinate-strip locator (handles slight VMware window border / titlebar shifts
        # while covering full 3-4 digit X,Y like '610,1841' or '1075,2631')
        y1, y2 = min(fh - 1, 95), min(fh, 185)
        for x_end in (min(fw, 118), min(fw, 108)):
            x1 = 2
            if (y2 - y1) < 20 or (x_end - x1) < 40:
                continue
            band = frame.image_bgr[y1:y2, x1:x_end]
            row_max_ch = band.max(axis=2)
            row_med = np.median(row_max_ch, axis=1)
            dark_rows = np.where(row_med <= 42.0)[0]
            if dark_rows.size < 8:
                continue
            splits = np.where(np.diff(dark_rows) > 2)[0] + 1
            runs = sorted(np.split(dark_rows, splits), key=lambda r: r.size, reverse=True)
            for run in runs:
                if run.size < 8:
                    continue
                ry1 = max(0, y1 + int(run[0]) - 2)
                ry2 = min(fh, y1 + int(run[-1]) + 3)
                alt = self.read_crop(frame.image_bgr[ry1:ry2, x1:x_end])
                if self._is_plausible_world_coord(alt):
                    return alt

        # Final fallback if frame actually uses proportional HUD scaling
        crop_norm, _ = frame.crop_norm(self.roi)
        res_norm = self.read_crop(crop_norm)
        return res_norm if self._is_plausible_world_coord(res_norm) else None

    def read_int_roi(self, frame: FramePacket, roi: tuple[float, float, float, float]) -> int | None:
        """Read a single integer HUD stat (e.g. bottom-left EA value: 94 vs 179) using digit templates."""
        if not self.templates:
            return None
        crops: list[np.ndarray] = []
        if self.hud_fixed_pixel_ui and roi[1] >= 0.80:
            fh, fw = frame.height, frame.width
            x1 = max(0, int(round(roi[0] * 1920.0)))
            x2 = min(fw, max(x1 + 10, int(round(roi[2] * 1920.0))))
            y1 = max(0, fh - int(round((1.0 - roi[1]) * 1080.0)))
            y2 = min(fh, max(y1 + 6, fh - int(round((1.0 - roi[3]) * 1080.0))))
            if y2 > y1 and x2 > x1:
                crops.append(frame.image_bgr[y1:y2, x1:x2])
        crop_n, _ = frame.crop_norm(roi)
        crops.append(crop_n)

        for crop in crops:
            if crop.size == 0:
                continue
            glyphs = segment_glyphs(crop)
            if not (1 <= len(glyphs) <= 4):
                continue
            digits: list[str] = []
            scores: list[float] = []
            ok = True
            for _x, _y, w, h, canvas in glyphs:
                label, score = self._match_glyph(h, w, canvas)
                if not label.isdigit():
                    ok = False
                    break
                digits.append(label)
                scores.append(score)
            if ok and scores and min(scores) >= max(0.65, self.min_confidence):
                return int("".join(digits))
        return None
