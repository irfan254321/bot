"""Threaded screen capture (mss + optional DXGI fallback) and offline replay source."""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vikings_bot.capture.window import GameWindow, WindowTracker
from vikings_bot.config import NormROI, roi_to_pixels

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FramePacket:
    image_bgr: np.ndarray
    timestamp: float
    origin_x: int = 0
    origin_y: int = 0
    source_name: str = ""

    @property
    def height(self) -> int:
        return int(self.image_bgr.shape[0])

    @property
    def width(self) -> int:
        return int(self.image_bgr.shape[1])

    @property
    def mean_luminance(self) -> float:
        # 16x16 strided subsample is ~256x faster than averaging all 6.2M BGR bytes at 30 FPS
        return float(self.image_bgr[::16, ::16].mean())

    def crop_norm(self, roi: NormROI) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        x1, y1, x2, y2 = roi_to_pixels(roi, self.width, self.height)
        return self.image_bgr[y1:y2, x1:x2], (x1, y1, x2, y2)


class FrameSource(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def latest_frame(self) -> FramePacket | None: ...


class MssCaptureSource(FrameSource):
    """Background threaded screen grabber via mss with latest-frame ring buffer."""

    def __init__(
        self,
        window_tracker: WindowTracker,
        target_fps: float = 30.0,
        black_frame_threshold: float = 2.0,
        black_frame_limit: int = 15,
    ) -> None:
        self.window_tracker = window_tracker
        self.target_fps = max(1.0, target_fps)
        self.black_frame_threshold = black_frame_threshold
        self.black_frame_limit = black_frame_limit

        self._lock = threading.Lock()
        self._latest: FramePacket | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.consecutive_black_frames: int = 0
        self.capture_fps_actual: float = 0.0
        self._last_window: GameWindow | None = None

    @property
    def current_window(self) -> GameWindow | None:
        return self._last_window

    @property
    def black_frame_warning(self) -> bool:
        return self.consecutive_black_frames >= self.black_frame_limit

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="MssCaptureThread", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def latest_frame(self) -> FramePacket | None:
        with self._lock:
            return self._latest

    def _run_loop(self) -> None:
        import mss

        period = 1.0 / self.target_fps
        fps_frames = 0
        fps_t0 = time.perf_counter()

        with mss.mss() as sct:
            while not self._stop_event.is_set():
                t0 = time.perf_counter()
                win = self.window_tracker.find_window()
                self._last_window = win
                if win is not None:
                    region = win.monitor_dict
                else:
                    # Fallback to primary monitor in fullscreen if window discovery is pending
                    mon = sct.monitors[1]
                    region = {
                        "left": int(mon["left"]),
                        "top": int(mon["top"]),
                        "width": int(mon["width"]),
                        "height": int(mon["height"]),
                    }
                try:
                    raw = sct.grab(region)
                    img_bgr = np.asarray(raw, dtype=np.uint8)[:, :, :3].copy()
                    pkt = FramePacket(
                        image_bgr=img_bgr,
                        timestamp=time.time(),
                        origin_x=region["left"],
                        origin_y=region["top"],
                        source_name="mss",
                    )
                    if pkt.mean_luminance < self.black_frame_threshold:
                        self.consecutive_black_frames += 1
                        if self.consecutive_black_frames == self.black_frame_limit:
                            logger.warning(
                                "Detected %d consecutive black frames under mss. "
                                "Exclusive DirectX fullscreen may block GDI capture; "
                                "try DXGI backend (dxcam) or Borderless Fullscreen.",
                                self.consecutive_black_frames,
                            )
                    else:
                        self.consecutive_black_frames = 0
                    with self._lock:
                        self._latest = pkt
                except Exception as exc:
                    logger.debug("mss capture error: %s", exc)

                fps_frames += 1
                now = time.perf_counter()
                if now - fps_t0 >= 1.0:
                    self.capture_fps_actual = fps_frames / (now - fps_t0)
                    fps_frames = 0
                    fps_t0 = now

                sleep_time = period - (now - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)


class DxgiCaptureSource(FrameSource):
    """Optional Desktop Duplication (DXGI) capture via dxcam for exclusive fullscreen."""

    def __init__(self, target_fps: float = 30.0) -> None:
        self.target_fps = int(max(1, round(target_fps)))
        self._camera = None
        self.capture_fps_actual: float = 0.0

    def start(self) -> None:
        import dxcam  # type: ignore[import-not-found]

        self._camera = dxcam.create(output_color="BGR")
        self._camera.start(target_fps=self.target_fps, video_mode=True)

    def stop(self) -> None:
        if self._camera is not None:
            self._camera.stop()
            del self._camera
            self._camera = None

    def latest_frame(self) -> FramePacket | None:
        if self._camera is None:
            return None
        frame = self._camera.get_latest_frame()
        if frame is None:
            return None
        return FramePacket(image_bgr=frame, timestamp=time.time(), origin_x=0, origin_y=0, source_name="dxgi")


class ReplayFrameSource(FrameSource):
    """Sequential offline image loader for deterministic testing and visual replay."""

    def __init__(self, directory: str | Path, name_filter: str = "", step_sec: float = 0.5) -> None:
        self.directory = Path(directory)
        self.name_filter = name_filter.lower()
        self.step_sec = step_sec
        self.files: list[Path] = []
        self._index: int = 0
        self._sim_ts: float = 1000.0

    def start(self) -> None:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        found = [
            p
            for p in self.directory.iterdir()
            if p.is_file() and p.suffix.lower() in exts and (not self.name_filter or self.name_filter in p.name.lower())
        ]
        # Sort by natural timestamp + numeric suffix inside parentheses if present
        found.sort(key=lambda p: (p.stat().st_mtime, p.name))
        self.files = found
        self._index = 0
        self._sim_ts = 1000.0

    def stop(self) -> None:
        pass

    def latest_frame(self) -> FramePacket | None:
        if self._index >= len(self.files):
            return None
        path = self.files[self._index]
        self._index += 1
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        pkt = FramePacket(
            image_bgr=img,
            timestamp=self._sim_ts,
            origin_x=0,
            origin_y=0,
            source_name=path.name,
        )
        self._sim_ts += self.step_sec
        return pkt


class PrintWindowCaptureSource(FrameSource):
    """Win32 PrintWindow (PW_RENDERFULLCONTENT) background window capture for unfocused Host AFK testing."""

    def __init__(
        self,
        window_tracker: WindowTracker,
        target_fps: float = 20.0,
        black_frame_threshold: float = 2.0,
        black_frame_limit: int = 15,
    ) -> None:
        self.window_tracker = window_tracker
        self.target_fps = max(1.0, target_fps)
        self.black_frame_threshold = black_frame_threshold
        self.black_frame_limit = black_frame_limit

        self._lock = threading.Lock()
        self._latest: FramePacket | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.consecutive_black_frames: int = 0
        self.capture_fps_actual: float = 0.0
        self._last_window: GameWindow | None = None

    @property
    def current_window(self) -> GameWindow | None:
        return self._last_window

    @property
    def black_frame_warning(self) -> bool:
        return self.consecutive_black_frames >= self.black_frame_limit

    @staticmethod
    def grab_hwnd_bgr(hwnd: int, width: int, height: int) -> np.ndarray | None:
        """Capture a single BGRA->BGR frame directly from an HWND via 64-bit-safe Win32 PrintWindow / BitBlt."""
        import ctypes
        from ctypes import wintypes

        if hwnd <= 0 or width < 64 or height < 64:
            return None

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

        user32.GetDC.argtypes = [wintypes.HWND]
        user32.GetDC.restype = wintypes.HDC
        user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        user32.ReleaseDC.restype = ctypes.c_int
        user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
        user32.PrintWindow.restype = wintypes.BOOL

        gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
        gdi32.CreateCompatibleDC.restype = wintypes.HDC
        gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
        gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
        gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        gdi32.SelectObject.restype = wintypes.HGDIOBJ
        gdi32.BitBlt.argtypes = [
            wintypes.HDC,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HDC,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.DWORD,
        ]
        gdi32.BitBlt.restype = wintypes.BOOL
        gdi32.GetDIBits.argtypes = [
            wintypes.HDC,
            wintypes.HBITMAP,
            wintypes.UINT,
            wintypes.UINT,
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.UINT,
        ]
        gdi32.GetDIBits.restype = ctypes.c_int
        gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        gdi32.DeleteObject.restype = wintypes.BOOL
        gdi32.DeleteDC.argtypes = [wintypes.HDC]
        gdi32.DeleteDC.restype = wintypes.BOOL

        class _BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        hwnd_c = wintypes.HWND(int(hwnd))
        wnd_dc = user32.GetDC(hwnd_c)
        if not wnd_dc:
            return None
        mem_dc = gdi32.CreateCompatibleDC(wnd_dc)
        if not mem_dc:
            user32.ReleaseDC(hwnd_c, wnd_dc)
            return None
        bmp = gdi32.CreateCompatibleBitmap(wnd_dc, int(width), int(height))
        if not bmp:
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(hwnd_c, wnd_dc)
            return None
        old_obj = gdi32.SelectObject(mem_dc, bmp)
        try:
            # PW_CLIENTONLY (0x1) | PW_RENDERFULLCONTENT (0x2) = 0x3
            ok = bool(user32.PrintWindow(hwnd_c, mem_dc, 0x00000003))
            if not ok:
                ok = bool(user32.PrintWindow(hwnd_c, mem_dc, 0x00000002))
            if not ok:
                SRCCOPY = 0x00CC0020
                ok = bool(gdi32.BitBlt(mem_dc, 0, 0, int(width), int(height), wnd_dc, 0, 0, SRCCOPY))
            if not ok:
                return None

            bmi = _BITMAPINFOHEADER()
            bmi.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
            bmi.biWidth = int(width)
            bmi.biHeight = -int(height)  # top-down DIB
            bmi.biPlanes = 1
            bmi.biBitCount = 32
            bmi.biCompression = 0  # BI_RGB

            buf = np.empty((int(height), int(width), 4), dtype=np.uint8)
            lines = gdi32.GetDIBits(
                mem_dc,
                bmp,
                0,
                int(height),
                buf.ctypes.data_as(ctypes.c_void_p),
                ctypes.byref(bmi),
                0,
            )
            if lines <= 0:
                return None
            return buf[:, :, :3].copy()
        finally:
            if old_obj:
                gdi32.SelectObject(mem_dc, old_obj)
            gdi32.DeleteObject(bmp)
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(hwnd_c, wnd_dc)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="PrintWindowCaptureThread", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def latest_frame(self) -> FramePacket | None:
        with self._lock:
            return self._latest

    def _run_loop(self) -> None:
        period = 1.0 / self.target_fps
        fps_frames = 0
        fps_t0 = time.perf_counter()

        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            win = self.window_tracker.find_window()
            self._last_window = win
            if win is not None:
                try:
                    img_bgr = self.grab_hwnd_bgr(win.hwnd, win.width, win.height)
                    if img_bgr is not None:
                        pkt = FramePacket(
                            image_bgr=img_bgr,
                            timestamp=time.time(),
                            origin_x=win.left,
                            origin_y=win.top,
                            source_name="printwindow",
                        )
                        if pkt.mean_luminance < self.black_frame_threshold:
                            self.consecutive_black_frames += 1
                        else:
                            self.consecutive_black_frames = 0
                        with self._lock:
                            self._latest = pkt
                except Exception as exc:
                    logger.debug("PrintWindow capture error: %s", exc)

            fps_frames += 1
            now = time.perf_counter()
            if now - fps_t0 >= 1.0:
                self.capture_fps_actual = fps_frames / (now - fps_t0)
                fps_frames = 0
                fps_t0 = now

            sleep_time = period - (now - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)


def create_capture_source(
    backend: str,
    window_tracker: WindowTracker,
    target_fps: float = 30.0,
    black_frame_threshold: float = 2.0,
    black_frame_limit: int = 15,
) -> FrameSource:
    backend_norm = backend.lower()
    if backend_norm == "printwindow":
        logger.info("Using Win32 PrintWindow background capture backend.")
        return PrintWindowCaptureSource(
            window_tracker=window_tracker,
            target_fps=target_fps,
            black_frame_threshold=black_frame_threshold,
            black_frame_limit=black_frame_limit,
        )
    if backend_norm == "dxgi":
        return DxgiCaptureSource(target_fps=target_fps)
    if backend_norm == "auto":
        try:
            import dxcam  # type: ignore[import-not-found]

            # Verify DXGI Desktop Duplication actually initializes (fails on virtual GPU VMs without WDDM 2.0)
            probe = dxcam.create(output_color="BGR")
            if probe is not None:
                del probe
                logger.info("Using DXGI (dxcam) capture backend for fullscreen compatibility.")
                return DxgiCaptureSource(target_fps=target_fps)
        except Exception:
            pass
        logger.info("Using mss capture backend.")
        return MssCaptureSource(
            window_tracker=window_tracker,
            target_fps=target_fps,
            black_frame_threshold=black_frame_threshold,
            black_frame_limit=black_frame_limit,
        )
    return MssCaptureSource(
        window_tracker=window_tracker,
        target_fps=target_fps,
        black_frame_threshold=black_frame_threshold,
        black_frame_limit=black_frame_limit,
    )
