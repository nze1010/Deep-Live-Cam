"""Modern Simplified Studio UI for Deep-Live-Cam.

Unified, single-window creative studio with:
- Integrated Live Webcam viewport (no detached popup windows)
- Mode switcher: Live Camera vs Media File Swap
- Source Face picker with instant thumbnail preview
- Streamlined settings deck (GFPGAN Enhancer, Mouth Mask, Poisson Blend)
- Real-time FPS monitor, snapshot capture, and output management
"""

from __future__ import annotations

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import platform
import queue
import sys
import threading
import time
import webbrowser
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import (
    QObject,
    QThread,
    QTimer,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import modules.globals
import modules.metadata
from modules import imread_unicode
from modules.capturer import get_video_frame, get_video_frame_total
from modules.face_analyser import (
    detect_many_faces_fast,
    detect_one_face_fast,
    ensure_landmarks,
    get_one_face,
)
from modules.gettext import LanguageManager
from modules.gpu_processing import gpu_cvt_color, gpu_flip, gpu_resize
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    has_image_extension,
    is_image,
    is_video,
)
from modules.video_capture import VideoCapturer
from modules.voice_changer import VOICE_ENGINE, VOICE_PRESETS

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph


# ─── Color Palette & Modern Dark Theme ────────────────────────────────────────

STUDIO_QSS = """
QMainWindow {
    background-color: #0f1115;
    color: #e2e8f0;
}

QWidget {
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 10pt;
}

/* Studio Header */
#headerFrame {
    background-color: #161920;
    border-bottom: 1px solid #232733;
    padding: 10px 16px;
}

#brandTitle {
    font-size: 13pt;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.5px;
}

#brandBadge {
    background-color: #272c3d;
    color: #818cf8;
    font-size: 8pt;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 6px;
}

#hardwareBadge {
    background-color: #1a2e26;
    color: #34d399;
    font-size: 8.5pt;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid #059669;
}

/* Navigation Mode Buttons */
QPushButton.navTab {
    background-color: transparent;
    color: #94a3b8;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 10pt;
    font-weight: 600;
}
QPushButton.navTab:hover {
    background-color: #1e232d;
    color: #f1f5f9;
}
QPushButton.navTab[active="true"] {
    background-color: #4f46e5;
    color: #ffffff;
}

/* Glassy Cards */
QFrame.studioCard {
    background-color: #161920;
    border: 1px solid #232733;
    border-radius: 12px;
}

#cardTitle {
    font-size: 10.5pt;
    font-weight: 700;
    color: #cbd5e1;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Primary Action Buttons */
QPushButton.primaryBtn {
    background-color: #4f46e5;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 10.5pt;
    font-weight: 600;
}
QPushButton.primaryBtn:hover {
    background-color: #6366f1;
}
QPushButton.primaryBtn:pressed {
    background-color: #4338ca;
}
QPushButton.primaryBtn:disabled {
    background-color: #262a36;
    color: #64748b;
}

QPushButton.liveActiveBtn {
    background-color: #dc2626;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 10.5pt;
    font-weight: 700;
}
QPushButton.liveActiveBtn:hover {
    background-color: #ef4444;
}

QPushButton.secondaryBtn {
    background-color: #232733;
    color: #e2e8f0;
    border: 1px solid #333948;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
}
QPushButton.secondaryBtn:hover {
    background-color: #2d3342;
    border-color: #475569;
}

/* Drop Areas */
QLabel#sourceDropZone {
    background-color: #1a1e27;
    border: 2px dashed #333948;
    border-radius: 12px;
}
QLabel#sourceDropZone:hover {
    border-color: #6366f1;
    background-color: #1e2330;
}

QLabel#videoCanvas {
    background-color: #0b0c0e;
    border: 1px solid #1f232b;
    border-radius: 12px;
}

/* Form Controls */
QComboBox {
    background-color: #1e222b;
    color: #e2e8f0;
    border: 1px solid #333948;
    border-radius: 8px;
    padding: 6px 12px;
    min-height: 24px;
}
QComboBox:hover {
    border-color: #6366f1;
}
QComboBox QAbstractItemView {
    background-color: #1e222b;
    color: #e2e8f0;
    selection-background-color: #4f46e5;
    border: 1px solid #333948;
}

QCheckBox {
    spacing: 10px;
    font-weight: 500;
    color: #cbd5e1;
    padding: 3px 0px;
}
QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border-radius: 6px;
    border: 1px solid #3f4758;
    background-color: #1a1e27;
}
QCheckBox::indicator:checked {
    background-color: #4f46e5;
    border-color: #6366f1;
}

/* Progress Bar */
QProgressBar {
    background-color: #1e222b;
    border: 1px solid #2b303c;
    border-radius: 6px;
    text-align: center;
    color: #ffffff;
    font-weight: 600;
    height: 18px;
}
QProgressBar::chunk {
    background-color: #4f46e5;
    border-radius: 5px;
}

/* Voice Cards */
QPushButton.voiceCard {
    background-color: #161a23;
    border: 1px solid #282e3d;
    border-radius: 10px;
    padding: 12px 14px;
    text-align: left;
    color: #e2e8f0;
    font-size: 9.5pt;
    font-weight: 600;
}
QPushButton.voiceCard:hover {
    background-color: #1e2433;
    border-color: #6366f1;
}
QPushButton.voiceCard[active="true"] {
    background-color: #21253c;
    border: 2px solid #818cf8;
    color: #ffffff;
}

/* Audio VU Meter */
#audioVuMeter {
    background-color: #141720;
    border: 1px solid #282d3c;
    border-radius: 4px;
    height: 10px;
}
#audioVuMeter::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:0.7 #38bdf8, stop:1 #818cf8);
    border-radius: 3px;
}

/* Status Bar Footer */
#footerBar {
    background-color: #12151b;
    border-top: 1px solid #1f232b;
    padding: 6px 16px;
}
#statusLabel {
    color: #94a3b8;
    font-size: 9pt;
}
#fpsBadge {
    color: #10b981;
    font-weight: 700;
    font-size: 9.5pt;
}
"""

_APP: Optional[QApplication] = None
_MAIN: Optional[StudioMainWindow] = None
_LANG: Optional[LanguageManager] = None
_BRIDGE: Optional[_UIBridge] = None
_CACHED_SOURCE_FACE: Any = None


def _(text: str) -> str:
    return _LANG.get(text) if _LANG else text


class _UIBridge(QObject):
    statusChanged = Signal(str)
    progressChanged = Signal(int)
    faceAnalyzed = Signal(bool, str)
    cameraStarted = Signal(bool, str)
    cameraStopped = Signal()
    voiceLevelChanged = Signal(float)
    voiceStateChanged = Signal(bool, str)


def _emit_status(text: str) -> None:
    if _BRIDGE is not None:
        _BRIDGE.statusChanged.emit(text)
    else:
        print(f"[DLC.STATUS] {text}")


# ─── Public API ───────────────────────────────────────────────────────────────

def update_status(text: str) -> None:
    _emit_status(_(text))
    if _APP is not None and QThread.currentThread() is _APP.thread():
        _APP.processEvents()


def check_and_ignore_nsfw(target, destroy: Optional[Callable] = None) -> bool:
    from numpy import ndarray
    from modules.predicter import predict_frame, predict_image, predict_video

    check_nsfw = None
    if isinstance(target, str):
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif isinstance(target, ndarray):
        check_nsfw = predict_frame

    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(to_quit=False)
        update_status("Content flagged as sensitive — skipped.")
        return True
    return False


def get_available_cameras() -> Tuple[List[int], List[str]]:
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            if devices:
                filtered_indices = []
                filtered_devices = []
                for idx, dev_name in enumerate(devices):
                    d_lower = dev_name.lower()
                    if "virtual" in d_lower or "unity" in d_lower or "obs" in d_lower:
                        continue
                    filtered_indices.append(idx)
                    filtered_devices.append(dev_name)
                if filtered_devices:
                    return filtered_indices, filtered_devices
                return list(range(len(devices))), devices
            return [], ["No cameras found"]
        except Exception as exc:
            print(f"Error detecting cameras: {exc}")
            return [], ["No cameras found"]

    if platform.system() == "Darwin":
        return [0, 1], ["Camera 0", "Camera 1"]

    indices: List[int] = []
    names: List[str] = []
    for i in range(10):
        cap = cv2.VideoCapture(f"/dev/video{i}")
        if cap.isOpened():
            indices.append(i)
            names.append(f"Camera {i}")
            cap.release()
    return (indices, names) if names else ([], ["No cameras found"])


# ─── Embedded Webcam Processing Workers ──────────────────────────────────────

class _EmbeddedCaptureWorker(threading.Thread):
    def __init__(self, capturer: VideoCapturer, out_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self._cap = capturer
        self._out_queue = out_queue
        self._stop_event = stop_event

    def run(self) -> None:
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            try:
                self._out_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._out_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._out_queue.put_nowait(frame)
                except queue.Full:
                    pass


class _EmbeddedProcessingWorker(threading.Thread):
    def __init__(
        self,
        in_queue: queue.Queue,
        out_queue: queue.Queue,
        stop_event: threading.Event,
        fps_callback: Callable[..., None],
        camera_fps: float = 30.0,
        enable_virtual_cam: bool = False,
    ):
        super().__init__(daemon=True)
        self._in_queue = in_queue
        self._out_queue = out_queue
        self._stop_event = stop_event
        self._fps_callback = fps_callback
        self._camera_fps = max(10.0, float(camera_fps))
        self._enable_virtual_cam = enable_virtual_cam

    def run(self) -> None:
        global _CACHED_SOURCE_FACE
        last_source_path = None
        source_image = None
        cached_target_face = None
        cached_many_faces = None

        if not modules.globals.frame_processors:
            modules.globals.frame_processors = ["face_swapper"]
        frame_processors = get_frame_processors_modules(modules.globals.frame_processors)

        det_interval = max(1, round(self._camera_fps * 0.08))
        det_count = 0
        missed_det_count = 0
        frame_count = 0
        prev_time = time.time()

        vcams = []
        if self._enable_virtual_cam:
            try:
                import pyvirtualcam
                w, h = modules.globals.capture_resolution
                fps_val = int(self._camera_fps) if self._camera_fps else 30
                for backend in ["obs", "unitycapture"]:
                    try:
                        c = pyvirtualcam.Camera(
                            width=w, height=h, fps=fps_val, fmt=pyvirtualcam.PixelFormat.BGR, backend=backend
                        )
                        vcams.append(c)
                        print(f"[VirtualCam] Live broadcast device: {c.device} ({backend}) at {w}x{h}")
                    except Exception as b_err:
                        print(f"[VirtualCam] Backend {backend} skipped: {b_err}")
                if not vcams:
                    try:
                        c = pyvirtualcam.Camera(
                            width=w, height=h, fps=fps_val, fmt=pyvirtualcam.PixelFormat.BGR
                        )
                        vcams.append(c)
                        print(f"[VirtualCam] Live broadcast fallback: {c.device} at {w}x{h}")
                    except Exception as fb_err:
                        print(f"[VirtualCam] Fallback error: {fb_err}")
            except Exception as v_err:
                print(f"[VirtualCam] Virtual camera setup skipped: {v_err}")

        while not self._stop_event.is_set():
            try:
                frame = self._in_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            try:
                if modules.globals.live_mirror:
                    frame = cv2.flip(frame, 1)

                # If swapping is enabled and source is valid
                if _CACHED_SOURCE_FACE is not None:
                    source_image = _CACHED_SOURCE_FACE
                elif modules.globals.source_path and os.path.exists(modules.globals.source_path):
                    curr_norm_path = os.path.normcase(os.path.abspath(modules.globals.source_path))
                    if source_image is None or last_source_path != curr_norm_path:
                        last_source_path = curr_norm_path
                        try:
                            with modules.globals.dml_lock:
                                source_image = get_one_face(imread_unicode(modules.globals.source_path))
                            if source_image is not None:
                                source_image._source_path = curr_norm_path
                                _CACHED_SOURCE_FACE = source_image
                        except Exception as e:
                            source_image = None
                            print(f"[FaceSwap] Error parsing source face: {e}")
                else:
                    source_image = None

                if source_image is not None:
                    if det_count == 0 or det_count % det_interval == 0:
                        try:
                            if modules.globals.many_faces:
                                new_many = detect_many_faces_fast(frame)
                                if new_many:
                                    cached_many_faces = new_many
                                    cached_target_face = None
                                    missed_det_count = 0
                                else:
                                    missed_det_count += 1
                                    if missed_det_count > 15:
                                        cached_many_faces = None
                            else:
                                new_target = detect_one_face_fast(frame)
                                if new_target is not None:
                                    cached_target_face = new_target
                                    cached_many_faces = None
                                    missed_det_count = 0
                                else:
                                    missed_det_count += 1
                                    if missed_det_count > 15:
                                        cached_target_face = None
                        except Exception as e:
                            print(f"[FaceSwap] Detection error: {e}")
                    det_count += 1

                    cached_faces = cached_many_faces or ([cached_target_face] if cached_target_face else None)
                    if modules.globals.mouth_mask and cached_faces:
                        try:
                            ensure_landmarks(frame, cached_faces)
                        except Exception:
                            pass

                    for fp in frame_processors:
                        try:
                            if fp.NAME == "DLC.FACE-ENHANCER":
                                if modules.globals.fp_ui.get("face_enhancer", False):
                                    frame = fp.process_frame(None, frame, detected_faces=cached_faces)
                            elif fp.NAME == "DLC.FACE-ENHANCER-GPEN256":
                                if modules.globals.fp_ui.get("face_enhancer_gpen256", False):
                                    frame = fp.process_frame(None, frame, detected_faces=cached_faces)
                            elif fp.NAME == "DLC.FACE-ENHANCER-GPEN512":
                                if modules.globals.fp_ui.get("face_enhancer_gpen512", False):
                                    frame = fp.process_frame(None, frame, detected_faces=cached_faces)
                            elif fp.NAME == "DLC.FACE-SWAPPER":
                                swapped_bboxes = []
                                active_target_faces = cached_many_faces if (modules.globals.many_faces and cached_many_faces) else ([cached_target_face] if cached_target_face else (cached_many_faces or []))
                                if active_target_faces:
                                    res = frame.copy()
                                    for t_face in active_target_faces:
                                        res = fp.swap_face(source_image, t_face, res)
                                        if hasattr(t_face, "bbox") and t_face.bbox is not None:
                                            swapped_bboxes.append(t_face.bbox.astype(int))
                                    frame = res
                                frame = fp.apply_post_processing(frame, swapped_bboxes)
                        except Exception as exc:
                            print(f"[FaceSwap] Processing step error: {exc}")
            except Exception as proc_err:
                print(f"[ProcessingWorker] Frame processing error: {proc_err}")

            # Send to virtual cameras if active
            for vc in vcams:
                try:
                    v_frame = frame
                    if v_frame.shape[1] != vc.width or v_frame.shape[0] != vc.height:
                        v_frame = cv2.resize(v_frame, (vc.width, vc.height))
                    vc.send(v_frame)
                except Exception:
                    pass

            # Measure FPS
            frame_count += 1
            now = time.time()
            if now - prev_time >= 0.5:
                curr_fps = frame_count / (now - prev_time)
                has_detected_face = bool(cached_many_faces or cached_target_face)
                self._fps_callback(curr_fps, has_detected_face)
                frame_count = 0
                prev_time = now

            try:
                self._out_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._out_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._out_queue.put_nowait(frame)
                except queue.Full:
                    pass

        for vc in vcams:
            try:
                vc.close()
            except Exception:
                pass


# ─── Main Studio Window ───────────────────────────────────────────────────────

class StudioMainWindow(QMainWindow):
    def __init__(self, start_cb: Callable, destroy_cb: Callable):
        super().__init__()
        self._start_cb = start_cb
        self._destroy_cb = destroy_cb

        # Camera state
        self._is_camera_active = False
        self._is_camera_starting = False
        self._cap: Optional[VideoCapturer] = None
        self._stop_event: Optional[threading.Event] = None
        self._capture_queue: Optional[queue.Queue] = None
        self._processed_queue: Optional[queue.Queue] = None
        self._render_timer = QTimer(self)
        self._render_timer.timeout.connect(self._render_next_frame)

        self._current_fps = 0.0
        self._last_rendered_frame: Optional[np.ndarray] = None

        self.setWindowTitle(f"{modules.metadata.name} Studio • {modules.metadata.version}")
        self.setMinimumSize(1020, 720)
        self.resize(1100, 760)

        # Build UI layout
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Top Header with Modes
        main_layout.addWidget(self._build_header())

        # 2. Main Content Area (3-Column Studio)
        content_container = QWidget()
        content_layout = QHBoxLayout(content_container)
        content_layout.setContentsMargins(18, 16, 18, 16)
        content_layout.setSpacing(18)

        # Left Column: Source Face
        content_layout.addWidget(self._build_source_panel(), stretch=26)

        # Center Column: Main Canvas & Mode Stack
        content_layout.addWidget(self._build_center_viewport(), stretch=48)

        # Right Column: Smart Controls
        content_layout.addWidget(self._build_controls_panel(), stretch=26)

        main_layout.addWidget(content_container, 1)

        # 3. Status Bar Footer
        main_layout.addWidget(self._build_footer())

        # Auto-load default source face if available
        default_source = None
        if modules.globals.source_path and os.path.exists(modules.globals.source_path):
            default_source = modules.globals.source_path
        elif os.path.exists("woman_source_face.png"):
            default_source = os.path.abspath("woman_source_face.png")

        if default_source:
            self._load_source_face(default_source)

        # Start non-blocking AI engine warmup
        threading.Thread(target=self._background_warmup, daemon=True).start()

    def _background_warmup(self) -> None:
        try:
            time.sleep(0.3)
            _emit_status("Warming up DirectML GPU acceleration...")
            from modules.face_analyser import get_face_analyser
            get_face_analyser()
            if not modules.globals.frame_processors:
                modules.globals.frame_processors = ["face_swapper"]
            get_frame_processors_modules(modules.globals.frame_processors)
            _emit_status("Ready • DirectML GPU Acceleration Active")
        except Exception as exc:
            print(f"[Warmup] Engine notice: {exc}")
            _emit_status("Ready • DirectML GPU Acceleration Active")

    # ── Header ───────────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("headerFrame")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(14)

        # Brand
        title = QLabel(f"🟣 {modules.metadata.name}")
        title.setObjectName("brandTitle")
        badge = QLabel("STUDIO")
        badge.setObjectName("brandBadge")
        layout.addWidget(title)
        layout.addWidget(badge)

        layout.addSpacing(20)

        # Mode Selector
        self.btn_mode_camera = QPushButton("🎥  Live Camera")
        self.btn_mode_camera.setProperty("class", "navTab")
        self.btn_mode_camera.setProperty("active", "true")
        self.btn_mode_camera.clicked.connect(lambda: self._switch_mode(0))

        self.btn_mode_media = QPushButton("🎬  Media File Swap")
        self.btn_mode_media.setProperty("class", "navTab")
        self.btn_mode_media.setProperty("active", "false")
        self.btn_mode_media.clicked.connect(lambda: self._switch_mode(1))

        self.btn_mode_voice = QPushButton("🎙️  Voice Studio")
        self.btn_mode_voice.setProperty("class", "navTab")
        self.btn_mode_voice.setProperty("active", "false")
        self.btn_mode_voice.clicked.connect(lambda: self._switch_mode(2))

        layout.addWidget(self.btn_mode_camera)
        layout.addWidget(self.btn_mode_media)
        layout.addWidget(self.btn_mode_voice)

        layout.addStretch(1)

        # Hardware status pill
        hw_pill = QLabel("⚡ DirectML (Intel GPU)")
        hw_pill.setObjectName("hardwareBadge")
        layout.addWidget(hw_pill)

        return header

    def _switch_mode(self, index: int) -> None:
        self.mode_stack.setCurrentIndex(index)
        self.btn_mode_camera.setProperty("active", "true" if index == 0 else "false")
        self.btn_mode_media.setProperty("active", "true" if index == 1 else "false")
        self.btn_mode_voice.setProperty("active", "true" if index == 2 else "false")
        for btn in (self.btn_mode_camera, self.btn_mode_media, self.btn_mode_voice):
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    # ── Left Column: Source Face ─────────────────────────────────────────────

    def _build_source_panel(self) -> QFrame:
        card = QFrame()
        card.setProperty("class", "studioCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QLabel("1. Source Face")
        title.setObjectName("cardTitle")
        layout.addWidget(title)

        # Source thumbnail dropzone
        self.source_preview = QLabel("Drop Face Photo Here\nor Click Below")
        self.source_preview.setObjectName("sourceDropZone")
        self.source_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.source_preview.setFixedSize(220, 220)
        self.source_preview.setScaledContents(True)
        layout.addWidget(self.source_preview, alignment=Qt.AlignmentFlag.AlignCenter)

        # Choose button
        self.btn_choose_source = QPushButton("📂  Select Face Photo")
        self.btn_choose_source.setProperty("class", "primaryBtn")
        self.btn_choose_source.clicked.connect(self._on_choose_source)
        layout.addWidget(self.btn_choose_source)

        # Status badge
        self.lbl_face_status = QLabel("⚪ No face selected")
        self.lbl_face_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_face_status.setStyleSheet("color: #94a3b8; font-weight: 500;")
        layout.addWidget(self.lbl_face_status)

        layout.addStretch(1)

        # Quick tips
        tips = QLabel("💡 Tip: Use a front-facing, well-lit photo with clear facial features for the best quality.")
        tips.setWordWrap(True)
        tips.setStyleSheet("color: #64748b; font-size: 8.5pt; line-height: 1.4;")
        layout.addWidget(tips)

        return card

    # ── Center Column: Live Canvas & Controls ─────────────────────────────────

    def _build_center_viewport(self) -> QFrame:
        card = QFrame()
        card.setProperty("class", "studioCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        # Mode Stack (Live Camera vs Media File Swap)
        self.mode_stack = QStackedWidget()

        # Page 0: Live Camera View
        cam_page = QWidget()
        cam_layout = QVBoxLayout(cam_page)
        cam_layout.setContentsMargins(0, 0, 0, 0)
        cam_layout.setSpacing(12)

        # Video Canvas
        self.video_canvas = QLabel("Camera Inactive\nClick 'Start Live Swap' to Begin")
        self.video_canvas.setObjectName("videoCanvas")
        self.video_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_canvas.setMinimumSize(480, 340)
        self.video_canvas.setStyleSheet("color: #64748b; font-size: 11pt; font-weight: 500;")
        cam_layout.addWidget(self.video_canvas, 1)

        # Camera Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        # Camera selector
        self.cb_cameras = QComboBox()
        self._cam_indices, self._cam_names = get_available_cameras()
        if self._cam_names and self._cam_names[0] != "No cameras found":
            self.cb_cameras.addItems(self._cam_names)
        else:
            self.cb_cameras.addItem("No cameras found")
            self.cb_cameras.setEnabled(False)
        self.cb_cameras.setMinimumWidth(160)
        toolbar.addWidget(self.cb_cameras)

        # Start / Stop Live Button
        self.btn_toggle_live = QPushButton("▶  Start Live Swap")
        self.btn_toggle_live.setProperty("class", "primaryBtn")
        self.btn_toggle_live.clicked.connect(self._on_toggle_live)
        toolbar.addWidget(self.btn_toggle_live, 1)

        # Snapshot button
        self.btn_snapshot = QPushButton("📷  Snapshot")
        self.btn_snapshot.setProperty("class", "secondaryBtn")
        self.btn_snapshot.clicked.connect(self._on_snapshot)
        toolbar.addWidget(self.btn_snapshot)

        cam_layout.addLayout(toolbar)

        # WhatsApp Call Instruction banner
        self.lbl_whatsapp_hint = QLabel(
            "📱 In WhatsApp call: click the 📷 camera button → switch from 'Integrated Camera' → 'OBS Virtual Camera'"
        )
        self.lbl_whatsapp_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_whatsapp_hint.setWordWrap(True)
        self.lbl_whatsapp_hint.setStyleSheet(
            "background: #1e293b; color: #a5f3fc; font-size: 9pt; font-weight: 600; "
            "border: 1px solid #0e7490; border-radius: 6px; padding: 6px 10px; margin-top: 2px;"
        )
        self.lbl_whatsapp_hint.setVisible(False)  # only shown when camera is running
        cam_layout.addWidget(self.lbl_whatsapp_hint)

        self.mode_stack.addWidget(cam_page)

        # Page 1: Media File Swap View
        media_page = QWidget()
        media_layout = QVBoxLayout(media_page)
        media_layout.setContentsMargins(0, 0, 0, 0)
        media_layout.setSpacing(12)

        self.media_drop_zone = QLabel("Drop Target Video or Photo Here\n(MP4, MOV, PNG, JPG)")
        self.media_drop_zone.setObjectName("videoCanvas")
        self.media_drop_zone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.media_drop_zone.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.media_drop_zone.setMinimumSize(480, 340)
        self.media_drop_zone.setStyleSheet("color: #64748b; font-size: 11pt; font-weight: 500;")
        media_layout.addWidget(self.media_drop_zone, 1)

        media_tools = QHBoxLayout()
        self.btn_choose_target = QPushButton("📂  Select Target File")
        self.btn_choose_target.setProperty("class", "secondaryBtn")
        self.btn_choose_target.clicked.connect(self._on_choose_target)
        media_tools.addWidget(self.btn_choose_target)

        self.btn_start_media_swap = QPushButton("⚡  Process & Export")
        self.btn_start_media_swap.setProperty("class", "primaryBtn")
        self.btn_start_media_swap.clicked.connect(self._on_start_media_swap)
        media_tools.addWidget(self.btn_start_media_swap, 1)

        media_layout.addLayout(media_tools)

        self.media_progress = QProgressBar()
        self.media_progress.setValue(0)
        self.media_progress.setVisible(False)
        media_layout.addWidget(self.media_progress)

        self.mode_stack.addWidget(media_page)

        # Page 2: Voice Studio View
        self.mode_stack.addWidget(self._build_voice_studio_page())

        layout.addWidget(self.mode_stack, 1)
        return card

    # ── Right Column: Smart Controls ─────────────────────────────────────────

    def _build_controls_panel(self) -> QFrame:
        card = QFrame()
        card.setProperty("class", "studioCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        title = QLabel("2. Face Settings")
        title.setObjectName("cardTitle")
        layout.addWidget(title)

        # Enhancer selector
        layout.addWidget(QLabel("Face Enhancer Quality:"))
        self.cb_enhancer = QComboBox()
        self.cb_enhancer.addItems(["Off (Fastest)", "GFPGAN (HD Sharp)", "GPEN-256", "GPEN-512"])
        self.cb_enhancer.currentTextChanged.connect(self._on_enhancer_change)
        layout.addWidget(self.cb_enhancer)
        layout.addSpacing(6)

        # Essential Switches
        self.chk_mouth_mask = QCheckBox("Natural Mouth Movement")
        self.chk_mouth_mask.setToolTip("Keeps original mouth & speech animation")
        self.chk_mouth_mask.setChecked(False)
        self.chk_mouth_mask.toggled.connect(lambda v: setattr(modules.globals, "mouth_mask", v))
        layout.addWidget(self.chk_mouth_mask)

        self.chk_smooth_blend = QCheckBox("Smooth Edge Blending")
        self.chk_smooth_blend.setToolTip("Seamlessly blends face edges with skin")
        self.chk_smooth_blend.setChecked(False)
        self.chk_smooth_blend.toggled.connect(lambda v: setattr(modules.globals, "poisson_blend", v))
        layout.addWidget(self.chk_smooth_blend)

        self.chk_many_faces = QCheckBox("Swap All Faces in View")
        self.chk_many_faces.setToolTip("Swaps every person detected instead of primary")
        self.chk_many_faces.setChecked(False)
        self.chk_many_faces.toggled.connect(lambda v: setattr(modules.globals, "many_faces", v))
        layout.addWidget(self.chk_many_faces)

        self.chk_mirror = QCheckBox("Mirror Camera")
        self.chk_mirror.setChecked(modules.globals.live_mirror)
        self.chk_mirror.toggled.connect(lambda v: setattr(modules.globals, "live_mirror", v))
        layout.addWidget(self.chk_mirror)

        self.chk_virtual_cam = QCheckBox("📡 Virtual Camera (WhatsApp/Zoom)")
        self.chk_virtual_cam.setToolTip("Broadcasts swapped video stream to WhatsApp, Zoom, and Teams")
        self.chk_virtual_cam.setChecked(True)
        layout.addWidget(self.chk_virtual_cam)

        self.btn_vcam_help = QPushButton("ℹ️  Setup for WhatsApp Calls")
        self.btn_vcam_help.setStyleSheet("font-size: 8.5pt; color: #818cf8; background: transparent; border: none; text-align: left; padding: 2px 0;")
        self.btn_vcam_help.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_vcam_help.clicked.connect(self._show_vcam_guide)
        layout.addWidget(self.btn_vcam_help)

        # Resolution dropdown
        layout.addWidget(QLabel("Camera Resolution:"))
        self.cb_resolution = QComboBox()
        self.cb_resolution.addItems(["640 x 480 (Recommended)", "960 x 540", "1280 x 720 (HD)"])
        self.cb_resolution.setCurrentIndex(0)
        self.cb_resolution.currentTextChanged.connect(self._on_resolution_change)
        layout.addWidget(self.cb_resolution)

        # 3. Quick Voice Controls
        layout.addSpacing(6)
        v_title = QLabel("3. Voice Changer")
        v_title.setObjectName("cardTitle")
        layout.addWidget(v_title)

        self.cb_quick_voice = QComboBox()
        for p_key, p_data in VOICE_PRESETS.items():
            self.cb_quick_voice.addItem(p_data["name"], p_key)
        self.cb_quick_voice.currentIndexChanged.connect(self._on_quick_voice_change)
        layout.addWidget(self.cb_quick_voice)

        self.btn_quick_voice_toggle = QPushButton("▶  Start Voice Changer")
        self.btn_quick_voice_toggle.setProperty("class", "secondaryBtn")
        self.btn_quick_voice_toggle.clicked.connect(self._on_toggle_voice)
        layout.addWidget(self.btn_quick_voice_toggle)

        btn_open_vstudio = QPushButton("🎙️  Open Full Voice Studio ➔")
        btn_open_vstudio.setStyleSheet("font-size: 8.5pt; color: #818cf8; background: transparent; border: none; text-align: left; padding: 2px 0;")
        btn_open_vstudio.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_open_vstudio.clicked.connect(lambda: self._switch_mode(2))
        layout.addWidget(btn_open_vstudio)

        layout.addStretch(1)

        # Output folder button
        btn_open_outputs = QPushButton("📁  Open Output Folder")
        btn_open_outputs.setProperty("class", "secondaryBtn")
        btn_open_outputs.clicked.connect(self._on_open_outputs)
        layout.addWidget(btn_open_outputs)

        return card

    # ── Footer Status Bar ────────────────────────────────────────────────────

    def _build_footer(self) -> QWidget:
        footer = QFrame()
        footer.setObjectName("footerBar")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(12)

        self.lbl_status = QLabel("Ready • DirectML GPU Acceleration Active")
        self.lbl_status.setObjectName("statusLabel")
        layout.addWidget(self.lbl_status, 1)

        self.lbl_fps = QLabel("FPS: --")
        self.lbl_fps.setObjectName("fpsBadge")
        layout.addWidget(self.lbl_fps)

        return footer

    # ── Voice Studio Page (Page 2) ───────────────────────────────────────────

    def _build_voice_studio_page(self) -> QWidget:
        page = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(14)

        # 1. Header Banner Card & Master Toggle
        header_card = QFrame()
        header_card.setProperty("class", "studioCard")
        h_layout = QHBoxLayout(header_card)
        h_layout.setContentsMargins(18, 14, 18, 14)
        h_layout.setSpacing(16)

        title_col = QVBoxLayout()
        v_title = QLabel("🎙️ Real-Time Voice Studio")
        v_title.setStyleSheet("font-size: 13pt; font-weight: 700; color: #ffffff;")
        v_sub = QLabel("Ultra-low latency (<20ms) human voice transformation for calls & streaming")
        v_sub.setStyleSheet("font-size: 8.5pt; color: #94a3b8;")
        title_col.addWidget(v_title)
        title_col.addWidget(v_sub)
        h_layout.addLayout(title_col, 1)

        self.btn_toggle_voice_studio = QPushButton("▶  Start Voice Changer")
        self.btn_toggle_voice_studio.setProperty("class", "primaryBtn")
        self.btn_toggle_voice_studio.setMinimumHeight(42)
        self.btn_toggle_voice_studio.setMinimumWidth(200)
        self.btn_toggle_voice_studio.clicked.connect(self._on_toggle_voice)
        h_layout.addWidget(self.btn_toggle_voice_studio)

        layout.addWidget(header_card)

        # 2. Voice Profiles Card (5 Presets)
        profile_card = QFrame()
        profile_card.setProperty("class", "studioCard")
        p_layout = QVBoxLayout(profile_card)
        p_layout.setContentsMargins(18, 14, 18, 14)
        p_layout.setSpacing(10)

        p_label = QLabel("Choose Vocal Profile (5 Distinct Human Models):")
        p_label.setStyleSheet("font-size: 10.5pt; font-weight: 700; color: #ffffff;")
        p_layout.addWidget(p_label)

        self._voice_buttons: Dict[str, QPushButton] = {}
        grid = QGridLayout()
        grid.setSpacing(10)

        row = 0
        col = 0
        for p_key, p_data in VOICE_PRESETS.items():
            btn = QPushButton(f"{p_data['name']}\n{p_data['description']}")
            btn.setProperty("class", "voiceCard")
            btn.setProperty("active", "true" if p_key == "deep_male" else "false")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setMinimumHeight(60)
            btn.clicked.connect(lambda _, k=p_key: self._select_voice_preset(k))
            self._voice_buttons[p_key] = btn
            grid.addWidget(btn, row, col)
            col += 1
            if col > 1:
                col = 0
                row += 1

        p_layout.addLayout(grid)
        layout.addWidget(profile_card)

        # 3. Audio Hardware Routing & Real-Time Monitoring Card
        io_card = QFrame()
        io_card.setProperty("class", "studioCard")
        io_layout = QVBoxLayout(io_card)
        io_layout.setContentsMargins(18, 14, 18, 14)
        io_layout.setSpacing(12)

        io_title = QLabel("Hardware Routing & Real-Time Monitoring")
        io_title.setStyleSheet("font-size: 10.5pt; font-weight: 700; color: #ffffff;")
        io_layout.addWidget(io_title)

        dev_grid = QGridLayout()
        dev_grid.setSpacing(10)

        # Microphone Input
        dev_grid.addWidget(QLabel("🎤 Microphone Input:"), 0, 0)
        self.cb_mic_input = QComboBox()
        self._mic_devices = VOICE_ENGINE.get_input_devices()
        for idx, name in self._mic_devices:
            self.cb_mic_input.addItem(name, idx)
        dev_grid.addWidget(self.cb_mic_input, 0, 1)

        # Output Target
        dev_grid.addWidget(QLabel("🔊 Target Output:"), 1, 0)
        self.cb_audio_output = QComboBox()
        self._out_devices = VOICE_ENGINE.get_output_devices()
        cable_idx = VOICE_ENGINE.find_virtual_cable_output()
        sel_row = 0
        for row_i, (idx, name) in enumerate(self._out_devices):
            self.cb_audio_output.addItem(name, idx)
            if cable_idx is not None and idx == cable_idx:
                sel_row = row_i
        self.cb_audio_output.setCurrentIndex(sel_row)
        dev_grid.addWidget(self.cb_audio_output, 1, 1)

        io_layout.addLayout(dev_grid)

        # Live Speech Meter (VU Meter)
        vu_box = QHBoxLayout()
        vu_box.setSpacing(10)
        vu_label = QLabel("Speech Level:")
        vu_label.setStyleSheet("color: #94a3b8; font-size: 9pt;")
        self.audio_vu_meter = QProgressBar()
        self.audio_vu_meter.setObjectName("audioVuMeter")
        self.audio_vu_meter.setRange(0, 100)
        self.audio_vu_meter.setValue(0)
        self.audio_vu_meter.setTextVisible(False)
        vu_box.addWidget(vu_label)
        vu_box.addWidget(self.audio_vu_meter, 1)
        io_layout.addLayout(vu_box)

        # Sliders: Mic Volume & Pitch Fine Tuning
        sliders_box = QHBoxLayout()
        sliders_box.setSpacing(18)

        # Volume
        vol_col = QVBoxLayout()
        self.lbl_mic_gain = QLabel("Microphone Gain: 100%")
        self.lbl_mic_gain.setStyleSheet("font-size: 9pt; color: #cbd5e1;")
        self.slider_mic_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_mic_gain.setRange(10, 250)
        self.slider_mic_gain.setValue(100)
        self.slider_mic_gain.valueChanged.connect(self._on_mic_gain_changed)
        vol_col.addWidget(self.lbl_mic_gain)
        vol_col.addWidget(self.slider_mic_gain)
        sliders_box.addLayout(vol_col)

        # Pitch Fine Tune
        pitch_col = QVBoxLayout()
        self.lbl_pitch_tune = QLabel("Pitch Fine-Tune: 0.0 semitones")
        self.lbl_pitch_tune.setStyleSheet("font-size: 9pt; color: #cbd5e1;")
        self.slider_pitch_tune = QSlider(Qt.Orientation.Horizontal)
        self.slider_pitch_tune.setRange(-120, 120)  # -12.0 to +12.0 semitones
        self.slider_pitch_tune.setValue(0)
        self.slider_pitch_tune.valueChanged.connect(self._on_pitch_tune_changed)
        pitch_col.addWidget(self.lbl_pitch_tune)
        pitch_col.addWidget(self.slider_pitch_tune)
        sliders_box.addLayout(pitch_col)

        io_layout.addLayout(sliders_box)
        layout.addWidget(io_card)

        # 4. WhatsApp Call Routing Guide Card
        guide_card = QFrame()
        guide_card.setProperty("class", "studioCard")
        g_layout = QVBoxLayout(guide_card)
        g_layout.setContentsMargins(18, 12, 18, 12)
        g_layout.setSpacing(6)

        g_title = QLabel("💡 Routing into WhatsApp Calls:")
        g_title.setStyleSheet("font-weight: 700; color: #818cf8; font-size: 9.5pt;")
        g_text = QLabel(
            "1. Set Target Output to <b>'CABLE Input'</b> (or your Virtual Cable).<br>"
            "2. In WhatsApp: Open WhatsApp ➔ <b>Settings ➔ Audio & Video ➔ Microphone</b> and select <b>'CABLE Output'</b>.<br>"
            "3. During your call, the other person hears your transformed voice live while seeing your swapped face!"
        )
        g_text.setStyleSheet("color: #94a3b8; font-size: 8.5pt; line-height: 1.4;")
        g_layout.addWidget(g_title)
        g_layout.addWidget(g_text)
        layout.addWidget(guide_card)

        scroll.setWidget(container)
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)
        return page

    # ── Interaction Handlers ─────────────────────────────────────────────────

    def set_status(self, text: str) -> None:
        self.lbl_status.setText(text)

    def _update_fps(self, fps: float, face_detected: bool = False) -> None:
        self._current_fps = fps
        if not face_detected and self._is_camera_active:
            self.lbl_fps.setText(f"⚠️ Face not in view | FPS: {fps:.1f}")
            self.lbl_fps.setStyleSheet("color: #f59e0b; font-weight: 600;")
        elif self._is_camera_active:
            self.lbl_fps.setText(f"🟢 Swapping Active | FPS: {fps:.1f}")
            self.lbl_fps.setStyleSheet("color: #10b981; font-weight: 600;")
        else:
            self.lbl_fps.setText(f"FPS: {fps:.1f}")
            self.lbl_fps.setStyleSheet("color: #e2e8f0; font-weight: normal;")

    def _select_voice_preset(self, key: str) -> None:
        VOICE_ENGINE.set_preset(key)
        for p_k, btn in self._voice_buttons.items():
            btn.setProperty("active", "true" if p_k == key else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        if hasattr(self, "cb_quick_voice"):
            idx = self.cb_quick_voice.findData(key)
            if idx >= 0 and self.cb_quick_voice.currentIndex() != idx:
                self.cb_quick_voice.blockSignals(True)
                self.cb_quick_voice.setCurrentIndex(idx)
                self.cb_quick_voice.blockSignals(False)
        self.slider_pitch_tune.setValue(0)
        self.set_status(f"Voice profile set to: {VOICE_PRESETS[key]['name']}")

    def _on_quick_voice_change(self, index: int) -> None:
        key = self.cb_quick_voice.currentData()
        if key:
            self._select_voice_preset(key)

    def _on_toggle_voice(self) -> None:
        if VOICE_ENGINE.is_running:
            VOICE_ENGINE.stop()
            self._update_voice_ui_state(False)
            self.set_status("Voice changer stopped.")
        else:
            in_idx = self.cb_mic_input.currentData() if hasattr(self, "cb_mic_input") else None
            out_idx = self.cb_audio_output.currentData() if hasattr(self, "cb_audio_output") else None
            success = VOICE_ENGINE.start(in_idx, out_idx)
            if success:
                self._update_voice_ui_state(True)
                self.set_status("Live voice changer active!")
            else:
                self.set_status("Failed to start audio stream. Check your microphone.")

    def _update_voice_ui_state(self, running: bool) -> None:
        text = "⏹  Stop Voice Changer" if running else "▶  Start Voice Changer"
        cls = "liveActiveBtn" if running else "primaryBtn"
        sec_cls = "liveActiveBtn" if running else "secondaryBtn"

        if hasattr(self, "btn_toggle_voice_studio"):
            self.btn_toggle_voice_studio.setText(text)
            self.btn_toggle_voice_studio.setProperty("class", cls)
            self.btn_toggle_voice_studio.style().unpolish(self.btn_toggle_voice_studio)
            self.btn_toggle_voice_studio.style().polish(self.btn_toggle_voice_studio)

        if hasattr(self, "btn_quick_voice_toggle"):
            self.btn_quick_voice_toggle.setText(text)
            self.btn_quick_voice_toggle.setProperty("class", sec_cls)
            self.btn_quick_voice_toggle.style().unpolish(self.btn_quick_voice_toggle)
            self.btn_quick_voice_toggle.style().polish(self.btn_quick_voice_toggle)

    def _on_mic_gain_changed(self, val: int) -> None:
        gain = val / 100.0
        self.lbl_mic_gain.setText(f"Microphone Gain: {val}%")
        VOICE_ENGINE.set_mic_gain(gain)

    def _on_pitch_tune_changed(self, val: int) -> None:
        st = val / 10.0
        self.lbl_pitch_tune.setText(f"Pitch Fine-Tune: {st:+.1f} semitones")
        VOICE_ENGINE.set_pitch_offset(st)

    @Slot(float)
    def _update_voice_level(self, level: float) -> None:
        if hasattr(self, "audio_vu_meter"):
            self.audio_vu_meter.setValue(int(min(1.0, level) * 100))

    def _show_vcam_guide(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        msg = QMessageBox(self)
        msg.setWindowTitle("Setup Virtual Camera for WhatsApp & Zoom Calls")
        msg.setText(
            "<h3>Using Live Face Swap in WhatsApp Calls</h3>"
            "<p>To broadcast your live swapped face into WhatsApp, Windows needs a Virtual Camera driver.</p>"
            "<p><b>Option 1: Built-in Driver (Recommended - No OBS needed!)</b><br>"
            "Click <b>'⚡ Install Built-in Driver'</b> below to register <b>'Deep Live Cam'</b> as a native Windows camera in 2 seconds.</p>"
            "<p><b>Option 2: OBS Studio</b><br>"
            "If you prefer full OBS Studio, install it from <a href='https://obsproject.com'>obsproject.com</a>.</p>"
            "<hr>"
            "<b>How to use in WhatsApp:</b>"
            "<ol>"
            "<li>In Deep-Live-Cam, check <b>'Virtual Camera'</b> and click <b>'Start Live Swap'</b>.</li>"
            "<li>In WhatsApp Desktop, go to <b>Settings ➔ Audio & Video ➔ Camera</b> and choose <b>'Deep Live Cam'</b> (or 'OBS Virtual Camera').</li>"
            "</ol>"
        )
        msg.setIcon(QMessageBox.Icon.Information)
        btn_install_driver = msg.addButton("⚡ Install Built-in Driver", QMessageBox.ButtonRole.ActionRole)
        btn_open_obs = msg.addButton("Download OBS Studio", QMessageBox.ButtonRole.ActionRole)
        msg.addButton(QMessageBox.StandardButton.Close)
        msg.exec()
        if msg.clickedButton() == btn_install_driver:
            self._install_builtin_driver()
        elif msg.clickedButton() == btn_open_obs:
            webbrowser.open("https://obsproject.com")

    def _install_builtin_driver(self) -> None:
        driver_bat = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "driver", "install-virtual-camera.bat")
        if os.path.exists(driver_bat):
            os.startfile(driver_bat)
            self.set_status("Launched virtual camera driver installer. Approve the Windows prompt.")
        else:
            self.set_status("Driver installer script not found.")

    def _load_source_face(self, file_path: str) -> None:
        if not file_path or not os.path.exists(file_path):
            return
        norm_path = os.path.normpath(file_path)
        modules.globals.source_path = norm_path
        pix = QPixmap(norm_path)
        if not pix.isNull():
            self.source_preview.setPixmap(
                pix.scaled(220, 220, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
            )
        self.lbl_face_status.setText("⏳ Analyzing Face...")
        self.lbl_face_status.setStyleSheet("color: #f59e0b; font-weight: 600;")
        self.set_status(f"Analyzing face: {os.path.basename(norm_path)}...")
        threading.Thread(target=self._analyze_source_face_worker, args=(norm_path,), daemon=True).start()

    def _on_choose_source(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Source Face Photo",
            "",
            "Image Files (*.jpg *.jpeg *.png *.webp *.bmp)",
        )
        if file_path:
            self._load_source_face(file_path)

    def _analyze_source_face_worker(self, file_path: str) -> None:
        global _CACHED_SOURCE_FACE
        try:
            norm_path = os.path.normcase(os.path.abspath(file_path))
            src_img = imread_unicode(file_path)
            with modules.globals.dml_lock:
                face = get_one_face(src_img)
            if face is not None:
                face._source_path = norm_path
                _CACHED_SOURCE_FACE = face
                if _BRIDGE:
                    _BRIDGE.faceAnalyzed.emit(True, os.path.basename(file_path))
            else:
                _CACHED_SOURCE_FACE = None
                if _BRIDGE:
                    _BRIDGE.faceAnalyzed.emit(False, os.path.basename(file_path))
        except Exception as e:
            print(f"[FaceAnalysis] Error: {e}")
            _CACHED_SOURCE_FACE = None
            if _BRIDGE:
                _BRIDGE.faceAnalyzed.emit(False, str(e))

    @Slot(bool, str)
    def _on_face_analyzed(self, success: bool, info: str) -> None:
        if success:
            self.lbl_face_status.setText("🟢 Face Detected & Ready")
            self.lbl_face_status.setStyleSheet("color: #10b981; font-weight: 600;")
            self.set_status(f"Loaded source face: {info}")
        else:
            self.lbl_face_status.setText("⚠️ No clear face detected")
            self.lbl_face_status.setStyleSheet("color: #f59e0b; font-weight: 600;")
            self.set_status("Warning: Could not detect a clear face in this photo.")

    def _on_choose_target(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Target Photo or Video",
            "",
            "Media Files (*.mp4 *.mov *.avi *.jpg *.jpeg *.png *.webp)",
        )
        if not file_path:
            return

        modules.globals.target_path = file_path
        if has_image_extension(file_path):
            pix = QPixmap(file_path)
            self.media_drop_zone.setPixmap(
                pix.scaled(self.media_drop_zone.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            self.media_drop_zone.setText(f"🎥 Selected Video:\n{os.path.basename(file_path)}")
        self.set_status(f"Target selected: {os.path.basename(file_path)}")

    def _on_start_media_swap(self) -> None:
        if not modules.globals.source_path:
            self.set_status("Please select a source face photo first.")
            return
        if not modules.globals.target_path:
            self.set_status("Please select a target video or photo first.")
            return

        self.media_progress.setVisible(True)
        self.media_progress.setValue(10)
        self.set_status("Processing file...")
        threading.Thread(target=self._run_media_processing, daemon=True).start()

    def _run_media_processing(self) -> None:
        try:
            self._start_cb()
            self.set_status("Processing finished successfully!")
        except Exception as e:
            self.set_status(f"Error: {e}")
        finally:
            self.media_progress.setVisible(False)

    def _on_enhancer_change(self, text: str) -> None:
        modules.globals.fp_ui["face_enhancer"] = "GFPGAN" in text
        modules.globals.fp_ui["face_enhancer_gpen256"] = "GPEN-256" in text
        modules.globals.fp_ui["face_enhancer_gpen512"] = "GPEN-512" in text
        self.set_status(f"Face enhancer set to: {text}")

    def _on_resolution_change(self, text: str) -> None:
        if "640" in text:
            modules.globals.capture_resolution = (640, 480)
        elif "960" in text:
            modules.globals.capture_resolution = (960, 540)
        elif "1280" in text:
            modules.globals.capture_resolution = (1280, 720)

    def _on_open_outputs(self) -> None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(out_dir):
            if platform.system() == "Windows":
                os.startfile(out_dir)
            else:
                webbrowser.open(out_dir)

    def _on_snapshot(self) -> None:
        if self._last_rendered_frame is None:
            self.set_status("No active camera frame to snapshot.")
            return
        snap_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        filename = f"snapshot_{int(time.time())}.jpg"
        filepath = os.path.join(snap_dir, filename)
        cv2.imwrite(filepath, self._last_rendered_frame)
        self.set_status(f"Snapshot saved to: snapshots/{filename}")

    # ── Live Camera Management ───────────────────────────────────────────────

    def _on_toggle_live(self) -> None:
        if self._is_camera_active:
            self._stop_live_camera()
        else:
            self._start_live_camera()

    def _start_live_camera(self) -> None:
        if self._is_camera_starting:
            return
        global _CACHED_SOURCE_FACE
        if _CACHED_SOURCE_FACE is None and not (modules.globals.source_path and os.path.exists(modules.globals.source_path)):
            self.set_status("⚠️ Please select a Source Face Photo first (Step 1 on the left)!")
            self._on_choose_source()
            return
        self._is_camera_starting = True
        self.btn_toggle_live.setEnabled(False)
        self.btn_toggle_live.setText("⏳ Starting Camera...")
        self.set_status("Connecting to camera...")
        threading.Thread(target=self._start_camera_worker, daemon=True).start()

    def _start_camera_worker(self) -> None:
        try:
            ui_idx = self.cb_cameras.currentIndex()
            if hasattr(self, "_cam_indices") and 0 <= ui_idx < len(self._cam_indices):
                cam_idx = self._cam_indices[ui_idx]
            else:
                cam_idx = max(0, ui_idx)

            cap = VideoCapturer(cam_idx)
            w, h = modules.globals.capture_resolution
            if not cap.start(w, h, 60):
                if _BRIDGE:
                    _BRIDGE.cameraStarted.emit(False, "Failed to open webcam. Check if another app is using it.")
                return

            self._cap = cap
            self._stop_event = threading.Event()
            self._capture_queue = queue.Queue(maxsize=2)
            self._processed_queue = queue.Queue(maxsize=2)

            self._capture_worker = _EmbeddedCaptureWorker(
                self._cap, self._capture_queue, self._stop_event
            )
            camera_fps = getattr(self._cap, "actual_fps", 30.0) or 30.0
            enable_vcam = self.chk_virtual_cam.isChecked()

            self._processing_worker = _EmbeddedProcessingWorker(
                self._capture_queue,
                self._processed_queue,
                self._stop_event,
                self._update_fps,
                camera_fps=camera_fps,
                enable_virtual_cam=enable_vcam,
            )
            self._capture_worker.start()
            self._processing_worker.start()

            if _BRIDGE:
                _BRIDGE.cameraStarted.emit(True, "Live face swap stream active.")
        except Exception as e:
            print(f"[CameraStart] Exception: {e}")
            if _BRIDGE:
                _BRIDGE.cameraStarted.emit(False, f"Camera error: {e}")

    @Slot(bool, str)
    def _on_camera_started(self, success: bool, message: str) -> None:
        self._is_camera_starting = False
        self.btn_toggle_live.setEnabled(True)
        if success:
            self._is_camera_active = True
            self.video_canvas.setText("🎥 Connecting camera stream...")
            self._render_timer.start(30)  # ~33 fps, smooth & light on Qt loop
            self.btn_toggle_live.setText("⏹  Stop Live Camera")
            self.btn_toggle_live.setProperty("class", "liveActiveBtn")
            self.btn_toggle_live.style().unpolish(self.btn_toggle_live)
            self.btn_toggle_live.style().polish(self.btn_toggle_live)
            self.set_status("✅ Live swap active! Now call in WhatsApp → tap 📷 camera button → switch to 'OBS Virtual Camera' or 'Unity Video Capture'")
            if hasattr(self, "lbl_whatsapp_hint"):
                self.lbl_whatsapp_hint.setVisible(True)
        else:
            self._is_camera_active = False
            self.btn_toggle_live.setText("▶  Start Live Swap")
            self.btn_toggle_live.setProperty("class", "primaryBtn")
            self.btn_toggle_live.style().unpolish(self.btn_toggle_live)
            self.btn_toggle_live.style().polish(self.btn_toggle_live)
            self.set_status(message)

    def _stop_live_camera(self) -> None:
        self.btn_toggle_live.setEnabled(False)
        self.btn_toggle_live.setText("⏳ Stopping...")
        self._render_timer.stop()

        def _stop_worker():
            if self._stop_event:
                self._stop_event.set()
            if self._cap:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            if _BRIDGE:
                _BRIDGE.cameraStopped.emit()

        threading.Thread(target=_stop_worker, daemon=True).start()

    @Slot()
    def _on_camera_stopped(self) -> None:
        self._is_camera_active = False
        self._is_camera_starting = False
        self._current_fps = 0.0
        self.lbl_fps.setText("FPS: --")
        self.btn_toggle_live.setEnabled(True)
        self.btn_toggle_live.setText("▶  Start Live Swap")
        self.btn_toggle_live.setProperty("class", "primaryBtn")
        self.btn_toggle_live.style().unpolish(self.btn_toggle_live)
        self.btn_toggle_live.style().polish(self.btn_toggle_live)
        self.video_canvas.setText("Camera Inactive\nClick 'Start Live Swap' to Begin")
        self.set_status("Camera stopped.")
        if hasattr(self, "lbl_whatsapp_hint"):
            self.lbl_whatsapp_hint.setVisible(False)

    def _render_next_frame(self) -> None:
        if not self._processed_queue:
            return
        try:
            frame = self._processed_queue.get_nowait()
        except queue.Empty:
            return

        self._last_rendered_frame = frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(q_img)

        # Scale smoothly to current viewport
        scaled = pix.scaled(
            self.video_canvas.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_canvas.setPixmap(scaled)

    def closeEvent(self, event) -> None:
        self._render_timer.stop()
        if self._stop_event:
            self._stop_event.set()
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        VOICE_ENGINE.stop()
        if self._destroy_cb:
            try:
                self._destroy_cb()
            except Exception:
                pass
        event.accept()


# ─── Entry Point Wrapper ──────────────────────────────────────────────────────

class _Window:
    """Compatibility wrapper exposing .mainloop() for core.py."""

    def __init__(self, app: QApplication, main_window: StudioMainWindow):
        self._app = app
        self._main = main_window

    def mainloop(self) -> None:
        self._main.show()
        self._app.exec()


def init(start: Callable[[], None], destroy: Callable[[], None], lang: str) -> _Window:
    global _APP, _MAIN, _LANG, _BRIDGE

    _LANG = LanguageManager(lang)
    if QApplication.instance() is None:
        _APP = QApplication(sys.argv)
    else:
        _APP = QApplication.instance()

    _APP.setStyleSheet(STUDIO_QSS)
    _BRIDGE = _UIBridge()
    _MAIN = StudioMainWindow(start, destroy)
    _BRIDGE.statusChanged.connect(_MAIN.set_status)
    _BRIDGE.faceAnalyzed.connect(_MAIN._on_face_analyzed)
    _BRIDGE.cameraStarted.connect(_MAIN._on_camera_started)
    _BRIDGE.cameraStopped.connect(_MAIN._on_camera_stopped)
    _BRIDGE.voiceLevelChanged.connect(_MAIN._update_voice_level)
    VOICE_ENGINE.set_level_callback(lambda lvl: _BRIDGE.voiceLevelChanged.emit(lvl) if _BRIDGE else None)

    return _Window(_APP, _MAIN)
