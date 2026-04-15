"""
Object detection via YOLOv8n (ultralytics).

Architecture
------------
Detection is fully async — it never touches the GUI thread:

  VideoPanel.push_frame()
      └─ detector.submit(frame)   # drops frame if detector is busy
              ↓
      _detect_thread              # runs YOLO inference
              ↓
      _detections (lock-guarded)  # latest results

Each call to push_frame() also calls detector.latest() to retrieve whatever
results are available and draws them onto the frame before display.  Because
inference is slower than the camera frame rate the overlay may lag by a frame
or two — that is expected and invisible in practice.

Model
-----
YOLOv8n ("nano") is used by default: ~6 MB, runs on CPU, fast enough for
real-time preview on modern hardware and usable on a Raspberry Pi 4.
The model file is downloaded to ~/.config/Ultralytics/ on first use.

Colors
------
Each COCO class gets a consistent BGR color derived from its index via the
golden-ratio hue method so boxes are always visually distinct.
"""

from __future__ import annotations

import colorsys
import logging
import queue
import threading
from typing import Callable, Optional

import cv2
import numpy as np

# Silence ultralytics' verbose startup / inference prints.
logging.getLogger("ultralytics").setLevel(logging.WARNING)

_MODEL_NAME = "yolov8n.pt"
_CONF_THRESHOLD = 0.40          # ignore detections below this confidence
_BOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICKNESS = 1


def _class_color(cls_id: int) -> tuple[int, int, int]:
    """Return a consistent BGR color for *cls_id* using the golden-ratio hue."""
    hue = (cls_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


# ---------------------------------------------------------------------------

class Detection:
    __slots__ = ("x1", "y1", "x2", "y2", "label", "conf", "color")

    def __init__(
        self,
        x1: int, y1: int, x2: int, y2: int,
        label: str, conf: float, cls_id: int,
    ) -> None:
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.label = label
        self.conf = conf
        self.color = _class_color(cls_id)


# ---------------------------------------------------------------------------

class ObjectDetector:
    """
    Wraps YOLOv8n.  Thread-safe; safe to call from the GUI thread.

    Lifecycle
    ---------
    enable(on_ready)  →  loads model in background, then starts detect thread
    disable()         →  stops detect thread, clears detections
    submit(frame)     →  feeds a frame (non-blocking, drops if busy)
    latest()          →  returns the most recent list[Detection]
    draw(frame, dets) →  static helper — draws boxes onto a BGR frame in-place
    """

    def __init__(self) -> None:
        self._model = None
        self._enabled: bool = False
        self._loading: bool = False

        self._detections: list[Detection] = []
        self._lock = threading.Lock()

        self._input_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── public ────────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def loading(self) -> bool:
        return self._loading

    def enable(self, on_ready: Optional[Callable[[], None]] = None) -> None:
        """
        Turn detection on.  If the model hasn't been loaded yet it is
        downloaded / loaded in a background thread; *on_ready* is called
        (on that same background thread) when inference is ready.
        """
        if self._enabled:
            return
        if self._model is None:
            self._loading = True
            threading.Thread(
                target=self._load_and_start,
                args=(on_ready,),
                daemon=True,
            ).start()
        else:
            self._start_detect_thread()
            if on_ready:
                on_ready()

    def disable(self) -> None:
        """Stop inference and clear the overlay."""
        self._enabled = False
        self._stop.set()
        with self._lock:
            self._detections = []
        # drain input queue so the thread exits promptly
        while not self._input_q.empty():
            try:
                self._input_q.get_nowait()
            except queue.Empty:
                break

    def submit(self, frame: np.ndarray) -> None:
        """Feed *frame* to the detector (non-blocking; drops if busy)."""
        if not self._enabled:
            return
        try:
            self._input_q.put_nowait(frame)
        except queue.Full:
            pass

    def latest(self) -> list[Detection]:
        """Return the most recent detection results (may be empty)."""
        with self._lock:
            return list(self._detections)

    @staticmethod
    def draw(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """
        Draw bounding boxes and labels onto *frame* (modified in-place).
        Returns the same frame for convenience.
        """
        for d in detections:
            color = d.color
            # Box
            cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, _BOX_THICKNESS)

            # Label background
            label_text = f"{d.label} {d.conf:.0%}"
            (tw, th), baseline = cv2.getTextSize(
                label_text, _FONT, _FONT_SCALE, _FONT_THICKNESS
            )
            lx, ly = d.x1, max(d.y1 - th - baseline - 4, 0)
            cv2.rectangle(
                frame,
                (lx, ly),
                (lx + tw + 6, ly + th + baseline + 4),
                color,
                cv2.FILLED,
            )

            # Label text (black for contrast)
            cv2.putText(
                frame,
                label_text,
                (lx + 3, ly + th + 2),
                _FONT,
                _FONT_SCALE,
                (0, 0, 0),
                _FONT_THICKNESS,
                cv2.LINE_AA,
            )
        return frame

    # ── internal ──────────────────────────────────────────────────────────────

    def _load_and_start(self, on_ready: Optional[Callable[[], None]]) -> None:
        # Hide all CUDA devices before importing torch so PyTorch doesn't
        # attempt GPU initialisation at all.  This avoids the "compute
        # capability not supported" warnings and keeps inference portable
        # (works unchanged on CPU-only machines and Raspberry Pi).
        import os
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

        from ultralytics import YOLO  # deferred — keeps startup fast
        self._model = YOLO(_MODEL_NAME)
        self._model.to("cpu")   # explicit: always run on CPU
        self._loading = False
        self._start_detect_thread()
        if on_ready:
            on_ready()

    def _start_detect_thread(self) -> None:
        self._enabled = True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._detect_loop, daemon=True
        )
        self._thread.start()

    def _detect_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._input_q.get(timeout=0.1)
            except queue.Empty:
                continue

            results = self._model.predict(
                frame, verbose=False, conf=_CONF_THRESHOLD, device="cpu"
            )

            detections: list[Detection] = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self._model.names[cls_id]
                    detections.append(
                        Detection(x1, y1, x2, y2, label, conf, cls_id)
                    )

            with self._lock:
                self._detections = detections
