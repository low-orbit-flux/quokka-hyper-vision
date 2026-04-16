"""
License plate detection and OCR.

Architecture
------------
Two-stage pipeline, no external model downloads required:

  VideoPanel.push_frame()
      └─ recognizer.submit(frame)     # drops frame if busy
              ↓
      _recognize_thread               # detection → crop → OCR
              ↓
      _plates (lock-guarded)          # latest results

Stage 1 — detection (OpenCV morphology, no model download)
    Sobel-X edges highlight vertical strokes (character boundaries).
    A morphological close merges those strokes into plate-shaped blobs.
    Contours are filtered by aspect ratio (1.5–7×) and minimum area.

Stage 2 — OCR (EasyOCR, models downloaded once to ~/.EasyOCR/)
    Each candidate crop is upscaled if narrow, then read with EasyOCR.
    Results are filtered to alphanumeric plate-like strings (4–12 chars).

Both stages run on CPU only.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
from typing import Callable, Optional

import cv2
import numpy as np

_OCR_MIN_W = 120        # upscale crop to at least this width before OCR
_OCR_CONF = 0.35        # discard EasyOCR results below this confidence
_MIN_AREA_FRAC = 0.002  # plate must be ≥0.2 % of frame area
_ASPECT_LO = 1.5        # plate width/height lower bound
_ASPECT_HI = 7.0        # plate width/height upper bound
_MAX_CANDIDATES = 10    # process at most this many regions per frame

# Plate text: 4–12 chars, mostly uppercase letters and digits, one optional
# separator (space or hyphen) — covers most North American and European formats.
_PLATE_RE = re.compile(r'^[A-Z0-9]{2,6}[\s\-]?[A-Z0-9]{2,6}$')

_BOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICKNESS = 1
_PLATE_COLOR = (0, 200, 255)    # orange (BGR)

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

class PlateMatch:
    __slots__ = ("x1", "y1", "x2", "y2", "text")

    def __init__(self, x1: int, y1: int, x2: int, y2: int, text: str) -> None:
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.text = text


# ---------------------------------------------------------------------------

class PlateRecognizer:
    """
    OpenCV morphology plate finder + EasyOCR.  Thread-safe.

    Lifecycle
    ---------
    enable(on_ready)      →  loads EasyOCR, starts daemon thread
    disable()             →  stops thread, clears results
    submit(frame)         →  non-blocking; drops if busy
    latest()              →  returns the most recent list[PlateMatch]
    draw(frame, plates)   →  static helper — draws onto BGR frame in-place
    """

    def __init__(self) -> None:
        self._reader = None         # EasyOCR Reader
        self._enabled: bool = False
        self._loading: bool = False

        self._plates: list[PlateMatch] = []
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
        if self._enabled:
            return
        self._loading = True
        threading.Thread(
            target=self._load_and_start,
            args=(on_ready,),
            daemon=True,
        ).start()

    def disable(self) -> None:
        self._enabled = False
        self._stop.set()
        with self._lock:
            self._plates = []
        while not self._input_q.empty():
            try:
                self._input_q.get_nowait()
            except queue.Empty:
                break

    def submit(self, frame: np.ndarray) -> None:
        if not self._enabled:
            return
        try:
            self._input_q.put_nowait(frame)
        except queue.Full:
            pass

    def latest(self) -> list[PlateMatch]:
        with self._lock:
            return list(self._plates)

    @staticmethod
    def draw(frame: np.ndarray, plates: list[PlateMatch]) -> np.ndarray:
        """Draw plate bounding boxes and OCR text onto *frame* (in-place)."""
        for p in plates:
            cv2.rectangle(
                frame, (p.x1, p.y1), (p.x2, p.y2),
                _PLATE_COLOR, _BOX_THICKNESS,
            )
            (tw, th), baseline = cv2.getTextSize(
                p.text, _FONT, _FONT_SCALE, _FONT_THICKNESS
            )
            lx = p.x1
            ly = max(p.y1 - th - baseline - 4, 0)
            cv2.rectangle(
                frame,
                (lx, ly),
                (lx + tw + 6, ly + th + baseline + 4),
                _PLATE_COLOR,
                cv2.FILLED,
            )
            cv2.putText(
                frame, p.text,
                (lx + 3, ly + th + 2),
                _FONT, _FONT_SCALE,
                (0, 0, 0),
                _FONT_THICKNESS, cv2.LINE_AA,
            )
        return frame

    # ── internal ──────────────────────────────────────────────────────────────

    def _load_and_start(self, on_ready: Optional[Callable[[], None]]) -> None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

        # easyocr uses the legacy pkg_resources API; suppress the
        # DeprecationWarning it emits on import.
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            import easyocr  # deferred — large import

        _LOG.info("Loading EasyOCR (downloads models on first use)…")
        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)

        self._loading = False
        self._start_recognize_thread()
        if on_ready:
            on_ready()

    def _start_recognize_thread(self) -> None:
        self._enabled = True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._recognize_loop, daemon=True
        )
        self._thread.start()

    # ── Stage 1: morphological plate finder ───────────────────────────────────

    @staticmethod
    def _find_candidates(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Return (x1,y1,x2,y2) bounding boxes of plate-like rectangular regions.

        Algorithm
        ---------
        1. Grayscale + CLAHE (contrast normalisation for varying lighting).
        2. Bilateral blur (preserves edges while killing noise).
        3. Sobel-X: strong response at vertical strokes = character edges.
        4. Morphological close with a wide horizontal kernel: merges the
           character-edge columns into a solid plate-shaped blob.
        5. Contour filter: aspect ratio 1.5–7×, area ≥ 0.2 % of frame.
        """
        fh, fw = frame.shape[:2]
        min_area = fw * fh * _MIN_AREA_FRAC

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        blur = cv2.bilateralFilter(gray, 9, 17, 17)

        sobel = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)

        _, thresh = cv2.threshold(
            sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Horizontal kernel stretches character edges into a plate-wide blob.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int, int]] = set()

        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            if len(candidates) >= _MAX_CANDIDATES:
                break
            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0 or w * h < min_area:
                continue
            aspect = w / h
            if not (_ASPECT_LO <= aspect <= _ASPECT_HI):
                continue

            # Small padding to capture the full plate.
            px = max(4, int(w * 0.04))
            py = max(2, int(h * 0.08))
            x1 = max(0, x - px)
            y1 = max(0, y - py)
            x2 = min(fw, x + w + px)
            y2 = min(fh, y + h + py)
            box = (x1, y1, x2, y2)
            if box not in seen:
                seen.add(box)
                candidates.append(box)

        return candidates

    # ── Stage 2: OCR + filter ─────────────────────────────────────────────────

    def _recognize_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._input_q.get(timeout=0.1)
            except queue.Empty:
                continue

            candidates = self._find_candidates(frame)

            plates: list[PlateMatch] = []
            for x1, y1, x2, y2 in candidates:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Upscale narrow crops so OCR has enough pixels per character.
                cw = crop.shape[1]
                if cw < _OCR_MIN_W:
                    scale = _OCR_MIN_W / cw
                    crop = cv2.resize(
                        crop, (0, 0), fx=scale, fy=scale,
                        interpolation=cv2.INTER_CUBIC,
                    )

                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                raw = self._reader.readtext(rgb, detail=1, paragraph=False)

                # Collect text fragments that pass confidence threshold.
                fragments = [
                    text.strip().upper()
                    for (_bbox, text, conf) in raw
                    if conf >= _OCR_CONF and text.strip()
                ]
                combined = "".join(fragments)

                # Keep only results that look like a licence plate.
                if _PLATE_RE.match(combined):
                    plates.append(PlateMatch(x1, y1, x2, y2, combined))

            with self._lock:
                self._plates = plates
