"""
Face detection and recognition via the face_recognition library (dlib).

Architecture
------------
Same non-blocking pattern as detector.py:

  VideoPanel.push_frame()
      └─ recognizer.submit(frame)   # drops frame if busy
              ↓
      _recognize_thread             # runs face detection + recognition
              ↓
      _matches (lock-guarded)       # latest results

Known faces
-----------
Place portrait images in the faces/ directory at the project root.
Name each file after the person: "Alice.jpg", "Bob.png", etc.
The recognizer loads these on enable(); re-enable to reload.

If no known faces are found, detection-only mode is used (all faces
labelled "Unknown").

Performance
-----------
Frames are scaled to 1/4 resolution before face detection to keep
inference fast enough for real-time preview on CPU.  face_recognition
uses dlib's HOG face-detection model — significantly faster than the
CNN model and sufficient for preview use.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

_FACE_SCALE = 0.25                  # process at 1/4 resolution for speed
_TOLERANCE = 0.55                   # face comparison threshold (lower = stricter)
_BOX_THICKNESS = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.55
_FONT_THICKNESS = 1

# BGR colors
_UNKNOWN_COLOR = (80, 80, 210)      # muted red for unrecognised faces
_KNOWN_COLORS = [                   # cycle through for known individuals
    (200, 160,  50),
    ( 50, 200, 100),
    (200,  50, 150),
    ( 50, 150, 200),
    (100, 200,  50),
    (180,  80, 200),
    ( 80, 200, 180),
]

_FACES_DIR = Path(__file__).parent.parent / "faces"

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------

class FaceMatch:
    __slots__ = ("x1", "y1", "x2", "y2", "name", "color")

    def __init__(
        self,
        x1: int, y1: int, x2: int, y2: int,
        name: str,
        known: bool,
        person_idx: int = 0,
    ) -> None:
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.name = name
        self.color = (
            _KNOWN_COLORS[person_idx % len(_KNOWN_COLORS)] if known
            else _UNKNOWN_COLOR
        )


# ---------------------------------------------------------------------------

class FaceRecognizer:
    """
    Wraps face_recognition (dlib).  Thread-safe; safe to call from GUI thread.

    Lifecycle
    ---------
    enable(on_ready)  →  loads known faces + starts recognize thread
    disable()         →  stops thread, clears results
    submit(frame)     →  feeds a frame (non-blocking, drops if busy)
    latest()          →  returns the most recent list[FaceMatch]
    draw(frame, matches) →  static helper — draws boxes onto BGR frame in-place
    """

    def __init__(self) -> None:
        self._lib = None            # face_recognition module, loaded lazily
        self._enabled: bool = False
        self._loading: bool = False

        self._known_encodings: list = []
        self._known_names: list[str] = []
        self._person_index: dict[str, int] = {}

        self._matches: list[FaceMatch] = []
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
        Turn recognition on.  Model and known faces are loaded in a background
        thread; *on_ready* is called on that thread when inference is ready.
        """
        if self._enabled:
            return
        self._loading = True
        threading.Thread(
            target=self._load_and_start,
            args=(on_ready,),
            daemon=True,
        ).start()

    def disable(self) -> None:
        """Stop inference and clear the overlay."""
        self._enabled = False
        self._stop.set()
        with self._lock:
            self._matches = []
        while not self._input_q.empty():
            try:
                self._input_q.get_nowait()
            except queue.Empty:
                break

    def submit(self, frame: np.ndarray) -> None:
        """Feed *frame* to the recognizer (non-blocking; drops if busy)."""
        if not self._enabled:
            return
        try:
            self._input_q.put_nowait(frame)
        except queue.Full:
            pass

    def latest(self) -> list[FaceMatch]:
        """Return the most recent face-match results (may be empty)."""
        with self._lock:
            return list(self._matches)

    @staticmethod
    def draw(frame: np.ndarray, matches: list[FaceMatch]) -> np.ndarray:
        """
        Draw face bounding boxes and name labels onto *frame* (in-place).
        Returns the same frame for convenience.
        """
        for m in matches:
            color = m.color
            cv2.rectangle(frame, (m.x1, m.y1), (m.x2, m.y2), color, _BOX_THICKNESS)

            (tw, th), baseline = cv2.getTextSize(
                m.name, _FONT, _FONT_SCALE, _FONT_THICKNESS
            )
            lx = m.x1
            ly = max(m.y1 - th - baseline - 4, 0)
            cv2.rectangle(
                frame,
                (lx, ly),
                (lx + tw + 6, ly + th + baseline + 4),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                m.name,
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
        # Keep GPU hidden — same reason as detector.py (sm_61 / CPU-only).
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

        import face_recognition  # deferred — dlib import is slow (~1 s)
        self._lib = face_recognition

        # ── load known faces ──────────────────────────────────────────────
        self._known_encodings = []
        self._known_names = []
        unique_names: list[str] = []

        if _FACES_DIR.is_dir():
            for img_path in sorted(_FACES_DIR.iterdir()):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                try:
                    img = face_recognition.load_image_file(str(img_path))
                    # Try progressively more aggressive upsampling before giving up.
                    # upsample=1 works for clear frontal photos; upsample=2 catches
                    # smaller or slightly off-angle faces at the cost of ~4× more CPU.
                    locs = None
                    for upsample in (1, 2):
                        locs = face_recognition.face_locations(
                            img, number_of_times_to_upsample=upsample
                        )
                        if locs:
                            break

                    if locs:
                        encs = face_recognition.face_encodings(img, locs)
                    else:
                        encs = []

                    if encs:
                        name = img_path.stem
                        self._known_encodings.append(encs[0])
                        self._known_names.append(name)
                        if name not in unique_names:
                            unique_names.append(name)
                    else:
                        _LOG.warning(
                            "No face found in %s — use a clear, frontal portrait",
                            img_path.name,
                        )
                except Exception as exc:
                    _LOG.warning("Could not load %s: %s", img_path.name, exc)

        self._person_index = {name: i for i, name in enumerate(unique_names)}

        if unique_names:
            _LOG.info("Face recognizer: loaded %d encoding(s) for: %s",
                      len(self._known_encodings), ", ".join(unique_names))
        else:
            _LOG.info("Face recognizer: no known faces — detection-only mode")

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

    def _recognize_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._input_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Downscale for faster HOG detection.
            small = cv2.resize(frame, (0, 0), fx=_FACE_SCALE, fy=_FACE_SCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            locations = self._lib.face_locations(rgb_small, model="hog")

            encodings: list = []
            if locations and self._known_encodings:
                encodings = self._lib.face_encodings(rgb_small, locations)

            inv = 1.0 / _FACE_SCALE
            matches: list[FaceMatch] = []

            for i, (top, right, bottom, left) in enumerate(locations):
                x1 = int(left   * inv)
                y1 = int(top    * inv)
                x2 = int(right  * inv)
                y2 = int(bottom * inv)

                name = "Unknown"
                known = False
                person_idx = 0

                if i < len(encodings) and self._known_encodings:
                    distances = self._lib.face_distance(
                        self._known_encodings, encodings[i]
                    )
                    best = int(np.argmin(distances))
                    if distances[best] <= _TOLERANCE:
                        name = self._known_names[best]
                        known = True
                        person_idx = self._person_index.get(name, 0)

                matches.append(FaceMatch(x1, y1, x2, y2, name, known, person_idx))

            with self._lock:
                self._matches = matches
