"""
Multi-camera manager: opens N cameras simultaneously, each on its own
capture thread.  The latest frame per camera is kept in a lock-guarded
attribute so it can be read from any thread at any time.

Usage
-----
    mgr = MultiCameraManager()
    mgr.start_all(cameras)          # list[CameraInfo]
    frame = mgr.latest(index)       # np.ndarray | None
    mgr.stop_all()
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np

from .camera_manager import CameraInfo, _cv2_backend, _silent


class _CameraCapture:
    """One camera: one daemon thread, one lock-guarded latest frame."""

    def __init__(self, index: int) -> None:
        self.index = index
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None

    def latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest

    # ── capture loop ──────────────────────────────────────────────────────────

    def _loop(self) -> None:
        with _silent():
            cap = cv2.VideoCapture(self.index, _cv2_backend())
        if not cap.isOpened():
            return
        try:
            while not self._stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                with self._lock:
                    self._latest = frame
        finally:
            cap.release()


class MultiCameraManager:
    """
    Opens and streams multiple cameras simultaneously.
    Thread-safe: latest() may be called from any thread.
    """

    def __init__(self) -> None:
        self._captures: dict[int, _CameraCapture] = {}

    def start_all(self, cameras: list[CameraInfo]) -> None:
        """Open all *cameras*. Stops any previously running captures first."""
        self.stop_all()
        for cam in cameras:
            cap = _CameraCapture(cam.index)
            cap.start()
            self._captures[cam.index] = cap

    def stop_all(self) -> None:
        for cap in self._captures.values():
            cap.stop()
        self._captures.clear()

    @property
    def indices(self) -> list[int]:
        return sorted(self._captures.keys())

    def latest(self, index: int) -> Optional[np.ndarray]:
        """Return the most recent frame for *index*, or None if not ready."""
        cap = self._captures.get(index)
        return cap.latest() if cap is not None else None
