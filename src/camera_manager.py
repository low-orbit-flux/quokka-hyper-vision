"""
Camera enumeration and frame-capture backend.

Design
------
* CameraInfo        – lightweight dataclass describing a detected device
* CameraManager     – opens/closes cameras; runs capture on a daemon thread
                      so the GUI thread is never blocked.

Frame pipeline
--------------
  capture thread  →  _frame_queue (maxsize=2, drops stale frames)  →  get_frame()

Enumeration strategy
--------------------
On Linux, USB webcams typically expose two or more /dev/videoN nodes per
physical device: one for actual video capture and one (or more) for metadata
or controls.  Both nodes appear in sysfs with the same human-readable name,
and both will pass cv2.VideoCapture.isOpened().  The only reliable filter is
to try to read a frame: metadata nodes return ret=False immediately.

All probing is done with CAP_V4L2 on Linux (skips the FFMPEG / obsensor
fallback chains that produce walls of error output for indices that don't
exist).  OpenCV's C-level diagnostics are suppressed at the OS file-descriptor
level during enumeration so they never reach the terminal.
"""

from __future__ import annotations

import contextlib
import glob
import os
import platform
import queue
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cv2_backend() -> int:
    """Return the preferred cv2 capture backend for this platform."""
    return cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY


@contextlib.contextmanager
def _silent():
    """
    Suppress OpenCV's C++ diagnostics by redirecting fd 2 (stderr) to
    /dev/null for the duration of the block.  Python-level stderr is
    unaffected because we operate at the OS file-descriptor level.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def _can_capture(index: int) -> bool:
    """
    Return True only if the device at *index* can produce a real video frame.
    isOpened() is not enough — metadata nodes open but never yield frames.
    """
    with _silent():
        cap = cv2.VideoCapture(index, _cv2_backend())
        if not cap.isOpened():
            cap.release()
            return False
        ret, _ = cap.read()
        cap.release()
    return ret


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class CameraInfo:
    index: int
    name: str
    url: Optional[str] = None   # set for remote (agent) cameras; None for local


# ---------------------------------------------------------------------------
# CameraManager
# ---------------------------------------------------------------------------

class CameraManager:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2)
        self.current_index: Optional[int] = None

    # ── enumeration ──────────────────────────────────────────────────────────

    def enumerate_cameras(self) -> list[CameraInfo]:
        """Return only devices that can actually deliver video frames."""
        if platform.system() == "Linux":
            cameras = self._enumerate_linux()
            if cameras:
                return cameras
        return self._probe_cameras()

    def _enumerate_linux(self) -> list[CameraInfo]:
        """
        Walk sysfs for device names, then validate each node with a real
        frame-read so metadata/control nodes are excluded.
        """
        cameras: list[CameraInfo] = []
        for name_path in sorted(glob.glob("/sys/class/video4linux/video*/name")):
            try:
                idx = int(name_path.split("/video4linux/video")[1].split("/")[0])
            except (IndexError, ValueError):
                continue

            try:
                with open(name_path) as fh:
                    label = fh.read().strip()
            except OSError:
                label = f"Camera {idx}"

            if _can_capture(idx):
                cameras.append(CameraInfo(index=idx, name=label))

        return cameras

    def _probe_cameras(self) -> list[CameraInfo]:
        """Fallback: probe indices 0-9 (Windows / macOS / RPi)."""
        cameras: list[CameraInfo] = []
        for i in range(10):
            if _can_capture(i):
                cameras.append(CameraInfo(index=i, name=f"Camera {i}"))
        return cameras

    # ── streaming ─────────────────────────────────────────────────────────────

    def start(self, index: int) -> None:
        """Open *index* and begin streaming. Stops any previous stream first."""
        self.stop()
        self.current_index = index
        self._stop_event.clear()
        # Discard stale frames from the previous session.
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        self._thread = threading.Thread(
            target=self._capture_loop, args=(index,), daemon=True
        )
        self._thread.start()

    def start_remote(self, url: str) -> None:
        """Open a remote MJPEG stream URL and begin capturing frames."""
        self.stop()
        self.current_index = None
        self._stop_event.clear()
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        self._thread = threading.Thread(
            target=self._capture_remote_loop, args=(url,), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the capture thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        self.current_index = None

    def get_frame(self) -> Optional[np.ndarray]:
        """Non-blocking; returns the latest frame or None."""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    # ── capture thread ────────────────────────────────────────────────────────

    def _capture_loop(self, index: int) -> None:
        with _silent():
            cap = cv2.VideoCapture(index, _cv2_backend())

        if not cap.isOpened():
            return

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                # Drop the oldest frame when the GUI is running behind.
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_queue.put(frame)
        finally:
            cap.release()

    def _capture_remote_loop(self, url: str) -> None:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            return
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_queue.put(frame)
        finally:
            cap.release()
