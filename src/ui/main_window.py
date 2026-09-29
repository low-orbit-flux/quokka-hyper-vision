"""
MainWindow – root CTk window, owns the camera manager, processor pipeline,
object detector, sidebar, and video panel.

Frame-poll loop
---------------
Every POLL_MS milliseconds the `_poll` method runs on the GUI thread:
  1. Calls camera_manager.get_frame() (non-blocking).
  2. If a frame arrived, hands it to video_panel.push_frame().
  3. Re-schedules itself with after().

Camera enumeration runs on a background thread to avoid freezing the UI;
results are delivered back to the GUI thread via after(0, callback).

Detection toggle
----------------
Clicking the sidebar button calls _toggle_detection().
  - First press:  starts model load in a background thread (button → "Loading…")
  - Model ready:  inference thread starts (button → "ON")
  - Second press: inference stops (button → "OFF")
Model stays in memory once loaded so subsequent enable/disable is instant.
"""

from __future__ import annotations

import threading
from typing import Optional

import customtkinter as ctk

from ..agent_client import discover, load_config
from ..camera_manager import CameraInfo, CameraManager
from ..detector import MODELS, ObjectDetector
from ..face_recognizer import FaceRecognizer
from ..multi_camera import MultiCameraManager
from ..plate_recognizer import PlateRecognizer
from ..processors import Pipeline
from ..stream_server import StreamServer
from .sidebar import Sidebar
from .video_panel import VideoPanel

POLL_MS = 16  # ~60 fps poll; actual frame rate is limited by the camera


class MainWindow(ctk.CTk):
    def __init__(self, config_path: Optional[str] = None) -> None:
        super().__init__()
        self.title("Camera Viewer")
        self.geometry("1280x720")
        self.minsize(800, 480)

        self._camera = CameraManager()
        self._pipeline = Pipeline()
        self._detector = ObjectDetector()
        self._face_recognizer = FaceRecognizer()
        self._plate_recognizer = PlateRecognizer()
        self._multi_camera = MultiCameraManager()
        self._stream_server: Optional[StreamServer] = None
        self._cameras: list[CameraInfo] = []        # local cameras (from enumeration)
        self._remote_cameras: list[CameraInfo] = [] # remote cameras (from agents/config)
        self._poll_id: Optional[str] = None

        self._build_layout()
        self._start_poll()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Kick off camera detection once the event loop is running.
        self.after(100, self._enumerate_cameras)

        # Connect to agent URLs from config file, if provided.
        if config_path:
            self.after(300, lambda: self._connect_config_agents(config_path))

    # ── layout ───────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._sidebar = Sidebar(
            self,
            on_camera_selected=self._on_camera_selected,
            on_refresh=self._enumerate_cameras,
            on_detection_toggle=self._toggle_detection,
            on_face_toggle=self._toggle_face,
            on_plate_toggle=self._toggle_plates,
            on_stream_toggle=self._toggle_stream,
            on_add_agent=self._add_agent,
        )
        self._sidebar.grid(row=0, column=0, padx=(8, 4), pady=8, sticky="ns")

        self._video = VideoPanel(
            self,
            pipeline=self._pipeline,
            detector=self._detector,
            face_recognizer=self._face_recognizer,
            plate_recognizer=self._plate_recognizer,
        )
        self._video.grid(row=0, column=1, padx=(4, 8), pady=8, sticky="nsew")

    # ── camera enumeration ────────────────────────────────────────────────────

    def _enumerate_cameras(self) -> None:
        self._sidebar.set_status("Detecting cameras…")
        self._cameras = []
        self._refresh_camera_list()

        def _run() -> None:
            cameras = self._camera.enumerate_cameras()
            self.after(0, lambda: self._on_cameras_ready(cameras))

        threading.Thread(target=_run, daemon=True).start()

    def _on_cameras_ready(self, cameras: list[CameraInfo]) -> None:
        self._cameras = cameras
        self._refresh_camera_list()
        if cameras:
            self._sidebar.set_status(f"{len(cameras)} camera(s) detected.")
        else:
            self._sidebar.set_status("No cameras found.")

    def _refresh_camera_list(self) -> None:
        """Rebuild the sidebar camera list from local + remote cameras."""
        self._sidebar.show_cameras(self._cameras + self._remote_cameras)

    # ── camera selection ──────────────────────────────────────────────────────

    def _on_camera_selected(self, cam: CameraInfo) -> None:
        self._video.show_placeholder("Opening camera…")
        self._sidebar.set_status(f"Opening {cam.name}…")
        self._sidebar.highlight(cam.index)
        if cam.url:
            self._camera.start_remote(cam.url)
        else:
            self._camera.start(cam.index)

    # ── agent / remote camera management ─────────────────────────────────────

    def _add_agent(self) -> None:
        """Show URL-input dialog then discover cameras from the entered address."""
        dialog = ctk.CTkInputDialog(
            title="Add Camera Server",
            text=(
                "Enter the server URL:\n"
                "(e.g.  http://192.168.1.50:8080)"
            ),
        )
        url = dialog.get_input()
        if not url:
            return
        url = url.strip()
        if not url:
            return

        self._sidebar.set_status(f"Connecting to {url}…")
        threading.Thread(
            target=self._discover_agent_bg,
            args=(url, None),
            daemon=True,
        ).start()

    def _discover_agent_bg(self, url: str, name_hint: Optional[str]) -> None:
        """Background thread: discover cameras then hand off to GUI thread."""
        cameras = discover(url, name_hint)
        if cameras:
            self.after(0, lambda: self._on_agent_connected(cameras, url))
        else:
            self.after(0, lambda: self._on_agent_connect_failed(url))

    def _on_agent_connected(self, cameras: list[CameraInfo], url: str) -> None:
        self._remote_cameras.extend(cameras)
        self._refresh_camera_list()
        n = len(cameras)
        host = url.split("//")[-1].split("/")[0]
        self._sidebar.set_status(
            f"Connected: {n} camera{'s' if n != 1 else ''} from {host}"
        )

    def _on_agent_connect_failed(self, url: str) -> None:
        self._sidebar.set_status(f"Could not connect to {url}")

    def _connect_config_agents(self, path: str) -> None:
        """Load a config file and connect to each listed URL in the background."""
        try:
            entries = load_config(path)
        except Exception as exc:
            self._sidebar.set_status(f"Config error: {exc}")
            return

        if not entries:
            return

        self._sidebar.set_status(f"Connecting to {len(entries)} server(s) from config…")

        def _run() -> None:
            for url, name_hint in entries:
                cameras = discover(url, name_hint)
                if cameras:
                    self.after(0, lambda c=cameras, u=url: self._on_agent_connected(c, u))

        threading.Thread(target=_run, daemon=True).start()

    # ── detection toggle ──────────────────────────────────────────────────────

    def _toggle_detection(self) -> None:
        if self._detector.loading:
            return  # ignore clicks while model is loading

        if self._detector.enabled:
            self._detector.disable()
            self._sidebar.set_detection_state("off")
        else:
            model_name = MODELS[self._sidebar.get_model_key()]
            self._sidebar.set_detection_state("loading")
            self._detector.enable(
                model_name=model_name,
                on_ready=lambda: self.after(0, self._on_detector_ready),
            )

    def _on_detector_ready(self) -> None:
        """Called on the GUI thread once the model is loaded and running."""
        self._sidebar.set_detection_state("on")

    # ── face recognition toggle ───────────────────────────────────────────────

    def _toggle_face(self) -> None:
        if self._face_recognizer.loading:
            return

        if self._face_recognizer.enabled:
            self._face_recognizer.disable()
            self._sidebar.set_face_state("off")
        else:
            self._sidebar.set_face_state("loading")
            self._face_recognizer.enable(
                on_ready=lambda: self.after(0, self._on_face_ready)
            )

    def _on_face_ready(self) -> None:
        """Called on the GUI thread once the face recognizer is ready."""
        self._sidebar.set_face_state("on")

    # ── plate recognition toggle ──────────────────────────────────────────────

    def _toggle_plates(self) -> None:
        if self._plate_recognizer.loading:
            return

        if self._plate_recognizer.enabled:
            self._plate_recognizer.disable()
            self._sidebar.set_plate_state("off")
        else:
            self._sidebar.set_plate_state("loading")
            self._plate_recognizer.enable(
                on_ready=lambda: self.after(0, self._on_plates_ready)
            )

    def _on_plates_ready(self) -> None:
        """Called on the GUI thread once the plate recognizer is ready."""
        self._sidebar.set_plate_state("on")

    # ── stream server toggle ──────────────────────────────────────────────────

    def _toggle_stream(self) -> None:
        if self._stream_server is not None and self._stream_server.running:
            self._stream_server.stop()
            self._multi_camera.stop_all()
            self._stream_server = None
            self._sidebar.set_stream_state("off")
        else:
            if not self._cameras:
                self._sidebar.set_status("No cameras detected — refresh first.")
                return
            self._sidebar.set_stream_state("starting")
            self._multi_camera.start_all(self._cameras)
            camera_names = {cam.index: cam.name for cam in self._cameras}
            self._stream_server = StreamServer(
                get_frame=self._multi_camera.latest,
                camera_names=camera_names,
            )
            self._stream_server.start(
                on_ready=lambda: self.after(0, self._on_stream_ready)
            )

    def _on_stream_ready(self) -> None:
        """Called on the GUI thread once the stream server is listening."""
        self._sidebar.set_stream_state("on")
        if self._stream_server is not None:
            self._sidebar.set_stream_url(self._stream_server.url)

    # ── frame poll loop ───────────────────────────────────────────────────────

    def _start_poll(self) -> None:
        self._poll()

    def _poll(self) -> None:
        frame = self._camera.get_frame()
        if frame is not None:
            self._video.push_frame(frame)
            # Update status label once on first frame after camera open.
            if self._sidebar._status.cget("text").startswith("Opening"):
                idx = self._camera.current_index
                name = f"Camera {idx}" if idx is not None else "camera"
                self._sidebar.set_status(f"Streaming: {name}")

        self._poll_id = self.after(POLL_MS, self._poll)

    # ── shutdown ──────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
        self._detector.disable()
        self._face_recognizer.disable()
        self._plate_recognizer.disable()
        if self._stream_server is not None:
            self._stream_server.stop()
        self._multi_camera.stop_all()
        self._camera.stop()
        self.destroy()
