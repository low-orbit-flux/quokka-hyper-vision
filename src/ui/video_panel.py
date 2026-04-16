"""
Right-hand video panel.

Uses a plain tk.Canvas + ImageTk.PhotoImage for video display.
CTkLabel + CTkImage is unreliable for high-frequency image updates because
tkinter can GC the internal PhotoImage between frames; the canvas approach
keeps the reference anchored on the panel instance and updates in-place
with itemconfig(), which is both stable and fast.
"""

from __future__ import annotations

import tkinter as tk
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk

from ..processors import Pipeline

if TYPE_CHECKING:
    from ..detector import ObjectDetector
    from ..face_recognizer import FaceRecognizer
    from ..plate_recognizer import PlateRecognizer

_PLACEHOLDER = "← Select a camera to start streaming"
_PLACEHOLDER_COLOUR = "#6b6b6b"


class VideoPanel(ctk.CTkFrame):
    def __init__(
        self,
        master,
        pipeline: Pipeline,
        detector: "ObjectDetector | None" = None,
        face_recognizer: "FaceRecognizer | None" = None,
        plate_recognizer: "PlateRecognizer | None" = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("corner_radius", 8)
        super().__init__(master, **kwargs)

        self._pipeline = pipeline
        self._detector = detector
        self._face_recognizer = face_recognizer
        self._plate_recognizer = plate_recognizer
        self._panel_w: int = 1
        self._panel_h: int = 1
        self._photo: ImageTk.PhotoImage | None = None   # MUST stay alive
        self._canvas_img_id: int | None = None
        self._showing_video: bool = False

        self._build()
        self.bind("<Configure>", self._on_resize)

    # ── construction ─────────────────────────────────────────────────────────

    def _build(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        bg = self._get_bg()
        self._canvas = tk.Canvas(
            self, bg=bg, highlightthickness=0, bd=0
        )
        self._canvas.grid(row=0, column=0, sticky="nsew")

        # Placeholder text drawn directly on the canvas.
        self._text_id = self._canvas.create_text(
            0, 0,
            text=_PLACEHOLDER,
            fill=_PLACEHOLDER_COLOUR,
            font=("Helvetica", 16),
            anchor="center",
        )
        self._recentre_text()

    # ── public API ────────────────────────────────────────────────────────────

    def show_placeholder(self, msg: str = _PLACEHOLDER) -> None:
        """Switch back to text mode (call when camera not yet selected)."""
        self._showing_video = False
        self._canvas.itemconfigure(self._text_id, text=msg, state="normal")
        if self._canvas_img_id is not None:
            self._canvas.itemconfigure(self._canvas_img_id, state="hidden")
        self._recentre_text()

    def push_frame(self, bgr_frame: np.ndarray) -> None:
        """Process *bgr_frame* and display it. Called from the GUI thread."""
        if self._panel_w < 4 or self._panel_h < 4:
            return

        # Run through CV processor pipeline (no-op by default).
        frame = self._pipeline.run(bgr_frame)

        # Object detection + face recognition: submit, then draw overlays.
        needs_copy = True
        if self._detector is not None and self._detector.enabled:
            self._detector.submit(frame)
            detections = self._detector.latest()
            if detections:
                if needs_copy:
                    frame = frame.copy()
                    needs_copy = False
                self._detector.draw(frame, detections)

        if self._face_recognizer is not None and self._face_recognizer.enabled:
            self._face_recognizer.submit(frame)
            face_matches = self._face_recognizer.latest()
            if face_matches:
                if needs_copy:
                    frame = frame.copy()
                    needs_copy = False
                self._face_recognizer.draw(frame, face_matches)

        if self._plate_recognizer is not None and self._plate_recognizer.enabled:
            self._plate_recognizer.submit(frame)
            plates = self._plate_recognizer.latest()
            if plates:
                if needs_copy:
                    frame = frame.copy()
                    needs_copy = False
                self._plate_recognizer.draw(frame, plates)

        # BGR → RGB, scale to fit panel preserving aspect ratio.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fh, fw = rgb.shape[:2]
        scale = min(self._panel_w / fw, self._panel_h / fh)
        new_w = max(1, int(fw * scale))
        new_h = max(1, int(fh * scale))

        pil_img = Image.fromarray(rgb).resize((new_w, new_h), Image.BILINEAR)

        # Create PhotoImage and pin it to self._photo BEFORE updating the canvas
        # so there is never a moment where the canvas references a dead image.
        new_photo = ImageTk.PhotoImage(image=pil_img)

        cx = self._panel_w // 2
        cy = self._panel_h // 2

        if self._canvas_img_id is None:
            self._canvas_img_id = self._canvas.create_image(
                cx, cy, anchor="center", image=new_photo
            )
        else:
            self._canvas.coords(self._canvas_img_id, cx, cy)
            self._canvas.itemconfigure(self._canvas_img_id, image=new_photo, state="normal")

        self._photo = new_photo   # replace old ref AFTER canvas update

        if not self._showing_video:
            self._canvas.itemconfigure(self._text_id, state="hidden")
            self._showing_video = True

    # ── internal ──────────────────────────────────────────────────────────────

    def _on_resize(self, event) -> None:
        self._panel_w = event.width
        self._panel_h = event.height
        self._recentre_text()

    def _recentre_text(self) -> None:
        self._canvas.coords(self._text_id, self._panel_w // 2, self._panel_h // 2)

    def _get_bg(self) -> str:
        """Match the canvas background to the current CTk appearance."""
        mode = ctk.get_appearance_mode()
        return "#1a1a1a" if mode == "Dark" else "#ebebeb"
