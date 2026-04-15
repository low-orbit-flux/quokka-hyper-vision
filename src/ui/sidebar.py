"""
Left sidebar: camera list + status + refresh button + detection toggle.
"""

from __future__ import annotations

from typing import Callable

import customtkinter as ctk

from ..camera_manager import CameraInfo


class Sidebar(ctk.CTkFrame):
    def __init__(
        self,
        master,
        on_camera_selected: Callable[[CameraInfo], None],
        on_refresh: Callable[[], None],
        on_detection_toggle: Callable[[], None] | None = None,
        **kwargs,
    ) -> None:
        kwargs.setdefault("width", 220)
        kwargs.setdefault("corner_radius", 8)
        super().__init__(master, **kwargs)

        self._on_camera_selected = on_camera_selected
        self._on_refresh = on_refresh
        self._on_detection_toggle = on_detection_toggle
        self._buttons: list[tuple[CameraInfo, ctk.CTkButton]] = []
        self._selected_index: int | None = None

        self._build()

    # ── construction ─────────────────────────────────────────────────────────

    def _build(self) -> None:
        self.grid_propagate(False)
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self,
            text="Cameras",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, padx=14, pady=(14, 6), sticky="ew")

        self._scroll = ctk.CTkScrollableFrame(self, label_text="")
        self._scroll.grid(row=1, column=0, padx=8, pady=4, sticky="nsew")
        self._scroll.grid_columnconfigure(0, weight=1)

        self._placeholder = ctk.CTkLabel(
            self._scroll,
            text="Detecting cameras…",
            text_color="gray",
            anchor="w",
        )
        self._placeholder.grid(row=0, column=0, padx=6, pady=6, sticky="ew")

        self._status = ctk.CTkLabel(
            self,
            text="",
            text_color="gray",
            wraplength=196,
            justify="left",
            anchor="w",
            font=ctk.CTkFont(size=12),
        )
        self._status.grid(row=2, column=0, padx=14, pady=(2, 2), sticky="ew")

        ctk.CTkButton(
            self,
            text="Refresh",
            command=self._on_refresh,
            height=32,
        ).grid(row=3, column=0, padx=12, pady=(4, 4), sticky="ew")

        # ── detection section ────────────────────────────────────────────────
        ctk.CTkLabel(
            self,
            text="Detection",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).grid(row=4, column=0, padx=14, pady=(10, 2), sticky="ew")

        self._detect_btn = ctk.CTkButton(
            self,
            text="⬤  Detection: OFF",
            command=self._on_detection_toggle,
            height=32,
            fg_color=("gray75", "gray30"),
            hover_color=("gray65", "gray40"),
            text_color=("gray30", "gray80"),
        )
        self._detect_btn.grid(row=5, column=0, padx=12, pady=(0, 12), sticky="ew")

    # ── public API ────────────────────────────────────────────────────────────

    def set_status(self, msg: str) -> None:
        self._status.configure(text=msg)

    def show_cameras(self, cameras: list[CameraInfo]) -> None:
        """Rebuild the camera button list."""
        # clear old buttons
        for _, btn in self._buttons:
            btn.destroy()
        self._buttons.clear()
        self._selected_index = None

        self._placeholder.grid_forget()

        if not cameras:
            self._placeholder.configure(text="No cameras found.")
            self._placeholder.grid(row=0, column=0, padx=6, pady=6, sticky="ew")
            return

        for row, cam in enumerate(cameras):
            btn = ctk.CTkButton(
                self._scroll,
                text=cam.name,
                anchor="w",
                fg_color="transparent",
                border_width=1,
                hover=True,
                command=lambda c=cam: self._select(c),
            )
            btn.grid(row=row, column=0, padx=4, pady=3, sticky="ew")
            self._buttons.append((cam, btn))

    def highlight(self, index: int) -> None:
        """Mark *index* as the active camera."""
        self._selected_index = index
        for cam, btn in self._buttons:
            if cam.index == index:
                btn.configure(fg_color=("gray75", "gray25"), border_width=0)
            else:
                btn.configure(fg_color="transparent", border_width=1)

    def set_detection_state(self, state: str) -> None:
        """
        Update detection button appearance.
        state: "off" | "loading" | "on"
        """
        if state == "on":
            self._detect_btn.configure(
                text="⬤  Detection: ON",
                fg_color=("#2d6a2d", "#2d8c2d"),
                hover_color=("#255525", "#257025"),
                text_color="white",
            )
        elif state == "loading":
            self._detect_btn.configure(
                text="◌  Loading model…",
                fg_color=("#7a6000", "#a07800"),
                hover_color=("#7a6000", "#a07800"),
                text_color="white",
            )
        else:  # off
            self._detect_btn.configure(
                text="⬤  Detection: OFF",
                fg_color=("gray75", "gray30"),
                hover_color=("gray65", "gray40"),
                text_color=("gray30", "gray80"),
            )

    # ── internal ──────────────────────────────────────────────────────────────

    def _select(self, cam: CameraInfo) -> None:
        self.highlight(cam.index)
        self._on_camera_selected(cam)
