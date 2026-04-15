#!/usr/bin/env python3
"""
Camera Viewer – entry point.

Run with:
    python main.py

Or on systems where python3 is the command:
    python3 main.py
"""

import customtkinter as ctk
from src.ui.main_window import MainWindow


def main() -> None:
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
