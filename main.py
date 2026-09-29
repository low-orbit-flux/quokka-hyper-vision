#!/usr/bin/env python3
"""
Camera Viewer — entry point.

Modes
-----
GUI (default):
    python main.py
    python main.py --config cameras.json   # connect to remote servers at startup

Headless MJPEG stream server:
    python main.py --server
    python main.py --server --port 9090
    python main.py --server --port 9090 --host 127.0.0.1

In server mode the application enumerates all attached cameras, opens each
one on a background thread, and serves MJPEG streams via HTTP — no display
required.  Press Ctrl-C to stop.

Config file format (JSON):
    [
        "http://192.168.1.50:8080",
        {"url": "http://192.168.1.51:8080", "name": "Office Server"}
    ]

Config file format (plain text, one URL per line):
    # Living room
    http://192.168.1.50:8080
    http://192.168.1.51:8080
"""

from __future__ import annotations

import argparse
import sys


def _run_gui(config_path: str | None = None) -> None:
    import customtkinter as ctk
    from src.ui.main_window import MainWindow

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = MainWindow(config_path=config_path)
    app.mainloop()


def _run_server(host: str, port: int) -> None:
    import signal
    import threading

    from src.camera_manager import CameraManager
    from src.multi_camera import MultiCameraManager
    from src.stream_server import StreamServer

    print("Enumerating cameras…", flush=True)
    cam_mgr = CameraManager()
    cameras = cam_mgr.enumerate_cameras()

    if not cameras:
        print("No cameras found. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(cameras)} camera(s):", flush=True)
    for cam in cameras:
        print(f"  [{cam.index}] {cam.name}", flush=True)

    multi = MultiCameraManager()
    multi.start_all(cameras)

    camera_names = {cam.index: cam.name for cam in cameras}
    server = StreamServer(
        get_frame=multi.latest,
        camera_names=camera_names,
        host=host,
        port=port,
    )

    stop_event = threading.Event()

    def _on_ready() -> None:
        print(f"\nStreaming at  {server.url}", flush=True)
        print(f"Index page:   {server.url}/", flush=True)
        print("Press Ctrl-C to stop.\n", flush=True)

    def _handle_signal(sig, frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    server.start(on_ready=_on_ready)
    stop_event.wait()

    print("Stopping…", flush=True)
    server.stop()
    multi.stop_all()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Camera Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run as a headless MJPEG stream server (no GUI)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        metavar="ADDR",
        help="Bind address for the HTTP server (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        metavar="PORT",
        help="TCP port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        help=(
            "Config file listing remote camera-server URLs to connect at startup. "
            "Accepts JSON array or plain text (one URL per line). "
            "Only applies in GUI mode."
        ),
    )
    args = parser.parse_args()

    if args.server:
        _run_server(host=args.host, port=args.port)
    else:
        _run_gui(config_path=args.config)


if __name__ == "__main__":
    main()
