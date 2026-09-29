"""
MJPEG HTTP stream server.

Uses Flask + werkzeug's make_server (threaded, cleanly stoppable).

Routes
------
  GET /                   HTML index — embeds all camera streams
  GET /camera/<N>         MJPEG stream  (multipart/x-mixed-replace)
  GET /camera/<N>/snap    Single JPEG snapshot

Usage
-----
    server = StreamServer(
        get_frame=multi_camera_manager.latest,
        camera_names={0: "Logitech C922", 2: "USB Camera"},
        host="0.0.0.0",
        port=8080,
    )
    server.start(on_ready=lambda: print(server.url))
    # ... later ...
    server.stop()
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

_LOG = logging.getLogger(__name__)

_JPEG_QUALITY = 75
_STREAM_FPS   = 15                      # max frames/sec per MJPEG client
_FRAME_SLEEP  = 1.0 / _STREAM_FPS

# Suppress werkzeug per-request logs; errors still shown.
logging.getLogger("werkzeug").setLevel(logging.ERROR)


def _lan_ip() -> str:
    """Best-effort: return the LAN IP this machine is reachable on."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

_PAGE = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Camera Viewer</title>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0 }}
    body {{ background:#111; color:#ddd; font-family:sans-serif; padding:20px }}
    h1 {{ color:#fff; margin-bottom:24px; font-size:1.4em }}
    .grid {{ display:flex; flex-wrap:wrap; gap:16px }}
    .cam {{ border:1px solid #333; border-radius:8px; padding:14px;
            background:#1a1a1a; min-width:280px; flex:1 }}
    .cam h2 {{ font-size:.9em; color:#999; margin-bottom:10px }}
    .cam img {{ width:100%; display:block; border-radius:4px }}
    .snap {{ font-size:.75em; color:#5af; text-decoration:none; margin-left:8px }}
    .snap:hover {{ text-decoration:underline }}
  </style>
</head>
<body>
  <h1>Camera Viewer — Live Streams</h1>
  <div class="grid">
{cameras}
  </div>
</body>
</html>
"""

_CAM_BLOCK = """\
    <div class="cam">
      <h2>{name}<a class="snap" href="/camera/{idx}/snap">snapshot</a></h2>
      <img src="/camera/{idx}" alt="{name}">
    </div>
"""


# ---------------------------------------------------------------------------

class StreamServer:
    """
    Threaded MJPEG HTTP server.

    Parameters
    ----------
    get_frame : Callable[[int], np.ndarray | None]
        Called per-frame to fetch the latest BGR image for camera *index*.
        Must be thread-safe (called from Flask worker threads).
    camera_names : dict[int, str]
        Maps camera index → display name for the HTML index page.
    host : str
        Bind address.  "0.0.0.0" exposes on all interfaces.
    port : int
        TCP port.
    """

    def __init__(
        self,
        get_frame: Callable[[int], Optional[np.ndarray]],
        camera_names: dict[int, str],
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        self._get_frame   = get_frame
        self._camera_names = dict(camera_names)
        self._host        = host
        self._port        = port
        self._wsgi        = None          # werkzeug BaseWSGIServer
        self._thread: Optional[threading.Thread] = None
        self._running     = False

    # ── public ────────────────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._running

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        """Human-readable URL on the LAN."""
        return f"http://{_lan_ip()}:{self._port}"

    def start(self, on_ready: Optional[Callable[[], None]] = None) -> None:
        """
        Start the server on a daemon thread.
        *on_ready* is called from that thread once the socket is bound.
        """
        if self._running:
            return
        self._thread = threading.Thread(
            target=self._serve, args=(on_ready,), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the server.  Returns immediately; thread exits shortly after."""
        self._running = False
        if self._wsgi is not None:
            self._wsgi.shutdown()
            self._wsgi = None

    # ── internal ──────────────────────────────────────────────────────────────

    def _serve(self, on_ready: Optional[Callable[[], None]]) -> None:
        from flask import Flask, Response
        from werkzeug.serving import make_server

        app = Flask(__name__)

        # Capture local refs so closures don't hold `self` after stop().
        get_frame    = self._get_frame
        camera_names = self._camera_names
        server_ref   = self        # for _running check in generator

        @app.route("/cameras")
        def cameras_json():
            from flask import jsonify
            return jsonify([
                {"index": idx, "name": name}
                for idx, name in sorted(camera_names.items())
            ])

        @app.route("/")
        def index():
            blocks = "".join(
                _CAM_BLOCK.format(name=camera_names[i], idx=i)
                for i in sorted(camera_names)
            )
            return _PAGE.format(cameras=blocks)

        @app.route("/camera/<int:idx>")
        def stream(idx: int):
            if idx not in camera_names:
                return "Camera not found", 404
            return Response(
                server_ref._mjpeg_gen(idx),
                mimetype="multipart/x-mixed-replace;boundary=frame",
            )

        @app.route("/camera/<int:idx>/snap")
        def snapshot(idx: int):
            if idx not in camera_names:
                return "Camera not found", 404
            frame = get_frame(idx)
            if frame is None:
                return "No frame available yet", 503
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
            )
            if not ok:
                return "Encode error", 500
            return Response(buf.tobytes(), mimetype="image/jpeg")

        try:
            self._wsgi = make_server(self._host, self._port, app, threaded=True)
            self._running = True
            _LOG.info("Stream server listening on %s:%d", self._host, self._port)
            if on_ready:
                on_ready()
            self._wsgi.serve_forever()
        except Exception:
            _LOG.exception("Stream server failed")
        finally:
            self._running = False

    def _mjpeg_gen(self, idx: int):
        """Generator that yields MJPEG boundary frames for one camera."""
        while self._running:
            t0 = time.monotonic()

            frame = self._get_frame(idx)
            if frame is not None:
                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
                )
                if ok:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + buf.tobytes()
                        + b"\r\n"
                    )

            elapsed = time.monotonic() - t0
            remainder = _FRAME_SLEEP - elapsed
            if remainder > 0:
                time.sleep(remainder)
