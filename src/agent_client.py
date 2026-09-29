"""
Remote camera-server discovery and config-file loading.

discover(url)
-------------
Given a URL the user typed, returns a list of CameraInfo objects whose
`url` field points to a playable MJPEG stream.

Discovery order
~~~~~~~~~~~~~~~
1. Try GET <url>/cameras  — our own server's JSON inventory endpoint.
   Returns [{"index": N, "name": "..."},...].  Each entry becomes one
   CameraInfo pointing at <url>/camera/<N>.

2. Fallback: probe *url* directly.  A successful HTTP response (any
   content-type) means the server is reachable; we treat the URL itself
   as a single MJPEG source.  OpenCV's VideoCapture handles the actual
   decoding at playback time.

Both steps run with a 5-second timeout so the GUI stays responsive.

load_config(path)
-----------------
Reads a config file and returns a list of (url, name_hint) pairs.

Supported formats
~~~~~~~~~~~~~~~~~
JSON array — items may be plain strings or {"url": ..., "name": ...}:
    [
        "http://192.168.1.50:8080",
        {"url": "http://192.168.1.51:8080", "name": "Office"}
    ]

Plain text — one URL per line, lines starting with # are comments:
    # Living room
    http://192.168.1.50:8080
    http://192.168.1.51:8080
"""

from __future__ import annotations

import itertools
import json
import threading
import urllib.error
import urllib.request
from typing import Optional

from .camera_manager import CameraInfo

# Thread-safe counter that generates unique negative indices for remote cameras.
# Negative values avoid collision with local device indices (0, 1, 2, …).
_index_iter = itertools.count(-1, -1)
_index_lock = threading.Lock()


def _next_index() -> int:
    with _index_lock:
        return next(_index_iter)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

_TIMEOUT = 5  # seconds


def discover(base_url: str, name_hint: Optional[str] = None) -> list[CameraInfo]:
    """
    Connect to *base_url* and return the cameras it exposes.

    Parameters
    ----------
    base_url :
        A server root (``http://host:8080``) or a direct MJPEG stream URL.
    name_hint :
        Optional display name override for the single-stream fallback.

    Returns
    -------
    List of CameraInfo (with `url` set).  Empty list on failure.
    """
    url = base_url.rstrip("/")

    # ── Strategy 1: our server's JSON inventory ───────────────────────────────
    cameras = _try_json_discovery(url)
    if cameras:
        return cameras

    # ── Strategy 2: direct MJPEG stream ──────────────────────────────────────
    if _probe_url(url):
        return [CameraInfo(
            index=_next_index(),
            name=name_hint or url,
            url=url,
        )]

    return []


def _try_json_discovery(base_url: str) -> list[CameraInfo]:
    """Try GET <base_url>/cameras and parse our server's JSON response."""
    try:
        req = urllib.request.Request(
            f"{base_url}/cameras",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
        if not isinstance(data, list):
            return []
        cameras = []
        for item in data:
            cameras.append(CameraInfo(
                index=_next_index(),
                name=item.get("name", f"Camera {item.get('index', '?')}"),
                url=f"{base_url}/camera/{item['index']}",
            ))
        return cameras
    except Exception:
        return []


def _probe_url(url: str) -> bool:
    """
    Return True if *url* accepts an HTTP connection within the timeout.
    Immediately closes the response body — suitable for MJPEG streams that
    never finish sending.
    """
    # Prefer HEAD (avoids consuming any body at all).
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, method=method)
            with urllib.request.urlopen(req, timeout=_TIMEOUT):
                return True
        except urllib.error.HTTPError as exc:
            # An HTTP error (4xx/5xx) still means the server is reachable.
            if exc.code < 500:
                return True
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# Config file loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> list[tuple[str, Optional[str]]]:
    """
    Parse a config file and return ``[(url, name_hint), ...]``.

    *name_hint* is ``None`` unless the config entry explicitly names the stream.
    """
    with open(path, encoding="utf-8") as fh:
        raw = fh.read().strip()

    # Auto-detect JSON vs plain-text.
    if raw.startswith(("[", "{")):
        return _parse_json_config(raw)
    return _parse_text_config(raw)


def _parse_json_config(text: str) -> list[tuple[str, Optional[str]]]:
    data = json.loads(text)
    if isinstance(data, dict):
        # {"servers": [...]} wrapper is also accepted.
        data = data.get("servers", data.get("cameras", []))
    result: list[tuple[str, Optional[str]]] = []
    for item in data:
        if isinstance(item, str):
            result.append((item, None))
        elif isinstance(item, dict) and "url" in item:
            result.append((item["url"], item.get("name")))
    return result


def _parse_text_config(text: str) -> list[tuple[str, Optional[str]]]:
    result: list[tuple[str, Optional[str]]] = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            result.append((line, None))
    return result
