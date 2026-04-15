# camera1 — Project Notes for Claude

## What this project is

A desktop GUI app that lists cameras detected on the system and streams live video from whichever one the user selects. Built to run on Linux, macOS, Windows, and Raspberry Pi. Planned future features: object detection and facial recognition via OpenCV / CV pipeline.

---

## Tech stack

| Concern | Choice | Why |
|---|---|---|
| GUI | **CustomTkinter** (Python) | Works everywhere including RPi (pure tkinter underneath); modern dark theme; no heavy Qt dependency |
| Camera capture | **OpenCV** (`cv2`) | De-facto standard for CV work; same library we'll use for detection/recognition later |
| Frame display | **`tk.Canvas` + `ImageTk.PhotoImage`** | `CTkLabel + CTkImage` GCs the PhotoImage between frames at video rates → `TclError: image "pyimageN" doesn't exist`. Canvas + pinned `self._photo` reference is stable |
| Virtual env | `.venv` at project root | `./run.sh` creates and activates it automatically |

---

## Project layout

```
camera1/
├── run.sh                  # launcher — creates venv if needed, activates, runs main.py
├── main.py                 # entry point
├── requirements.txt
├── src/
│   ├── camera_manager.py   # camera enumeration + threaded capture
│   ├── processors.py       # CV pipeline (PassThrough by default; add detectors here)
│   └── ui/
│       ├── main_window.py  # root CTk window, poll loop, wires sidebar ↔ video panel
│       ├── sidebar.py      # left panel: camera list, highlight, refresh, status
│       └── video_panel.py  # right panel: tk.Canvas video display
└── rust/                   # original Rust/iced prototype (kept for reference)
    ├── Cargo.toml
    ├── Cargo.lock
    └── src/main.rs
```

---

## How to run

```bash
./run.sh
```

That's it. The script:
1. Creates `.venv` if it doesn't exist
2. Activates it
3. `pip install -q -r requirements.txt`
4. `exec python main.py`

---

## Key bugs found and fixed (important for future work)

### 1 — Linux exposes multiple `/dev/videoN` nodes per physical webcam
USB webcams register two (or more) sysfs entries: one for video capture, one for metadata/controls. Both pass `cv2.VideoCapture.isOpened()`. Only the capture node returns frames on `cap.read()`.

**Fix:** `_can_capture(index)` in `camera_manager.py` attempts an actual `cap.read()` and only returns `True` if a frame comes back.

### 2 — OpenCV probing noise fills the terminal
When `_probe_cameras()` iterates indices 0–9, OpenCV tries V4L2, FFMPEG, obsensor, and other backends for every missing device → wall of C++ error output.

**Fix:** `_silent()` context manager redirects fd 2 (stderr) to `/dev/null` at the OS level for the duration of all probe/open calls. Python-level stderr is unaffected.

### 3 — Force `CAP_V4L2` on Linux
Prevents OpenCV from falling through to FFMPEG and obsensor backends on Linux, which eliminates the multi-backend retry spam entirely.

### 4 — `CTkLabel` + `CTkImage` is not safe for video rates
CTkLabel internally calls `self._label.configure(text=...)` after every image configure, which triggers tkinter to re-evaluate the image reference. At 30fps, the `PhotoImage` inside `CTkImage` gets GC'd before tkinter finishes → `TclError: image "pyimageN" doesn't exist`.

**Fix:** `VideoPanel` uses a plain `tk.Canvas`. Frames are pushed via `canvas.itemconfigure(image=new_photo)` and the reference is held on `self._photo` (instance variable, never GC'd).

---

## Adding CV processors

`src/processors.py` defines a `Pipeline` that `VideoPanel` runs every frame before display. To add a detector:

```python
# src/processors.py (or a new file)
class MyDetector:
    def process(self, frame: np.ndarray) -> np.ndarray:
        # draw boxes, annotations, etc. on frame using cv2
        return frame
```

Register it in `MainWindow.__init__`:

```python
self._pipeline.add(MyDetector())
```

The frame the detector returns is what gets displayed. No other changes needed.

---

## Rust prototype

The original implementation used **iced 0.13** + **nokhwa 0.10** (V4L2 backend). It works but was replaced with Python for easier CV integration. Key notes from that build:

- iced 0.12 has a `glyphon 0.5` integer-overflow panic on first render; must use **iced 0.13** which uses the patched `iced_glyphon 0.6` fork
- `iced::widget::image::Handle::from_pixels` was renamed to `from_rgba` in iced 0.13
- `scrollable` content must not use `Length::Fill` on the scroll axis in iced 0.13
- `iced_wgpu` image atlas uses `Rgba8Unorm[Srgb]` texture format — `from_rgba` bytes are `[R, G, B, A]` order (confirmed by reading `iced_graphics-0.12.1/src/image.rs`)
- Camera subscription uses `Subscription::run_with_id(idx, iced::stream::channel(...))` pattern; changing `idx` tears down and restarts the subscription automatically

---

## Environment

- OS: Linux (Arch), kernel 6.18.20-1-lts
- Python: 3.14.3
- Display: Wayland (XWayland at :1 for tkinter)
- Camera tested: Logitech C922 Pro Stream Webcam
