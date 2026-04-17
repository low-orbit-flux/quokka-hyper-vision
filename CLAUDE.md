# camera1 — Project Notes for Claude

## What this project is

A desktop GUI app that lists cameras detected on the system and streams live video from whichever one the user selects. Built to run on Linux, macOS, Windows, and Raspberry Pi. Object detection (YOLO), facial recognition (dlib), and licence plate OCR are all implemented as independent toggleable features.

---

## Tech stack

| Concern | Choice | Why |
|---|---|---|
| GUI | **CustomTkinter** (Python) | Works everywhere including RPi (pure tkinter underneath); modern dark theme; no heavy Qt dependency |
| Camera capture | **OpenCV** (`cv2`) | De-facto standard for CV work; same library used for detection |
| Frame display | **`tk.Canvas` + `ImageTk.PhotoImage`** | `CTkLabel + CTkImage` GCs the PhotoImage between frames at video rates → `TclError: image "pyimageN" doesn't exist`. Canvas + pinned `self._photo` reference is stable |
| Object detection | **ultralytics YOLO** (selectable model) | Easy API; three models selectable from sidebar (see below) |
| Facial recognition | **face_recognition** (dlib) | Simple API; HOG detector fast on CPU; encoding-based identity matching |
| Licence plates | **OpenCV morphology** + **EasyOCR** | Sobel+close finds plate-shaped regions; EasyOCR reads text; no model download beyond EasyOCR itself |
| Inference device | **CPU only** (`device="cpu"`) | GTX 1080 is CUDA sm_61, pip PyTorch only ships sm_75+; CPU is portable and works on RPi |
| Virtual env | `.venv` at project root | `./run.sh` creates and activates it automatically |

---

## Project layout

```
camera1/
├── run.sh                  # launcher — creates venv if needed, activates, runs main.py
├── main.py                 # entry point
├── requirements.txt
├── faces/                  # known-face images for recognition (see below)
├── src/
│   ├── camera_manager.py   # camera enumeration + threaded capture
│   ├── processors.py       # CV pipeline (PassThrough by default; add processors here)
│   ├── detector.py         # YOLO object detector (async, CPU-only, selectable model)
│   ├── face_recognizer.py  # dlib face detector + recognizer (async, CPU-only)
│   ├── plate_recognizer.py # OpenCV morphology + EasyOCR (async, CPU-only)
│   └── ui/
│       ├── main_window.py  # root CTk window, poll loop, wires sidebar ↔ video panel
│       ├── sidebar.py      # left panel: camera list, refresh, detection/face/plate toggles
│       └── video_panel.py  # right panel: tk.Canvas video display + overlays
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

The script:
1. Creates `.venv` if it doesn't exist
2. Activates it
3. `pip install -q -r requirements.txt`
4. `exec python main.py`

---

## Object detection

Three models are selectable from the sidebar segmented button (only when detection is OFF):

| Label | Model file | Classes | Size | Notes |
|---|---|---|---|---|
| **Fast** | `yolov8n.pt` | 80 COCO | ~6 MB | Fastest; good for RPi |
| **Balanced** | `yolov8s.pt` | 80 COCO | ~22 MB | Default; mAP +7 pp vs nano |
| **600+ types** | `yolov8n-oiv7.pt` | 601 Open Images | ~6 MB | Many more object categories |

All models download automatically to `~/.config/Ultralytics/` on first use. Switching models while detection is OFF discards the old model and reloads on next enable.

- **First press** → button turns amber "◌  Loading…" while model loads
- **Model ready** → button turns green "⬤  Detection: ON"; coloured bounding boxes + labels drawn on feed
- **Second press** → inference stops, overlay clears, button goes grey "OFF"
- Confidence threshold: 0.35 (slightly permissive to match larger models)

### Detection architecture (non-blocking)

```
GUI poll (16 ms)
  └─ video_panel.push_frame()
       ├─ detector.submit(frame)   ← drops if detector busy (non-blocking)
       └─ detector.latest()        ← reads last results (lock-guarded, instant)
            └─ ObjectDetector.draw() onto frame copy before display

ObjectDetector._detect_loop  (daemon thread)
  └─ model.predict(device="cpu") at CPU inference speed
       └─ stores results under threading.Lock
```

Video feed always runs at full camera frame rate. Detection overlays update at inference speed independently.

---

## Key bugs found and fixed

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

### 5 — PyTorch CUDA incompatibility (GTX 1080, sm_61)
The pip-distributed PyTorch (ultralytics dependency) ships kernels for sm_75+ only. The GTX 1080 is sm_61 → `RuntimeError: GET was unable to find an engine to execute this computation`.

**Fix:** Set `CUDA_VISIBLE_DEVICES=""` before torch imports (hides all GPUs, eliminates warning spam) and pass `device="cpu"` to both `model.to()` and `model.predict()`. CPU inference is portable and fast enough for preview.

### 6 — EasyOCR emits `pkg_resources` DeprecationWarning on import
EasyOCR uses the legacy `pkg_resources` API internally, printing a deprecation warning to stdout on every import.

**Fix:** Wrap the `import easyocr` in `warnings.catch_warnings()` with `filterwarnings("ignore", category=DeprecationWarning)` in `plate_recognizer._load_and_start()`.

### 7 — HuggingFace `keremberke/yolov8n-license-plate-detection` requires auth
The original plate detection approach used `hf_hub_download` to fetch a YOLO plate model from HuggingFace, which returns 401 Unauthorized for unauthenticated users.

**Fix:** Replaced with a model-free OpenCV morphological approach (Sobel-X + horizontal morph close) for plate region detection. No model download needed beyond EasyOCR.

### 8 — dlib HOG misses faces in lower-resolution or off-angle photos
`face_recognition.face_encodings(img)` uses 1× upsampling by default, which silently skips faces in smaller or slightly tilted portrait images (logging "No face found — skipped").

**Fix:** The known-face loader in `face_recognizer._load_and_start()` retries with `number_of_times_to_upsample=2` before giving up. 2× upsampling catches most real-world portraits at the cost of ~4× more CPU for the one-time load step.

---

## Adding CV processors (pipeline hook)

`src/processors.py` defines a `Pipeline` that `VideoPanel` runs every frame before display and before detection. To add a pre-processing step:

```python
class MyProcessor:
    def process(self, frame: np.ndarray) -> np.ndarray:
        # transform frame (e.g. denoise, equalise histogram)
        return frame
```

Register in `MainWindow.__init__`:

```python
self._pipeline.add(MyProcessor())
```

---

## Facial recognition

- Button: **"⬤  Faces: OFF"** in the sidebar, below Detection
- **First press** → button turns amber "◌  Loading…" while dlib loads (~1 s)
- **Model ready** → button turns blue "⬤  Faces: ON"; face boxes drawn on feed
- **Second press** → inference stops, overlay clears, button goes grey "OFF"
- Works in two modes automatically:
  - **Detection-only** (no images in `faces/`) — all faces labelled "Unknown"
  - **Recognition** — place portrait images in `faces/` named `Alice.jpg`, `Bob.png`, etc. to identify individuals

### Adding known faces

Drop one or more clear portrait images into the `faces/` directory:

```
faces/
├── Alice.jpg
└── Bob.png
```

The file stem is used as the display name. Re-enable the feature to reload. If a face isn't found at 1× upsampling, the loader retries at 2× before skipping.

### Face recognition architecture (non-blocking)

```
GUI poll (16 ms)
  └─ video_panel.push_frame()
       ├─ face_recognizer.submit(frame)  ← drops if busy (non-blocking)
       └─ face_recognizer.latest()       ← reads last results (lock-guarded)
            └─ FaceRecognizer.draw() onto frame copy before display

FaceRecognizer._recognize_loop  (daemon thread)
  └─ frame scaled to 25% → HOG face detection → dlib encodings → compare
       └─ coordinates scaled back up; results stored under threading.Lock
```

---

## Licence plate recognition

- Button: **"⬤  Plates: OFF"** in the sidebar, below Face Recognition
- **First press** → amber "◌  Loading…" while EasyOCR loads (~100 MB, cached in `~/.EasyOCR/` on first use)
- **Model ready** → orange "⬤  Plates: ON"; plate boxes + OCR text drawn on feed
- **Second press** → stops, overlay clears

### Plate recognition architecture

```
GUI poll (16 ms)
  └─ video_panel.push_frame()
       ├─ plate_recognizer.submit(frame)
       └─ plate_recognizer.latest()
            └─ PlateRecognizer.draw() onto frame copy

PlateRecognizer._recognize_loop  (daemon thread)
  ├─ Stage 1 (OpenCV, no model): run at 100% + 50% scale, merge with NMS
  │    CLAHE → bilateral blur → Sobel-X → morph close (17×3 kernel)
  │    → contours filtered by aspect ratio (1.5–7×) and min area (0.2%)
  └─ Stage 2 (EasyOCR): for each candidate crop:
       upscale if <120px wide → CLAHE + sharpen → readtext(allowlist=A-Z0-9)
       → regex filter (must contain digit, 5–8 chars or split format) → PlateMatch
```

Accuracy improvements applied:
- **`allowlist`** on EasyOCR — only A–Z and 0–9, eliminates punctuation/lowercase misreads
- **Per-crop CLAHE + sharpening** — normalises contrast on the crop itself before OCR
- **Multi-scale detection** — runs at both full and half resolution, deduped with IoU NMS
- **Tighter regex** — requires at least one digit; separates "with-separator" from "no-separator" formats

EasyOCR is the bottleneck (~0.3–1 fps per crop on CPU); the frame-drop queue keeps the video feed unaffected.

### Installation note — dlib

`face-recognition` depends on `dlib` which requires a C++ build toolchain:

```bash
# Arch Linux
sudo pacman -S cmake base-devel

# Debian/Ubuntu/Raspberry Pi OS
sudo apt install cmake build-essential
```

dlib compiles during `pip install face-recognition` (takes ~2 min on a Pi 4). `./run.sh` triggers this automatically on first run after the package is added.

---

## Rust prototype

The original implementation used **iced 0.13** + **nokhwa 0.10** (V4L2 backend). Replaced with Python for easier CV integration. Key notes:

- iced 0.12 has a `glyphon 0.5` integer-overflow panic on first render; must use **iced 0.13** (`iced_glyphon 0.6` fork)
- `Handle::from_pixels` renamed to `Handle::from_rgba` in iced 0.13
- `scrollable` content must not use `Length::Fill` on the scroll axis in iced 0.13
- `iced_wgpu` atlas uses `Rgba8Unorm[Srgb]` — bytes are `[R, G, B, A]` (verified in `iced_graphics-0.12.1/src/image.rs`)
- Camera subscription: `Subscription::run_with_id(idx, iced::stream::channel(...))` — changing `idx` auto-restarts subscription

---

## Environment

- OS: Linux (Arch), kernel 6.18.20-1-lts
- Python: 3.14.3
- GPU: NVIDIA GeForce GTX 1080 (sm_61 — **not** compatible with pip PyTorch; CPU inference used)
- Display: Wayland session (XWayland at `:1` for tkinter)
- Camera tested: Logitech C922 Pro Stream Webcam
