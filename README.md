# PDF Autoscroller

Gaze-controlled hands-free scrolling for PDF reading using MediaPipe face landmarks and a webcam.

## How it works

1. **Calibration** — look at 9 dots on screen (~2 sec each). The system learns the mapping from your eye position to screen coordinates using quadratic least-squares regression.

2. **Tracking** — real-time loop detects your face via MediaPipe FaceLandmarker, computes iris position relative to eye corners, and maps it to a screen coordinate.

3. **Auto-scroll** — look at the **bottom** of the screen to scroll down, **top** to scroll up, **center** to stop. Scroll speed is proportional to how far from center you look.

## Requirements

- Python 3.9+
- Webcam
- Windows (uses MSMF camera backend; untested on other OS)

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The model file (`face_landmarker.task`) downloads automatically on first run.

## Usage

```bash
python main.py
```

### Controls

| Key | Action |
|---|---|
| `P` | Pause / resume auto-scroll |
| `+` / `-` | Increase / decrease max scroll speed |
| `Q` | Quit |

### Tips

- Sit ~2 feet from the camera with good lighting
- Keep your face centered in the frame
- Adjust speed with `+`/`-` to match your reading pace
- If scrolling feels jerky, press `-` to lower max speed
- Press `P` to pause scrolling while you look away

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point — calibration + gaze loop + scroll logic |
| `gaze_detector.py` | MediaPipe FaceLandmarker wrapper (478 landmarks, iris indices) |
| `gaze_ratio.py` | Iris-projection gaze ratio computation |
| `calibration.py` | 9-point quadratic regression calibration |
| `smoother.py` | Kalman filter for smoothing gaze cursor |
| `test_camera.py` | Quick webcam test |

## Known issues

- **No face detected** — make sure you're visible, well-lit, and no other app is using the camera
- **Camera locked after crash** — close other camera apps or restart the program

## Project

CMSC 191 Final Project.
