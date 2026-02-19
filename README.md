# Shooting Form Studio

Pose-based basketball shooting form analysis with:
- Streamlit web app (`web_app.py`)
- CLI video processor (`main.py`)

The project uses MediaPipe pose landmarks and compares joint angles (elbow, shoulder, hip, knee).

## Features

- Build real player models from uploaded NBA clips
- Single-video comparison against:
  - saved player model, or
  - custom reference clip
- Automatic release-sync sub-feature in single-video compare:
  - aligns user and reference clips at release frame
  - generates side-by-side slow-motion comparison video
  - shows angle-difference summary + frame scrubber
- Bilingual UI toggle (English / Korean)
- CLI support for:
  - single-video pose annotation
  - two-video side-by-side pose comparison

## Project Structure

- `web_app.py`: Streamlit app
- `main.py`: CLI processor
- `models/pose_landmarker_full.task`: MediaPipe Tasks pose model (required for some environments)
- `models/nba_player_models.json`: saved player model database
- `outputs/`: output videos

## Requirements

- Python 3.10+
- `opencv-python`
- `mediapipe`
- `numpy`
- `pandas`
- `streamlit`
- Optional: `ultralytics` (only for `main.py --ball`)

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install opencv-python mediapipe numpy pandas streamlit
# optional
pip install ultralytics
```

## Run Web App

```bash
streamlit run web_app.py
```

### Web Workflow

1. `NBA Model Builder` tab:
   - upload NBA clips
   - build/update player model
2. `Single Video Compare` tab:
   - upload your video
   - choose saved model or upload custom reference clip
   - run comparison
3. If custom reference clip is provided:
   - release-sync comparison runs automatically as sub-feature
   - slow-motion synchronized video is generated

## Run CLI

### Single Video Annotation

```bash
python main.py --input path/to/input.mp4 --output outputs/annotated.mp4
```

Useful flags:
- `--show`
- `--max_frames 300`
- `--draw_indices`

Optional ball mode (single-video only):

```bash
python main.py --input input.mp4 --output outputs/annotated.mp4 --ball --ball_model yolov8n.pt
```

### Two-Video Comparison (CLI)

```bash
python main.py --input path/to/video_a.mp4 --input2 path/to/video_b.mp4 --output outputs/compare.mp4
```

## Notes

- If your MediaPipe build does not expose `mp.solutions.pose`, this project falls back to MediaPipe Tasks API using `models/pose_landmarker_full.task`.
- Keep camera angle similar between compared videos for better landmark consistency.

## Troubleshooting

- `Failed to import 'mediapipe'`
  - install/upgrade: `pip install --upgrade mediapipe`
- No visible comparison video in web
  - app provides frame scrubber fallback even when codec playback is unavailable
- Poor comparison quality
  - use side-view clips with clear full-body visibility

