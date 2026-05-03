# Shooting Form Analysis

This repository now has two working entry points:

- `web_app.py` (recommended): Streamlit app for comparing your shooting form with a selected NBA player.
- `main.py`: CLI video annotator for pose tracking (and optional basketball detection with YOLO).

## 1) Web app (recommended)

### Install
```bash
pip install streamlit mediapipe opencv-python numpy pandas
```

### Run
```bash
streamlit run web_app.py
```

### How to use
1. Choose an NBA player.
2. Choose a reference mode:
   - **Use built-in NBA profile** (works immediately), or
   - **Upload custom NBA reference clip**.
3. Upload your own shooting video.
4. Click **Analyze form**.

The app compares release-frame joint angles (elbow, shoulder, hip, knee) and shows angle gaps.

## 2) CLI annotator

### Install
```bash
pip install mediapipe opencv-python numpy
```

Optional for ball detection:
```bash
pip install ultralytics
```

### Run
```bash
python main.py --input /path/to/video.mp4 --output outputs/annotated.mp4
```

With ball detection:
```bash
python main.py --input /path/to/video.mp4 --output outputs/annotated.mp4 --ball
```
