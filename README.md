# Shooting Form Studio

A Streamlit app and command-line tool for pose-based basketball shooting-form comparison.

The project uses MediaPipe pose landmarks to estimate elbow, shoulder, hip, and knee angles from uploaded video. It is an experimental coaching aid, not a biomechanical or medical assessment.

## Current Features

### Streamlit app (`web_app.py`)

- Analyze one uploaded shooting clip.
- Compare estimated joint angles with:
  - a saved metric profile, or
  - an uploaded reference clip.
- Show the selected release-candidate frame and angle gaps.
- Generate a slow-motion comparison video without retaining the full video in memory.
- Build a local metric profile from uploaded reference clips.

### Command line (`main.py`)

- Annotate one video with pose landmarks.
- Compare two videos side by side.
- Optionally run basketball detection with Ultralytics YOLO.

## Important Limitations

- The current release candidate is the frame where the wrist on the more visible body side reaches its highest image position. The code does not verify the shooting hand or ball release.
- Camera angle, framing, occlusion, and clip timing strongly affect the result.
- A saved metric profile contains angle targets, not an NBA player's source video. In that mode the comparison video repeats the user's motion on both sides and labels the right side as a metric target.
- The included Stephen Curry profile is a single experimental sample and should not be treated as a professional benchmark.
- Profiles created in the app are written to `models/nba_player_models.json`. On hosts with ephemeral filesystems, including Render Free, those changes disappear after a restart, redeploy, or idle spin-down.
- The interface is currently English-only.
- Generated comparison files are deleted after Streamlit reads them, but uploaded source clips currently remain in the instance temporary directory until the process restarts.
- Comparison video uses the MP4V codec. Browser preview support varies; the download remains available when inline playback fails.

## Requirements

- Python 3.12 for the tested deployment
- `opencv-python-headless`
- `mediapipe`
- `numpy`
- `pandas`
- `streamlit`
- Optional: `ultralytics` for CLI `--ball` mode

Pinned deployment versions are listed in `requirements.txt`.

## Local Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt pytest
```

Run the web app:

```bash
streamlit run web_app.py
```

Run the smoke tests:

```bash
python -m py_compile web_app.py main.py
python -m pytest -q
```

## Web Workflow

1. Open the **Analyze** tab.
2. Upload your shooting clip.
3. Choose a saved metric profile or upload a reference clip.
4. Select whether to render a slow-motion comparison.
5. Click **Analyze form**.

Use clips with a clear full-body view and a similar camera angle for both videos.

## CLI Examples

Single-video annotation:

```bash
python main.py --input path/to/input.mp4 --output outputs/annotated.mp4
```

Two-video comparison:

```bash
python main.py --input path/to/video_a.mp4 --input2 path/to/video_b.mp4 --output outputs/compare.mp4
```

Optional ball detection:

```bash
python -m pip install ultralytics
python main.py --input input.mp4 --output outputs/annotated.mp4 --ball --ball_model yolov8n.pt
```

## Deployment

The repository includes a Render Docker Blueprint:

- `Dockerfile`: installs the Python and Linux runtime dependencies.
- `render.yaml`: defines the web service, health check, and deploy policy.
- `/_stcore/health`: Streamlit health endpoint used by Render.
- `.github/workflows/quality.yml`: Python 3.12 compile, test, and MediaPipe runtime smoke checks.

A `render.yaml` file does not create a service by itself. Connect this repository as a Blueprint in the Render dashboard, wait for the `quality` check to pass, and then use the generated `*.onrender.com` address.

Render Free services sleep after inactivity and can take about a minute to start again.
