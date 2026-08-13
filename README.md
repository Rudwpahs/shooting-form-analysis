# Shooting Form Analysis

GitHub-hosted basketball shooting form analyzer.

**Stack:** Python (Flask) + HTML · MediaPipe 3D joint angles (degrees only) · SQLite matching

## Features

1. Upload 1–3 camera views (side / front / oblique)
2. Select the person in the clip
3. Lift pose to 3D and store **joint angles only** (no height / limb length)
4. Match against player angle profiles in DB

## Run locally

```bash
pip install -r requirements.txt
python run.py
# http://127.0.0.1:7860
```

## Deploy (GitHub → Render)

Push to `main`. Render Blueprint uses `Dockerfile` + `render.yaml`.

See [DEPLOYMENT.md](DEPLOYMENT.md).

## Project layout

| Path | Role |
|------|------|
| `app/` | Pose, 3D angles, analyze, similarity, DB, Flask API |
| `static/` | HTML / CSS / JS UI |
| `models/` | MediaPipe `.task` + seed JSON |
| `data/` | SQLite (runtime) |
| `Dockerfile` | Production image (gunicorn) |

Legacy Streamlit (`web_app.py`) / Next site (`website/`) are not used for deploy.
