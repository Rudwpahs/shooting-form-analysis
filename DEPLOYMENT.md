# Deployment (GitHub → Render)

This repo deploys from GitHub using Render Docker.

## Render

1. Open https://dashboard.render.com
2. **New** → **Blueprint** (or Web Service)
3. Connect `Rudwpahs/shooting-form-analysis`
4. Render uses `render.yaml` + `Dockerfile`
5. Auto-deploys on commit to `main`

The container runs:

```bash
gunicorn -b 0.0.0.0:$PORT -w 1 -t 300 --threads 4 app.server:app
```

App stack:
- Python Flask API (`app/`)
- Static HTML UI (`static/`)
- SQLite angle DB (`data/`, created at runtime)
- MediaPipe pose model (`models/pose_landmarker_full.task`)

## Local

```bash
pip install -r requirements.txt
python run.py
# http://127.0.0.1:7860
```

## Local Docker

```bash
docker build -t shooting-form-analysis .
docker run --rm -p 10000:10000 -e PORT=10000 shooting-form-analysis
```

Open http://localhost:10000
