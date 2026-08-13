# Shooting Form Analysis

GitHub-hosted basketball shooting form analyzer.

**Stack:** Python (Flask) + HTML · MediaPipe 3D joint angles (degrees only) · SQLite matching

Experimental coaching aid — not a biomechanical or medical assessment.

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

```bash
python -m unittest discover -s tests -v
```

## Deploy (GitHub → Render)

Push to `main`. Render Blueprint uses `Dockerfile` + `render.yaml`.

See [DEPLOYMENT.md](DEPLOYMENT.md).

Render Free services sleep after inactivity and can take about a minute to start again. SQLite profiles on ephemeral disks reset after redeploy/spin-down unless you attach a persistent disk.

## Project layout

| Path | Role |
|------|------|
| `app/` | Pose, 3D angles, analyze, similarity, DB, Flask API |
| `static/` | HTML / CSS / JS UI (UI UX Pro Max sports system) |
| `models/` | MediaPipe `.task` + seed JSON |
| `data/` | SQLite (runtime) |
| `Dockerfile` | Production image (gunicorn) |
| `design-system/` | Persisted UI UX Pro Max rules |

## Limitations

- Release candidate ≈ frame where the shooting-side wrist is highest in the image.
- Camera angle, framing, and occlusion affect results.
- Seeded player profiles are unofficial self-measured angle samples, not affiliated with any league/athlete.
- Legacy Streamlit (`web_app.py`) / Next site (`website/`) are not used for deploy.
