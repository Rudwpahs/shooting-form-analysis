# Shooting Form Analysis

GitHub-hosted basketball shooting form analyzer.

**Stack:** Python (Flask) + HTML · MediaPipe 33-landmark motion · phase-aligned DTW matching · SQLite

Experimental coaching aid — not a biomechanical or medical assessment.

## Features

1. Upload 1–3 camera views (side / front / oblique)
2. Select the person in the clip
3. Normalize all 33 landmarks by torso scale and shooting hand (no height / limb-length advantage)
4. Match the complete catch-to-follow-through motion with phase-aligned DTW
5. Compare with a selected player and independently find the nearest player
6. Retarget player and user motion onto the same general-adult skeleton
7. Rotate, zoom, scrub, and play catch-to-follow-through 3D landmark timelines

## Run locally

```bash
pip install -r requirements.txt
python run.py
# http://127.0.0.1:7860
```

```bash
python -m unittest discover -s tests -v
python scripts/validate_motion_dataset.py --min-clips 3
```

Player models are learned from quality-filtered public shooting/form footage.
The committed candidate catalog stores metadata and source URLs only; downloaded
videos are temporary and are not redistributed. To rebuild data, install
`requirements-data.txt`, run `scripts/build_youtube_catalog.py`, then
`scripts/discover_allstar_players.py`.

## Deploy (GitHub → Render)

Push to `main`. Render Blueprint uses `Dockerfile` + `render.yaml`.

See [DEPLOYMENT.md](DEPLOYMENT.md).

Render Free services sleep after inactivity and can take about a minute to start again. SQLite profiles on ephemeral disks reset after redeploy/spin-down unless you attach a persistent disk.

## Project layout

| Path | Role |
|------|------|
| `app/` | Pose, 3D angles, analyze, similarity, DB, Flask API |
| `static/` | HTML / CSS / JS UI (UI UX Pro Max sports system) |
| `models/` | MediaPipe `.task`, learned motion profiles, source catalog, validation report |
| `scripts/` | YouTube metadata discovery, profile learning, deterministic data validation |
| `data/` | SQLite (runtime) |
| `Dockerfile` | Production image (gunicorn) |
| `design-system/` | Persisted UI UX Pro Max rules |

## Limitations

- Release is selected from a temporally valid wrist-rise/arm-extension window; unusual edits can still confuse it.
- Camera angle, framing, and occlusion affect results.
- Monocular landmark depth is an estimate, not calibrated multi-camera biomechanics.
- Skeletons use fixed ordinary-adult proportions; they visualize angle motion and do not reproduce an athlete's actual body dimensions.
- Player profiles are unofficial measurements from public footage, not affiliated with any league/athlete.
- Legacy Streamlit (`web_app.py`) / Next site (`website/`) are not used for deploy.
