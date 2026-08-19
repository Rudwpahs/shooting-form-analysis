# Shooting Form Analysis

GitHub-hosted basketball shooting form analyzer.

**Stack:** Python (Flask) + HTML · MediaPipe 33-landmark motion · phase-aligned DTW matching · SQLite

Experimental coaching aid — not a biomechanical or medical assessment.

## Features

1. Upload 1–3 camera views (side / front / oblique)
2. Select the person in the clip
3. Normalize all 33 landmarks by torso scale and shooting hand (no height / limb-length advantage)
4. Verify a basketball shot with shooter continuity plus hand–ball separation and upward ball motion evidence
5. Match the complete catch-to-follow-through motion only against human-approved, provenance-complete player references
6. Return single-view pose analysis as **unverified** when ball or capture evidence is insufficient
7. Use calibrated, synchronized multi-view DLT triangulation only after a 3D quality gate
8. Publish a player skeleton only when its profile is `verified_3d` and a canonical model is present

## Run locally

```bash
pip install -r requirements.txt
python run.py
# http://127.0.0.1:7860
```

```bash
pip install -r requirements-data.txt  # required for full reconstruction tests / data scripts
python -m unittest discover -s tests -v
python scripts/validate_motion_dataset.py --min-clips 3
python scripts/import_reference_manifest.py data/reference_clips/curry_reference_v1.jsonl --strict
```

Legacy player profiles remain visible for review but are now isolated as `unverified_legacy`; they cannot be matched or rendered as 3D models. A matchable profile requires at least three independent, human-approved clips with a complete source, real-footage, player-identity, shot-event, and frame-label record. A `verified_3d` profile additionally requires a published canonical model from calibrated, synchronized multi-view capture.

The committed candidate catalog stores metadata and source URLs only; downloaded videos are temporary and are not redistributed. Do not use automated search results as player reference data without filling the provenance contract and manual review. See [Verified 3D Capture](docs/VERIFIED_3D_CAPTURE.md) for the exact manifest, capture, calibration, synchronization, and publication procedure.

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

- The baseline basketball detector is color/circle based; it can reject valid shots under motion blur, unusual ball colors, or poor lighting. Rejection is safer than treating a pose-only wrist peak as a confirmed shot.
- Camera angle, framing, fast movement, and occlusion affect pose results.
- Monocular landmark depth is an estimate, not calibrated multi-camera biomechanics. The API labels it pose-only and will not publish it as verified 3D.
- 3D requires same-trial calibrated, synchronized views, ball-verified shot evidence, and reprojection/bone/temporal quality gates.
- Skeletons use fixed ordinary-adult proportions unless a verified subject-specific model is supplied; they do not reproduce an athlete's actual body dimensions.
- `verified_2d` player references are unofficial coaching references, not athlete identification or league-affiliated measurements.
- Legacy Streamlit (`web_app.py`) / Next site (`website/`) are not used for deploy.
