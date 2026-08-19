# Initial Roster 3D Completion Protocol

## Scope

This protocol covers the original 16-player roster: Stephen Curry, Devin Booker, Kevin Durant, Donovan Mitchell, Anthony Edwards, Tyrese Maxey, Luka Dončić, Jamal Murray, Jalen Brunson, Jaylen Brown, Kawhi Leonard, Norman Powell, De'Aaron Fox, Shai Gilgeous-Alexander, LeBron James, and Victor Wembanyama.

## Current gate

The legacy profiles contain 60 unique source URLs across the 16 players. All 16 remain `blocked_unverified`: none has human identity/shot review, source-license status, ball-hand separation evidence, two calibrated camera views, or a passing canonical 3D validation. A legacy pose profile must not be relabeled as a completed player-specific 3D model.

## YouTube pose-candidate layer

The 16-player roster has a separate `models/youtube_single_view_pose_profiles.json` layer. Each profile retains a reviewed public YouTube candidate and its camera-relative pose metrics. It is an explicit `single_view_youtube_3d_pose_estimate`, not a calibrated metric 3D reconstruction. The model boundary is preserved in every profile: `calibration_status=not_available` and `metric_3d_status=not_available`. This layer is suitable for comparing visible shooting mechanics, while the completion gate below remains required for any verified 3D claim.

## Review queue

Generate the queue from the legacy source metadata:

```bash
python3 scripts/generate_initial_roster_review_queue.py
```

The command creates `data/initial_roster_review_queue.jsonl`. Each row is a pending candidate and is deliberately non-importable. A reviewer must establish: real footage, player identity, actual shot event, source license/use status, catch/release/follow-through frames, visible-ball and occlusion measures, and a camera-calibration reference.

## Completion gate per player

A player may move to `verified_2d` only after at least three independent approved clips pass the provenance contract. A player may move to `verified_3d` only after the approved clips also support calibrated multi-view reconstruction and pass reprojection, bone-consistency, and temporal-quality checks. The source pipeline enforces this separation; it does not synthesize approval from titles or pose smoothness alone.

## Output rule

The iOS app may display a player-specific 3D reference only for `verified_3d`. Until then, it may use the anonymous reference archetypes but must not expose a player name, source URL, or an unverified 3D model.
