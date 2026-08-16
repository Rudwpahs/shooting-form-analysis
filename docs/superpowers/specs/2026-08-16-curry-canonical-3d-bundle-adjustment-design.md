# Curry Canonical 3D Bundle-Adjustment Design

## Scope

Build and validate a Stephen Curry-only canonical 3D shooting motion from multiple non-synchronized front and side clips. The output is 48 normalized motion frames, 33 MediaPipe landmarks, XYZ coordinates, per-landmark dispersion, confidence, and provenance.

This iteration does not replace the existing player matcher, user analysis pipeline, SQLite schema, or renderer. The existing angle-derived skeleton remains the fallback for Curry when the new model is absent or fails validation, and remains the only path for every other player.

## Existing data flow and root cause

The current flow is:

1. `app/pose.py` extracts both `image_landmarks` and `world_landmarks`.
2. `app/analyze.py` normally computes joint angles from `world_landmarks`.
3. The motion timeline is built from `image_landmarks` after `app.motion.normalize_landmarks()` independently recenters every frame on its pelvis and independently scales every frame by its torso length.
4. `app.motion.build_motion_prototype()` phase-aligns timelines to 48 frames and takes a coordinate median, but only within one detected camera view.
5. `app.skeleton.canonical_landmarks_from_pose()` reduces the 33-landmark pose to 18 joints and retargets its directions onto fixed adult proportions.
6. Every rendered frame is independently shifted until its lowest foot is at Y=0.
7. The API selects one stored timeline, converts it through the fixed-body retargeter, and passes it to the existing canvas renderer.

Consequently, the displayed skeleton and stored 3D angles come from different coordinate sources; multi-view data is not fused across the full motion; jump translation is discarded; and the final body is a synthetic fixed-proportion body rather than Curry's reconstructed motion.

The current Curry data contains five side-view source clips and no front-view timeline. A prior profile and the current discovery script retain public front-view candidates. `persist_player_profile()` currently replaces the complete `views` object, which explains why the earlier front entry disappeared.

## Architecture

### New reconstruction module

Add `app/reconstruction3d.py` with focused, independently testable units:

- fixed-scale clip normalization;
- phase resampling to 48 frames;
- robust observation filtering;
- yaw-aware orthographic projection and initialization;
- SciPy sparse bundle adjustment;
- player-specific bone-length estimation;
- validation metrics;
- canonical model serialization and loading;
- conversion to the existing skeleton API schema.

SciPy is imported lazily inside the optimizer so serving a prebuilt model does not initialize the optimizer. The dependency is nevertheless pinned in `requirements.txt` so CI exercises the real optimization path and Render builds remain reproducible.

### Curry builder

Add `scripts/build_curry_canonical3d.py`. It will:

1. read current Curry side sources and curated front sources;
2. download and analyze candidate clips without committing video files;
3. reject clips with low visibility, incomplete shot spans, implausible body geometry, or a detected view inconsistent with the assigned view;
4. retain at least three accepted clips per front and side view;
5. normalize, phase-align, robustly filter, initialize, optimize, and validate the model;
6. write only a validation-passed Curry entry to `models/canonical_3d_models.json`;
7. emit before/after metrics for the final report.

The builder does not overwrite `models/nba_player_models.json`. Existing source metadata and fallback timelines remain intact.

### Minimal runtime integration

`GET /api/players/<player_key>/skeleton` first checks for a validation-passed entry in `canonical_3d_models.json`. For `stephen_curry`, it returns the new 33-landmark timeline. If loading, shape validation, or the validation flag fails, it executes the existing DB and `build_skeleton_timeline()` path unchanged.

The new landmarks use stable semantic MediaPipe names and a body-bone list compatible with the existing generic renderer. `static/skeleton3d.js` is changed only if an actual renderer incompatibility is demonstrated by a regression test or manual inspection.

## Observation model

Coordinates are defined as:

- X: Curry's left-right direction;
- Y: vertical, positive upward;
- Z: rim direction / front-back direction.

For clip `c` with yaw `theta_c`, normalized screen horizontal coordinate `u` observes:

`u_c(t,i) = X(t,i) cos(theta_c) + Z(t,i) sin(theta_c)`

and normalized screen vertical coordinate observes:

`v_c(t,i) = Y(t,i)`

Front, side, and oblique defaults are 0, pi/2, and pi/4 radians. Explicit per-clip yaw metadata overrides the default. At least two sufficiently separated yaw groups are required to activate the final Curry model.

The initial X/Z estimate is the visibility- and quality-weighted least-squares solution:

`[X, Z]^T = (A^T W A)^-1 A^T W u`

Y is initialized with a robust weighted median of all accepted front and side observations.

## Fixed scale and origin

No frame receives its own scale. Each clip receives one scale computed from high-visibility observations over the complete accepted shot span.

Candidate measures are pelvis-to-shoulder center, shoulder width, hip width, left/right hip-to-knee, and left/right knee-to-ankle. View-sensitive widths are downweighted for side views. Per-measure frame outliers are removed with MAD, and the remaining normalized segment estimates are combined by weighted median into one clip scale.

The clip origin is Curry's pelvis center at catch, not the pelvis center of every frame. X and Z use the catch pelvis origin. Y also uses the catch pelvis origin, preserving dip, rise, jump, and landing translation. After optimization, one global floor offset is computed from reliable catch-phase ankle/heel/toe observations and applied to every frame equally. There is no per-frame floor alignment.

## Phase synchronization

Existing shot-span detection supplies catch, dip, release, and follow-through. Existing normalized anchors remain catch=0.00, dip=0.25, rise=0.48, release=0.75, follow-through=1.00.

An optional set-point anchor is estimated between rise and release from shooting-wrist vertical velocity and elbow flexion. When it is unreliable, it is recorded as inferred metadata and placed at normalized progress 0.60. This does not alter existing API phase labels or fallback timelines.

Every accepted clip is piecewise-linearly resampled between anchors to exactly 48 frames before cross-clip aggregation.

## Robust filtering

Filtering occurs before and during optimization:

1. reject landmark observations below the visibility threshold;
2. reject clip frames that imply implausible torso or major-limb ratios;
3. apply a temporal Hampel test to landmark velocity to remove isolated tracking jumps;
4. at each normalized frame and landmark, reject cross-clip coordinate residuals beyond the MAD threshold;
5. use SciPy `least_squares(loss="soft_l1")` so remaining moderate outliers do not dominate.

Missing observations remain missing; they are not replaced with zero. Optimization fills them from other views, bone constraints, and temporal continuity while reducing their output confidence.

## Sparse bundle adjustment

The optimization variable is `P[48,33,3]`. Player-specific canonical major-bone lengths are initialized from the median of high-confidence reconstructed frames.

The residual vector minimizes:

`E = E_projection + lambda_bone E_bone + lambda_time E_acceleration + lambda_world E_world_angle + lambda_anchor E_gauge`

Where:

- `E_projection` is the weighted front/side/oblique screen-coordinate residual;
- `E_bone` penalizes deviation of major body bones from Curry's median bone lengths;
- `E_acceleration` penalizes second differences, preserving velocity while suppressing jitter;
- `E_world_angle` softly checks elbow, shoulder, hip, and knee angles against robust MediaPipe world-landmark angles without treating monocular world Z as ground truth;
- `E_gauge` fixes the catch pelvis origin and prevents translation/rotation drift.

`scipy.optimize.least_squares` uses a sparse Jacobian pattern because each observation touches one point, each bone residual touches two points in one frame, and each temporal residual touches the same point in three adjacent frames. Solver weights and tolerances are stored in model metadata.

## Model format

`models/canonical_3d_models.json` stores one `stephen_curry` entry containing:

- model/version and `quality_mode="multi_view_3d"`;
- 48 frames x 33 landmarks x XYZ;
- per-frame phase and normalized progress;
- per-landmark, per-axis MAD;
- observation count and confidence;
- player-specific major-bone lengths;
- clip URL, detected view, yaw, quality, and accepted/rejected status;
- projection, bone, temporal, symmetry, and angle-validation metrics;
- optimizer settings and `validation.passed`.

Physical height is not stored or used. Coordinates are body-scale normalized.

## Validation and activation gates

The builder compares the new Curry model with the current rendered Curry skeleton and reports:

- major-bone length coefficient of variation and maximum range;
- frame-to-frame velocity P95, maximum, and maximum/P95 spike ratio;
- left/right paired-bone consistency;
- release elbow, shoulder, hip, and knee angle error against robust world-angle observations;
- front and side projection RMSE;
- pelvis vertical range;
- landmark coverage and confidence.

Activation requires all of the following:

- exactly 48 finite frames and 33 XYZ landmarks;
- at least three accepted front and three accepted side sources;
- major-bone coefficient of variation no greater than 1%;
- projection RMSE no greater than 0.08 body-scale units for each required view;
- no non-finite optimizer result;
- no extreme temporal spike introduced by optimization;
- release joint-angle mean absolute error no greater than 8 degrees across the four validated joints.

If a gate fails, the script writes a diagnostic report but does not activate or replace the existing Curry skeleton.

## Tests

Add `tests/test_reconstruction3d.py` covering:

- exact synthetic front/side XYZ reconstruction;
- front and side reprojection;
- fixed scale across a clip and preservation of pelvis vertical motion;
- bone-length stability across the complete timeline;
- temporal continuity;
- robustness to one severe outlier clip;
- yaw-aware oblique least-squares reconstruction;
- model shape, variance, confidence, and serialization;
- invalid-model fallback.

Extend existing smoke coverage to verify that a valid Curry canonical model is preferred while unknown/other players retain the old path. Existing skeleton, motion, matching, health, and deployment tests must continue to pass.

## Expected file changes

- Add `app/reconstruction3d.py`.
- Add `scripts/build_curry_canonical3d.py`.
- Add `tests/test_reconstruction3d.py`.
- Add generated `models/canonical_3d_models.json` after validation passes.
- Modify `app/analyze.py` only to optionally retain raw image/world observations for the offline Curry builder.
- Modify `app/server.py` only to prefer a validated Curry canonical model before the existing fallback.
- Modify `requirements.txt` to pin SciPy.
- Modify existing smoke tests for the Curry preference/fallback contract.
- Modify `static/skeleton3d.js` only if validation proves necessary.

## Out of scope

- Rebuilding other NBA player models.
- Replacing existing player matching.
- Multi-view reconstruction of user uploads.
- Destructive DB migrations.
- UI redesign.
- Treating monocular MediaPipe world Z as measured depth.

These can be added after the Curry prototype demonstrates acceptable projection, kinematic, and temporal metrics.
