# Liveness API

A lightweight face-liveness detection API used by `stillwater-api`'s
`POST /v1/verify/face` to independently re-verify that a face-capture came
from a live human in front of the camera, not a printed photo or a screen
replay. Runs on CPU, packaged as a Docker container for an ARM64 (aarch64)
EC2 instance — same shape as the sibling `moderation-api` service.

## Why 3 posed frames, not 1

A single still image cannot prove liveness: a well-lit, in-focus photo held
straight at the camera passes every per-frame quality check (face detected,
eyes open, facing camera, well lit) that a live face would, and no single
photo can prove that 3 different poses came from the same live moment. The
signal that actually distinguishes a live head turn from a spoof is that a
flat photo/screen, physically rotated, does not reproduce the foreshortening
and occlusion a real 3D head turn produces at a given yaw angle. This API
requires exactly 3 frames in a fixed order — front-facing (eyes open), head
turned right, head turned left — and verifies each against the specific pose
it's supposed to show.

## API

### `POST /v1/liveness`

Request body — exactly 3 base64-encoded JPEG/PNG frames, in this fixed
order: front-facing, turned right, turned left:

```json
{ "frames": ["<base64 front>", "<base64 right>", "<base64 left>"] }
```

Response body:

```json
{
  "live": true,
  "faceScore": 0.94,
  "checks": {
    "antispoof": 0.91,
    "front": { "facingCamera": true, "eyesOpen": true, "wellLit": true, "yaw": 3.2 },
    "right": { "turned": true, "wellLit": true, "yaw": 34.1 },
    "left": { "turned": true, "wellLit": true, "yaw": -31.7 }
  }
}
```

`live` is the single boolean callers should act on. `checks` is diagnostic
detail only, for logs/debugging — it is not a second source of truth.

- **antispoof** — mean real-vs-spoof score across the 3 frames from
  MiniVision's Silent-Face-Anti-Spoofing model (via DeepFace's
  `anti_spoofing=True`), a per-frame texture/print/screen classifier.
- **front** — the actual liveness check for frame 0: near-zero head yaw
  (via MediaPipe Face Mesh landmarks + solvePnP), eyes open
  (eye-aspect-ratio), and well-lit (face-ROI brightness). Eyes-open isn't
  checked on the turned frames — one eye is occluded/foreshortened in
  profile, making eye-aspect-ratio unreliable there.
- **right** / **left** — frames 1/2 must each show head yaw past a minimum
  threshold in their expected direction (positive = subject's right,
  negative = subject's left — see `head_pose_degrees()`), plus well-lit.

`live` requires antispoof plus all three pose checks to pass, and exactly
one detected face in every frame.

## Running

```bash
docker compose up --build -d
```

This builds the image for `linux/arm64` and starts the API on port `9091`,
bound to `127.0.0.1` only.

To stop it:

```bash
docker compose down
```

## Local development (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 9091
```

## Tuning

The thresholds in `main.py` (`EAR_OPEN_THRESHOLD`, `FRONT_YAW_MAX_DEG`,
`TURN_YAW_MIN_DEG`, `TURN_YAW_MAX_DEG`, `MIN_BRIGHTNESS`/`MAX_BRIGHTNESS`,
`ANTISPOOF_REAL_THRESHOLD`) are starting points, not calibrated against a
labeled dataset. Expect to adjust them after testing against real spoof
attempts (see `test.py`) and real users in varied lighting — in particular,
`TURN_YAW_MIN_DEG`/`TURN_YAW_MAX_DEG` trade off how deliberate a turn must
be against how reliably MediaPipe's landmarks and solvePnP's pose estimate
hold up as the face approaches profile.
