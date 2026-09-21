"""Local liveness-detection API.

Given exactly 3 frames from a face-verification flow — front-facing (eyes
open), head turned right, head turned left, in that order — decides whether
they came from a live human in front of the camera right now, as opposed to
a printed photo or a video/photo replayed on a second screen.

A single still image can't answer that question — a well-lit, in-focus photo
held straight at the camera passes every per-frame quality check a live face
would, and no single photo can prove all 3 required poses came from the same
live moment. The signal that actually distinguishes a live head turn from a
spoof is that a flat photo/screen, physically rotated, does not reproduce the
foreshortening and occlusion a real 3D head turn produces at a given yaw
angle — so per-frame head-pose verification (see head_pose_degrees()) against
the specific pose each frame is supposed to show is the actual liveness
check, not just a quality gate.
"""

import base64
import io
import logging
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("liveness")

app = FastAPI(title="Local Liveness Detection API")

# ── model loading (once, at container startup) ──────────────────────────────

import mediapipe as mp  # noqa: E402

_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=2,  # 2, not 1, so a second face can be detected and rejected
    refine_landmarks=True,
    min_detection_confidence=0.5,
)

from deepface import DeepFace  # noqa: E402

# Landmark indices into MediaPipe's 468/478-point face mesh (with
# refine_landmarks=True for iris points). See
# https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

EAR_OPEN_THRESHOLD = 0.18
# Frame 0 (front) must be within this of straight-ahead. Frames 1/2 (right,
# left) must fall between the min and max — far enough to be a deliberate
# turn, not so far that mesh/pose detection at a near-profile angle gets
# unreliable. Starting points, not calibrated — see README's Tuning section.
FRONT_YAW_MAX_DEG = 15.0
TURN_YAW_MIN_DEG = 20.0
TURN_YAW_MAX_DEG = 60.0
MIN_BRIGHTNESS = 60.0
MAX_BRIGHTNESS = 200.0
ANTISPOOF_REAL_THRESHOLD = 0.5


class LivenessRequest(BaseModel):
    # Exactly 3 base64-encoded JPEG/PNG frames, in a fixed order: front-facing
    # (eyes open), head turned right, head turned left.
    frames: list[str] = Field(..., min_length=3, max_length=3)


def decode_frame(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("frame could not be decoded as an image")
    return img


def eye_aspect_ratio(landmarks, indices, w, h) -> float:
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
    vertical_1 = np.linalg.norm(pts[1] - pts[5])
    vertical_2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])
    if horizontal == 0:
        return 0.0
    # float(...): np.linalg.norm returns numpy.float64, and a numpy.bool_
    # downstream of that (e.g. an EAR >= threshold comparison) breaks
    # FastAPI's JSON encoder, which only recognizes native Python types.
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def head_pose_degrees(landmarks, w, h) -> tuple[float, float]:
    """Returns (yaw, pitch) in degrees via solvePnP against a generic 3D
    face model. Positive yaw is turned to the subject's right."""
    image_points = np.array(
        [
            (landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h),
            (landmarks[CHIN].x * w, landmarks[CHIN].y * h),
            (landmarks[LEFT_EYE_CORNER].x * w, landmarks[LEFT_EYE_CORNER].y * h),
            (landmarks[RIGHT_EYE_CORNER].x * w, landmarks[RIGHT_EYE_CORNER].y * h),
            (landmarks[LEFT_MOUTH_CORNER].x * w, landmarks[LEFT_MOUTH_CORNER].y * h),
            (landmarks[RIGHT_MOUTH_CORNER].x * w, landmarks[RIGHT_MOUTH_CORNER].y * h),
        ],
        dtype=np.float64,
    )
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ],
        dtype=np.float64,
    )
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1))
    ok, rotation_vec, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    sy = (rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2) ** 0.5
    pitch = np.degrees(np.arctan2(-rotation_mat[2, 0], sy))
    yaw = np.degrees(np.arctan2(rotation_mat[1, 0], rotation_mat[0, 0]))
    return float(yaw), float(pitch)


def face_brightness(img: np.ndarray, bbox) -> float:
    x, y, w, h = bbox
    x, y = max(x, 0), max(y, 0)
    roi = img[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


@app.post("/v1/liveness")
async def check_liveness(request: LivenessRequest):
    try:
        frames = [decode_frame(b64) for b64 in request.frames]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    mesh_results = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = _face_mesh.process(rgb)
        mesh_results.append(result)

    # Exactly one face, in every frame.
    if any(r.multi_face_landmarks is None or len(r.multi_face_landmarks) != 1 for r in mesh_results):
        face_counts = [0 if r.multi_face_landmarks is None else len(r.multi_face_landmarks) for r in mesh_results]
        logger.info("live=False reason=face_count_per_frame=%s (want exactly 1 in every frame)", face_counts)
        return {
            "live": False,
            "faceScore": 0.0,
            "checks": {"antispoof": 0.0, "front": None, "right": None, "left": None},
        }

    per_frame_landmarks = [r.multi_face_landmarks[0].landmark for r in mesh_results]

    antispoof_scores = []
    face_scores = []
    yaws = []

    for frame, landmarks in zip(frames, per_frame_landmarks):
        h, w = frame.shape[:2]

        yaw, _pitch = head_pose_degrees(landmarks, w, h)
        yaws.append(yaw)

        try:
            faces = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="opencv",
                enforce_detection=False,
                anti_spoofing=True,
            )
        except Exception:
            logger.exception("DeepFace.extract_faces failed on a frame; treating it as antispoof=0, faceScore=0")
            faces = []

        if faces:
            primary = faces[0]
            antispoof_scores.append(float(primary.get("antispoof_score", 0.0)))
            face_scores.append(float(primary.get("confidence", 0.0)))
        else:
            antispoof_scores.append(0.0)
            face_scores.append(0.0)

    def frame_brightness_ok(frame: np.ndarray, landmarks) -> bool:
        h, w = frame.shape[:2]
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        bbox = (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))
        brightness = face_brightness(frame, bbox)
        return MIN_BRIGHTNESS <= brightness <= MAX_BRIGHTNESS

    # Frame 0 must be front-facing with eyes open; frames 1/2 must each show
    # a deliberate turn in their expected direction (see head_pose_degrees:
    # positive yaw = turned to the subject's right). A flat photo physically
    # rotated does not reproduce the foreshortening a real head turn does at
    # a given yaw, which is what makes this an actual liveness check rather
    # than just a pose quality gate.
    front_h, front_w = frames[0].shape[:2]
    front_landmarks = per_frame_landmarks[0]
    left_ear = eye_aspect_ratio(front_landmarks, LEFT_EYE, front_w, front_h)
    right_ear = eye_aspect_ratio(front_landmarks, RIGHT_EYE, front_w, front_h)
    front_eyes_open = ((left_ear + right_ear) / 2.0) >= EAR_OPEN_THRESHOLD
    front_facing = abs(yaws[0]) <= FRONT_YAW_MAX_DEG
    front_well_lit = frame_brightness_ok(frames[0], per_frame_landmarks[0])
    front_ok = front_facing and front_eyes_open and front_well_lit

    right_turned = TURN_YAW_MIN_DEG <= yaws[1] <= TURN_YAW_MAX_DEG
    right_well_lit = frame_brightness_ok(frames[1], per_frame_landmarks[1])
    right_ok = right_turned and right_well_lit

    left_turned = -TURN_YAW_MAX_DEG <= yaws[2] <= -TURN_YAW_MIN_DEG
    left_well_lit = frame_brightness_ok(frames[2], per_frame_landmarks[2])
    left_ok = left_turned and left_well_lit

    # Aggregate rather than per-frame-independent, so one noisy antispoof
    # reading at a wider turn angle doesn't by itself sink an otherwise-good
    # attempt.
    avg_antispoof = float(np.mean(antispoof_scores)) if antispoof_scores else 0.0
    avg_face_score = float(np.mean(face_scores)) if face_scores else 0.0
    antispoof_ok = avg_antispoof >= ANTISPOOF_REAL_THRESHOLD

    live = antispoof_ok and front_ok and right_ok and left_ok

    logger.info(
        "live=%s | antispoof=%.3f(ok=%s) | front(facing=%s eyesOpen=%s wellLit=%s yaw=%.1f) | "
        "right(turned=%s wellLit=%s yaw=%.1f) | left(turned=%s wellLit=%s yaw=%.1f) | faceScores=%s",
        bool(live),
        avg_antispoof,
        antispoof_ok,
        front_facing,
        front_eyes_open,
        front_well_lit,
        yaws[0],
        right_turned,
        right_well_lit,
        yaws[1],
        left_turned,
        left_well_lit,
        yaws[2],
        [round(s, 3) for s in face_scores],
    )

    # bool()/float() at this boundary regardless of whether every upstream
    # computation already returned a native type: `and`/`or` pass through
    # whichever operand they last evaluated as-is, so one numpy.bool_ or
    # numpy.float64 anywhere upstream would otherwise reach here unnoticed
    # and break FastAPI's JSON encoder (as it did before these casts existed).
    return {
        "live": bool(live),
        "faceScore": float(avg_face_score),
        "checks": {
            "antispoof": float(avg_antispoof),
            "front": {"facingCamera": bool(front_facing), "eyesOpen": bool(front_eyes_open), "wellLit": bool(front_well_lit), "yaw": float(yaws[0])},
            "right": {"turned": bool(right_turned), "wellLit": bool(right_well_lit), "yaw": float(yaws[1])},
            "left": {"turned": bool(left_turned), "wellLit": bool(left_well_lit), "yaw": float(yaws[2])},
        },
    }
