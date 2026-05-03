import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception:
    mp = None

try:
    from ultralytics import YOLO
except Exception:  # ultralytics is optional unless --ball is used.
    YOLO = None


def create_pose_detector():
    """Create a MediaPipe Pose detector across package variants."""
    if mp is None:
        return None, "`mediapipe` is not installed. Install with: pip install mediapipe"

    # 1) Standard MediaPipe API.
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        return mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ), None

    # 2) Fallback for environments where `solutions` is not exposed at top-level.
    try:
        from mediapipe.python.solutions.pose import Pose as PoseCls  # type: ignore

        return PoseCls(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ), None
    except Exception:
        return None, (
            "Could not initialize MediaPipe Pose from this `mediapipe` installation. "
            "Please reinstall the official package: `pip uninstall mediapipe -y && pip install mediapipe`."
        )


POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (24, 26),
    (25, 27), (26, 28),
    (27, 29), (28, 30),
    (29, 31), (30, 32),
    (27, 31), (28, 32),
]


def draw_pose(frame: np.ndarray, landmarks, draw_indices: bool = False) -> np.ndarray:
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in POSE_CONNECTIONS:
        if 0 <= a < len(pts) and 0 <= b < len(pts):
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)

    for i, (x, y) in enumerate(pts):
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
        if draw_indices:
            cv2.putText(frame, str(i), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return frame


def detect_people(frame_bgr: np.ndarray, pose_detector) -> List:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result = pose_detector.process(rgb)
    if result.pose_landmarks:
        return [result.pose_landmarks.landmark]
    return []


def draw_ball_box(frame: np.ndarray, xyxy, conf: float):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, f"ball {conf:.2f}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


def get_ball_best(yolo_res, ball_cls_id: int):
    best = None
    if yolo_res.boxes is None or len(yolo_res.boxes) == 0:
        return None

    for b in yolo_res.boxes:
        cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
        if cls_id != ball_cls_id:
            continue
        conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
        xyxy = b.xyxy[0].cpu().numpy()
        if best is None or conf > best[0]:
            best = (conf, xyxy)
    return best


def choose_owner(pose_landmarks_list, ball_center: Tuple[float, float], frame_wh: Tuple[int, int]) -> Optional[int]:
    if not pose_landmarks_list:
        return None

    w, h = frame_wh
    bx, by = ball_center
    hand_idxs = [15, 16, 17, 18, 19, 20, 21, 22]

    best_i = None
    best_d = None
    for i, lms in enumerate(pose_landmarks_list):
        dmin = 1e18
        for idx in hand_idxs:
            lm = lms[idx]
            px, py = lm.x * w, lm.y * h
            d = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
            dmin = min(dmin, d)
        if best_d is None or dmin < best_d:
            best_d = dmin
            best_i = i

    if best_d is not None and best_d < 180:
        return best_i
    return None


def main():
    parser = argparse.ArgumentParser(description="Pose + optional ball tracking annotator")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="outputs/annotated.mp4", help="Output video path")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--draw_indices", action="store_true")

    parser.add_argument("--ball", action="store_true", help="Enable basketball detection")
    parser.add_argument("--ball_model", default="yolov8n.pt")
    parser.add_argument("--ball_conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input video not found: {in_path}")

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    pose_detector, pose_error = create_pose_detector()
    if pose_detector is None:
        cap.release()
        writer.release()
        raise RuntimeError(pose_error)

    ball_model = None
    ball_cls_id = 32
    if args.ball:
        if YOLO is None:
            raise RuntimeError("--ball was provided, but ultralytics is not installed. Install with: pip install ultralytics")
        ball_model = YOLO(args.ball_model)

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            pose_list = detect_people(frame, pose_detector)
            annotated = frame.copy()
            owner_idx = None

            if ball_model is not None:
                yolo_res = ball_model(annotated, conf=args.ball_conf, imgsz=args.imgsz, verbose=False)[0]
                best_ball = get_ball_best(yolo_res, ball_cls_id)
                if best_ball is not None:
                    conf, bxyxy = best_ball
                    draw_ball_box(annotated, bxyxy, conf)
                    bx1, by1, bx2, by2 = bxyxy
                    owner_idx = choose_owner(pose_list, ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0), (width, height))

            if owner_idx is not None and owner_idx < len(pose_list):
                draw_pose(annotated, pose_list[owner_idx], draw_indices=args.draw_indices)
                cv2.putText(annotated, f"Owner: person#{owner_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            elif pose_list:
                draw_pose(annotated, pose_list[0], draw_indices=args.draw_indices)
                cv2.putText(annotated, "Owner: none", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            writer.write(annotated)

            if args.show:
                cv2.imshow("Pose Annotator", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        writer.release()
        pose_detector.close()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Done. Saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
