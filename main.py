import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as exc:
    raise RuntimeError(
        "Failed to import 'mediapipe'. Install the official package with: pip install mediapipe"
    ) from exc

try:
    from ultralytics import YOLO
except Exception:  # ultralytics is optional unless --ball is used.
    YOLO = None


@dataclass
class _PoseLandmarksContainer:
    landmark: List


@dataclass
class _PoseProcessResult:
    pose_landmarks: Optional[_PoseLandmarksContainer]


class _TasksPoseAdapter:
    def __init__(self, landmarker, fps: float):
        self._landmarker = landmarker
        self._timestamp_ms = 0
        self._frame_interval_ms = max(1, int(round(1000.0 / max(fps, 1e-6))))

    def process(self, frame_rgb: np.ndarray) -> _PoseProcessResult:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(image, self._timestamp_ms)
        self._timestamp_ms += self._frame_interval_ms
        if result.pose_landmarks:
            return _PoseProcessResult(_PoseLandmarksContainer(result.pose_landmarks[0]))
        return _PoseProcessResult(None)

    def close(self):
        self._landmarker.close()


def _create_solutions_pose_detector():
    pose_cls = mp.solutions.pose.Pose
    return pose_cls(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _create_tasks_pose_detector(fps: float):
    try:
        vision = mp.tasks.vision
        base_options = mp.tasks.BaseOptions
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe is installed but neither solutions.pose nor tasks APIs are usable."
        ) from exc

    model_path = Path(__file__).resolve().parent / "models" / "pose_landmarker_full.task"
    if not model_path.exists():
        raise RuntimeError(
            "PoseLandmarker model file not found at "
            f"{model_path}. Add the .task model or install a mediapipe build with solutions.pose."
        )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    return _TasksPoseAdapter(landmarker, fps=fps)


def create_pose_detector(fps: float = 30.0):
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        return _create_solutions_pose_detector()
    return _create_tasks_pose_detector(fps=fps)


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


def detect_primary_pose(frame_bgr: np.ndarray, pose_detector):
    people = detect_people(frame_bgr, pose_detector)
    return people[0] if people else None


def landmark_xy(landmarks, idx: int, w: int, h: int) -> np.ndarray:
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return float("nan")
    cos_angle = float(np.dot(ba, bc) / denom)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def get_shooting_side(landmarks) -> str:
    left_vis = landmarks[15].visibility + landmarks[13].visibility + landmarks[11].visibility
    right_vis = landmarks[16].visibility + landmarks[14].visibility + landmarks[12].visibility
    return "left" if left_vis > right_vis else "right"


def collect_pose_metrics(landmarks, w: int, h: int) -> Optional[Dict[str, float]]:
    if landmarks is None or len(landmarks) < 29:
        return None

    side = get_shooting_side(landmarks)
    s_idx, e_idx, w_idx, h_idx, k_idx, a_idx = (11, 13, 15, 23, 25, 27) if side == "left" else (12, 14, 16, 24, 26, 28)

    shoulder = landmark_xy(landmarks, s_idx, w, h)
    elbow = landmark_xy(landmarks, e_idx, w, h)
    wrist = landmark_xy(landmarks, w_idx, w, h)
    hip = landmark_xy(landmarks, h_idx, w, h)
    knee = landmark_xy(landmarks, k_idx, w, h)
    ankle = landmark_xy(landmarks, a_idx, w, h)

    metrics = {
        "Elbow": calculate_angle(shoulder, elbow, wrist),
        "Shoulder": calculate_angle(elbow, shoulder, hip),
        "Hip": calculate_angle(shoulder, hip, knee),
        "Knee": calculate_angle(hip, knee, ankle),
    }
    if any(np.isnan(list(metrics.values()))):
        return None
    return metrics


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


def resize_to_height(frame: np.ndarray, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h == target_h:
        return frame
    new_w = max(1, int(round(w * target_h / h)))
    return cv2.resize(frame, (new_w, target_h), interpolation=cv2.INTER_AREA)


def combine_side_by_side(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
    target_h = max(frame_a.shape[0], frame_b.shape[0])
    left = resize_to_height(frame_a, target_h)
    right = resize_to_height(frame_b, target_h)
    return np.hstack([left, right])


def draw_diff_overlay(canvas: np.ndarray, m1: Optional[Dict[str, float]], m2: Optional[Dict[str, float]]):
    line_h = 28
    lines = ["Differences (abs):"]
    if m1 is None or m2 is None:
        lines.append("Not enough landmarks in one/both videos")
    else:
        for key in ["Elbow", "Shoulder", "Hip", "Knee"]:
            lines.append(f"{key}: {abs(m1[key] - m2[key]):5.1f} deg")

    overlay_h = 10 + line_h * len(lines)
    cv2.rectangle(canvas, (10, 10), (380, 10 + overlay_h), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (20, 35 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def process_single_video(args):
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

    pose_detector = create_pose_detector(fps=fps)

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


def process_compare_videos(args):
    in1 = Path(args.input)
    in2 = Path(args.input2)
    if not in1.exists():
        raise FileNotFoundError(f"Input video not found: {in1}")
    if not in2.exists():
        raise FileNotFoundError(f"Input video not found: {in2}")
    if args.ball:
        raise RuntimeError("--ball is not supported in two-video compare mode.")

    cap1 = cv2.VideoCapture(str(in1))
    cap2 = cv2.VideoCapture(str(in2))
    if not cap1.isOpened():
        raise RuntimeError(f"Failed to open video: {in1}")
    if not cap2.isOpened():
        raise RuntimeError(f"Failed to open video: {in2}")

    fps1 = cap1.get(cv2.CAP_PROP_FPS) or 30.0
    fps2 = cap2.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = min(fps1, fps2) if fps1 > 0 and fps2 > 0 else 30.0

    pose1 = create_pose_detector(fps=fps1)
    pose2 = create_pose_detector(fps=fps2)

    ok1, frame1 = cap1.read()
    ok2, frame2 = cap2.read()
    if not ok1 or not ok2:
        cap1.release()
        cap2.release()
        pose1.close()
        pose2.close()
        raise RuntimeError("Failed to read first frame from one/both videos.")

    combined0 = combine_side_by_side(frame1, frame2)
    out_h, out_w = combined0.shape[:2]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (out_w, out_h))

    frame_idx = 0
    try:
        while ok1 and ok2:
            vis1 = frame1.copy()
            vis2 = frame2.copy()

            lms1 = detect_primary_pose(frame1, pose1)
            lms2 = detect_primary_pose(frame2, pose2)

            m1 = None
            m2 = None

            if lms1 is not None:
                draw_pose(vis1, lms1, draw_indices=args.draw_indices)
                h1, w1 = vis1.shape[:2]
                m1 = collect_pose_metrics(lms1, w1, h1)
            if lms2 is not None:
                draw_pose(vis2, lms2, draw_indices=args.draw_indices)
                h2, w2 = vis2.shape[:2]
                m2 = collect_pose_metrics(lms2, w2, h2)

            cv2.putText(vis1, "Video A", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
            cv2.putText(vis2, "Video B", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

            combined = combine_side_by_side(vis1, vis2)
            draw_diff_overlay(combined, m1, m2)
            writer.write(combined)

            if args.show:
                cv2.imshow("Pose Compare", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break

            ok1, frame1 = cap1.read()
            ok2, frame2 = cap2.read()
    finally:
        cap1.release()
        cap2.release()
        writer.release()
        pose1.close()
        pose2.close()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Done. Saved compare output to: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Pose annotator (single video) or two-video pose comparison")
    parser.add_argument("--input", required=True, help="Input video path (Video A)")
    parser.add_argument("--input2", default="", help="Optional second input video path (Video B)")
    parser.add_argument("--output", default="outputs/annotated.mp4", help="Output video path")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--draw_indices", action="store_true")

    parser.add_argument("--ball", action="store_true", help="Enable basketball detection (single-video mode only)")
    parser.add_argument("--ball_model", default="yolov8n.pt")
    parser.add_argument("--ball_conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    if args.input2:
        process_compare_videos(args)
    else:
        process_single_video(args)


if __name__ == "__main__":
    main()
