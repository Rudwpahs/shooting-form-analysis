import argparse
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ultralytics import YOLO


# -----------------------------
# YouTube download (yt-dlp)
# -----------------------------
def download_youtube(url: str, out_dir: str = "videos") -> str:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    template = str(out_path / "%(title).200s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "-o",
        template,
        url,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"yt-dlp 다운로드 실패:\n{proc.stderr}")

    text = (proc.stdout or "") + "\n" + (proc.stderr or "")

    m = re.search(r'Merging formats into\s+"(.+?)"', text)
    if m:
        return m.group(1)

    m = re.search(r"Destination:\s+(.+)", text)
    if m:
        return m.group(1).strip()

    files = sorted(out_path.glob("*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise RuntimeError("다운로드는 된 것 같은데 파일을 찾지 못했습니다. videos 폴더를 확인해 주세요.")
    return str(files[0])


# -----------------------------
# File picker
# -----------------------------
def pick_video_file() -> str:
    try:
        from tkinter import Tk
        from tkinter.filedialog import askopenfilename

        Tk().withdraw()
        path = askopenfilename(
            title="비디오 파일 선택",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv"),
                ("All files", "*.*"),
            ],
        )
        return path
    except Exception as e:
        raise RuntimeError(
            "파일 선택창(tkinter)을 열지 못했습니다.\n"
            "대신 커맨드에서 --input으로 비디오 경로를 지정해 주세요."
        ) from e


def ask_youtube_or_file() -> Tuple[str, str]:
    """
    실행 시: 유튜브 사용할지 먼저 묻고,
    아니오(Enter 포함)면 바로 파일 선택창을 띄웁니다.
    반환: (mode, path) mode는 "youtube" 또는 "file"
    """
    ans = input("유튜브 링크로 분석할까요? (y/N): ").strip().lower()
    if ans == "y":
        url = input("유튜브 URL을 입력하세요: ").strip()
        if not url:
            raise RuntimeError("유튜브 URL이 비었습니다.")
        path = download_youtube(url, out_dir="videos")
        return "youtube", path

    # 기본: 파일 선택창
    path = pick_video_file()
    if not path:
        raise RuntimeError("비디오 선택이 취소되었습니다.")
    return "file", path


# -----------------------------
# Pose connections
# -----------------------------
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


def draw_pose(
    bgr_frame: np.ndarray,
    landmarks,
    connections=POSE_CONNECTIONS,
    draw_indices: bool = False,
    line_color=(0, 255, 0),
    point_color=(0, 0, 255),
) -> np.ndarray:
    h, w = bgr_frame.shape[:2]
    pts = []
    for lm in landmarks:
        x_px = int(lm.x * w)
        y_px = int(lm.y * h)
        pts.append((x_px, y_px))

    for a, b in connections:
        if 0 <= a < len(pts) and 0 <= b < len(pts):
            cv2.line(bgr_frame, pts[a], pts[b], line_color, 2, cv2.LINE_AA)

    for i, (x, y) in enumerate(pts):
        cv2.circle(bgr_frame, (x, y), 4, point_color, -1, cv2.LINE_AA)
        if draw_indices:
            cv2.putText(
                bgr_frame, str(i), (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )
    return bgr_frame


def build_landmarker(model_path: str, num_poses: int):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


# -----------------------------
# Ball helpers
# -----------------------------
def draw_ball_box(frame: np.ndarray, xyxy, conf: float, track_id=None):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)
    label = f"ball {conf:.2f}"
    if track_id is not None:
        label = f"ball#{int(track_id)} {conf:.2f}"
    cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)


def get_ball_class_id(model: YOLO) -> int:
    try:
        for k, v in model.names.items():
            if v == "sports ball":
                return int(k)
    except Exception:
        pass
    return 32


def get_ball_best(yolo_res, ball_cls_id: int):
    best = None  # (conf, xyxy, tid)
    if yolo_res.boxes is None or len(yolo_res.boxes) == 0:
        return None

    for b in yolo_res.boxes:
        cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
        if cls_id != ball_cls_id:
            continue
        conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
        xyxy = b.xyxy[0].cpu().numpy()
        tid = None
        if hasattr(b, "id") and b.id is not None:
            tid = int(b.id.item()) if hasattr(b.id, "item") else int(b.id)

        if best is None or conf > best[0]:
            best = (conf, xyxy, tid)
    return best


# -----------------------------
# Owner (possession)
# -----------------------------
def landmark_px(landmarks, idx: int, w: int, h: int) -> Tuple[int, int]:
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


def choose_owner(
    pose_landmarks_list,
    ball_center: Tuple[float, float],
    frame_wh: Tuple[int, int],
    ball_box: np.ndarray,
    last_owner: Optional[int],
    hold_frames_left: int,
    owner_hold: int = 8,
) -> Tuple[Optional[int], int]:
    w, h = frame_wh
    bx, by = ball_center

    x1, y1, x2, y2 = ball_box
    br = 0.5 * max(10.0, float(max(x2 - x1, y2 - y1)))
    dist_thresh = br * 2.2 + 60.0  # 튜닝 포인트

    hand_idxs = [15, 16, 17, 18, 19, 20, 21, 22]  # 손/손목 주변

    best_i = None
    best_d = None

    for i, lms in enumerate(pose_landmarks_list):
        dmin = 1e18
        for idx in hand_idxs:
            px, py = landmark_px(lms, idx, w, h)
            d = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
            if d < dmin:
                dmin = d

        if best_d is None or dmin < best_d:
            best_d = dmin
            best_i = i

    # 너무 멀면 기존 소유자 조금 유지
    if best_d is None or best_d > dist_thresh:
        if last_owner is not None and hold_frames_left > 0:
            return last_owner, hold_frames_left - 1
        return None, 0

    # 소유자 바뀌면 hold 초기화
    if last_owner != best_i:
        return best_i, owner_hold
    return best_i, owner_hold


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ask", action="store_true", help="실행 시 유튜브 여부 먼저 묻고, 아니면 파일 선택창")

    parser.add_argument("--youtube", default="", help="유튜브 URL(있으면 자동 다운로드 후 분석)")
    parser.add_argument("--input", default="", help="입력 비디오 경로 (없으면 파일 선택창)")
    parser.add_argument("--output", default="outputs/annotated.mp4", help="출력 비디오 경로")

    parser.add_argument("--model", default="models/pose_landmarker_full.task", help="포즈 .task 모델 경로")
    parser.add_argument("--show", action="store_true", help="화면 표시(q로 종료)")
    parser.add_argument("--max_frames", type=int, default=0, help="0이면 끝까지")
    parser.add_argument("--draw_indices", action="store_true", help="포즈 랜드마크 인덱스 표시")
    parser.add_argument("--num_poses", type=int, default=4, help="화면에서 추적할 최대 사람 수")

    parser.add_argument("--ball", action="store_true", help="농구공 박스 그리기")
    parser.add_argument("--ball_model", default="yolov8n.pt", help="YOLO 모델 경로 또는 이름")
    parser.add_argument("--ball_conf", type=float, default=0.12, help="공 탐지 confidence threshold")
    parser.add_argument("--ball_track", action="store_true", help="YOLO track 사용(persist)")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO 입력 해상도(작은 공이면 1280 권장)")

    args = parser.parse_args()

    # ✅ 입력 비디오 결정 우선순위
    # 1) --ask : 유튜브 먼저 물어보고, 아니면 파일선택창
    # 2) --youtube
    # 3) --input
    # 4) 파일선택창
    if args.ask:
        _, input_path = ask_youtube_or_file()
    elif args.youtube.strip():
        input_path = download_youtube(args.youtube.strip(), out_dir="videos")
    else:
        input_path = args.input.strip()
        if not input_path:
            input_path = pick_video_file()
            if not input_path:
                raise RuntimeError("비디오 선택이 취소되었습니다. 다시 실행해 선택해 주세요.")

    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"입력 비디오가 없습니다: {in_path}")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"모델 파일(.task)이 없습니다: {model_path}\n"
            f"models 폴더에 pose_landmarker_full.task를 먼저 받아주세요."
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("출력 비디오 파일을 생성하지 못했습니다(코덱/경로/권한 문제 가능).")

    landmarker = build_landmarker(str(model_path), num_poses=args.num_poses)

    ball_yolo = None
    ball_cls_id = None
    if args.ball:
        ball_yolo = YOLO(args.ball_model)
        ball_cls_id = get_ball_class_id(ball_yolo)

    owner_idx: Optional[int] = None
    owner_hold_left = 0

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # 1) pose inference (multi-person)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)
            pose_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_list = pose_result.pose_landmarks or []

            # 2) ball detection (원본 frame_bgr로!)
            best_ball = None
            if args.ball and ball_yolo is not None:
                if args.ball_track:
                    yolo_res = ball_yolo.track(
                        frame_bgr,
                        persist=True,
                        conf=args.ball_conf,
                        imgsz=args.imgsz,
                        verbose=False,
                    )[0]
                else:
                    yolo_res = ball_yolo(
                        frame_bgr,
                        conf=args.ball_conf,
                        imgsz=args.imgsz,
                        verbose=False,
                    )[0]

                best_ball = get_ball_best(yolo_res, ball_cls_id)

            annotated = frame_bgr.copy()

            # 공 박스 + 소유자 판단
            if best_ball is not None:
                bconf, bxyxy, btid = best_ball
                draw_ball_box(annotated, bxyxy, bconf, track_id=btid)

                if len(pose_list) > 0:
                    bx1, by1, bx2, by2 = bxyxy
                    bx = float(bx1 + bx2) / 2.0
                    by = float(by1 + by2) / 2.0

                    owner_idx, owner_hold_left = choose_owner(
                        pose_landmarks_list=pose_list,
                        ball_center=(bx, by),
                        frame_wh=(width, height),
                        ball_box=bxyxy,
                        last_owner=owner_idx,
                        hold_frames_left=owner_hold_left,
                        owner_hold=8,
                    )
            else:
                if owner_idx is not None and owner_hold_left > 0:
                    owner_hold_left -= 1
                else:
                    owner_idx = None

            # 소유자만 스켈레톤 표시
            if owner_idx is not None and 0 <= owner_idx < len(pose_list):
                annotated = draw_pose(
                    annotated,
                    pose_list[owner_idx],
                    draw_indices=args.draw_indices,
                )
                cv2.putText(
                    annotated,
                    f"Owner: person#{owner_idx}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    annotated,
                    "Owner: none",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            writer.write(annotated)

            if args.show:
                cv2.imshow("Ball Owner Skeleton (press q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break

    finally:
        cap.release()
        writer.release()
        landmarker.close()
        if args.show:
            cv2.destroyAllWindows()

    print(f"[완료] 저장됨: {out_path.resolve()}")


if __name__ == "__main__":
    main()
