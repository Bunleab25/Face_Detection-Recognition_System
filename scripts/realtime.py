from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics.models.yolo.model import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime YOLO face detection (webcam/video).")
    parser.add_argument(
        "--model",
        default="runs/train/yolo11n_fast/weights/best.pt",
        help="Path to a YOLO weights file (.pt). Default: runs/train/yolo11n_fast/weights/best.pt",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (e.g. 0) or video file path. Default: 0",
    )
    parser.add_argument("--device", default="0", help="Device: 0,1,... or 'cpu'. Default: 0")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size. Default: 640")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold. Default: 0.25")
    parser.add_argument(
        "--window",
        default="Face Detector (Realtime)",
        help="OpenCV window title. Default: Face Detector (Realtime)",
    )
    return parser.parse_args()


def _open_capture(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        index = int(source)
        try:
            return cv2.VideoCapture(index, cv2.CAP_DSHOW)
        except Exception:
            return cv2.VideoCapture(index)

    return cv2.VideoCapture(source)


def main() -> int:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))

    cap = _open_capture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    last_time = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        results = model.predict(
            source=frame,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=False,
        )

        annotated = results[0].plot()

        now = time.perf_counter()
        dt = now - last_time
        last_time = now
        fps = (1.0 / dt) if dt > 0 else 0.0

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(args.window, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
