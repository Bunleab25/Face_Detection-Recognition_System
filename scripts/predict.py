from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics.models.yolo.model import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO prediction on an image/folder/video.")
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Path to a YOLO weights file (.pt). Default: yolo11n.pt",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image path, folder, video path, or webcam index (e.g. 0).",
    )
    parser.add_argument("--device", default="0", help="Device: 0,1,... or 'cpu'. Default: 0")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size. Default: 640")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold. Default: 0.25")
    parser.add_argument("--project", default="runs/detect", help="Output project folder. Default: runs/detect")
    parser.add_argument("--name", default="predict", help="Run name under project/. Default: predict")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))

    # Ultralytics accepts a path, directory, URL, or webcam index.
    # If source is a number (e.g., "0"), pass it as an int.
    source: str | int
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    results = model.predict(
        source=source,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        project=args.project,
        name=args.name,
        save=True,
    )

    # Print where outputs went (Ultralytics sets save_dir per run).
    if results:
        print("saved_to", results[0].save_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
