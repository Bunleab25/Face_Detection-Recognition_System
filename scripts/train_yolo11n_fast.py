from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Efficient YOLOv11n training script (object detection)."
    )

    parser.add_argument(
        "--data",
        default="face_datasets/data.yaml",
        help="Path to YOLO data.yaml (train/val/test + nc + names).",
    )
    parser.add_argument(
        "--model",
        default="models/yolo11n.pt",
        help="Base model weights (.pt). Default: models/yolo11n.pt",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Device: 0,1,... or 'cpu'. Default: 0",
    )

    # Speed/efficiency knobs
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size. Default: 640",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Epochs. Default: 100",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size. Use -1 for auto. Default: -1",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Dataloader workers. Default: 8",
    )
    parser.add_argument(
        "--cache",
        choices=["ram", "disk", "false"],
        default="disk",
        help="Cache images/labels for faster epochs. Default: disk",
    )
    parser.add_argument(
        "--amp",
        type=int,
        choices=[0, 1],
        default=1,
        help="Automatic Mixed Precision (1=on, 0=off). Default: 1",
    )

    # Output
    parser.add_argument("--project", default="runs/train", help="Output root. Default: runs/train")
    parser.add_argument("--name", default="yolo11n_fast", help="Run name. Default: yolo11n_fast")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow existing output directory without error.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    started_at = datetime.now().astimezone()
    t0 = time.perf_counter()
    print(f"train_started_at {started_at.isoformat(timespec='seconds')}")

    train_status = "running"
    raised: Exception | None = None
    exit_code = 0

    # Map cache arg to Ultralytics expected values
    cache_value: str | bool
    if args.cache == "false":
        cache_value = False
    else:
        cache_value = args.cache

    try:
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_path}")

        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Download it or pass --model <path>."
            )

        model = YOLO(str(model_path))
        model.train(
            data=str(data_path),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            cache=cache_value,
            amp=bool(args.amp),
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
        )
        train_status = "completed"
        exit_code = 0
    except KeyboardInterrupt:
        train_status = "interrupted"
        exit_code = 130
    except Exception as exc:  # noqa: BLE001
        train_status = "error"
        raised = exc
        exit_code = 1
    finally:
        finished_at = datetime.now().astimezone()
        elapsed_s = time.perf_counter() - t0
        print(f"train_status {train_status}")
        print(f"train_finished_at {finished_at.isoformat(timespec='seconds')}")
        print(f"train_elapsed_seconds {elapsed_s:.1f}")

    if raised is not None:
        raise raised

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
