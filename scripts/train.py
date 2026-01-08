from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import time

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO face detector on a YOLO-format dataset.")
    parser.add_argument("--data", default="face_datasets/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model weights (.pt). Default: yolo11n.pt")
    parser.add_argument("--device", default="0", help="Device: 0,1,... or 'cpu'. Default: 0")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size. Default: 640")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs. Default: 50")
    parser.add_argument("--batch", type=int, default=16, help="Batch size. Default: 16")
    parser.add_argument("--project", default="runs/train", help="Output project folder. Default: runs/train")
    parser.add_argument("--name", default="face", help="Run name under project/. Default: face")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    started_at = datetime.now().astimezone()
    t0 = time.perf_counter()
    print(f"train_started_at {started_at.isoformat(timespec='seconds')}")

    train_status = "running"
    raised: Exception | None = None
    exit_code = 0

    try:
        data_path = Path(args.data)
        if not data_path.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_path}")

        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = YOLO(str(model_path))
        model.train(
            data=str(data_path),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
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
