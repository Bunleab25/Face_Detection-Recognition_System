"""Step-based training loop scaffold with a Darknet-style schedule.

Implements:
- Batch size: 64
- Total steps (iterations): 20,000
- Optimizer: SGD
- LR: starts at 1e-3
- Custom scheduler: exponential decay every 4,000 steps

This script is intentionally model/dataset-agnostic: plug in your YOLOv3 model,
loss, and dataset (images + targets) where indicated.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


DEFAULT_NUM_WORKERS = 0 if os.name == "nt" else 4


def count_images(images_dir: str | os.PathLike) -> int:
    p = Path(images_dir)
    if not p.exists():
        raise FileNotFoundError(f"Images directory not found: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Not a directory: {p}")

    return sum(1 for f in p.rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXTS)


def epoch_equivalent(total_steps: int, batch_size: int, num_images: int) -> float:
    # One epoch ~= seeing num_images samples once.
    # total samples seen = total_steps * batch_size
    return (total_steps * batch_size) / max(1, num_images)


def steps_per_epoch(num_images: int, batch_size: int, drop_last: bool) -> int:
    if drop_last:
        return max(1, num_images // batch_size)
    return max(1, math.ceil(num_images / batch_size))


def make_step_decay_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    decay_every_steps: int = 4000,
    gamma: float = 0.1,
) -> MultiStepLR:
    """Darknet-style step schedule using MultiStepLR.

    Paper text: "After being set at 0.001, the learning rate decreased exponentially every 4000 steps".

    Darknet's common interpretation is geometric decay at fixed iteration boundaries, i.e.
    multiply the LR by a constant ("scales") at each step milestone.

    Default milestones for 20,000 total steps and 4,000-step decay:
      [4000, 8000, 12000, 16000]

    Note: we drive MultiStepLR with the *global step* counter (not epochs).
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")
    if decay_every_steps <= 0:
        raise ValueError("decay_every_steps must be > 0")
    if not (0.0 < gamma <= 1.0):
        raise ValueError("gamma must be in (0, 1]")

    milestones = list(range(decay_every_steps, total_steps, decay_every_steps))

    # last_epoch=0 ensures that when we call scheduler.step() once per optimizer step,
    # the first decay triggers at global_step == milestones[0] (e.g., 4000).
    return MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=0)


# -------------------------
# Minimal placeholders
# -------------------------


class DummyYoloDataset(Dataset):
    """Placeholder dataset.

    Replace with your actual YOLO dataset yielding:
      image: Tensor[C,H,W]
      target: whatever your loss expects
    """

    def __init__(self, num_samples: int = 1000, image_size: int = 416):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.rand(3, self.image_size, self.image_size)
        y = torch.zeros(1)  # placeholder target
        return x, y


class DummyYoloModel(nn.Module):
    """Placeholder model.

    Replace with a YOLOv3 model implementation.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def dummy_loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Replace with YOLOv3 loss: box + obj + cls losses.
    return (pred - 0.0).pow(2).mean()


@dataclass
class TrainConfig:
    batch_size: int = 64
    total_steps: int = 20_000
    lr0: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4
    decay_every_steps: int = 4000
    gamma: float = 0.1
    num_workers: int = DEFAULT_NUM_WORKERS
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    drop_last: bool = False


def train(cfg: TrainConfig, *, dataset: Dataset) -> None:
    device = torch.device(cfg.device)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=cfg.drop_last,
    )

    model = DummyYoloModel().to(device)

    optimizer = SGD(
        model.parameters(),
        lr=cfg.lr0,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=False,
    )

    # Required if we create a scheduler with last_epoch != -1.
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])

    scheduler = make_step_decay_scheduler(
        optimizer,
        total_steps=cfg.total_steps,
        decay_every_steps=cfg.decay_every_steps,
        gamma=cfg.gamma,
    )

    model.train()
    global_step = 0

    print(f"milestones (steps)    : {sorted(scheduler.milestones)}")
    print(f"gamma                : {cfg.gamma}")

    # Loop indefinitely over epochs until we reach total_steps.
    while global_step < cfg.total_steps:
        for images, targets in loader:
            if global_step >= cfg.total_steps:
                break

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            preds = model(images)
            loss = dummy_loss_fn(preds, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1

            # IMPORTANT: step per-iteration.
            # We configure MultiStepLR so that calling step() once per iteration drops
            # LR at exactly 4000/8000/12000/16000 steps (Darknet-style milestones).
            scheduler.step()

            if global_step % 100 == 0 or global_step in {1, cfg.total_steps}:
                lr = optimizer.param_groups[0]["lr"]
                print(f"step={global_step:6d}/{cfg.total_steps}  lr={lr:.6g}  loss={loss.item():.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv3-style step-based LR schedule (PyTorch scaffold)")

    parser.add_argument("--images-dir", type=str, default="face_datasets/train/images")
    parser.add_argument("--num-images", type=int, default=0, help="If set, overrides auto-count")

    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20_000)

    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    parser.add_argument("--decay-every", type=int, default=4000)
    parser.add_argument("--gamma", type=float, default=0.1, help="Multiply LR by gamma every decay-every steps")

    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--drop-last", action="store_true", default=False)
    parser.add_argument("--no-drop-last", dest="drop_last", action="store_false")

    args = parser.parse_args()

    cfg = TrainConfig(
        batch_size=args.batch,
        total_steps=args.steps,
        lr0=args.lr0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        decay_every_steps=args.decay_every,
        gamma=args.gamma,
        num_workers=args.num_workers,
        device=args.device,
        drop_last=args.drop_last,
    )

    n = args.num_images if args.num_images > 0 else count_images(args.images_dir)
    eq_epochs = epoch_equivalent(cfg.total_steps, cfg.batch_size, n)
    spe = steps_per_epoch(n, cfg.batch_size, cfg.drop_last)
    implied_epochs_int = math.ceil(cfg.total_steps / spe)

    print(f"train images (N)      : {n}")
    print(f"batch size           : {cfg.batch_size}")
    print(f"total steps          : {cfg.total_steps}")
    print(f"epoch equivalent     : {eq_epochs:.4f} (steps*batch/N)")
    print(f"steps per epoch      : {spe} (drop_last={cfg.drop_last})")
    print(f"epochs to reach steps: {implied_epochs_int} (ceil(steps / steps_per_epoch))")

    # Replace this dataset with your YOLO dataset.
    dataset = DummyYoloDataset(num_samples=n)

    train(cfg, dataset=dataset)


if __name__ == "__main__":
    main()
