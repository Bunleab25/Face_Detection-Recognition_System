"""ArcFace training loop with step-based LR schedule.

Integrates:
- Step 1: Aligned 112×112 faces (from face_alignment.py)
- Step 2: ArcFace model (from model.py)
- YOLOv3-style SGD + step-based LR decay

Configuration matches the paper's approach:
- Batch size: 64
- Optimizer: SGD with momentum
- LR schedule: Step decay every N steps
- Loss: Cross-entropy on ArcFace logits
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from model import create_arcface_model


class AlignedFaceDataset(Dataset):
    """Dataset wrapper for pre-aligned 112×112 faces.

    Expects directory structure:
        root/
            identity1/
                img1.jpg
                img2.jpg
            identity2/
                img3.jpg
            ...

    Args:
        root: Path to dataset root with identity subdirectories.
        transform: Optional transform (default: normalize only, no geometric transforms).
    """

    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        # Use torchvision.datasets.ImageFolder for simplicity
        default_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Scale to [-1, 1]
        ])
        self.transform = transform or default_transform
        
        # Find valid image files (skip empty classes)
        self.dataset = datasets.ImageFolder(
            str(self.root), 
            transform=self.transform,
            is_valid_file=lambda p: Path(p).suffix.lower() in ['.jpg', '.jpeg', '.png']
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def class_to_idx(self):
        return self.dataset.class_to_idx


def get_data_augmentation(mode: str = "none") -> transforms.Compose:
    """Get data augmentation pipeline.

    ArcFace requires aligned faces, so ONLY appearance augmentation is allowed.
    NO geometric transforms (rotation, scaling, cropping).

    Args:
        mode: "none", "light", or "strong".

    Returns:
        Composed transform.
    """
    if mode == "none":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    elif mode == "light":
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    elif mode == "strong":
        return transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    else:
        raise ValueError(f"Unknown augmentation mode: {mode}")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    """Train for one epoch.

    Args:
        model: ArcFace model.
        dataloader: Training data loader.
        criterion: Loss function (CrossEntropyLoss).
        optimizer: SGD optimizer.
        device: Device.
        epoch: Current epoch number.

    Returns:
        avg_loss: Average loss for the epoch.
        avg_acc: Average accuracy for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", dynamic_ncols=True)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass: get ArcFace logits + embeddings
        logits, embeddings = model(images, labels)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        total_correct += predicted.eq(labels).sum().item()
        total_samples += images.size(0)

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * total_correct / total_samples:.2f}%",
        })

    avg_loss = total_loss / total_samples
    avg_acc = 100.0 * total_correct / total_samples
    return avg_loss, avg_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ArcFace face recognition model.")

    # Data
    parser.add_argument("--data-dir", type=str, required=True, help="Path to aligned face dataset root")
    parser.add_argument("--augmentation", type=str, default="light", choices=["none", "light", "strong"])

    # Model
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--embedding-size", type=int, default=512)
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet pretrained backbone")
    parser.add_argument("--scale", type=float, default=64.0, help="ArcFace scale parameter")
    parser.add_argument("--margin", type=float, default=0.5, help="ArcFace angular margin (radians)")
    parser.add_argument("--dropout", type=float, default=0.0)

    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lr-milestones", type=int, nargs="+", default=[20, 35, 45], help="LR decay epochs")
    parser.add_argument("--lr-gamma", type=float, default=0.1, help="LR decay factor")

    # System
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="runs/arcface_train")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Dataset
    print(f"Loading dataset from: {args.data_dir}")
    transform = get_data_augmentation(args.augmentation)
    train_dataset = AlignedFaceDataset(args.data_dir, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    num_classes = len(train_dataset.classes)
    print(f"Dataset: {len(train_dataset)} images, {num_classes} identities")
    print(f"Augmentation: {args.augmentation}")

    # Model
    print(f"\nCreating model:")
    print(f"  Backbone: {args.backbone}")
    print(f"  Embedding size: {args.embedding_size}")
    print(f"  ArcFace scale: {args.scale}, margin: {args.margin}")
    model = create_arcface_model(
        num_classes=num_classes,
        embedding_size=args.embedding_size,
        backbone=args.backbone,
        pretrained=args.pretrained,
        scale=args.scale,
        margin=args.margin,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Optimizer & Scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=False,
    )

    scheduler = MultiStepLR(
        optimizer,
        milestones=args.lr_milestones,
        gamma=args.lr_gamma,
    )

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"LR schedule: {args.lr} → decay by {args.lr_gamma} at epochs {args.lr_milestones}")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        avg_loss, avg_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Step LR scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print epoch summary
        print(f"Epoch {epoch}/{args.epochs}: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%, LR={current_lr:.6f}")

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint_path = output_dir / f"arcface_{args.backbone}_epoch_{epoch}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
                "acc": avg_acc,
                "args": vars(args),
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

            # Also save just the backbone (for inference)
            backbone_path = output_dir / f"backbone_{args.backbone}_epoch_{epoch}.pth"
            torch.save(model.backbone.state_dict(), backbone_path)

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Final model saved to: {output_dir}")


if __name__ == "__main__":
    main()
