"""Batch face alignment for ArcFace training dataset preparation.

Preprocesses face datasets (LFW, WIDER, etc.) by:
1. Detecting faces with YOLO
2. Aligning to canonical 112×112 template
3. Saving aligned crops in organized directory structure

This is the CRITICAL preprocessing step for ArcFace - it requires aligned faces,
not just resized crops.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from face_alignment import FaceAligner


@dataclass
class AlignmentStats:
    total_images: int = 0
    faces_detected: int = 0
    faces_aligned: int = 0
    failed_images: int = 0


def process_image_file(
    args: tuple[Path, Path, FaceAligner, bool],
) -> tuple[int, int]:
    """Process a single image (multiprocessing worker function).

    Args:
        args: (input_path, output_dir, aligner, save_first_only)

    Returns:
        (num_detected, num_aligned)
    """
    input_path, output_dir, aligner, save_first_only = args

    try:
        img = cv2.imread(str(input_path))
        if img is None:
            return (0, 0)

        aligned_faces = aligner.process_image(img)
        if len(aligned_faces) == 0:
            return (0, 0)

        # Save aligned face(s).
        # For identity datasets like LFW, typically save first face only.
        # For detection datasets, save all.
        num_to_save = 1 if save_first_only else len(aligned_faces)

        for i, face in enumerate(aligned_faces[:num_to_save]):
            if save_first_only:
                # Preserve original filename (one face per identity image).
                out_name = input_path.stem + ".jpg"
            else:
                # Multi-face: append index.
                out_name = f"{input_path.stem}_{i:03d}.jpg"

            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), face)

        return (len(aligned_faces), num_to_save)

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return (0, 0)


def align_dataset(
    input_dir: Path,
    output_dir: Path,
    yolo_model: str,
    device: str = "0",
    conf_threshold: float = 0.5,
    landmark_method: str = "fallback",
    save_first_only: bool = True,
    num_workers: int = 1,
    file_pattern: str = "**/*.jpg",
) -> AlignmentStats:
    """Batch-align a face dataset.

    Args:
        input_dir: Root directory with face images.
        output_dir: Output directory for aligned 112×112 crops.
        yolo_model: Path to YOLO .pt weights.
        device: Device for YOLO ("0", "cpu", etc.).
        conf_threshold: Face detection confidence threshold.
        landmark_method: "cv2_face" or "fallback".
        save_first_only: If True, save only the first detected face per image (for identity datasets).
        num_workers: Number of parallel workers (1 = sequential).
        file_pattern: Glob pattern for input images.

    Returns:
        stats: Alignment statistics.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all image files.
    image_files = sorted(input_dir.glob(file_pattern))
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {input_dir} matching {file_pattern}")

    print(f"Found {len(image_files)} images in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"YOLO model: {yolo_model}")
    print(f"Device: {device}")
    print(f"Landmark method: {landmark_method}")
    print(f"Save first face only: {save_first_only}")
    print(f"Workers: {num_workers}")

    # Create aligner (note: multiprocessing requires careful handling of YOLO model).
    # For num_workers > 1, each worker creates its own aligner to avoid pickling issues.
    aligner = FaceAligner(
        yolo_model_path=yolo_model,
        device=device,
        conf_threshold=conf_threshold,
        landmark_method=landmark_method,  # type: ignore
    )

    stats = AlignmentStats(total_images=len(image_files))

    if num_workers == 1:
        # Sequential processing.
        for img_path in tqdm(image_files, desc="Aligning faces"):
            detected, aligned = process_image_file((img_path, output_dir, aligner, save_first_only))
            if detected > 0:
                stats.faces_detected += detected
                stats.faces_aligned += aligned
            else:
                stats.failed_images += 1
    else:
        # Parallel processing (careful: YOLO model may not be thread-safe; use processes).
        # Note: Creating aligner per-worker inside process_image_file to avoid pickling.
        # This is inefficient; better approach is to use multiprocessing.Pool with initializer.
        # For simplicity, we'll use sequential for now (parallel YOLO inference is complex).
        raise NotImplementedError("Parallel processing not yet implemented (YOLO model pickling issues)")

    print(f"\nAlignment complete:")
    print(f"  Total images processed: {stats.total_images}")
    print(f"  Faces detected: {stats.faces_detected}")
    print(f"  Faces aligned & saved: {stats.faces_aligned}")
    print(f"  Failed images: {stats.failed_images}")

    return stats


def align_lfw_dataset(
    lfw_root: Path,
    output_root: Path,
    yolo_model: str,
    device: str = "0",
) -> None:
    """Align LFW dataset preserving identity structure.

    LFW has structure: lfw-deepfunneled/lfw-deepfunneled/<identity>/<image>.jpg
    We want output: aligned_lfw/<identity>/<image>.jpg (all 112×112).

    Args:
        lfw_root: Path to lfw-deepfunneled/lfw-deepfunneled/.
        output_root: Output directory.
        yolo_model: YOLO model path.
        device: Device.
    """
    lfw_root = Path(lfw_root)
    output_root = Path(output_root)

    if not lfw_root.exists():
        raise FileNotFoundError(f"LFW root not found: {lfw_root}")

    # LFW structure: each subdirectory is an identity.
    identity_dirs = [d for d in lfw_root.iterdir() if d.is_dir()]
    print(f"Found {len(identity_dirs)} identities in LFW dataset")

    aligner = FaceAligner(
        yolo_model_path=yolo_model,
        device=device,
        conf_threshold=0.5,
        landmark_method="fallback",
    )

    total_images = 0
    total_aligned = 0

    for identity_dir in tqdm(identity_dirs, desc="Processing identities"):
        identity_name = identity_dir.name
        output_identity_dir = output_root / identity_name
        output_identity_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(identity_dir.glob("*.jpg")) + sorted(identity_dir.glob("*.png"))

        for img_path in image_files:
            total_images += 1
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                aligned_faces = aligner.process_image(img)
                if len(aligned_faces) > 0:
                    # Save first face (LFW images typically contain one face).
                    out_path = output_identity_dir / img_path.name
                    cv2.imwrite(str(out_path), aligned_faces[0])
                    total_aligned += 1

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print(f"\nLFW alignment complete:")
    print(f"  Total images: {total_images}")
    print(f"  Successfully aligned: {total_aligned}")
    print(f"  Output directory: {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch face alignment for ArcFace preprocessing.")

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory with face images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for aligned 112×112 faces",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="models/yolo11n.pt",
        help="Path to YOLO .pt weights (default: models/yolo11n.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device for YOLO inference (default: 0)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Face detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--landmark-method",
        type=str,
        default="fallback",
        choices=["cv2_face", "fallback"],
        help="Landmark detection method (default: fallback)",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Save all detected faces (default: save first only)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.jpg",
        help="File pattern for input images (default: **/*.jpg)",
    )
    parser.add_argument(
        "--lfw-mode",
        action="store_true",
        help="LFW dataset mode (preserves identity structure)",
    )

    args = parser.parse_args()

    if args.lfw_mode:
        align_lfw_dataset(
            lfw_root=Path(args.input_dir),
            output_root=Path(args.output_dir),
            yolo_model=args.yolo_model,
            device=args.device,
        )
    else:
        align_dataset(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            yolo_model=args.yolo_model,
            device=args.device,
            conf_threshold=args.conf,
            landmark_method=args.landmark_method,
            save_first_only=not args.save_all,
            num_workers=1,
            file_pattern=args.pattern,
        )


if __name__ == "__main__":
    main()
