"""Face verification inference using trained ArcFace model.

Once trained, we use only the backbone + embedding layer (discard ArcFace loss).
Compares faces using cosine similarity of L2-normalized embeddings.

Usage:
    # Single pair verification
    python inference_arcface.py --model checkpoint.pth --face1 img1.jpg --face2 img2.jpg

    # Batch verification from pairs file
    python inference_arcface.py --model checkpoint.pth --pairs pairs.txt --dataset-root lfw_aligned_112/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import create_arcface_model


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load trained ArcFace model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint (.pth file).
        device: Device to load model on.

    Returns:
        model: Loaded model in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model config from checkpoint
    args = checkpoint.get("args", {})
    num_classes = checkpoint["model_state_dict"]["metric_loss.weight"].shape[0]

    # Create model
    model = create_arcface_model(
        num_classes=num_classes,
        embedding_size=args.get("embedding_size", 512),
        backbone=args.get("backbone", "resnet18"),
        pretrained=False,  # Load from checkpoint
        scale=args.get("scale", 64.0),
        margin=args.get("margin", 0.5),
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Identities: {num_classes}")
    print(f"  Backbone: {args.get('backbone', 'resnet18')}")

    return model


def load_image(image_path: str | Path, size: int = 112) -> torch.Tensor:
    """Load and preprocess image for ArcFace inference.

    Args:
        image_path: Path to image file.
        size: Target size (default: 112 for aligned faces).

    Returns:
        tensor: (1, 3, size, size) preprocessed image tensor.
    """
    img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    return transform(img).unsqueeze(0)  # Add batch dimension


def get_embedding(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Extract L2-normalized embedding from face image.

    Args:
        model: Trained ArcFace model.
        image_tensor: (1, 3, 112, 112) preprocessed image.
        device: Device.

    Returns:
        embedding: (1, 512) L2-normalized feature vector.
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        # Inference mode: no labels, returns embeddings only
        embedding = model(image_tensor, labels=None)
    return embedding.cpu()


def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Compute cosine similarity between two embeddings.

    Args:
        emb1: (1, D) embedding vector (L2-normalized).
        emb2: (1, D) embedding vector (L2-normalized).

    Returns:
        similarity: Cosine similarity in range [-1, 1].
    """
    # Since embeddings are already L2-normalized, cosine = dot product
    similarity = torch.sum(emb1 * emb2, dim=1).item()
    return similarity


def verify_pair(
    model: torch.nn.Module,
    face1_path: str,
    face2_path: str,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[bool, float]:
    """Verify if two face images belong to the same person.

    Args:
        model: Trained ArcFace model.
        face1_path: Path to first face image.
        face2_path: Path to second face image.
        device: Device.
        threshold: Similarity threshold (default: 0.5).

    Returns:
        is_same: True if same person, False otherwise.
        similarity: Cosine similarity score.
    """
    # Load and preprocess images
    img1 = load_image(face1_path)
    img2 = load_image(face2_path)

    # Extract embeddings
    emb1 = get_embedding(model, img1, device)
    emb2 = get_embedding(model, img2, device)

    # Compute similarity
    similarity = cosine_similarity(emb1, emb2)
    is_same = similarity > threshold

    return is_same, similarity


def evaluate_lfw_pairs(
    model: torch.nn.Module,
    pairs_file: str,
    dataset_root: str,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[float, int, int]:
    """Evaluate on LFW pairs.txt format.

    Format:
        name1 idx1 name2 idx2  # Different people
        name idx1 idx2         # Same person

    Args:
        model: Trained model.
        pairs_file: Path to pairs.txt.
        dataset_root: Root directory of LFW dataset.
        device: Device.
        threshold: Verification threshold.

    Returns:
        accuracy: Verification accuracy.
        correct: Number of correct predictions.
        total: Total number of pairs.
    """
    pairs_file = Path(pairs_file)
    dataset_root = Path(dataset_root)

    correct = 0
    total = 0

    with open(pairs_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                # Same person: name idx1 idx2
                name, idx1, idx2 = parts
                path1 = dataset_root / name / f"{name}_{int(idx1):04d}.jpg"
                path2 = dataset_root / name / f"{name}_{int(idx2):04d}.jpg"
                ground_truth = True
            elif len(parts) == 4:
                # Different people: name1 idx1 name2 idx2
                name1, idx1, name2, idx2 = parts
                path1 = dataset_root / name1 / f"{name1}_{int(idx1):04d}.jpg"
                path2 = dataset_root / name2 / f"{name2}_{int(idx2):04d}.jpg"
                ground_truth = False
            else:
                continue

            # Verify
            is_same, similarity = verify_pair(model, str(path1), str(path2), device, threshold)

            if is_same == ground_truth:
                correct += 1
            total += 1

            if total % 100 == 0:
                print(f"Processed {total} pairs, Accuracy: {100.0 * correct / total:.2f}%")

    accuracy = 100.0 * correct / total
    return accuracy, correct, total


def main() -> None:
    parser = argparse.ArgumentParser(description="ArcFace face verification inference.")

    # Model
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Single pair verification
    parser.add_argument("--face1", type=str, help="Path to first face image")
    parser.add_argument("--face2", type=str, help="Path to second face image")

    # Batch evaluation (LFW pairs)
    parser.add_argument("--pairs", type=str, help="Path to LFW pairs.txt file")
    parser.add_argument("--dataset-root", type=str, help="Root directory of face dataset")

    # Threshold
    parser.add_argument("--threshold", type=float, default=0.5, help="Cosine similarity threshold")

    args = parser.parse_args()

    # Load model
    device = torch.device(args.device)
    model = load_model(args.model, device)

    # Single pair verification
    if args.face1 and args.face2:
        print(f"\nVerifying pair:")
        print(f"  Face 1: {args.face1}")
        print(f"  Face 2: {args.face2}")

        is_same, similarity = verify_pair(model, args.face1, args.face2, device, args.threshold)

        print(f"\nResult:")
        print(f"  Similarity: {similarity:.4f}")
        print(f"  Threshold: {args.threshold}")
        if is_same:
            print(f"  ✅ MATCH - Same person (similarity > threshold)")
        else:
            print(f"  ❌ NO MATCH - Different people (similarity ≤ threshold)")

    # Batch evaluation
    elif args.pairs and args.dataset_root:
        print(f"\nEvaluating on pairs file: {args.pairs}")
        print(f"Dataset root: {args.dataset_root}")
        print(f"Threshold: {args.threshold}\n")

        accuracy, correct, total = evaluate_lfw_pairs(
            model, args.pairs, args.dataset_root, device, args.threshold
        )

        print(f"\nResults:")
        print(f"  Correct: {correct}/{total}")
        print(f"  Accuracy: {accuracy:.2f}%")

    else:
        print("Error: Must provide either (--face1 and --face2) or (--pairs and --dataset-root)")
        parser.print_help()


if __name__ == "__main__":
    main()
