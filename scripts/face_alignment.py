"""Face alignment preprocessing for ArcFace.

ArcFace requires faces aligned to a canonical 112×112 template, not just resized.
Standard steps:
1. Detect face bbox (via YOLO)
2. Detect 5 facial landmarks (2 eyes, nose, 2 mouth corners)
3. Apply similarity transform to align & crop to 112×112

This module provides:
- Landmark detection (via cv2.face or fallback heuristic from bbox)
- Similarity transform alignment
- Integration with YOLOv11 detection pipeline
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

# ArcFace standard 112×112 template with canonical 5-point positions.
# Coordinates are (x, y) in normalized [0, 1] space, then scaled to 112×112.
# Reference: InsightFace arcface_torch preprocessing.
ARCFACE_DST_POINTS_112 = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)


def estimate_similarity_transform(
    src_pts: NDArray[np.float32],
    dst_pts: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Estimate 2×3 similarity transformation matrix (rotation + scale + translation).

    Args:
        src_pts: (N, 2) source points (detected landmarks in original image).
        dst_pts: (N, 2) destination points (canonical template positions).

    Returns:
        M: (2, 3) affine transformation matrix suitable for cv2.warpAffine.
    """
    if src_pts.shape != dst_pts.shape or src_pts.shape[1] != 2:
        raise ValueError(f"Points must be (N, 2), got src={src_pts.shape}, dst={dst_pts.shape}")

    # Use OpenCV's estimateAffinePartial2D for similarity transform (4 DOF: rotation, scale, tx, ty).
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
    if M is None:
        # Fallback to least-squares if RANSAC fails.
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    if M is None:
        raise RuntimeError("Failed to estimate similarity transform")
    return M


def align_face_from_landmarks(
    img: NDArray[np.uint8],
    landmarks: NDArray[np.float32],
    output_size: int = 112,
    dst_template: NDArray[np.float32] | None = None,
) -> NDArray[np.uint8]:
    """Align and crop a face to a canonical template using 5-point landmarks.

    Args:
        img: Input image (H, W, 3) BGR.
        landmarks: (5, 2) facial landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth].
        output_size: Output square image size (default 112 for ArcFace).
        dst_template: (5, 2) canonical landmark positions (default ArcFace 112×112 template).

    Returns:
        aligned: (output_size, output_size, 3) BGR aligned face.
    """
    if dst_template is None:
        dst_template = ARCFACE_DST_POINTS_112

    if landmarks.shape != (5, 2):
        raise ValueError(f"Expected (5, 2) landmarks, got {landmarks.shape}")

    M = estimate_similarity_transform(landmarks, dst_template)
    aligned = cv2.warpAffine(img, M, (output_size, output_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return aligned


def detect_landmarks_cv2_face(
    img: NDArray[np.uint8],
    face_bbox: Tuple[int, int, int, int],
) -> NDArray[np.float32] | None:
    """Detect 5 facial landmarks using OpenCV's face module (requires opencv-contrib-python).

    Args:
        img: Input image BGR.
        face_bbox: (x1, y1, x2, y2) bounding box.

    Returns:
        landmarks: (5, 2) array [left_eye, right_eye, nose, left_mouth, right_mouth], or None if detection fails.
    """
    try:
        # Lazy import to avoid hard dependency.
        facemark = cv2.face.createFacemarkLBF()
        model_path = cv2.data.haarcascades + "lbfmodel.yaml"  # type: ignore
        facemark.loadModel(model_path)
    except (AttributeError, cv2.error):
        # cv2.face not available or model not found.
        return None

    x1, y1, x2, y2 = face_bbox
    faces = np.array([[x1, y1, x2 - x1, y2 - y1]], dtype=np.int32)
    ok, landmarks = facemark.fit(img, faces)
    if not ok or len(landmarks) == 0:
        return None

    # landmarks[0] shape: (68, 2) for 68-point model; we need 5-point subset.
    # Standard subset for 5-point (approximate):
    #   left_eye = mean(landmarks[36:42]), right_eye = mean(landmarks[42:48]), nose = landmarks[30],
    #   left_mouth = landmarks[48], right_mouth = landmarks[54]
    pts = landmarks[0].reshape(-1, 2)
    if len(pts) < 68:
        return None

    left_eye = pts[36:42].mean(axis=0)
    right_eye = pts[42:48].mean(axis=0)
    nose = pts[30]
    left_mouth = pts[48]
    right_mouth = pts[54]

    return np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)


def detect_landmarks_fallback(
    face_bbox: Tuple[int, int, int, int],
) -> NDArray[np.float32]:
    """Fallback: estimate 5-point landmarks from bounding box geometry.

    This is a coarse approximation for when no landmark detector is available.
    Assumes frontal face with standard proportions.

    Args:
        face_bbox: (x1, y1, x2, y2).

    Returns:
        landmarks: (5, 2) estimated positions.
    """
    x1, y1, x2, y2 = face_bbox
    w = x2 - x1
    h = y2 - y1

    # Heuristic proportions (approximate frontal face).
    left_eye = np.array([x1 + 0.35 * w, y1 + 0.35 * h], dtype=np.float32)
    right_eye = np.array([x1 + 0.65 * w, y1 + 0.35 * h], dtype=np.float32)
    nose = np.array([x1 + 0.5 * w, y1 + 0.55 * h], dtype=np.float32)
    left_mouth = np.array([x1 + 0.38 * w, y1 + 0.75 * h], dtype=np.float32)
    right_mouth = np.array([x1 + 0.62 * w, y1 + 0.75 * h], dtype=np.float32)

    return np.array([left_eye, right_eye, nose, left_mouth, right_mouth], dtype=np.float32)


class FaceAligner:
    """High-level API for face detection + alignment.

    Integrates YOLO face detection with landmark-based alignment for ArcFace.
    """

    def __init__(
        self,
        yolo_model_path: str | Path,
        device: str = "cpu",
        conf_threshold: float = 0.5,
        landmark_method: Literal["cv2_face", "fallback"] = "fallback",
        output_size: int = 112,
    ):
        """Initialize face aligner.

        Args:
            yolo_model_path: Path to YOLO .pt weights (e.g., yolo11n.pt or face-specific).
            device: Device for YOLO inference ("cpu", "0", "cuda", etc.).
            conf_threshold: Confidence threshold for face detection.
            landmark_method: "cv2_face" (requires opencv-contrib-python) or "fallback" (bbox-based).
            output_size: Output aligned face size (default 112 for ArcFace).
        """
        from ultralytics import YOLO

        self.model = YOLO(str(yolo_model_path))
        self.device = device
        self.conf_threshold = conf_threshold
        self.landmark_method = landmark_method
        self.output_size = output_size

    def detect_faces(self, img: NDArray[np.uint8]) -> list[Tuple[int, int, int, int]]:
        """Detect faces in an image using YOLO.

        Args:
            img: Input image (H, W, 3) BGR.

        Returns:
            bboxes: List of (x1, y1, x2, y2) bounding boxes.
        """
        results = self.model.predict(img, device=self.device, conf=self.conf_threshold, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in boxes]

    def align_face(
        self,
        img: NDArray[np.uint8],
        face_bbox: Tuple[int, int, int, int],
    ) -> NDArray[np.uint8] | None:
        """Align a single detected face to canonical 112×112 template.

        Args:
            img: Input image BGR.
            face_bbox: (x1, y1, x2, y2).

        Returns:
            aligned: (output_size, output_size, 3) aligned face, or None if alignment fails.
        """
        if self.landmark_method == "cv2_face":
            landmarks = detect_landmarks_cv2_face(img, face_bbox)
            if landmarks is None:
                # Fallback if cv2.face fails.
                landmarks = detect_landmarks_fallback(face_bbox)
        else:
            landmarks = detect_landmarks_fallback(face_bbox)

        try:
            aligned = align_face_from_landmarks(img, landmarks, output_size=self.output_size)
            return aligned
        except (cv2.error, RuntimeError):
            return None

    def process_image(
        self,
        img: NDArray[np.uint8],
    ) -> list[NDArray[np.uint8]]:
        """Detect all faces in an image and return aligned crops.

        Args:
            img: Input image (H, W, 3) BGR.

        Returns:
            aligned_faces: List of (112, 112, 3) BGR aligned face crops.
        """
        bboxes = self.detect_faces(img)
        aligned_faces = []
        for bbox in bboxes:
            aligned = self.align_face(img, bbox)
            if aligned is not None:
                aligned_faces.append(aligned)
        return aligned_faces


def demo_alignment(image_path: str, output_dir: str = "runs/alignment", yolo_model: str = "yolo11n.pt", device: str = "0") -> None:
    """Demo: align faces in an image and save results.

    Args:
        image_path: Path to input image.
        output_dir: Directory to save aligned faces.
        yolo_model: Path to YOLO model weights.
        device: Device for inference.
    """
    import os

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    aligner = FaceAligner(
        yolo_model_path=yolo_model,
        device=device,
        landmark_method="fallback",
    )

    aligned_faces = aligner.process_image(img)
    print(f"Detected and aligned {len(aligned_faces)} face(s)")

    os.makedirs(output_dir, exist_ok=True)
    for i, face in enumerate(aligned_faces):
        out_path = os.path.join(output_dir, f"face_{i:03d}.jpg")
        cv2.imwrite(out_path, face)
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Face alignment demo for ArcFace preprocessing.")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="runs/alignment", help="Output directory (default: runs/alignment)")
    parser.add_argument("--yolo-model", type=str, default="yolo11n.pt", help="YOLO model path (default: yolo11n.pt)")
    parser.add_argument("--device", type=str, default="0", help="Device: 0, cpu, etc. (default: 0)")

    args = parser.parse_args()
    demo_alignment(args.image_path, args.output_dir, args.yolo_model, args.device)
