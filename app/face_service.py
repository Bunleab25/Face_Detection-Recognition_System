from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO  # type: ignore

from app.config import settings
from app.model import FaceRecognitionModel


class NoFaceDetectedError(RuntimeError):
    pass


@dataclass
class FaceDetection:
    box: Tuple[int, int, int, int]  # x1,y1,x2,y2
    conf: float

class FaceService:
    def __init__(self) -> None:
        self.device = torch.device(
            settings.device if torch.cuda.is_available() else "cpu"
        )
        print(f"Initializing FaceService on {self.device}...")

        # Load YOLO (use your trained weights via YOLO_MODEL_PATH)
        print(f"Loading YOLO weights from {settings.yolo_model_path}")
        self.detector = YOLO(settings.yolo_model_path)

        # Load ArcFace (use your trained checkpoint via ARCFACE_CKPT_PATH)
        self.recognizer = self._load_arcface(settings.arcface_ckpt_path).to(self.device).eval()

        # Preprocessing
        self.preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((settings.input_size, settings.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _load_arcface(self, ckpt_path: str) -> FaceRecognitionModel:
        if not ckpt_path:
            raise FileNotFoundError(
                "ArcFace checkpoint path is empty. Set ARCFACE_CKPT_PATH."
            )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"ArcFace checkpoint not found: {ckpt_path}. Set ARCFACE_CKPT_PATH."
            )

        print(f"Loading ArcFace weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get("model_state_dict")
                or checkpoint.get("state_dict")
                or checkpoint
            )
        else:
            raise ValueError("Unsupported ArcFace checkpoint format.")

        # Strip DDP prefix if present
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        # Infer num_classes if possible (ArcFace head weight is metric_loss.weight)
        num_classes = None
        for key, value in state_dict.items():
            if key.endswith("metric_loss.weight") and hasattr(value, "shape"):
                num_classes = int(value.shape[0])
                break
        if num_classes is None:
            num_classes = 1

        model = FaceRecognitionModel(backbone="resnet50", num_classes=num_classes)
        model.load_state_dict(state_dict, strict=False)
        return model

    def get_embedding(
        self, image_bytes: bytes
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")

        det = self._detect_best_face(img)
        if det is None:
            raise NoFaceDetectedError("No face detected")

        x1, y1, x2, y2 = det.box
        face_bgr = img[y1:y2, x1:x2]
        if face_bgr.size == 0:
            raise NoFaceDetectedError("Detected face crop is empty")

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_tensor = self.preprocess(face_rgb)  # type: ignore[arg-type]
        if not isinstance(face_tensor, torch.Tensor):
            raise RuntimeError("Preprocess pipeline did not return a torch.Tensor")
        face_tensor = face_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.recognizer(face_tensor)
            embedding = F.normalize(embedding, dim=1)
        embedding_np = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)

        return embedding_np, det.box, det.conf

    def _detect_best_face(self, frame_bgr: np.ndarray) -> Optional[FaceDetection]:
        results = self.detector(frame_bgr, verbose=False)
        best: Optional[FaceDetection] = None

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0].item())
                if conf < float(settings.yolo_conf):
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                h, w = frame_bgr.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                candidate = FaceDetection(box=(x1, y1, x2, y2), conf=conf)
                if best is None or candidate.conf > best.conf:
                    best = candidate

        return best
