"""
Real-Time Face Recognition System
Loads the face database and recognizes people in real-time via webcam

Usage: python main_system.py --database face_database.npy
"""

import cv2
import torch
import numpy as np
import os
import time
from typing import cast

from ultralytics import YOLO  # type: ignore
from torchvision import transforms
from scripts.model import FaceRecognitionModel

# --- CONFIGURATION ---
DB_FILE = "face_database.npy"
MODEL_PATH = "runs/arcface_train/arcface_resnet50_epoch_100.pth"
THRESHOLD = 0.5  # Adjust this: 0.4 (loose) to 0.7 (strict)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _load_database(db_file: str) -> dict:
    if not os.path.exists(db_file):
        return {}
    database = np.load(db_file, allow_pickle=True).item()
    return database if isinstance(database, dict) else {}


def _save_database(db_file: str, database: dict) -> None:
    np.save(db_file, np.array(database, dtype=object), allow_pickle=True)


def main(db_file=DB_FILE, threshold=THRESHOLD, camera_id=0, model_path=MODEL_PATH):
    # 1. Load Database (or create an empty one)
    known_faces = _load_database(db_file)
    print("Loading database...")
    print(f"✓ Loaded {len(known_faces)} people: {list(known_faces.keys())}")

    # 2. Load Models
    print("Loading detection model...")
    detector = YOLO("models/yolo11n.pt")
    
    print("Loading recognition model...")
    # Load checkpoint first to get num_classes
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        
        # Get num_classes from checkpoint
        if 'model_state_dict' in checkpoint:
            num_classes = checkpoint['model_state_dict']['metric_loss.weight'].shape[0]
        else:
            num_classes = checkpoint['metric_loss.weight'].shape[0]
        
        recognizer = FaceRecognitionModel(backbone='resnet50', num_classes=num_classes)
        
        if 'model_state_dict' in checkpoint:
            recognizer.load_state_dict(checkpoint['model_state_dict'])
        else:
            recognizer.load_state_dict(checkpoint)
    else:
        recognizer = FaceRecognitionModel(backbone='resnet50', num_classes=5749)
        
    recognizer.eval().to(DEVICE)
    print(f"✓ Model loaded on {DEVICE}")

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 3. Start Webcam
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {camera_id}")
        return
    
    print("✓ Camera opened")
    print(f"\n{'='*60}")
    print(f"  SMART AGRICULTURE SECURITY SYSTEM")
    print(f"{'='*60}")
    print(f"Known identities: {list(known_faces.keys())}")
    print(f"Similarity threshold: {threshold}")
    print(f"Device: {DEVICE}")
    print(f"\nControls:")
    print(f"  'q' - Quit")
    print(f"  's' - Snap + enroll new person")
    print(f"  'p' - Save screenshot")
    print(f"{'='*60}\n")
    print("Starting recognition...")

    fps_start_time = time.time()
    fps_counter = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        # Calculate FPS
        fps_counter += 1
        if (time.time() - fps_start_time) > 1:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()

        # Detect
        results = detector(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Crop Face
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0: 
                    continue

                # Get Live Embedding
                face_tensor = cast(torch.Tensor, preprocess(face_img)).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    live_emb = recognizer(face_tensor)
                    live_emb = torch.nn.functional.normalize(live_emb, dim=1).cpu().numpy()

                # Compare against ALL enrolled faces
                best_score = -1
                best_name = "Unknown"

                for name, saved_emb in known_faces.items():
                    # Cosine Similarity Calculation
                    score = np.dot(live_emb, saved_emb.T).item()
                    
                    if score > best_score:
                        best_score = score
                        best_name = name

                # Visualization
                if best_score > threshold:
                    color = (0, 255, 0)  # Green for match
                    label = f"{best_name} ({best_score:.2f})"
                else:
                    color = (0, 0, 255)  # Red for unknown
                    label = f"Unknown ({best_score:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Display FPS
        cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, "S: Enroll", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "P: Screenshot", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Smart Agriculture Security", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Screenshot saved: {filename}")
        elif key == ord('s'):
            snapped = frame.copy()
            cv2.imshow("Smart Agriculture Security", snapped)
            cv2.waitKey(100)

            print("\n" + "-" * 30)
            name = input("Enter name for this person: ").strip()
            if not name:
                print("Name cannot be empty.")
                continue

            # Detect best face in the snapped frame
            results = detector(snapped, verbose=False)
            best_conf = -1.0
            best_face_img = None
            best_box = None

            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0].item())
                    if conf < 0.5:
                        continue
                    if conf > best_conf:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        face_img = snapped[y1:y2, x1:x2]
                        if face_img.size == 0:
                            continue
                        best_conf = conf
                        best_face_img = face_img
                        best_box = (x1, y1, x2, y2)

            if best_face_img is None:
                print("\n[ERROR] No face detected! Please try again.")
                continue

            face_tensor = cast(torch.Tensor, preprocess(best_face_img)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                embedding = recognizer(face_tensor)
                embedding = torch.nn.functional.normalize(embedding, dim=1).cpu().numpy()

            known_faces[name] = embedding
            _save_database(db_file, known_faces)
            print(f"\n[SUCCESS] User '{name}' saved to {db_file}")
            print(f"Total users in database: {len(known_faces)}")
            print(f"Face box: {best_box} (conf={best_conf:.2f})")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Shutdown complete")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time face recognition system")
    parser.add_argument("--database", type=str, default=DB_FILE, 
                       help="Path to face database (.npy file)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, 
                       help="Similarity threshold (0.4-0.7)")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera ID (default: 0)")
    parser.add_argument("--device", type=str, default=DEVICE,
                       choices=['cuda', 'cpu'], help="Device to use")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                       help="Path to trained ArcFace checkpoint (.pth)")
    
    args = parser.parse_args()
    
    DEVICE = args.device
    
    main(db_file=args.database, threshold=args.threshold, camera_id=args.camera, model_path=args.model)
