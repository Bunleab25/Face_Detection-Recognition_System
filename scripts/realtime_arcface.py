"""Phase 2: Real-time face recognition using webcam."""

import sys
import time
import pickle
from pathlib import Path

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

sys.path.append(str(Path(__file__).parent))
from model import FaceRecognitionModel
from face_alignment import FaceAligner


class RealtimeFaceRecognizer:
    """Real-time face recognition system."""
    
    def __init__(self, model_path, database_path, device='cuda', threshold=0.5):
        """Initialize recognizer.
        
        Args:
            model_path: Path to trained ArcFace model.
            database_path: Path to pickled database with registered embeddings.
            device: 'cuda' or 'cpu'.
            threshold: Similarity threshold (0.5 recommended for ArcFace).
        """
        self.device = device
        self.threshold = threshold
        
        # Load model
        print("Loading ArcFace model...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Detect backbone
        if 'resnet50' in str(model_path):
            backbone = 'resnet50'
        elif 'resnet34' in str(model_path):
            backbone = 'resnet34'
        else:
            backbone = 'resnet18'
        
        self.model = FaceRecognitionModel(
            backbone=backbone,
            num_classes=checkpoint.get('num_classes', 5611),
            embedding_size=512,
            scale=64.0,
            margin=0.5,
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        print(f"✓ Model loaded: {backbone}")
        
        # Load database
        print("Loading face database...")
        with open(database_path, 'rb') as f:
            data = pickle.load(f)
        
        self.known_embeddings = data['embeddings']
        self.known_names = data['names']
        print(f"✓ Loaded {len(self.known_names)} known identities")
        
        # Initialize face aligner
        print("Initializing face detector & aligner...")
        self.aligner = FaceAligner(
            yolo_model_path='models/yolo11n.pt',
            device=device,
            conf_threshold=0.5,
        )
        print("✓ Aligner ready")
        
        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def get_embedding(self, aligned_face):
        """Get normalized embedding from aligned face.
        
        Args:
            aligned_face: BGR image (112x112) from aligner.
            
        Returns:
            embedding: (512,) normalized numpy array.
        """
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        
        # Transform and add batch dimension
        face_tensor = self.transform(face_rgb).unsqueeze(0).to(self.device)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.model(face_tensor)  # (1, 512)
            embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalize
            embedding = embedding.cpu().numpy().flatten()  # (512,)
        
        return embedding
    
    def recognize(self, live_embedding):
        """Match live embedding against database.
        
        Args:
            live_embedding: (512,) normalized embedding from webcam.
            
        Returns:
            (name, score): Best match name and similarity score.
        """
        best_score = -1
        best_name = "Unknown"
        
        for name, saved_embedding in self.known_embeddings.items():
            # Cosine similarity (dot product for normalized vectors)
            score = np.dot(live_embedding, saved_embedding)
            
            if score > best_score:
                best_score = score
                best_name = name
        
        # Apply threshold
        if best_score < self.threshold:
            return "Unknown", best_score
        
        return best_name, best_score
    
    def process_frame(self, frame):
        """Detect and recognize all faces in frame.
        
        Args:
            frame: BGR image from webcam.
            
        Returns:
            annotated_frame: Frame with boxes and labels.
            results: List of (bbox, name, score) tuples.
        """
        # 1. Detect faces with YOLOv11
        bboxes = self.aligner.detect_faces(frame)
        
        results = []
        annotated = frame.copy()
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            
            # 2. Align face (CRITICAL: proper alignment to 112x112)
            aligned_face = self.aligner.align_face(frame, bbox)
            
            if aligned_face is None:
                continue
            
            # 3. Get embedding
            live_embedding = self.get_embedding(aligned_face)
            
            # 4. Recognize
            name, score = self.recognize(live_embedding)
            
            results.append((bbox, name, score))
            
            # 5. Draw visualization
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Label with name and score
            label = f"{name} ({score:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Background for text
            cv2.rectangle(annotated, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         color, -1)
            
            # Text
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated, results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time face recognition')
    parser.add_argument('--database', type=str, default='face_database.pkl',
                       help='Path to registered faces database')
    parser.add_argument('--model', type=str, default='runs/arcface_train/arcface_resnet50_epoch_100.pth',
                       help='Path to trained ArcFace model')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Recognition threshold (0.5 recommended for ArcFace)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("PHASE 2: REAL-TIME RECOGNITION")
    print("=" * 60)
    
    # Check if database exists
    if not Path(args.database).exists():
        print(f"\n❌ Database not found: {args.database}")
        print("\nPlease register faces first:")
        print(f"  python scripts/register_face.py --image photo.jpg --name 'Your Name'")
        return 1
    
    # Initialize recognizer
    print()
    recognizer = RealtimeFaceRecognizer(
        model_path=args.model,
        database_path=args.database,
        device=args.device,
        threshold=args.threshold,
    )
    
    # Open webcam
    print(f"\nOpening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return 1
    
    print("✓ Camera opened")
    
    print("\n" + "=" * 60)
    print("CONTROLS:")
    print("=" * 60)
    print("  Press 'q' to quit")
    print("  Press 's' to save screenshot")
    print("=" * 60)
    print(f"\nKnown identities: {', '.join(recognizer.known_names)}")
    print(f"Threshold: {args.threshold}")
    print("\nStarting recognition...\n")
    
    # FPS tracking
    frame_count = 0
    fps_start = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break
        
        # Process frame
        annotated, results = recognizer.process_frame(frame)
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 10 == 0:
            elapsed = time.time() - fps_start
            fps = 10 / elapsed if elapsed > 0 else 0
            fps_start = time.time()
        
        # Draw FPS
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw face count
        cv2.putText(annotated, f"Faces: {len(results)}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('ArcFace Real-Time Recognition', annotated)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"Screenshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Shutdown complete")
    
    return 0


if __name__ == "__main__":
    exit(main())
