"""Real-time face recognition from webcam using trained ArcFace model."""

import sys
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from torchvision import transforms

sys.path.append(str(Path(__file__).parent))
from model import FaceRecognitionModel
from face_alignment import FaceAligner


class FaceRecognitionSystem:
    """Real-time face recognition system."""
    
    def __init__(self, model_path, identity_db_path, device='cuda', threshold=0.95):
        """Initialize face recognition system.
        
        Args:
            model_path: Path to trained ArcFace model.
            identity_db_path: Path to directory with known faces (identity folders).
            device: 'cuda' or 'cpu'.
            threshold: Similarity threshold for verification.
        """
        self.device = device
        self.threshold = threshold
        
        # Load model
        print("Loading model...")
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
        
        # Transform for preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # Initialize face aligner
        self.aligner = FaceAligner(
            yolo_model_path='yolo11n.pt',
            device=device,
            conf_threshold=0.5,
        )
        
        # Load identity database
        print("Loading identity database...")
        self.identity_db = {}
        self.load_identity_database(identity_db_path)
        print(f"✓ Loaded {len(self.identity_db)} known identities\n")
    
    def load_identity_database(self, db_path):
        """Load embeddings for all known identities."""
        db_path = Path(db_path)
        
        for identity_dir in db_path.iterdir():
            if not identity_dir.is_dir():
                continue
            
            # Get first image for this identity
            images = list(identity_dir.glob("*.jpg"))
            if not images:
                continue
            
            # Load and get embedding
            img = cv2.imread(str(images[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize if needed (should be 112x112 if pre-aligned)
            if img.shape[:2] != (112, 112):
                img = cv2.resize(img, (112, 112))
            
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(tensor).squeeze(0).cpu().numpy()
            
            self.identity_db[identity_dir.name] = embedding
    
    def get_embedding(self, face_img):
        """Extract embedding from face image (already aligned 112x112)."""
        # Convert to RGB if needed
        if len(face_img.shape) == 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        elif face_img.shape[2] == 4:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGRA2RGB)
        else:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Resize to 112x112
        if face_img.shape[:2] != (112, 112):
            face_img = cv2.resize(face_img, (112, 112))
        
        tensor = self.transform(face_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(tensor).squeeze(0).cpu().numpy()
        
        return embedding
    
    def recognize_face(self, face_img):
        """Recognize a face against the database.
        
        Args:
            face_img: Aligned face image (112x112).
            
        Returns:
            (name, similarity) or (None, 0) if no match.
        """
        embedding = self.get_embedding(face_img)
        
        best_match = None
        best_sim = -1
        
        for name, db_embedding in self.identity_db.items():
            sim = float(np.dot(embedding, db_embedding))
            if sim > best_sim:
                best_sim = sim
                best_match = name
        
        if best_sim >= self.threshold:
            return best_match, best_sim
        else:
            return "Unknown", best_sim
    
    def process_frame(self, frame):
        """Detect and recognize faces in a frame.
        
        Args:
            frame: BGR image from webcam.
            
        Returns:
            annotated_frame: Frame with bounding boxes and labels.
            results: List of (bbox, name, similarity) tuples.
        """
        # Detect faces
        bboxes = self.aligner.detect_faces(frame)
        
        results = []
        annotated = frame.copy()
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            
            # Align face
            aligned_face = self.aligner.align_face(frame, bbox)
            
            if aligned_face is not None:
                # Recognize
                name, similarity = self.recognize_face(aligned_face)
                results.append((bbox, name, similarity))
                
                # Draw bounding box
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{name} ({similarity:.2f})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated, results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time face recognition')
    parser.add_argument('--model', type=str, default='runs/arcface_train/arcface_resnet50_epoch_100.pth',
                       help='Path to trained model')
    parser.add_argument('--database', type=str, default='LFW_datasets/lfw_aligned_112',
                       help='Path to identity database (folder with identity subfolders)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    parser.add_argument('--threshold', type=float, default=0.95,
                       help='Recognition threshold')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera index (0 for default webcam)')
    
    args = parser.parse_args()
    
    # Initialize system
    print("Initializing face recognition system...")
    system = FaceRecognitionSystem(
        model_path=args.model,
        identity_db_path=args.database,
        device=args.device,
        threshold=args.threshold,
    )
    
    # Open webcam
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("\n" + "=" * 60)
    print("REAL-TIME FACE RECOGNITION")
    print("=" * 60)
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    print("=" * 60 + "\n")
    
    frame_count = 0
    fps_start = time.time()
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break
        
        # Process frame
        annotated, results = system.process_frame(frame)
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - fps_start)
            fps_start = time.time()
        
        # Draw FPS
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw face count
        cv2.putText(annotated, f"Faces: {len(results)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Face Recognition', annotated)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"recognition_{int(time.time())}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nShutdown complete")


if __name__ == "__main__":
    main()
