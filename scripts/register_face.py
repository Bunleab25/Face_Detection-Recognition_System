"""Phase 1: Register identities from single photos (1 photo per person)."""

import sys
from pathlib import Path
import pickle

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

sys.path.append(str(Path(__file__).parent))
from model import FaceRecognitionModel
from face_alignment import FaceAligner


class FaceDatabase:
    """Simple face database using embeddings."""
    
    def __init__(self):
        self.embeddings = {}  # {name: embedding_vector}
        self.names = []
    
    def register(self, name, embedding):
        """Register a person with their embedding."""
        self.embeddings[name] = embedding
        if name not in self.names:
            self.names.append(name)
    
    def save(self, path):
        """Save database to file."""
        data = {
            'embeddings': self.embeddings,
            'names': self.names
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"✓ Database saved to {path}")
    
    def load(self, path):
        """Load database from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.embeddings = data['embeddings']
        self.names = data['names']
        print(f"✓ Loaded {len(self.names)} identities from {path}")


def register_from_image(image_path, name, model, aligner, device='cuda'):
    """Register a person from a single image.
    
    Args:
        image_path: Path to image file.
        name: Name of the person.
        model: Trained ArcFace model.
        aligner: FaceAligner instance.
        device: 'cuda' or 'cpu'.
        
    Returns:
        embedding: 512-d normalized embedding vector or None if failed.
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    # Detect faces
    bboxes = aligner.detect_faces(img)
    
    if len(bboxes) == 0:
        print(f"Error: No face detected in {image_path}")
        return None
    
    if len(bboxes) > 1:
        print(f"Warning: Multiple faces detected, using first face")
    
    # Align face (CRITICAL for ArcFace accuracy)
    bbox = bboxes[0]
    aligned_face = aligner.align_face(img, bbox)
    
    if aligned_face is None:
        print(f"Error: Failed to align face in {image_path}")
        return None
    
    # Convert to RGB and prepare for model
    aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    face_tensor = transform(aligned_face).unsqueeze(0).to(device)
    
    # Get embedding
    model.eval()
    with torch.no_grad():
        embedding = model(face_tensor)  # (1, 512)
        embedding = F.normalize(embedding, p=2, dim=1)  # L2 normalize
        embedding = embedding.cpu().numpy().flatten()  # (512,)
    
    return embedding


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Register faces to database (1 photo each)')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image file')
    parser.add_argument('--name', type=str, required=True,
                       help='Name of the person')
    parser.add_argument('--database', type=str, default='face_database.pkl',
                       help='Path to database file (will be created if not exists)')
    parser.add_argument('--model', type=str, default='runs/arcface_train/arcface_resnet50_epoch_100.pth',
                       help='Path to trained ArcFace model')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PHASE 1: REGISTRATION")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading ArcFace model...")
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=False)
    
    # Detect backbone
    if 'resnet50' in args.model:
        backbone = 'resnet50'
    elif 'resnet34' in args.model:
        backbone = 'resnet34'
    else:
        backbone = 'resnet18'
    
    model = FaceRecognitionModel(
        backbone=backbone,
        num_classes=checkpoint.get('num_classes', 5611),
        embedding_size=512,
        scale=64.0,
        margin=0.5,
        pretrained=False,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    model.eval()
    print(f"   ✓ Model loaded: {backbone}")
    
    # Initialize aligner
    print("\n2. Initializing face aligner...")
    aligner = FaceAligner(
        yolo_model_path='models/yolo11n.pt',
        device=args.device,
        conf_threshold=0.5,
    )
    print("   ✓ Aligner ready")
    
    # Load or create database
    print("\n3. Loading database...")
    db = FaceDatabase()
    db_path = Path(args.database)
    
    if db_path.exists():
        db.load(str(db_path))
    else:
        print(f"   Creating new database: {args.database}")
    
    # Register new person
    print(f"\n4. Registering: {args.name}")
    print(f"   Image: {args.image}")
    
    embedding = register_from_image(args.image, args.name, model, aligner, args.device)
    
    if embedding is not None:
        db.register(args.name, embedding)
        print(f"\n   ✅ Successfully registered {args.name}")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   L2 norm: {np.linalg.norm(embedding):.4f}")
        
        # Save database
        print(f"\n5. Saving database...")
        db.save(str(db_path))
        
        print("\n" + "=" * 60)
        print("REGISTRATION COMPLETE")
        print("=" * 60)
        print(f"Total identities in database: {len(db.names)}")
        print(f"Names: {', '.join(db.names)}")
        print(f"\nNext step: Run real-time recognition:")
        print(f"  python scripts/realtime_arcface.py --database {args.database}")
    else:
        print("\n❌ Registration failed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
