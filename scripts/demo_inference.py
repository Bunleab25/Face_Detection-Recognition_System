"""Quick demo of ArcFace inference pipeline (using untrained weights for demonstration)."""

import sys
from pathlib import Path

import torch
import cv2
import numpy as np
from torchvision import transforms

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from model import FaceRecognitionModel


def load_and_preprocess_image(image_path: str, size: int = 112):
    """Load and preprocess a face image for ArcFace model.
    
    Args:
        image_path: Path to image file (should be pre-aligned 112x112 face).
        size: Expected size (default 112).
        
    Returns:
        tensor: (1, 3, 112, 112) preprocessed tensor ready for model.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize if needed (aligned faces should already be 112x112)
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size))
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    tensor = transform(img)
    return tensor.unsqueeze(0)  # Add batch dimension


def get_embedding(model, image_tensor, device='cpu'):
    """Extract L2-normalized embedding from face image.
    
    Args:
        model: FaceRecognitionModel in eval mode.
        image_tensor: (1, 3, 112, 112) preprocessed face image.
        device: 'cpu' or 'cuda'.
        
    Returns:
        embedding: (512,) L2-normalized embedding vector.
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        embedding = model(image_tensor)  # (1, 512)
        embedding = embedding.squeeze(0)  # (512,)
    return embedding.cpu().numpy()


def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings.
    
    For L2-normalized embeddings, this is simply the dot product.
    
    Args:
        emb1, emb2: (512,) L2-normalized embedding vectors.
        
    Returns:
        similarity: Scalar in [-1, 1], where 1 = identical, 0 = orthogonal, -1 = opposite.
    """
    return float(np.dot(emb1, emb2))


def main():
    device = 'cpu'  # Use CPU for demo
    
    # Create model with random weights (for demo)
    print("Creating ArcFace model (untrained weights for demonstration)...")
    model = FaceRecognitionModel(
        backbone='resnet18',
        num_classes=5611,  # LFW identity count
        embedding_size=512,
        scale=64.0,
        margin=0.5,
        pretrained=False,  # Random weights for demo
    )
    model = model.to(device)
    model.eval()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters\n")
    
    # Demo 1: Single face embedding
    print("=" * 60)
    print("DEMO 1: Extracting embedding from a single face")
    print("=" * 60)
    
    aligned_face_dir = Path("LFW_datasets/lfw_aligned_112")
    if not aligned_face_dir.exists():
        print(f"Error: {aligned_face_dir} not found. Run preprocessing first.")
        return
    
    # Find first available identity with images
    sample_identity = None
    for identity_dir in sorted(aligned_face_dir.iterdir()):
        if identity_dir.is_dir():
            images = list(identity_dir.glob("*.jpg"))
            if len(images) >= 2:
                sample_identity = identity_dir
                break
    
    if sample_identity is None:
        print("Error: No identity with at least 2 images found.")
        return
    
    images = sorted(sample_identity.glob("*.jpg"))
    img1_path = images[0]
    img2_path = images[1]
    
    print(f"Identity: {sample_identity.name}")
    print(f"Image 1: {img1_path.name}")
    print(f"Image 2: {img2_path.name}\n")
    
    # Extract embeddings
    tensor1 = load_and_preprocess_image(str(img1_path))
    tensor2 = load_and_preprocess_image(str(img2_path))
    
    emb1 = get_embedding(model, tensor1, device)
    emb2 = get_embedding(model, tensor2, device)
    
    print(f"Embedding 1 shape: {emb1.shape}")
    print(f"Embedding 1 L2 norm: {np.linalg.norm(emb1):.4f}")
    print(f"Embedding 2 shape: {emb2.shape}")
    print(f"Embedding 2 L2 norm: {np.linalg.norm(emb2):.4f}\n")
    
    # Demo 2: Face verification (same person)
    print("=" * 60)
    print("DEMO 2: Face verification (same identity)")
    print("=" * 60)
    
    similarity_same = cosine_similarity(emb1, emb2)
    print(f"Cosine similarity: {similarity_same:.4f}")
    print(f"Note: With untrained weights, similarity is random.")
    print(f"      After training, same person typically scores > 0.5\n")
    
    # Demo 3: Face verification (different person)
    print("=" * 60)
    print("DEMO 3: Face verification (different identity)")
    print("=" * 60)
    
    # Find another identity
    other_identity = None
    for identity_dir in sorted(aligned_face_dir.iterdir()):
        if identity_dir.is_dir() and identity_dir != sample_identity:
            images = list(identity_dir.glob("*.jpg"))
            if len(images) >= 1:
                other_identity = identity_dir
                break
    
    if other_identity:
        img3_path = list(other_identity.glob("*.jpg"))[0]
        print(f"Comparing {sample_identity.name} vs {other_identity.name}")
        print(f"Image 1: {img1_path.name}")
        print(f"Image 3: {img3_path.name}\n")
        
        tensor3 = load_and_preprocess_image(str(img3_path))
        emb3 = get_embedding(model, tensor3, device)
        
        similarity_diff = cosine_similarity(emb1, emb3)
        print(f"Cosine similarity: {similarity_diff:.4f}")
        print(f"Note: With untrained weights, similarity is random.")
        print(f"      After training, different people typically score < 0.3\n")
    
    # Demo 4: Verification decision with threshold
    print("=" * 60)
    print("DEMO 4: Verification decision with threshold")
    print("=" * 60)
    
    threshold = 0.4  # Typical threshold after training
    print(f"Using threshold: {threshold}\n")
    
    def verify(sim, thresh):
        return "MATCH" if sim > thresh else "NO MATCH"
    
    print(f"Same person:      {similarity_same:.4f} → {verify(similarity_same, threshold)}")
    if other_identity:
        print(f"Different people: {similarity_diff:.4f} → {verify(similarity_diff, threshold)}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Train the model on GPU for 50 epochs (~1-2 hours):")
    print("   python scripts/train_arcface.py --data-dir LFW_datasets/lfw_aligned_112 \\")
    print("          --backbone resnet18 --epochs 50 --device cuda\n")
    print("2. Run inference with trained model:")
    print("   python scripts/inference_arcface.py --model runs/arcface_train/arcface_resnet18_epoch_50.pth \\")
    print("          --face1 face1.jpg --face2 face2.jpg\n")
    print("3. Expected LFW accuracy after training: ~98-99%")


if __name__ == "__main__":
    main()
