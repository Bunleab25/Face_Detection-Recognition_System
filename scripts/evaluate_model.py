"""Quick evaluation of trained ArcFace model on random face pairs."""

import sys
import random
from pathlib import Path

import torch
import cv2
import numpy as np
from torchvision import transforms

sys.path.append(str(Path(__file__).parent))
from model import FaceRecognitionModel


def load_model(checkpoint_path, device='cpu'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get num_classes from checkpoint or use default LFW value
    num_classes = checkpoint.get('num_classes', 5611)
    
    # Detect backbone from checkpoint path
    if 'resnet50' in str(checkpoint_path):
        backbone = 'resnet50'
    elif 'resnet34' in str(checkpoint_path):
        backbone = 'resnet34'
    else:
        backbone = 'resnet18'
    
    model = FaceRecognitionModel(
        backbone=backbone,
        num_classes=num_classes,
        embedding_size=512,
        scale=64.0,
        margin=0.5,
        pretrained=False,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, backbone


def get_embedding(model, image_path, device='cpu'):
    """Extract embedding from face image."""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(tensor).squeeze(0).cpu().numpy()
    
    return embedding


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "runs/arcface_train/arcface_resnet50_epoch_100.pth"
    dataset_root = Path("LFW_datasets/lfw_aligned_112")
    
    print("Loading trained model...")
    model, backbone = load_model(model_path, device)
    print(f"✓ Model loaded from {model_path}")
    print(f"  Backbone: {backbone}")
    print(f"  Device: {device}\n")
    
    # Get all identities with at least 2 images
    identities_with_pairs = []
    all_identities = []
    
    for identity_dir in dataset_root.iterdir():
        if identity_dir.is_dir():
            images = list(identity_dir.glob("*.jpg"))
            if len(images) >= 2:
                identities_with_pairs.append(identity_dir)
            if len(images) >= 1:
                all_identities.append(identity_dir)
    
    print(f"Found {len(identities_with_pairs)} identities with 2+ images")
    print(f"Found {len(all_identities)} total identities\n")
    
    # Test 1: Same person pairs
    print("=" * 60)
    print("TEST 1: Same Person Verification (10 random pairs)")
    print("=" * 60)
    
    same_person_sims = []
    for i in range(10):
        identity = random.choice(identities_with_pairs)
        images = list(identity.glob("*.jpg"))
        img1, img2 = random.sample(images, 2)
        
        emb1 = get_embedding(model, img1, device)
        emb2 = get_embedding(model, img2, device)
        sim = float(np.dot(emb1, emb2))
        same_person_sims.append(sim)
        
        print(f"{i+1}. {identity.name[:30]:30s} → {sim:.4f}")
    
    avg_same = np.mean(same_person_sims)
    print(f"\nAverage similarity (same person): {avg_same:.4f}")
    print(f"Min: {min(same_person_sims):.4f}, Max: {max(same_person_sims):.4f}\n")
    
    # Test 2: Different person pairs
    print("=" * 60)
    print("TEST 2: Different Person Verification (10 random pairs)")
    print("=" * 60)
    
    diff_person_sims = []
    for i in range(10):
        person1, person2 = random.sample(all_identities, 2)
        img1 = random.choice(list(person1.glob("*.jpg")))
        img2 = random.choice(list(person2.glob("*.jpg")))
        
        emb1 = get_embedding(model, img1, device)
        emb2 = get_embedding(model, img2, device)
        sim = float(np.dot(emb1, emb2))
        diff_person_sims.append(sim)
        
        print(f"{i+1}. {person1.name[:15]:15s} vs {person2.name[:15]:15s} → {sim:.4f}")
    
    avg_diff = np.mean(diff_person_sims)
    print(f"\nAverage similarity (different people): {avg_diff:.4f}")
    print(f"Min: {min(diff_person_sims):.4f}, Max: {max(diff_person_sims):.4f}\n")
    
    # Suggest threshold
    print("=" * 60)
    print("THRESHOLD RECOMMENDATION")
    print("=" * 60)
    
    suggested_threshold = (avg_same + avg_diff) / 2
    print(f"Suggested threshold: {suggested_threshold:.4f}")
    print(f"  (midpoint between same={avg_same:.4f} and diff={avg_diff:.4f})\n")
    
    # Test accuracy with threshold
    correct_same = sum(s > suggested_threshold for s in same_person_sims)
    correct_diff = sum(s <= suggested_threshold for s in diff_person_sims)
    total_accuracy = (correct_same + correct_diff) / 20 * 100
    
    print(f"Accuracy on test set (20 pairs):")
    print(f"  Same person:  {correct_same}/10 correct ({correct_same*10}%)")
    print(f"  Diff person:  {correct_diff}/10 correct ({correct_diff*10}%)")
    print(f"  Overall:      {total_accuracy:.1f}%\n")
    
    print("=" * 60)
    print("USAGE EXAMPLE")
    print("=" * 60)
    print(f"python scripts/inference_arcface.py \\")
    print(f"    --model {model_path} \\")
    print(f"    --face1 face1.jpg --face2 face2.jpg \\")
    print(f"    --threshold {suggested_threshold:.4f}")


if __name__ == "__main__":
    main()
