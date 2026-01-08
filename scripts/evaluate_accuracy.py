import argparse
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from PIL import Image
from torchvision import transforms
import math

# --- Model Definition (from web_app/app/model.py) ---
class ArcFace(nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698.pdf) """
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class FaceRecognitionModel(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=5749, embedding_size=512):
        super(FaceRecognitionModel, self).__init__()
        
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights=None)
            self.backbone.fc = nn.Identity()
            self.embedding = nn.Linear(512, embedding_size)
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights=None)
            self.backbone.fc = nn.Identity()
            self.embedding = nn.Linear(512, embedding_size)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights=None)
            self.backbone.fc = nn.Identity()
            self.embedding = nn.Linear(2048, embedding_size)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.bn = nn.BatchNorm1d(embedding_size)
        self.metric_loss = ArcFace(embedding_size, num_classes)

    def forward(self, x, labels=None):
        features = self.backbone(x)
        features = self.embedding(features)
        features = self.bn(features)
        
        if labels is not None:
            return self.metric_loss(features, labels)
        return features

# --- Helper Functions ---

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'metric_loss.weight' in state_dict:
            num_classes = state_dict['metric_loss.weight'].shape[0]
        else:
            num_classes = 5749 
            
        if 'embedding.weight' in state_dict:
            in_features = state_dict['embedding.weight'].shape[1]
            if in_features == 2048:
                backbone = 'resnet50'
            elif in_features == 512:
                backbone = 'resnet18'
            else:
                backbone = 'resnet50'
        else:
            backbone = 'resnet50'
            
    else:
        state_dict = checkpoint
        num_classes = 5749
        backbone = 'resnet50'

    print(f"Initializing model with backbone={backbone}, num_classes={num_classes}")
    model = FaceRecognitionModel(backbone=backbone, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_image(image_path, size=112):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform(img).unsqueeze(0)

def get_embedding(model, image_tensor, device):
    with torch.no_grad():
        emb = model(image_tensor)
        emb = F.normalize(emb)
    return emb

def evaluate_lfw_csv(model, pairs_file, dataset_root, device):
    pairs_file = Path(pairs_file)
    dataset_root = Path(dataset_root)
    
    similarities = []
    labels = []
    
    print(f"Reading pairs from {pairs_file}...")
    with open(pairs_file, "r") as f:
        lines = f.readlines()[1:] # Skip header
        
    print(f"Processing {len(lines)} pairs...")
    
    count = 0
    total = len(lines)
    
    for line in lines:
        parts = line.strip().split(',')
        
        if len(parts) < 3:
            continue
            
        if parts[-1] == '': # Matched
            name = parts[0]
            idx1 = parts[1]
            idx2 = parts[2]
            
            path1 = dataset_root / name / f"{name}_{int(idx1):04d}.jpg"
            path2 = dataset_root / name / f"{name}_{int(idx2):04d}.jpg"
            label = 1
        else: # Mismatched
            name1 = parts[0]
            idx1 = parts[1]
            name2 = parts[2]
            idx2 = parts[3]
            
            path1 = dataset_root / name1 / f"{name1}_{int(idx1):04d}.jpg"
            path2 = dataset_root / name2 / f"{name2}_{int(idx2):04d}.jpg"
            label = 0
            
        if not path1.exists() or not path2.exists():
            continue
            
        try:
            img1 = load_image(path1).to(device)
            img2 = load_image(path2).to(device)
            
            emb1 = get_embedding(model, img1, device)
            emb2 = get_embedding(model, img2, device)
            
            sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
            
            similarities.append(sim)
            labels.append(label)
        except Exception as e:
            print(f"Error processing pair: {e}")
            continue
        
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{total} pairs...", end='\r')
            
    print(f"\nFinished processing {count} pairs.")
    
    if len(similarities) == 0:
        print("No pairs processed successfully.")
        return 0, 0
            
    similarities = np.array(similarities)
    labels = np.array(labels)
    
    # Find best threshold
    thresholds = np.arange(0, 1.0, 0.01)
    best_acc = 0
    best_thresh = 0
    
    for thresh in thresholds:
        preds = (similarities > thresh).astype(int)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            
    return best_acc * 100, best_thresh

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--pairs", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    try:
        model = load_model(args.model, device)
        acc, thresh = evaluate_lfw_csv(model, args.pairs, args.dataset_root, device)
        print(f"\nBest Accuracy: {acc:.2f}% at threshold {thresh:.2f}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
