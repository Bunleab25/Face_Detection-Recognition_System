# ArcFace Model Architecture - Complete Overview

## Component Hierarchy

```
FaceRecognitionModel
├── backbone: ResNet (ResNet18/34/50)
│   ├── conv1: Conv2d(3 → 64, 7×7, stride=2)
│   ├── bn1: BatchNorm2d(64)
│   ├── relu: ReLU
│   ├── maxpool: MaxPool2d(3×3, stride=2)
│   ├── layer1: Sequential[BasicBlock × 2]  # 64 channels
│   ├── layer2: Sequential[BasicBlock × 2]  # 128 channels
│   ├── layer3: Sequential[BasicBlock × 2]  # 256 channels
│   ├── layer4: Sequential[BasicBlock × 2]  # 512 channels
│   ├── avgpool: AdaptiveAvgPool2d(1×1)
│   └── fc: Identity()  # Removed original classifier
├── dropout: Dropout(p=0.0) or Identity()
├── embedding: Linear(512 → 512)
├── bn: BatchNorm1d(512)
└── metric_loss: ArcFace(512 → num_classes)
    ├── weight: Parameter(num_classes × 512)  # Class centers
    ├── scale: 64.0
    ├── margin: 0.5 radians
    └── forward(): Compute cos(θ + m) for target class
```

---

## Data Flow

### Training Mode

```
Input Image (B, 3, 112, 112)
      ↓
Conv1: (B, 64, 56, 56)       # 7×7 conv, stride 2
      ↓
MaxPool: (B, 64, 28, 28)     # 3×3 pool, stride 2
      ↓
Layer1: (B, 64, 28, 28)      # ResBlock × 2
      ↓
Layer2: (B, 128, 14, 14)     # ResBlock × 2, downsample
      ↓
Layer3: (B, 256, 7, 7)       # ResBlock × 2, downsample
      ↓
Layer4: (B, 512, 4, 4)       # ResBlock × 2, downsample
      ↓
AvgPool: (B, 512, 1, 1)      # Adaptive global pooling
      ↓
Flatten: (B, 512)            # Spatial to vector
      ↓
Dropout: (B, 512)            # Optional regularization
      ↓
Embedding: (B, 512)          # Linear projection
      ↓
BatchNorm: (B, 512)          # Normalize embedding stats
      ↓
L2 Normalize: (B, 512)       # Unit vectors (implicit in ArcFace)
      ↓
ArcFace Loss: (B, num_classes)  # cos(θ + m) for target, cos(θ) for others
      ↓
CrossEntropy Loss: scalar    # Optimization target
```

### Inference Mode

```
Input Image (B, 3, 112, 112)
      ↓
[Same backbone flow as training]
      ↓
Embedding: (B, 512)
      ↓
BatchNorm: (B, 512)
      ↓
L2 Normalize: (B, 512)       # Explicitly normalized
      ↓
Output: (B, 512)             # Face embedding for verification
```

---

## Tensor Shapes by Layer (Example: ResNet18, Batch=8)

| Layer | Output Shape | Receptive Field | Parameters |
|-------|--------------|-----------------|------------|
| Input | (8, 3, 112, 112) | 1×1 | 0 |
| conv1 | (8, 64, 56, 56) | 7×7 | 9,408 |
| maxpool | (8, 64, 28, 28) | 15×15 | 0 |
| layer1[0] | (8, 64, 28, 28) | 39×39 | 73,984 |
| layer1[1] | (8, 64, 28, 28) | 87×87 | 73,984 |
| layer2[0] | (8, 128, 14, 14) | 119×119 | 230,144 |
| layer2[1] | (8, 128, 14, 14) | 183×183 | 295,424 |
| layer3[0] | (8, 256, 7, 7) | 215×215 | 919,040 |
| layer3[1] | (8, 256, 7, 7) | 279×279 | 1,180,672 |
| layer4[0] | (8, 512, 4, 4) | 311×311 | 3,673,088 |
| layer4[1] | (8, 512, 4, 4) | 375×375 | 4,720,640 |
| avgpool | (8, 512, 1, 1) | - | 0 |
| flatten | (8, 512) | - | 0 |
| embedding | (8, 512) | - | 262,656 |
| bn | (8, 512) | - | 1,024 |
| **ArcFace** | **(8, num_classes)** | - | **512 × num_classes** |

**Total (ResNet18 + embedding):** ~11.7M parameters  
**ArcFace weight matrix:** +512 × num_classes (e.g., 512 × 5749 = 2.9M for LFW)

---

## ArcFace Forward Pass Details

### Step 1: Normalize Inputs
```python
embeddings = F.normalize(embeddings, p=2, dim=1)  # (B, 512) → unit vectors
weight_normalized = F.normalize(self.weight, p=2, dim=1)  # (num_classes, 512) → unit vectors
```

### Step 2: Compute Cosine Similarity
```python
cosine = F.linear(embeddings, weight_normalized)  # (B, num_classes)
# cosine[i, j] = cos(θ_ij) = embedding[i] · weight[j]
```

### Step 3: Compute Sine from Cosine
```python
sine = torch.sqrt(1.0 - torch.pow(cosine, 2))  # sin²(θ) + cos²(θ) = 1
```

### Step 4: Apply Angular Margin
```python
# cos(θ + m) = cos(θ)·cos(m) - sin(θ)·sin(m)
phi = cosine * self.cos_m - sine * self.sin_m  # (B, num_classes)
```

### Step 5: Create One-Hot Target Mask
```python
one_hot = torch.zeros_like(cosine)  # (B, num_classes)
one_hot.scatter_(1, labels.view(-1, 1), 1)  # Mark target class with 1
```

### Step 6: Apply Margin Only to Target Class
```python
output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
# Target class:     cos(θ + m)
# Non-target class: cos(θ)
```

### Step 7: Scale for Softmax
```python
output *= self.scale  # Multiply by s (default: 64.0)
```

### Step 8: Compute Loss (Outside ArcFace)
```python
loss = nn.CrossEntropyLoss()(output, labels)
```

---

## Mathematical Visualization

### Angular Margin Geometry (2D Example)

```
        Weight Vector (Class Center)
              ↑ W
              |
        m →   | ← θ (angle to embedding)
            ╱ |
          ╱   |
  Embedding   |
    (x) ------+-------- Feature Space Axis

Standard Softmax: Maximize cos(θ)
ArcFace:         Maximize cos(θ + m)  ← Harder constraint!
```

**Effect:**
- Standard: Decision boundary at θ = 90° (cos(90°) = 0)
- ArcFace (m=0.5): Decision boundary at θ ≈ 61.4° (cos(90° - 28.6°) = 0)
- **Result:** ~28.6° larger margin between classes

---

## Comparison: Softmax vs ArcFace

### Standard Softmax (VGG16 Paper)

```python
# Forward
features = backbone(x)               # (B, D)
logits = classifier(features)        # (B, num_classes)
loss = CrossEntropyLoss(logits, labels)

# Decision boundary
P(y=c|x) ∝ exp(W_c · x + b_c)

# Problem: No explicit margin between classes
```

### ArcFace (This Implementation)

```python
# Forward
features = backbone(x)               # (B, D)
embeddings = embedding_layer(features)  # (B, 512)
embeddings = normalize(embeddings)   # Unit vectors
logits = arcface(embeddings, labels) # (B, num_classes) with angular margin
loss = CrossEntropyLoss(logits, labels)

# Decision boundary
P(y=c|x) ∝ exp(s · cos(θ_c + m))  for target class
P(y=j|x) ∝ exp(s · cos(θ_j))      for other classes

# Advantage: Explicit angular margin m between classes
```

---

## Hyperparameter Effects

### Scale (s)

| Value | Effect | Training |
|-------|--------|----------|
| 30 | Softer gradients, slower convergence | More stable |
| 64 | **Recommended balance** | Standard |
| 128 | Sharper gradients, faster convergence | May oscillate |

### Margin (m)

| Value | Effect | Difficulty |
|-------|--------|------------|
| 0.3 | Easier optimization, less discriminative | Easy |
| 0.5 | **Recommended for faces** | Medium |
| 0.7 | Harder optimization, very discriminative | Hard |

### Embedding Size

| Value | Params | Accuracy | Speed |
|-------|--------|----------|-------|
| 128 | Lowest | Lower | Fastest |
| 256 | Low | Good | Fast |
| 512 | **Standard** | **Best** | **Medium** |
| 1024 | Highest | Marginal gain | Slowest |

---

## Memory & Compute Requirements

### GPU Memory (Batch Size = 64, ResNet18)

| Component | Memory |
|-----------|--------|
| Input (64 × 3 × 112 × 112) | ~9 MB |
| Activations (forward) | ~150 MB |
| Gradients (backward) | ~150 MB |
| Model parameters | ~50 MB |
| Optimizer state (Adam) | ~100 MB |
| **Total** | **~500 MB** |

**Recommendation:**
- **RTX 3060 (12GB):** Batch size 128-256
- **RTX 3090 (24GB):** Batch size 512+

### Training Speed (RTX 3060)

| Backbone | Batch=64 | Batch=128 | Batch=256 |
|----------|----------|-----------|-----------|
| ResNet18 | ~250 img/s | ~400 img/s | ~600 img/s |
| ResNet34 | ~180 img/s | ~300 img/s | ~450 img/s |
| ResNet50 | ~120 img/s | ~200 img/s | ~300 img/s |

---

## Code Examples

### Initialize Model
```python
from model import create_arcface_model

model = create_arcface_model(
    num_classes=5749,      # LFW identities
    embedding_size=512,
    backbone="resnet18",
    pretrained=True,
    scale=64.0,
    margin=0.5
).cuda()
```

### Training Step
```python
images, labels = next(train_loader)  # (B, 3, 112, 112), (B,)
images, labels = images.cuda(), labels.cuda()

optimizer.zero_grad()
logits, embeddings = model(images, labels)
loss = nn.CrossEntropyLoss()(logits, labels)
loss.backward()
optimizer.step()
```

### Inference (Verification)
```python
model.eval()
with torch.no_grad():
    emb1 = model.get_embeddings(face1.cuda())  # (1, 512)
    emb2 = model.get_embeddings(face2.cuda())  # (1, 512)
    
    similarity = torch.sum(emb1 * emb2)  # Cosine similarity
    is_same = similarity > 0.5  # Threshold
```

---

## Summary

✅ **Architecture:** ResNet18/34/50 backbone + 512-d embedding + ArcFace loss  
✅ **Input:** 112×112 aligned faces (from Step 1)  
✅ **Output:** 512-d L2-normalized embeddings for verification  
✅ **Loss:** Angular margin (m=0.5) with scale (s=64.0)  
✅ **Parameters:** ~12M (ResNet18) + 512 × num_classes (ArcFace matrix)  
✅ **Verified:** Forward pass works, embeddings normalized correctly  

**Next:** Integrate with training loop (Step 3)
