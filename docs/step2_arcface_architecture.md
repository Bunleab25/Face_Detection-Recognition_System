# Step 2: ArcFace Network Architecture

## ✅ Implementation Complete

I've created a complete ArcFace face recognition model with ResNet backbone and angular margin loss.

---

## 📁 Files Created

### 1. **`scripts/arcface_layer.py`**
ArcFace and CosFace loss implementations:

**ArcFace (Additive Angular Margin Loss):**
- Formula: `L = -log(exp(s·cos(θ + m)) / Σ exp(s·cos(θ)))`
- Adds angular margin `m` (default: 0.5 radians) to target class
- Scale factor `s` (default: 64.0)
- L2-normalized embeddings and weight vectors
- Easy margin option for unstable training

**CosFace (Large Margin Cosine Loss):**
- Alternative loss with cosine margin instead of angular
- Often simpler to train
- Formula: `L = -log(exp(s·(cos(θ) - m)) / Σ exp(s·cos(θ)))`

### 2. **`scripts/model.py`**
Complete face recognition model architecture:

**Architecture:**
```
Input (B, 3, 112, 112) → aligned faces
  ↓
ResNet Backbone (ResNet18/34/50)
  ↓
Dropout (optional)
  ↓
Embedding Layer (512-d)
  ↓
Batch Normalization
  ↓
ArcFace Loss (training only)
  ↓
Logits (B, num_classes)
```

**Key Features:**
- ✅ ResNet18/34/50 backbone (lighter than VGG16)
- ✅ 112×112 input (ArcFace standard)
- ✅ 512-d embeddings (face identity representation)
- ✅ L2-normalized features
- ✅ Dual mode: training (returns logits + embeddings) / inference (returns embeddings only)
- ✅ Optional dropout regularization
- ✅ ImageNet pretrained initialization

---

## 🔑 Key Differences from Original Paper

| Aspect | Original Paper (VGG16) | This Implementation (ArcFace) |
|--------|------------------------|-------------------------------|
| **Backbone** | VGG16 | ResNet18/34/50 |
| **Input size** | 224×224 (simple resize) | 112×112 (aligned faces) |
| **Loss** | Softmax | **ArcFace (angular margin)** |
| **Embeddings** | Classification features | **512-d L2-normalized** |
| **Purpose** | Classification | **Face verification/identification** |
| **Parameters** | ~138M (VGG16) | ~12M (ResNet18) |

---

## 🧪 Verification Test

**Model instantiated successfully:**
```
Total parameters: 11,952,192
Trainable parameters: 11,952,192

Training mode:
  Input: (8, 3, 112, 112)
  Labels: (8,)
  Output logits: (8, 1000)
  Embeddings: (8, 512)

Inference mode:
  Input: (8, 3, 112, 112)
  Embeddings: (8, 512)
  L2 norm: [1.0, 1.0, ..., 1.0]  ✅ Correctly normalized
```

---

## 📖 Usage

### Create Model

```python
from model import create_arcface_model

# For LFW dataset (5,749 identities)
model = create_arcface_model(
    num_classes=5749,
    embedding_size=512,
    backbone="resnet18",
    pretrained=True,
    loss_type="arcface",
    scale=64.0,
    margin=0.5,
    dropout=0.0
)
```

### Training Forward Pass

```python
# Input: aligned 112×112 faces
images = torch.randn(32, 3, 112, 112)  # Batch of 32
labels = torch.randint(0, 5749, (32,))  # Ground-truth identities

# Forward pass (training mode)
logits, embeddings = model(images, labels)

# Compute loss
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)
```

### Inference (Face Verification)

```python
model.eval()

# Extract embeddings for two faces
emb1 = model.get_embeddings(face1)  # (1, 512)
emb2 = model.get_embeddings(face2)  # (1, 512)

# Compute cosine similarity
similarity = torch.sum(emb1 * emb2, dim=1)  # Range: [-1, 1]

# Verification decision
threshold = 0.5  # Tune on validation set
is_same_person = similarity > threshold
```

---

## 🎯 Hyperparameters (Paper-Based Defaults)

### ArcFace Loss Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `scale` (s) | 64.0 | 30-64 | Feature scale (higher = sharper gradients) |
| `margin` (m) | 0.5 | 0.3-0.6 | Angular margin in radians (~28.6°) |
| `easy_margin` | False | - | Use simplified margin (for unstable training) |

### Model Parameters

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `embedding_size` | 512 | 128-512 | Embedding dimension |
| `backbone` | resnet18 | resnet18/34/50 | Feature extractor |
| `pretrained` | True | True/False | ImageNet initialization |
| `dropout` | 0.0 | 0.0-0.5 | Regularization dropout |

---

## 🔬 Model Variants

### Recommended Configurations

**1. Lightweight (fast training/inference):**
```python
model = create_arcface_model(
    num_classes=num_identities,
    embedding_size=512,
    backbone="resnet18",  # ~12M parameters
    pretrained=True,
    scale=64.0,
    margin=0.5
)
```

**2. Standard (balanced):**
```python
model = create_arcface_model(
    num_classes=num_identities,
    embedding_size=512,
    backbone="resnet34",  # ~22M parameters
    pretrained=True,
    scale=64.0,
    margin=0.5,
    dropout=0.1
)
```

**3. High-capacity (best accuracy):**
```python
model = create_arcface_model(
    num_classes=num_identities,
    embedding_size=512,
    backbone="resnet50",  # ~26M parameters
    pretrained=True,
    scale=64.0,
    margin=0.5,
    dropout=0.2
)
```

**4. CosFace (alternative loss):**
```python
model = create_arcface_model(
    num_classes=num_identities,
    embedding_size=512,
    backbone="resnet18",
    pretrained=True,
    loss_type="cosface",  # Use CosFace instead
    scale=64.0,
    margin=0.35,  # Cosine margin (different from angular)
)
```

---

## 🧮 ArcFace Math Explained

### Standard Softmax (Original Paper)
```
L = -log(exp(W_yi · x_i) / Σ_j exp(W_j · x_i))
```
- Linear decision boundaries
- No explicit margin between classes

### ArcFace (This Implementation)
```
L = -log(exp(s·cos(θ_yi + m)) / (exp(s·cos(θ_yi + m)) + Σ_{j≠yi} exp(s·cos(θ_j))))
```

**Where:**
- `θ`: Angle between feature `x` and weight `W` (computed via arccos of dot product)
- `m`: Angular margin (pushes classes apart in angular space)
- `s`: Scale factor (controls gradient magnitude)

**Geometric Interpretation:**
- Embeddings lie on unit hypersphere (L2-normalized)
- ArcFace pushes decision boundaries by angle `m`
- Creates larger inter-class margin → better discrimination

**Example:**
- Without margin: cos(30°) = 0.866
- With m=0.5 rad (~28.6°): cos(30° + 28.6°) = cos(58.6°) = 0.520
- Harder for model to satisfy → learns more discriminative features

---

## 📊 Expected Training Behavior

### Loss Curve
```
Epoch 1-5:   Loss ~10 → ~3   (learning class separations)
Epoch 5-20:  Loss ~3 → ~1    (fine-tuning margins)
Epoch 20-50: Loss ~1 → ~0.5  (convergence)
```

### Embedding Quality
- **Early training**: Embeddings scattered, low inter-class separation
- **Mid training**: Classes form clusters on hypersphere
- **Late training**: Tight intra-class clusters, large inter-class margins

### Verification Accuracy (LFW)
- **ResNet18**: ~98-99% @ 10 epochs (with pretrained backbone)
- **ResNet50**: ~99%+ @ 20 epochs

---

## 🛠️ Dependencies (Already Installed)

- ✅ `torch`
- ✅ `torchvision` (for ResNet models)
- ✅ Python 3.8+

---

## 🚀 Next Steps

1. **Create dataset loader** for aligned 112×112 faces (from Step 1)
2. **Implement training loop** with SGD optimizer (from YOLOv3 schedule)
3. **Add validation** with face verification metrics (cosine similarity)
4. **Train model** on LFW or WIDER FACE

See next step for complete training pipeline integration.

---

## 📝 Notes

### Why ResNet over VGG16?

1. **Fewer parameters**: ResNet18 (12M) vs VGG16 (138M)
2. **Better for small inputs**: Designed for efficient feature extraction
3. **Residual connections**: Better gradient flow for deep networks
4. **Industry standard**: All modern face recognition uses ResNet/MobileNet

### Why 112×112 instead of 224×224?

1. **ArcFace standard**: All pretrained models use 112×112
2. **Faster training**: 4× fewer pixels
3. **Aligned faces**: Don't need high resolution (faces already centered)
4. **Memory efficient**: 4× less GPU memory per batch

### Training vs Inference Mode

**Training:**
- Input: (B, 3, 112, 112) + labels
- Output: logits (B, num_classes) + embeddings (B, 512)
- Use: Compute cross-entropy loss on logits
- ArcFace layer: Active (applies angular margin)

**Inference:**
- Input: (B, 3, 112, 112)
- Output: embeddings (B, 512) - L2 normalized
- Use: Compute cosine similarity between embeddings
- ArcFace layer: Bypassed (not needed for verification)
