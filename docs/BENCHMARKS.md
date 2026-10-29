# Empirical Benchmark Report

Automated inference latency, throughput, and hardware efficiency benchmarks measured on local deployment environment.

## 1. Test Hardware & Environment

| Component | Specification |
|:----------|:--------------|
| **Host OS** | Linux 7.0.0-38-generic (x86_64) |
| **CPU** | 12th Gen Intel(R) Core(TM) i5-12450H |
| **GPU** | NVIDIA GeForce RTX 3050 Laptop GPU (CUDA 13.0) |
| **Framework** | PyTorch 2.13.0+cu130 |
| **Precision** | FP32 |

---

## 2. Model Inference Latency & Throughput

### YOLO Face Detector (`models/yolo_face.pt` — 5.19 MB)
- **Input Resolution**: 640×640 px

| Compute Device | Mean Latency | Median | P95 Latency | Throughput (FPS) | VRAM Peak | Acceleration |
|:---------------|:-------------|:-------|:------------|:-----------------|:----------|:-------------|
| **NVIDIA CUDA** | **7.47 ms** | 7.04 ms | 9.3 ms | **133.8 FPS** | 65.9 MB | **6.21x** |
| **Intel CPU** | 46.37 ms | 44.32 ms | 55.07 ms | 21.6 FPS | N/A | 1.0x (ref) |

### ArcFace Deep Feature Extractor (`models/arcface.pth` — 209.65 MB)
- **Architecture**: ResNet-50 Backbone (27,430,976 parameters)
- **Input Resolution**: 112×112 px
- **Embedding Space**: 512-Dimensional Hyperspherical Manifold ($L_2$ Normalized)

| Compute Device | Mean Latency | Median | P95 Latency | Embeddings / Sec | VRAM Peak | Acceleration |
|:---------------|:-------------|:-------|:------------|:-----------------|:----------|:-------------|
| **NVIDIA CUDA** | **5.53 ms** | 5.45 ms | 6.02 ms | **180.9 FPS** | 156.4 MB | **6.23x** |
| **Intel CPU** | 34.45 ms | 33.08 ms | 50.32 ms | 29.0 FPS | N/A | 1.0x (ref) |

---

## 3. High-Dimensional Vector Search Latency

Cosine distance calculation across 512-dimensional hyperspherical embeddings:

| Enrolled Identities | Mean Search Latency | Throughput (QPS) |
|:--------------------|:--------------------|:-----------------|
| **100 Identities** | 5.6 µs | 178,420 queries/sec |
| **1,000 Identities** | 2060.24 µs | 485 queries/sec |
| **10,000 Identities** | 5068.08 µs | 197 queries/sec |

---

## 4. End-to-End Pipeline Performance

Full pipeline execution: **1080p Video Ingest → YOLO Face Detection → Crop & Landmark Normalization → ArcFace 512-D Inference → 500-Profile Vector Cosine Match**:

| Environment | End-to-End Latency | Pipeline Throughput | Real-Time Capable (>24 FPS) |
|:------------|:-------------------|:-------------------|:-----------------------------|
| **CUDA (RTX 3050 Laptop GPU)** | **19.85 ms** | **50.4 FPS** | **YES (Real-Time)** |
| **CPU (Intel Core i5-12450H)** | 59.74 ms | 16.7 FPS | Fallback Mode |

---

## 5. Model Training & Learning Loss Curves

The model weights include full 100-epoch training history and validation trajectories.

### YOLOv11 Face Detector Training & Convergence (100 Epochs)

| Metric | Initial (Epoch 1) | Final (Epoch 100) | Peak |
|:-------|:------------------|:------------------|:-----|
| **Train Box Loss (IoU)** | 1.886 | **1.286** | Min: 1.286 |
| **Train Cls Loss (BCE)** | 1.763 | **0.610** | Min: 0.610 |
| **Train DFL Loss** | 1.224 | **0.970** | Min: 0.968 |
| **Val Box Loss** | 1.667 | **1.348** | Min: 1.347 |
| **Val Cls Loss** | 1.130 | **0.619** | Min: 0.619 |
| **Val DFL Loss** | 1.090 | **0.992** | Min: 0.992 |
| **Precision (B)** | 73.4% | **85.1%** | **85.2%** |
| **Recall (B)** | 41.5% | **58.3%** | **58.5%** |
| **mAP @ 0.50 (B)** | 47.2% | **67.2%** | **67.2%** |
| **mAP @ 0.50:0.95 (B)** | 22.7% | **37.0%** | **37.0%** |

![YOLO Training Curves](../assets/research/yolo_training_loss_curves.png)
*Figure 5.1: 6-panel training curves showing Box/Cls/DFL losses, precision/recall, mAP metrics, and SGD learning rate decay.*

#### Precision-Recall, F1-Score & Confusion Matrix

| Precision-Recall Curve | F1-Confidence Curve |
|:---:|:---:|
| ![PR Curve](../assets/research/yolo_pr_curve.png) | ![F1 Curve](../assets/research/yolo_f1_curve.png) |

![Confusion Matrix](../assets/research/yolo_confusion_matrix.png)
*Figure 5.2: Normalized detection confusion matrix (Face vs Background).*

---

### ArcFace ResNet-50 Feature Extractor Convergence

Trained on 5,749 identity classes with additive angular margin penalty ($s = 64.0, m = 0.50$) and MultiStepLR decay at epochs 20, 35, and 45.

![ArcFace Loss Curve](../assets/research/arcface_training_loss_curve.png)
*Figure 5.3: ArcFace cross-entropy loss convergence and learning rate step schedule.*

