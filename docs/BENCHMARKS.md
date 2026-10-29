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
| **CPU (Intel Core i5-11400H)** | 59.74 ms | 16.7 FPS | Fallback Mode |
