# Face Detection & Recognition System
### A Deep Metric Learning Framework via YOLO and ArcFace

This repository implements a real-time, hardware-accelerated face detection and identity recognition framework. The system integrates single-stage bounding box localization via **YOLO** with deep metric learning representations via **Additive Angular Margin Loss (ArcFace)** on a ResNet-50 backbone, backed by a FastAPI service and an asynchronous real-time telemetry dashboard.

---

## 1. Theoretical Background & Methodology

Open-set face identification requires mapping facial images to a compact latent representation where intra-class variance is minimized and inter-class divergence is maximized.

### Additive Angular Margin Loss (ArcFace)

Standard softmax cross-entropy separates feature vectors using linear hyperplanes but fails to enforce geodesic angular margins on open-set verification tasks. ArcFace (Deng et al., CVPR 2019) incorporates an additive angular margin penalty $m$ directly into the target logit angle on a normalized hypersphere:

$$\mathcal{L}_{\text{ArcFace}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s(\cos(\theta_{y_i} + m))}}{e^{s(\cos(\theta_{y_i} + m))} + \sum_{j \neq y_i} e^{s \cos \theta_j}}$$

- $s = 64.0$: Feature scale factor projecting embeddings onto hypersphere radius $s$.
- $m = 0.5$: Additive angular margin penalty (radians), enforcing strict decision boundaries between adjacent identity clusters.

![ArcFace Decision Margin Comparison](assets/research/arcface_geometric_margin_comparison.png)
*Figure 1: Comparison of decision margins across Softmax, SphereFace, CosFace, and ArcFace under binary classification (adapted from Deng et al., CVPR 2019).*

---

### Pipeline Architecture

1. **Face Localization**: YOLO detects face regions and outputs bounding box coordinates $(x_1, y_1, x_2, y_2)$ and confidence scores.
2. **Alignment & Preprocessing**: Facial crops are aligned and resized to $112 \times 112$ tensors, normalized to mean $[0.5, 0.5, 0.5]$ and std $[0.5, 0.5, 0.5]$.
3. **Hyperspherical Embedding**: ResNet-50 extracts a 512-dimensional vector $\mathbf{z} \in \mathbb{R}^{512}$, normalized via $L_2$-norm:
   $$\hat{\mathbf{z}} = \frac{\mathbf{z}}{\|\mathbf{z}\|_2}, \quad \|\hat{\mathbf{z}}\|_2 = 1$$
4. **Metric Verification**: Identities are matched against enrolled vectors using cosine similarity:
   $$\mathcal{S}(\hat{\mathbf{z}}_q, \hat{\mathbf{z}}_k) = \hat{\mathbf{z}}_q^\top \hat{\mathbf{z}}_k = \cos(\theta_{qk})$$
   A candidate match is confirmed if $\mathcal{S} \ge \tau$ (default threshold $\tau = 0.50$).

![ArcFace Training & Inference Pipeline](assets/research/arcface_pipeline_framework.png)
*Figure 2: ArcFace loss computation pipeline with normalized feature embeddings and class weight centers (adapted from Deng et al., CVPR 2019).*

---

## 2. Empirical Verification & Telemetry

### Real-Time Inference & Live Monitoring
Sub-70ms inference loop running on an **NVIDIA GeForce RTX 3050 Laptop GPU** (CUDA 13.0). The dashboard streams real-time bounding box localization, ArcFace similarity scores, and timestamped detection telemetry.

![Figure 3: Real-Time Stream Monitoring](assets/screenshots/fig1_realtime_monitoring_dashboard.png)
*Figure 3: Live webcam stream, bounding box overlay, and detection activity telemetry.*

### Identity Inspection & Verification
High-confidence verification on portrait representations (**99.8% Cosine Similarity**, **89.6% YOLO Confidence**) and unconstrained indoor conditions.

![Figure 4: Face Recognition Inspection](assets/screenshots/fig4_high_confidence_verification.png)
*Figure 4: Empirical identity matching with dual confidence indicators.*

---

## 3. Empirical Hardware Benchmarks

Automated benchmarks evaluated on **Ubuntu 24.04 LTS** (Kernel 7.0) powered by an **NVIDIA GeForce RTX 3050 Laptop GPU** (CUDA 13.0, 4GB VRAM) and an **Intel Core i5-12450H CPU**:

### Model Latency & Throughput (FP32)

| Stage / Model | Architecture | Compute Device | Mean Latency | Throughput | Peak VRAM | Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLO Face Detector** | YOLOv11 Face (5.19 MB) | **CUDA (RTX 3050)** | **7.47 ms** | **133.8 FPS** | 65.9 MB | **6.2x** |
| | | CPU (i5-12450H) | 46.37 ms | 21.6 FPS | N/A | 1.0x (ref) |
| **ArcFace Feature Extractor** | ResNet-50 (27.4M params) | **CUDA (RTX 3050)** | **5.53 ms** | **180.9 FPS** | 156.4 MB | **6.2x** |
| | | CPU (i5-12450H) | 34.45 ms | 29.0 FPS | N/A | 1.0x (ref) |
| **End-to-End Pipeline** | 1080p Ingest → Detect → 512-D Embed → 500-Profile Match | **CUDA (Hybrid)** | **19.85 ms** | **50.4 FPS** | 222.3 MB | **3.0x** |
| | | CPU (Fallback) | 59.74 ms | 16.7 FPS | N/A | 1.0x (ref) |

### Vector Cosine Matching Scaling
Cosine distance calculation across 512-dimensional hyperspherical embeddings:
- **100 Identities**: `5.60 µs` (~178,000 queries/sec)
- **1,000 Identities**: `2.06 ms` (~485 queries/sec)
- **10,000 Identities**: `5.07 ms` (~197 queries/sec)

> Detailed logs and metrics are exported to [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and [`benchmark_results.json`](benchmark_results.json).
> Reproduce benchmarks anytime via:
> ```bash
> python scripts/benchmark.py --iterations 50
> ```

---

## 4. Getting Started

### Prerequisites
- Python 3.9+ (Python 3.11 recommended)
- Git LFS (required to pull model checkpoints):
  ```bash
  git lfs install
  git lfs pull
  ```

### Quick Launch

#### Ubuntu / Linux
```bash
chmod +x start.sh
./start.sh
```

#### Windows
```cmd
start.bat
```

#### Manual Run
```bash
pip install -r requirements.txt
cp .env.example .env
python run.py
```

The web dashboard is served at **http://127.0.0.1:8000** (Default API key: `my-secret-key-123`).

---

## 5. REST API Endpoints

All protected endpoints require the `X-API-Key` header:

- `GET /status` — Hardware diagnostics, active compute device (`cuda`/`cpu`), and profile count.
- `GET /api/verify-key` — API key validation.
- `POST /recognize` — Face detection, bounding box extraction, and identity matching.
- `POST /enroll` — Extract and store 512-D embedding with user name.
- `GET /users` — List all registered profiles.
- `DELETE /users/{user_id}` — Remove an enrolled profile from SQLite database.

---

## 6. Project Layout

```text
├── app/
│   ├── config.py             # Settings, .env resolution & hardware selection
│   ├── database.py           # SQLAlchemy SQLite schema
│   ├── face_service.py       # Inference pipeline (YOLO + ArcFace ResNet-50)
│   ├── main.py               # FastAPI endpoints
│   ├── model.py              # PyTorch ArcFace model architecture
│   ├── security.py           # API key authentication
│   └── static/
│       └── index.html        # Telemetry dashboard
├── assets/
│   ├── research/             # Theoretical figures from CVPR 2019 reference
│   └── screenshots/          # Empirical monitoring & evaluation figures
├── models/                   # Git LFS checkpoints (arcface.pth, yolo_face.pt)
├── tests/                    # Automated test suite
├── start.sh                  # Linux launcher
├── start.bat                 # Windows launcher
└── run.py                    # Cross-platform runner
```

---

## 7. References

```bibtex
@inproceedings{deng2019arcface,
  title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={4690--4699},
  year={2019}
}

@inproceedings{he2016deep,
  title={Deep Residual Learning for Image Recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={770--778},
  year={2016}
}

@software{jocher2023yolo,
  title={Ultralytics YOLO},
  author={Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  year={2023},
  url={https://github.com/ultralytics/ultralytics}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
