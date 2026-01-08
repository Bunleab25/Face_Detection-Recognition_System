# Face Detection & Recognition System

A complete face detection and recognition system using **YOLOv11** for face detection and **ArcFace** for face recognition. Features real-time webcam processing, FastAPI web interface, and comprehensive training pipelines.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange.svg)
![YOLO](https://img.shields.io/badge/YOLO-v11-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **Face Detection**: YOLOv11-based high-performance face detector
- **Face Recognition**: ArcFace (ResNet50) for identity recognition
- **Real-time Processing**: Webcam support for live face detection & recognition
- **Web Interface**: FastAPI-based web application with REST API
- **Face Alignment**: Automatic 5-point landmark alignment for ArcFace
- **Database Management**: SQLite database for storing face embeddings
- **Multi-Platform**: CPU and GPU (CUDA) support

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Face Detection](#face-detection)
  - [Face Recognition](#face-recognition)
  - [Web Application](#web-application)
- [Training](#training)
- [Documentation](#documentation)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for faster processing)
- Webcam (for real-time features)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/face_detector.git
cd face_detector
```

2. **Create virtual environment**
```bash
python -m venv .venv
```

3. **Activate virtual environment**

Windows (PowerShell):
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Linux/Mac:
```bash
source .venv/bin/activate
```

4. **Install dependencies**

For CPU:
```bash
pip install -r requirements.txt
pip install -r requirements-cpu.txt
```

For GPU (CUDA 12.4):
```bash
pip install -r requirements.txt
pip install -r requirements-gpu.txt
```

For Web Application:
```bash
pip install -r requirements_fastapi.txt
```

## ⚡ Quick Start

### Face Detection (Webcam)
```bash
python scripts/realtime.py --source 0 --device 0
```

### Face Recognition (Webcam)
```bash
python main_system.py --database face_database.npy
```

### Web Application
```bash
uvicorn app.main:app --reload
```
Visit `http://localhost:8000` in your browser.

## 📁 Project Structure

```
face_detector/
├── app/                        # FastAPI web application
│   ├── main.py                 # API endpoints
│   ├── face_service.py         # Face processing service
│   ├── database.py             # Database models
│   ├── config.py               # Configuration
│   └── static/                 # Web UI assets
├── scripts/                    # Core scripts
│   ├── train.py                # YOLO training
│   ├── train_arcface.py        # ArcFace training
│   ├── predict.py              # Batch prediction
│   ├── realtime.py             # Real-time detection
│   ├── realtime_recognition.py # Real-time recognition
│   ├── face_alignment.py       # Face alignment
│   ├── register_identity.py    # Register new identity
│   └── model.py                # ArcFace model definition
├── docs/                       # Documentation
│   ├── arcface_quickstart.md
│   ├── face_alignment_guide.md
│   └── ...
├── main_system.py              # Standalone recognition system
├── requirements.txt            # Core dependencies
├── requirements-gpu.txt        # GPU (CUDA) dependencies
├── requirements-cpu.txt        # CPU-only dependencies
├── requirements_fastapi.txt    # Web app dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 📖 Usage

### Face Detection

**Single Image Prediction:**
```bash
python scripts/predict.py --source path/to/image.jpg --device 0
```

**Batch Processing:**
```bash
python scripts/predict.py --source path/to/images/ --device 0
```

**Real-time Webcam:**
```bash
python scripts/realtime.py --source 0 --device 0
```
Press `q` or `Esc` to quit.

### Face Recognition

**Register New Identity:**
```bash
python scripts/register_identity.py --name "John Doe" --image path/to/john.jpg
```

**Run Recognition System:**
```bash
python main_system.py --database face_database.npy --threshold 0.5
```

**Real-time Recognition:**
```bash
python scripts/realtime_recognition.py --database face_database.npy
```

### Web Application

**Start Server:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**API Endpoints:**
- `POST /enroll` - Enroll new user with face image
- `POST /recognize` - Recognize face from image
- `GET /users` - List all enrolled users
- `DELETE /users/{id}` - Delete user by ID

**Example cURL:**
```bash
# Enroll user
curl -X POST "http://localhost:8000/enroll" \
  -F "name=John Doe" \
  -F "file=@photo.jpg"

# Recognize face
curl -X POST "http://localhost:8000/recognize" \
  -F "file=@unknown.jpg"
```

### Face Alignment

**Align Single Image:**
```bash
python scripts/face_alignment.py input.jpg --output-dir aligned_faces --device 0
```

**Batch Dataset Alignment:**
```bash
python scripts/preprocess_arcface_dataset.py \
  --input-dir raw_faces/ \
  --output-dir aligned_faces_112/ \
  --yolo-model yolo11n.pt \
  --device 0
```

## 🎓 Training

### Train Face Detector (YOLO)

**Basic Training:**
```bash
python scripts/train.py \
  --data face_datasets/data.yaml \
  --model yolo11n.pt \
  --device 0 \
  --imgsz 640 \
  --epochs 50
```

**Fast Training (Optimized):**
```bash
python scripts/train_yolo11n_fast.py \
  --data face_datasets/data.yaml \
  --model yolo11n.pt \
  --device 0 \
  --epochs 100
```

### Train Face Recognition (ArcFace)

**Prepare Dataset:**
1. Place face images in `combined_dataset/PersonName/` folders
2. Run alignment:
```bash
python scripts/preprocess_arcface_dataset.py \
  --input-dir combined_dataset/ \
  --output-dir aligned_dataset_112/ \
  --lfw-mode
```

**Train Model:**
```bash
python scripts/train_arcface.py \
  --data-dir aligned_dataset_112/ \
  --epochs 100 \
  --batch-size 64 \
  --device 0
```

**Evaluate:**
```bash
python scripts/evaluate_accuracy.py \
  --model runs/arcface_train/arcface_resnet50_epoch_100.pth \
  --test-dir test_dataset/
```

## 📚 Documentation

Detailed documentation available in the `docs/` folder:

- [ArcFace Quick Start](docs/arcface_quickstart.md)
- [Face Alignment Guide](docs/face_alignment_guide.md)
- [ArcFace Architecture](docs/arcface_architecture_reference.md)
- [Training & Inference](docs/step3_4_training_inference.md)
- [Implementation Details](docs/arcface_alignment_implementation.md)

## 📦 Requirements

**Core Dependencies:**
- Python 3.8+
- PyTorch 2.6.0+
- Ultralytics 8.3.246
- OpenCV 4.12.0
- NumPy
- torchvision

**Optional Dependencies:**
- FastAPI (web interface)
- SQLAlchemy (database)
- uvicorn (ASGI server)

See `requirements*.txt` files for complete dependency lists.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Face detection
- [ArcFace](https://arxiv.org/abs/1801.07698) - Face recognition architecture
- [LFW Dataset](http://vis-www.cs.umass.edu/lfw/) - Training data
- [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) - Detection dataset

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ using PyTorch and YOLO**
