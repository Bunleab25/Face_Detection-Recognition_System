# Face Detection & Recognition System

A Python project for face detection and recognition using YOLO and ArcFace. It includes command-line tools for detection and model training, face recognition workflows, and a FastAPI web interface.

> **Status:** This is a research and learning project. Recognition quality depends on the model, data, and operating conditions; validate it for your use case before relying on its output.

## Features

- Face detection with YOLO-based models
- ArcFace embedding and recognition workflows
- Image, video, and webcam scripts
- FastAPI interface for enrollment and recognition
- CPU support, with optional CUDA acceleration
- Training, evaluation, and alignment guides in [`docs/`](docs/)

## Requirements

- Python 3.9 or later
- Git LFS for the model weights stored in this repository
- A webcam for live camera examples
- An NVIDIA GPU is optional

## Get started

Install Git LFS before cloning so the model files are downloaded rather than left as pointer files:

```bash
git lfs install
git clone https://github.com/Bunleab25/Face_Detection-Recognition_System.git
cd Face_Detection-Recognition_System
git lfs pull
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the CPU dependencies:

```bash
python -m pip install -r requirements.txt -r requirements-cpu.txt
```

For CUDA 12.4, install the base dependencies and the GPU PyTorch wheels instead:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-gpu.txt
```

The GPU requirements target PyTorch 2.6.0 with CUDA 12.4. For other CUDA versions, install the matching PyTorch and torchvision builds from the [official PyTorch selector](https://pytorch.org/get-started/locally/).

## Quick start

### Detect faces in an image

```bash
python scripts/predict.py \
  --model web_app/models/yolo_face.pt \
  --source path/to/image.jpg \
  --device cpu
```

Use `--device 0` to select the first CUDA GPU when CUDA is installed. Prediction outputs are saved under `runs/detect/`.

### Start the web application

Copy the example configuration and install the web dependencies:

```bash
cp .env.example .env
python -m pip install -r requirements_fastapi.txt
```

On Windows PowerShell, use `Copy-Item .env.example .env` instead of `cp`.

Run the server from the repository root:

```bash
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. The API currently provides:

- `POST /enroll` — enroll a person using a `name` form field and an image `file`
- `POST /recognize` — find the closest enrolled identity for an image `file`

Example enrollment request:

```bash
curl -X POST http://127.0.0.1:8000/enroll \
  -F "name=Example Person" \
  -F "file=@path/to/image.jpg"
```

The sample `.env` points to the model weights tracked with Git LFS. Set `DEVICE=cpu` for CPU inference or `DEVICE=cuda` when a compatible CUDA environment is available. Keep `.env` and local databases private; they are intentionally excluded from version control.

## Project layout

```text
app/                 FastAPI application and static web interface
docs/                Setup, architecture, alignment, and training guides
scripts/             Detection, recognition, training, and evaluation tools
web_app/models/      YOLO face detector and ArcFace weights (Git LFS)
main_system.py       Standalone recognition entry point
requirements*.txt    CPU, GPU, core, and web dependency lists
```

## Documentation

- [Quick start guide](QUICKSTART.md)
- [Project structure](PROJECT_STRUCTURE.md)
- [ArcFace quick start](docs/arcface_quickstart.md)
- [Face alignment guide](docs/face_alignment_guide.md)
- [ArcFace architecture reference](docs/arcface_architecture_reference.md)
- [Training and inference](docs/step3_4_training_inference.md)
- [Contributing](CONTRIBUTING.md)

## Privacy and responsible use

Face images and recognition embeddings are sensitive biometric data. Obtain informed consent, restrict access, protect stored data, and follow applicable laws and organizational policies. Do not use this project as the sole basis for consequential decisions.

## License

This project is licensed under the [MIT License](LICENSE).
