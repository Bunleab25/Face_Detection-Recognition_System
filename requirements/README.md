# Requirements Directory

This directory categorizes dependencies for different environments:

| File | Purpose |
|------|---------|
| [`base.txt`](file:///home/bunleab/Face_Detection-Recognition_System/requirements/base.txt) | Core computer vision, YOLO, and face preprocessing stack. |
| [`fastapi.txt`](file:///home/bunleab/Face_Detection-Recognition_System/requirements/fastapi.txt) | FastAPI, Uvicorn, SQLite/SQLAlchemy backend and API telemetry dependencies. |
| [`cpu.txt`](file:///home/bunleab/Face_Detection-Recognition_System/requirements/cpu.txt) | Standard CPU-only PyTorch wheels + base dependencies. |
| [`gpu.txt`](file:///home/bunleab/Face_Detection-Recognition_System/requirements/gpu.txt) | NVIDIA CUDA-accelerated PyTorch wheels + base dependencies. |

### Quick Install
To install everything needed for the local server and web dashboard:
```bash
pip install -r requirements.txt
```

For GPU acceleration:
```bash
pip install -r requirements/gpu.txt
pip install -r requirements/fastapi.txt
```
