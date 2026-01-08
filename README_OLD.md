# Face Detector (YOLO)

A minimal, reproducible project scaffold for training and running a YOLO-based **face detector** using the dataset in `face_datasets/`.

## Quickstart (Windows / PowerShell)

1) Activate the virtual environment:

```powershell
Set-Location "D:\face_detector"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

2) Install base dependencies:

```powershell
python -m pip install -r requirements.txt
```

3) (Optional) Install GPU-enabled PyTorch (recommended for training/inference speed):

```powershell
python -m pip install -r requirements-gpu.txt
```

4) Run a prediction on one image:

```powershell
python scripts\predict.py --source face_datasets\test\images --device 0
```

5) Run realtime detection from your webcam:

```powershell
python scripts\realtime.py --source 0 --device 0
```

Quit the realtime window with `q` or `Esc`.

Outputs are written to `runs/detect/...` by default.

## Training

This repo includes a Roboflow-exported YOLO dataset with one class: `face`.

```powershell
python scripts\train.py --data face_datasets\data.yaml --model yolo11n.pt --device 0 --imgsz 640 --epochs 50
```

For an “efficient defaults” run (YOLOv11n + GPU + AMP + caching):

```powershell
python scripts\train_yolo11n_fast.py --data face_datasets\data.yaml --model yolo11n.pt --device 0 --epochs 100
```
`scripts/train.py` prints:
- `train_started_at` (local time with timezone)
- `train_finished_at`
- `train_elapsed_seconds`

If you want a timestamped log file:

```powershell
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
python scripts\train.py --data face_datasets\data.yaml --model yolo11n.pt --device 0 --imgsz 640 --epochs 50 | Tee-Object -FilePath "runs\train\train_$ts.log"
```

## Face Alignment for ArcFace (NEW)

ArcFace face recognition requires **aligned faces** (eyes horizontal, canonical 112×112 template), not just resized images.

### Quick Demo

Align a single face image:

```powershell
python scripts\face_alignment.py "LFW_datasets\lfw-deepfunneled\lfw-deepfunneled\Aaron_Eckhart\Aaron_Eckhart_0001.jpg" --output-dir runs\test_alignment --device cpu
```

### Batch Preprocessing

Process entire LFW dataset (preserves identity directory structure):

```powershell
python scripts\preprocess_arcface_dataset.py --input-dir LFW_datasets\lfw-deepfunneled\lfw-deepfunneled --output-dir LFW_datasets\lfw_aligned_112 --yolo-model yolo11n.pt --device 0 --lfw-mode
```

Process WIDER FACE training set:

```powershell
python scripts\preprocess_arcface_dataset.py --input-dir face_datasets\train\images --output-dir face_datasets\train_aligned_112 --yolo-model yolo11n.pt --device 0 --save-all
```

**Output**: 112×112 aligned face crops ready for ArcFace training.

📖 **Full documentation**: [`docs/arcface_alignment_implementation.md`](docs/arcface_alignment_implementation.md)

## Dataset

- Dataset config: `face_datasets/data.yaml`
- Class names: `['face']`
- License noted by Roboflow export: **CC BY 4.0** (see `face_datasets/README.dataset.txt`)

## Notes

- `face_datasets/` and `runs/` are ignored by default in `.gitignore` (they can be large). If you want them tracked, remove those ignore entries.
- If you see PowerShell errors like `Successfully : The term 'Successfully' is not recognized...`, it usually means output text was pasted back into the prompt.
