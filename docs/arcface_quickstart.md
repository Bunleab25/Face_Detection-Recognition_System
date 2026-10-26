# ArcFace Preprocessing Quick Start

## Goal
Preprocess face images for ArcFace training by aligning them to a canonical 112×112 template.

## Why This Matters
❌ **Original paper**: Simply resized images to 224×224  
✅ **ArcFace**: Requires **aligned faces** with eyes horizontal at canonical positions

Proper alignment ensures:
- Consistent pose across all training samples
- Better feature learning for face embeddings
- Higher accuracy on verification/identification tasks

---

## Step-by-Step

### 1. Single Image Test (Verify It Works)

```bash
python scripts/face_alignment.py \
  "LFW_datasets/lfw-deepfunneled/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg" \
  --output-dir runs/test_alignment \
  --device cpu
```

**Expected output:**
```
Detected and aligned 1 face(s)
  Saved: runs/test_alignment\face_000.jpg
```

**Verify:**
- Output image is exactly 112×112 pixels
- Eyes appear horizontal
- Face is centered

---

### 2. Preprocess LFW Dataset

```bash
python scripts/preprocess_arcface_dataset.py \
  --input-dir LFW_datasets/lfw-deepfunneled/lfw-deepfunneled \
  --output-dir LFW_datasets/lfw_aligned_112 \
  --yolo-model models/yolo11n.pt \
  --device 0 \
  --lfw-mode
```

**Output:**
```
LFW_datasets/lfw_aligned_112/
  Aaron_Eckhart/
    Aaron_Eckhart_0001.jpg  # 112×112 aligned
    Aaron_Eckhart_0002.jpg
  Aaron_Guiel/
    ...
```

**Time estimate:** ~10-30 minutes for full LFW (13,233 images) on GPU

---

### 3. Preprocess WIDER FACE Training Set

```bash
python scripts/preprocess_arcface_dataset.py \
  --input-dir face_datasets/train/images \
  --output-dir face_datasets/train_aligned_112 \
  --yolo-model models/yolo11n.pt \
  --device 0 \
  --conf 0.5
```

**Time estimate:** Depends on dataset size; ~1-5 images/second on GPU

---

## Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--device` | `0` | GPU device (`0`, `1`, ...) or `cpu` |
| `--conf` | `0.5` | Face detection confidence threshold |
| `--landmark-method` | `fallback` | `fallback` (bbox-based) or `cv2_face` (requires opencv-contrib-python) |
| `--lfw-mode` | off | Preserve identity directory structure (for LFW/MS-Celeb-1M) |
| `--save-all` | off | Save all detected faces (default: first face only) |

---

## What Happens During Alignment

For each input image:

1. **Detect face** using YOLOv11 → bounding box `(x1, y1, x2, y2)`
2. **Estimate 5 landmarks**:
   - Left eye center
   - Right eye center
   - Nose tip
   - Left mouth corner
   - Right mouth corner
3. **Compute similarity transform**:
   - Rotation to make eyes horizontal
   - Scale to canonical face size
   - Translation to center face
4. **Warp image** to 112×112 aligned crop

**Result:** Eyes at standard positions `(38.29, 51.70)` and `(73.53, 51.50)` in output image

---

## Next: ArcFace Training

Once preprocessing is complete:

1. **Load aligned faces** (no resizing needed)
2. **Apply ONLY appearance augmentation**:
   - ✅ Color jitter, blur, noise
   - ❌ NO rotation, scaling, cropping
3. **Normalize**: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
4. **Input to network**: `(B, 3, 112, 112)` tensors

See [`docs/arcface_alignment_implementation.md`](arcface_alignment_implementation.md) for full integration guide.

---

## Troubleshooting

### "No faces detected"
- Lower `--conf` threshold (e.g., `--conf 0.3`)
- Check YOLO model is trained on faces
- Verify input images are valid

### "Alignment fails / distorted faces"
- Try `--landmark-method cv2_face` (requires `pip install opencv-contrib-python`)
- Check if faces are extremely non-frontal (ArcFace assumes mostly frontal)

### Slow processing
- Use GPU: `--device 0` instead of `--device cpu`
- Reduce resolution (if using high-res inputs, consider downscaling first)
- Batch processing is sequential (no parallel workers yet)

---

## Performance Benchmarks

Approximate processing speeds (RTX 3060):

| Dataset | Images | Time | Speed |
|---------|--------|------|-------|
| Single image | 1 | ~0.3s | - |
| LFW (small batch) | 100 | ~30s | 3.3 img/s |
| WIDER FACE subset | 1000 | ~5 min | 3-4 img/s |

**Bottleneck:** YOLO inference per-image. Future optimization: batch inference.

---

## Files Reference

- **Core module**: `scripts/face_alignment.py`
- **Batch script**: `scripts/preprocess_arcface_dataset.py`
- **Full docs**: `docs/arcface_alignment_implementation.md`
- **User guide**: `docs/face_alignment_guide.md`
