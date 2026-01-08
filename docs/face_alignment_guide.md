# Face Alignment for ArcFace

## Critical Difference: Alignment vs. Simple Resize

**The original paper** (likely ImageNet-based) simply resized images to 224×224.

**ArcFace requires face alignment** to work correctly:
1. ✅ **Detect face bounding box** (using YOLOv11)
2. ✅ **Detect 5 facial landmarks** (eyes, nose, mouth corners)
3. ✅ **Apply similarity transform** to rotate & scale face so eyes are horizontal
4. ✅ **Warp to canonical 112×112 template**

This ensures all faces are in a **consistent pose/scale** for the embedding network.

---

## Implementation

### Module: `scripts/face_alignment.py`

**Core components:**

1. **`FaceAligner` class**: High-level API integrating YOLO detection + landmark-based alignment
2. **`align_face_from_landmarks()`**: Applies similarity transform using 5-point landmarks
3. **Landmark detection**:
   - `detect_landmarks_cv2_face()`: Uses OpenCV's face module (requires `opencv-contrib-python`)
   - `detect_landmarks_fallback()`: Geometric estimation from bounding box (no extra dependencies)

**Canonical template**: ArcFace standard 112×112 with InsightFace-compatible landmark positions.

---

## Usage

### 1. Single Image Demo

```bash
python scripts/face_alignment.py <image_path> [output_dir]
```

**Example:**
```bash
python scripts/face_alignment.py AcrFace/lfw-deepfunneled/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg runs/alignment
```

**Output**: Aligned 112×112 face crops saved to `runs/alignment/face_000.jpg`

---

### 2. Batch Preprocessing for ArcFace Training

**Script**: `scripts/preprocess_arcface_dataset.py`

#### Option A: Process LFW Dataset (preserves identity structure)

```bash
python scripts/preprocess_arcface_dataset.py \
  --input-dir AcrFace/lfw-deepfunneled/lfw-deepfunneled \
  --output-dir AcrFace/lfw_aligned_112 \
  --yolo-model yolo11n.pt \
  --device 0 \
  --lfw-mode
```

**Output structure:**
```
AcrFace/lfw_aligned_112/
  Aaron_Eckhart/
    Aaron_Eckhart_0001.jpg  # 112×112 aligned
  Aaron_Guiel/
    Aaron_Guiel_0001.jpg
  ...
```

#### Option B: Process Any Face Dataset

```bash
python scripts/preprocess_arcface_dataset.py \
  --input-dir face_datasets/train/images \
  --output-dir face_datasets/train_aligned_112 \
  --yolo-model yolo11n.pt \
  --device 0 \
  --conf 0.5 \
  --save-all
```

**Flags:**
- `--save-all`: Save all detected faces (for multi-face images)
- `--conf`: Detection confidence threshold
- `--landmark-method`: `fallback` (default, bbox-based) or `cv2_face` (requires opencv-contrib-python)

---

## Integration with ArcFace Training

Once you have aligned 112×112 faces, your ArcFace training pipeline should:

1. **DataLoader**: Load pre-aligned 112×112 images (no resizing needed, maybe only normalization)
2. **No geometric augmentation**: Avoid rotation/scaling that breaks alignment
3. **Only appearance augmentation**: Color jitter, brightness, contrast, blur
4. **Input to network**: (B, 3, 112, 112) tensors

---

## Landmark Detection Methods

### Method 1: `fallback` (default)
- Estimates landmarks from bounding box geometry
- Assumes frontal faces with standard proportions
- ✅ No extra dependencies
- ⚠️ Less accurate for non-frontal poses

### Method 2: `cv2_face`
- Uses OpenCV's face module (LBF landmark detector)
- Requires: `pip install opencv-contrib-python`
- ✅ More accurate for varied poses
- ⚠️ Slower, requires additional model files

For production ArcFace training, consider using a dedicated landmark model like:
- **MediaPipe Face Mesh** (5-point subset)
- **Dlib 5-point detector**
- **RetinaFace** (includes landmarks)

---

## Verification

Test alignment quality visually:

```bash
python scripts/face_alignment.py AcrFace/lfw-deepfunneled/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg runs/test_alignment
```

Check that:
- Eyes are horizontal
- Face is centered
- Output is exactly 112×112 pixels
- No extreme distortion

---

## Next Steps

1. **Preprocess your training dataset**:
   ```bash
   python scripts/preprocess_arcface_dataset.py \
     --input-dir AcrFace/lfw-deepfunneled/lfw-deepfunneled \
     --output-dir AcrFace/lfw_aligned_112 \
     --lfw-mode
   ```

2. **Update your ArcFace training script** to load from `lfw_aligned_112/` instead of raw images

3. **Disable geometric augmentation** in your data transforms (keep only color/blur augmentation)

4. **Verify input shape**: Ensure your network expects (B, 3, 112, 112) aligned faces
