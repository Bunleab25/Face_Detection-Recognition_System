# Reference Datasets & Evaluation Protocols

This document details the standard benchmark datasets, acquisition steps, directory formats, and evaluation protocols for face detection (YOLO) and face recognition/verification (ArcFace).

---

## 1. Primary Reference Datasets

### A. Labeled Faces in the Wild (LFW) — Face Verification Benchmark
- **Paper**: *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments* (Huang et al., 2008)
- **Scope**: 13,233 web-collected face images across 5,749 identities.
- **Role in Project**: Training data and standard verification protocol. The local ArcFace checkpoint records an LFW-aligned training directory and 5,611 identity classes, so an LFW score is not an independent generalization result.
- **Official URL**: [http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/)
- **Standard Protocol**: 6,000 paired comparisons (3,000 matched genuine pairs, 3,000 mismatched impostor pairs) evaluated with 10-fold cross-validation.

#### Preprocessing & Alignment for ArcFace
```bash
# 1. Download LFW Deep Funneled images
wget http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
tar -xzf lfw-deepfunneled.tgz -C LFW_datasets/

# 2. Batch align & crop to 112x112 using RetinaFace/YOLO landmarks
python scripts/preprocess_arcface_dataset.py \
  --input-dir LFW_datasets/lfw-deepfunneled \
  --output-dir LFW_datasets/lfw_aligned_112 \
  --lfw-mode
```

#### LFW Evaluation Caveat

The standard 6,000-pair LFW protocol remains useful for diagnostics, but it is potentially contaminated for this checkpoint because the checkpoint metadata identifies LFW as its training source. Do not present an LFW score as an independent test result.

The evaluator below parses standard whitespace-delimited LFW pair files, caches each image embedding, selects thresholds only on the other folds, and reports fixed-threshold FAR/FRR, ROC AUC, EER, and cross-validated accuracy:

```bash
python scripts/evaluate_accuracy.py \
  --model models/arcface.pth \
  --pairs LFW_datasets/pairs.txt \
  --dataset-root LFW_datasets/lfw_aligned_112 \
  --output runs/eval/lfw_diagnostic.json
```

---

### CFP-W / CFP-FP — External Frontal-to-Profile Verification

CFP-W supplies 500 identities, 5,000 frontal images, 2,000 profile images, and official 10-fold protocols. Its FP protocol has 350 genuine and 350 impostor pairs per fold, with identities separated across folds. Calibrate each fold's threshold on the other nine folds and report the mean and standard deviation across held-out folds.

Download the dataset from the [official CFP-W page](http://www.cfpw.io/), extract it, then evaluate the full YOLO + ArcFace pipeline:

```bash
python scripts/evaluate_cfp.py \
  --dataset-root CFP/cfp-dataset \
  --protocol FP \
  --output runs/eval/cfp_fp.json
```

The report includes accuracy, ROC AUC, and EER on each official test fold, plus a diagnostic conditional score for pairs where both faces were detected. Detector failures count as rejections in the end-to-end result; the conditional diagnostic is not a substitute for that result.

#### Measured result for the local checkpoints

The current local YOLO + ArcFace pipeline scored **57.24% ± 1.17% accuracy**, **0.5870 ± 0.0143 ROC AUC**, and **42.80% ± 1.60% EER** on the official CFP-FP folds. Detection failed on 306 of 5,901 unique images, affecting 1,079 of 7,000 pairs. Recognition accuracy conditional on successful detection was **58.67% ± 1.21%**. These near-chance results do not support production identity verification; changing the threshold alone will not solve the poor separation.

---

### B. WIDER FACE — Face Detection Benchmark
- **Paper**: *WIDER FACE: A Face Detection Benchmark* (Yang et al., CVPR 2016)
- **Scope**: 32,203 images containing 393,703 annotated face bounding boxes across 61 event categories.
- **Evaluation Splits**: Divided into Easy, Medium, and Hard subsets based on scale, occlusions, and illumination extremes.
- **Role in Project**: Baseline training and validation benchmark for `models/yolo_face.pt`.
- **Official URL**: [http://shuoyang1213.me/WIDERFACE/](http://shuoyang1213.me/WIDERFACE/)

#### YOLO Dataset Directory Structure (`face_datasets/data.yaml`)
```yaml
path: face_datasets
train: images/train
val: images/val
test: images/test

names:
  0: face
```

Evaluate the detector using the official WIDER FACE validation difficulty splits and annotations:

```bash
python scripts/evaluate_wider.py \
  --images-dir WIDER_val/images \
  --eval-tools-zip eval_tools.zip \
  --model models/yolo_face.pt \
  --output runs/eval/wider_face_val.json
```

This evaluates the public validation set, not the hidden WIDER test set; do not describe it as an independent test if that validation split was used for model selection.

On the current local YOLO checkpoint, the fresh official-protocol validation AP at IoU 0.5 was **93.84% easy**, **91.50% medium**, and **80.15% hard**. These official WIDER AP values are not directly comparable to the stored Ultralytics mAP metrics because they use different evaluation protocols.

---

### C. Auxiliary Large-Scale Pretraining Benchmarks

| Dataset | Scale | Classes | Application | Reference |
|:--------|:------|:--------|:------------|:----------|
| **CASIA-WebFace** | 494,414 images | 10,575 subjects | Medium-scale ArcFace representation pretraining | Yi et al., 2014 |
| **MS-Celeb-1M (MS1MV2)** | 5.8M images | 85,742 subjects | Cleaned ArcFace deep representation benchmark | Guo et al., ECCV 2016 / Deng et al., CVPR 2019 |
| **CFP-FP** | 7,000 images | 500 subjects | Frontal-to-Profile extreme pose verification | Sengupta et al., WACV 2016 |
| **AgeDB-30** | 12,240 images | 568 subjects | Cross-age 30-year span verification benchmark | Moschoglou et al., CVPRW 2017 |

---

## 2. Evaluation Metrics & Definitions

### Detection Metrics (YOLO)
1. **Intersection over Union (IoU)**:
   ```text
   IoU = Area(B_pred ∩ B_gt) / Area(B_pred ∪ B_gt)
   ```
2. **Mean Average Precision (mAP@0.50)**:
   Area under the Precision-Recall curve across all evaluated classes at `IoU >= 0.50`.
3. **Distribution Focal Loss (DFL)** & **Complete IoU (CIoU)** for bounding box regression under extreme poses.

### Recognition Metrics (ArcFace)
1. **Cosine Similarity Metric**:
   ```text
   Similarity(z1, z2) = dot(z1, z2) / (||z1|| * ||z2||) = cos(θ)
   ```
2. **Verification Decision Rule**:
   - If `Similarity >= threshold` -> **Genuine Match (Same Person)**
   - If `Similarity < threshold` -> **Impostor Reject (Different Person)**
3. **False Acceptance Rate (FAR)** & **False Rejection Rate (FRR)**:
   - **FAR(threshold)**: Percentage of impostor pairs incorrectly accepted as genuine.
   - **FRR(threshold)**: Percentage of genuine pairs incorrectly rejected as impostors.
4. **Equal Error Rate (EER)**: The operational threshold where `FAR == FRR`.

---

## 3. Academic Citations (BibTeX)

```bibtex
@inproceedings{deng2019arcface,
  title     = {ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
  author    = {Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {4690--4699},
  year      = {2019}
}

@inproceedings{yang2016wider,
  title     = {WIDER FACE: A Face Detection Benchmark},
  author    = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {5525--5533},
  year      = {2016}
}

@techreport{huang2008lfw,
  title       = {Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments},
  author      = {Huang, Gary B. and Ramesh, Manu and Berg, Tamara and Learned-Miller, Erik},
  institution = {University of Massachusetts, Amherst},
  number      = {07-49},
  year        = {2008}
}
```
