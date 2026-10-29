# Reference Datasets & Evaluation Protocols

This document details the standard benchmark datasets, acquisition steps, directory formats, and evaluation protocols for face detection (YOLO) and face recognition/verification (ArcFace).

---

## 1. Primary Reference Datasets

### A. Labeled Faces in the Wild (LFW) — Face Verification Benchmark
- **Paper**: *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments* (Huang et al., 2008)
- **Scope**: 13,233 web-collected face images across 5,749 identities.
- **Role in Project**: Primary training set and verification benchmark for `models/arcface.pth` (5,749 classes).
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

#### Running LFW Verification Evaluation
```bash
python scripts/inference_arcface.py \
  --model models/arcface.pth \
  --pairs LFW_datasets/pairs.txt \
  --dataset-root LFW_datasets/lfw_aligned_112/
```

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

---

### C. Auxiliary Large-Scale Pretraining Benchmarks

| Dataset | Scale | Classes | Application | Reference |
|:--------|:------|:--------|:------------|:----------|
| **CASIA-WebFace** | 494,414 images | 10,575 subjects | Medium-scale ArcFace representation pretraining | Yi et al., 2014 |
| **MS-Celeb-1M (MS1MV2)** | 5.8M images | 85,742 subjects | Cleaned ArcFace deep representation benchmark | Guo et al., ECCV 2016 / Deng et al., CVPR 2019 |
| **CFP-FP** | 7,000 images | 500 subjects | Frontal-to-Profile extreme pose verification | Sengupta et al., WACV 2016 |
| **AgeDB-30** | 12,240 images | 568 subjects | Cross-age 30-year span verification benchmark | Moschoglou et al., CVPRW 2017 |

---

## 2. Evaluation Metrics & Theoretical Definitions

### Detection Metrics (YOLO)
1. **Intersection over Union (IoU)**:
   $$\text{IoU} = \frac{\text{Area}(\mathcal{B}_{\text{pred}} \cap \mathcal{B}_{\text{gt}})}{\text{Area}(\mathcal{B}_{\text{pred}} \cup \mathcal{B}_{\text{gt}})}$$
2. **Mean Average Precision (mAP@0.50)**:
   $$\text{AP} = \int_{0}^{1} p(r) \, dr, \quad \text{mAP} = \frac{1}{C}\sum_{k=1}^{C} \text{AP}_k$$
3. **Distribution Focal Loss (DFL)** & **Complete IoU (CIoU)** for robust bounding box regression under unconstrained poses.

### Recognition Metrics (ArcFace)
1. **Cosine Similarity Metric**:
   $$\mathcal{S}(\hat{\mathbf{z}}_1, \hat{\mathbf{z}}_2) = \frac{\mathbf{z}_1^\top \mathbf{z}_2}{\|\mathbf{z}_1\|_2 \|\mathbf{z}_2\|_2} = \cos(\theta)$$
2. **Verification Decision Rule**:
   $$\hat{y} = \begin{cases} 1 \quad (\text{Genuine Match}), & \text{if } \mathcal{S} \ge \tau \\ 0 \quad (\text{Impostor Reject}), & \text{if } \mathcal{S} < \tau \end{cases}$$
3. **False Acceptance Rate (FAR)** & **False Rejection Rate (FRR)**:
   $$\text{FAR}(\tau) = \frac{|\{ (\mathbf{z}_i, \mathbf{z}_j) \in \mathcal{D}_{\text{impostor}} \mid \mathcal{S} \ge \tau \}|}{|\mathcal{D}_{\text{impostor}}|}$$
   $$\text{FRR}(\tau) = \frac{|\{ (\mathbf{z}_i, \mathbf{z}_j) \in \mathcal{D}_{\text{genuine}} \mid \mathcal{S} < \tau \}|}{|\mathcal{D}_{\text{genuine}}|}$$
4. **Equal Error Rate (EER)**: The operational threshold where $\text{FAR}(\tau) = \text{FRR}(\tau)$.

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
