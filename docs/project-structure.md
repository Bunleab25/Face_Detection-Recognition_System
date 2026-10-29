# Project Structure

```text
.
├── app/                 FastAPI application, model service, and web assets
├── assets/              Empirical screenshots, UI telemetry, and research diagrams
├── docs/                Setup, usage, architecture, and training guides
├── models/              YOLO and ArcFace model weights (Git LFS)
├── requirements/        Modular dependency files (base, cpu, gpu, fastapi)
├── scripts/             Detection, recognition, training, and evaluation tools
├── tests/               Unit and integration test suites
├── CONTRIBUTING.md      Contribution guidelines
├── LICENSE              MIT license
├── README.md            Project overview and academic documentation
├── requirements.txt     Root dependency manifest
├── run.py               Cross-platform service bootstrap
├── setup.py             Interactive environment setup helper
├── start.bat            Windows one-click launcher
└── start.sh             Linux / Ubuntu launcher
```

Datasets, generated training runs, local databases, and environment files belong
outside version control. Model weights are tracked with Git LFS.

## Documentation

- [Empirical hardware benchmarks](BENCHMARKS.md)
- [Reference datasets & evaluation protocols](REFERENCE_DATASETS.md)
- [ArcFace quick start](arcface_quickstart.md)
- [Face alignment guide](face_alignment_guide.md)
- [ArcFace architecture reference](arcface_architecture_reference.md)
- [Training and inference](step3_4_training_inference.md)
