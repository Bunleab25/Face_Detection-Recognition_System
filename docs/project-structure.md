# Project Structure

```text
.
├── app/                 FastAPI application, model service, and web assets
├── docs/                Setup, usage, architecture, and training guides
├── models/              YOLO and ArcFace model weights (Git LFS)
├── scripts/             Detection, recognition, training, and evaluation tools
├── web_app/             Alternate web-app launcher
├── .env.example         Safe template for local model configuration
├── CONTRIBUTING.md      Contribution guidelines
├── LICENSE              MIT license
├── README.md            Project overview and getting started
├── main_system.py       Standalone recognition entry point
├── requirements*.txt    Dependency lists for supported configurations
└── setup.py             Interactive environment setup helper
```

Datasets, generated training runs, local databases, and environment files belong
outside version control. Model weights are tracked with Git LFS.

## Documentation

- [ArcFace quick start](arcface_quickstart.md)
- [Face alignment guide](face_alignment_guide.md)
- [ArcFace architecture reference](arcface_architecture_reference.md)
- [Training and inference](step3_4_training_inference.md)
