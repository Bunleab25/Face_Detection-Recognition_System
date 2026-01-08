# Project Structure Visualization

```
face_detector/
│
├── 📄 Core Documentation
│   ├── README.md                     # Main project documentation
│   ├── LICENSE                       # MIT License
│   ├── CONTRIBUTING.md               # Contribution guidelines
│   ├── CHECKLIST.md                  # Pre-push checklist
│   ├── GITHUB_PREP_SUMMARY.md        # GitHub setup summary
│   ├── GIT_QUICK_REFERENCE.md        # Git commands reference
│   ├── QUICKSTART.md                 # Quick start guide
│   └── README_OLD.md                 # Backup of original README
│
├── 📦 Dependencies
│   ├── requirements.txt              # Core dependencies
│   ├── requirements-gpu.txt          # GPU/CUDA dependencies
│   ├── requirements-cpu.txt          # CPU-only dependencies
│   └── requirements_fastapi.txt      # Web app dependencies
│
├── 🔧 Configuration
│   ├── .gitignore                    # Git ignore rules
│   ├── .gitattributes                # Git LFS and file handling
│   └── .env                          # Environment variables (not in git)
│
├── 🚀 Setup Scripts
│   ├── setup.py                      # Automated setup script
│   ├── init_git.ps1                  # Git setup (Windows)
│   └── init_git.sh                   # Git setup (Linux/Mac)
│
├── 🤖 Main Application
│   ├── main_system.py                # Standalone recognition system
│   └── yolo11n.pt                    # YOLO model weights
│
├── 🌐 Web Application
│   └── app/
│       ├── __init__.py
│       ├── main.py                   # FastAPI endpoints
│       ├── face_service.py           # Face processing service
│       ├── database.py               # Database models (SQLAlchemy)
│       ├── model.py                  # ML model definitions
│       ├── config.py                 # Application configuration
│       └── static/                   # Web UI assets
│           ├── index.html
│           ├── styles.css
│           └── script.js
│
├── 🔬 Scripts & Tools
│   └── scripts/
│       ├── 🎯 Detection
│       │   ├── predict.py            # Batch prediction
│       │   ├── realtime.py           # Real-time detection
│       │   └── train.py              # Train YOLO detector
│       │
│       ├── 👤 Recognition
│       │   ├── realtime_recognition.py   # Real-time recognition
│       │   ├── register_identity.py      # Register new person
│       │   ├── inference_arcface.py      # ArcFace inference
│       │   └── train_arcface.py          # Train ArcFace model
│       │
│       ├── 📐 Face Alignment
│       │   ├── face_alignment.py         # Single image alignment
│       │   └── preprocess_arcface_dataset.py  # Batch alignment
│       │
│       ├── 📊 Evaluation
│       │   ├── evaluate_model.py         # Model evaluation
│       │   ├── evaluate_accuracy.py      # Accuracy metrics
│       │   └── demo_inference.py         # Demo script
│       │
│       ├── 🏋️ Training
│       │   ├── train_yolo11n_fast.py     # Fast YOLO training
│       │   └── train_yolov3_step_schedule.py  # YOLOv3 training
│       │
│       └── 🧠 Model
│           └── model.py              # ArcFace model architecture
│
├── 📚 Documentation
│   └── docs/
│       ├── arcface_quickstart.md         # ArcFace quick start
│       ├── face_alignment_guide.md       # Alignment tutorial
│       ├── arcface_architecture_reference.md
│       ├── arcface_alignment_implementation.md
│       ├── ARCFACE_README.md
│       ├── step2_arcface_architecture.md
│       └── step3_4_training_inference.md
│
├── 🗄️ Data (Excluded from Git)
│   ├── face_datasets/                # YOLO training data
│   │   ├── train/
│   │   ├── valid/
│   │   ├── test/
│   │   └── data.yaml
│   │
│   ├── LFW_datasets/                 # LFW dataset
│   │   └── lfw-deepfunneled/
│   │
│   ├── combined_dataset/             # Combined face dataset
│   │   ├── Person1/
│   │   ├── Person2/
│   │   └── ...
│   │
│   └── known_faces/                  # Registered faces
│
├── 💾 Output & Models (Excluded from Git)
│   ├── runs/                         # Training outputs
│   │   ├── detect/
│   │   ├── train/
│   │   └── arcface_train/
│   │
│   ├── *.db                          # SQLite databases
│   ├── *.npy                         # NumPy face databases
│   └── *.pth                         # Model checkpoints
│
├── 🔄 CI/CD
│   └── .github/
│       └── workflows/
│           └── python-app.yml        # GitHub Actions workflow
│
└── 🗑️ Ignored Files (.gitignore)
    ├── __pycache__/                  # Python cache
    ├── .venv/                        # Virtual environment
    ├── *.log                         # Log files
    ├── .env                          # Environment variables
    └── ...

```

## 📊 File Categories

### Essential Files (Must Review Before Push)
- ✅ README.md - Main documentation
- ✅ LICENSE - Project license
- ✅ .gitignore - Exclude large files
- ✅ requirements*.txt - Dependencies

### Documentation
- 📄 12 markdown files
- 📂 docs/ folder with detailed guides

### Source Code
- 🐍 17+ Python scripts
- 🌐 Web application (FastAPI)
- 🧠 ML models (YOLO, ArcFace)

### Configuration
- ⚙️ Git configuration files
- 🔧 Setup and initialization scripts
- 📝 CI/CD workflows

### Data (Not in Git)
- 📁 3 dataset directories
- 💾 Model weights
- 🗃️ Database files

## 📏 Project Size Estimates

| Component | Size | In Git? |
|-----------|------|---------|
| Source Code | ~2 MB | ✅ Yes |
| Documentation | ~500 KB | ✅ Yes |
| Dependencies | N/A | ❌ No (install via pip) |
| Datasets | ~500 MB - 2 GB | ❌ No (download separately) |
| Model Weights | ~10-100 MB | ⚠️ Optional (use Git LFS) |
| Outputs/Runs | Variable | ❌ No (generated locally) |

## 🎯 Recommended Git Strategy

### Include in Git
- All `.py` source files
- Documentation (`.md` files)
- Configuration files
- Requirements files
- Small sample images (optional)

### Exclude from Git (in .gitignore)
- Virtual environments (`.venv/`)
- Datasets (`face_datasets/`, `LFW_datasets/`)
- Training outputs (`runs/`)
- Database files (`*.db`, `*.npy`)
- Python cache (`__pycache__/`)

### Use Git LFS For
- Pre-trained model weights (`*.pt`, `*.pth`)
- Large binary files
- Optional: archive datasets

## 📦 Repository Size Target

**Recommended:** < 100 MB (without LFS)  
**Maximum:** < 1 GB (with LFS)

Current repository (without datasets/models): **~2-5 MB** ✅

---

**Last Updated:** January 8, 2026
