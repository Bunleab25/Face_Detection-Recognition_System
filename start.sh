#!/usr/bin/env bash
# ==============================================================================
# Face Detection & Recognition System - Ubuntu Launcher
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo "  Face Detection & Recognition System (Ubuntu Launcher)"
echo "=================================================================="

# 1. Detect Python Environment
PYTHON_BIN=""
if [ -x "/home/bunleab/miniconda3/envs/ai-ml/bin/python" ]; then
    PYTHON_BIN="/home/bunleab/miniconda3/envs/ai-ml/bin/python"
    echo "✔ Using conda environment: ai-ml ($PYTHON_BIN)"
elif [ -d "$SCRIPT_DIR/.venv" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
    echo "✔ Using virtualenv: .venv"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="$(command -v python3)"
    echo "✔ Using system Python: $PYTHON_BIN"
else
    echo "❌ Error: Python 3 not found!"
    exit 1
fi

# 2. Check & Create .env if missing
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "ℹ Creating default .env configuration..."
    cat <<EOF > "$SCRIPT_DIR/.env"
# Face Recognition System Configuration
DEVICE=cuda
FACE_API_KEY=my-secret-key-123
YOLO_MODEL_PATH=models/yolo_face.pt
ARCFACE_CKPT_PATH=models/arcface.pth
YOLO_CONF=0.5
HOST=127.0.0.1
PORT=8000
EOF
    echo "✔ Generated .env (Default API Key: my-secret-key-123)"
else
    echo "✔ Found existing .env file"
fi

# Export FACE_API_KEY if present in .env
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# 3. Check Model Weights
if [ ! -f "$SCRIPT_DIR/models/arcface.pth" ] || [ $(stat -c%s "$SCRIPT_DIR/models/arcface.pth" 2>/dev/null || stat -f%z "$SCRIPT_DIR/models/arcface.pth") -lt 1000000 ]; then
    echo "⚠️ Warning: models/arcface.pth appears to be missing or a Git LFS pointer."
    echo "Attempting to pull model weights via git lfs..."
    git lfs pull || echo "Failed to pull LFS weights. Please run 'git lfs pull' manually."
fi

# 4. Launch Application
echo "------------------------------------------------------------------"
echo "Starting web server on http://${HOST:-127.0.0.1}:${PORT:-8000} ..."
echo "API Key configured: ${FACE_API_KEY:-Not set}"
echo "------------------------------------------------------------------"

exec "$PYTHON_BIN" run.py
