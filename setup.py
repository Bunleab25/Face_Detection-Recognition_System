"""
Setup script for Face Detection & Recognition System
Helps initialize the project environment
"""

import os
import sys
import subprocess
import platform

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required!")
        return False
    
    print("✓ Python version is compatible")
    return True

def check_cuda():
    """Check if CUDA is available"""
    print_header("Checking CUDA Availability")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Device Count: {torch.cuda.device_count()}")
            print(f"  Device Name: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠ CUDA is not available - will use CPU")
            return False
    except ImportError:
        print("⚠ PyTorch not installed yet")
        return False

def install_requirements(use_gpu=True):
    """Install required packages"""
    print_header("Installing Requirements")
    
    if use_gpu:
        requirements_files = [os.path.join("requirements", "gpu.txt"), os.path.join("requirements", "fastapi.txt")]
        print("Installing with GPU support...")
    else:
        requirements_files = [os.path.join("requirements", "cpu.txt"), os.path.join("requirements", "fastapi.txt")]
        print("Installing with CPU support...")
    
    # Fallback to root requirements.txt if directory files aren't found
    if not any(os.path.exists(rf) for rf in requirements_files):
        requirements_files = ["requirements.txt"]
    
    for req_file in requirements_files:
        if os.path.exists(req_file):
            print(f"\nInstalling from {req_file}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
            print(f"✓ {req_file} installed successfully")
        else:
            print(f"⚠ {req_file} not found, skipping...")
    
    print("\n✓ All requirements installed")

def create_directories():
    """Create necessary directories"""
    print_header("Creating Project Directories")
    
    dirs = [
        "models",
        "runs",
        "runs/detect",
        "runs/train",
        "runs/arcface_train",
        "known_faces",
        "face_datasets",
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created: {dir_path}")

def download_yolo_model():
    """Download YOLO model if not exists"""
    print_header("Checking YOLO Model")
    
    model_path = os.path.join("models", "yolo11n.pt")
    
    if os.path.exists(model_path):
        print(f"✓ {model_path} already exists")
    else:
        print(f"Downloading {model_path}...")
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            print(f"✓ {model_path} downloaded successfully")
        except Exception as e:
            print(f"⚠ Could not download model: {e}")
            print("  You can manually download it later")

def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("  Face Detection & Recognition System")
    print("  Setup Script")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Ask for GPU installation
    use_gpu = input("\nDo you have a CUDA-capable GPU? (y/n): ").lower().strip() == 'y'
    
    # Install requirements
    try:
        install_requirements(use_gpu)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error installing requirements: {e}")
        print("Please check the error messages above and try again.")
        sys.exit(1)
    
    # Check CUDA after installation
    if use_gpu:
        check_cuda()
    
    # Create directories
    create_directories()
    
    # Download YOLO model
    download_yolo_model()
    
    # Final message
    print_header("Setup Complete!")
    print("✓ Your environment is ready!")
    print("\nNext steps:")
    print("  1. For face detection: python scripts/realtime.py --source 0 --device 0")
    print("  2. For web app: uvicorn app.main:app --reload")
    print("  3. See README.md for more information")
    print("\nHappy coding! 🚀\n")

if __name__ == "__main__":
    main()
