#!/usr/bin/env python3
import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path

def check_dependencies():
    """Check and install required dependencies"""
    required = [
        "PySide6",
        "opencv-python",
        "opencv-contrib-python",
        "torch",
        "torchvision",
        "numpy",
        "onnxruntime",
        "onnxruntime-gpu",  # Optional for GPU support
        "tensorflow",
       # "protobuf<4.0.0"  # TensorFlow compatibility
        "protobuf"  # Tens
    ]

    for package in required:
        try:
            if package == "tensorflow":
                __import__("tensorflow")
            elif package == "onnxruntime":
                __import__("onnxruntime")
            else:
                __import__(package.lower().replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except:
                print(f"Failed to install {package}. Trying with --user flag...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package])

def create_directories():
    """Create necessary directories"""
    dirs = ["models", "haarcascades", "assets", "exports", "captures"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)

def download_required_files():
    """Download required model files if they don't exist"""
    files_to_download = {
        # Face detection models
        "haarcascades/haarcascade_frontalface_default.xml":
            "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",

        "models/deploy.prototxt":
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",

        "models/res10_300x300_ssd_iter_140000.caffemodel":
            "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",

        # Sample emotion models (you should replace these with your actual models)
        "models/emotion_model.onnx":
            "https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
    }

    for file_path, url in files_to_download.items():
        if not Path(file_path).exists():
            print(f"Downloading {file_path}...")
            try:
                os.makedirs(Path(file_path).parent, exist_ok=True)
                urllib.request.urlretrieve(url, file_path)
                print(f"Downloaded {file_path}")
            except Exception as e:
                print(f"Failed to download {file_path}: {str(e)}")
                print(f"You can manually download it from: {url}")

def check_models():
    """Check for model files"""
    model_extensions = ['.pt', '.onnx', '.pb', '.h5', '.pth']
    model_files = []

    for ext in model_extensions:
        model_files.extend(list(Path("models").glob(f"*{ext}")))

    if not model_files:
        print("\n  No emotion models found in models/ directory!")
        print("Please place your trained models in the models/ folder:")
        print("  - PyTorch: *.pt, *.pth")
        print("  - ONNX: *.onnx")
        print("  - TensorFlow: *.pb, *.h5")
        print("\nThe application will use mock data until models are added.")
    else:
        print(f"\n Found {len(model_files)} model files:")
        for model in model_files:
            print(f"  - {model.name}")

def check_gpu_support():
    """Check for GPU support"""
    print("\n" + "="*50)
    print("Hardware Acceleration Check")
    print("="*50)

    # Check PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"PyTorch GPU: CUDA {torch.version.cuda}")
            print(f"Device: {torch.cuda.get_device_name(0)}")
        else:
            print(" PyTorch GPU: Not available (CPU only)")
    except:
        print(" PyTorch not installed properly")

    # Check ONNX Runtime GPU
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            print("ONNX Runtime GPU: Available")
        else:
            print("ONNX Runtime GPU: Not available")
    except:
        print("ONNX Runtime not installed")

    # Check OpenCV CUDA
    try:
        import cv2
        cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_device_count > 0:
            print(f"OpenCV CUDA: {cuda_device_count} device(s)")
        else:
            print("OpenCV CUDA: Not available")
    except:
        print("OpenCV not installed properly")

def main():
    print("=" * 60)
    print("Emotion AI Studio Pro - Initialization")
    print("=" * 60)

    # Check dependencies
    print("\n📦 Checking dependencies...")
    check_dependencies()

    # Create directories
    print("\nCreating directories...")
    create_directories()

    # Download required files
    print("\nDownloading required files...")
    download_required_files()

    # Check models
    print("\nChecking emotion models...")
    check_models()

    # Check GPU support
    check_gpu_support()

    # Run the application
    print("\n" + "="*60)
    print("Starting BUKEmotionAI Studio Pro...")
    print("="*60 + "\n")

    try:
        from main import main as app_main
        sys.exit(app_main())
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure all dependencies are installed")
        print("2. Check if you have a compatible GPU driver")
        print("3. Try running with: python3 -m pip install --upgrade pip")
        print("4. Check Python version (requires 3.8+)\n")
        raise

if __name__ == "__main__":
    main()
