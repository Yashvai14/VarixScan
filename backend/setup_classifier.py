#!/usr/bin/env python3
"""
Quick Setup Script for High-Performance Varicose Vein Classifier
Helps you get started with the training system quickly.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3.8, 0):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_gpu_availability():
    """Check if CUDA GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠️  No CUDA GPU detected. Training will use CPU (slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet. GPU check will be performed after installation.")
        return False

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    requirements = [
        "torch==2.0.1",
        "torchvision==0.15.2", 
        "timm==0.9.7",
        "opencv-python==4.8.1.78",
        "albumentations==1.3.1",
        "Pillow==10.0.0",
        "scikit-image==0.21.0",
        "scikit-learn==1.3.0",
        "imbalanced-learn==0.11.0",
        "numpy==1.24.3",
        "pandas==2.0.3",
        "matplotlib==3.7.1",
        "seaborn==0.12.2"
    ]
    
    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All packages installed successfully!")
    return True

def create_data_structure():
    """Create the required data directory structure"""
    print("\n📁 Creating data directory structure...")
    
    data_dir = Path("data")
    varicose_dir = data_dir / "varicose"
    normal_dir = data_dir / "normal"
    
    # Create directories
    varicose_dir.mkdir(parents=True, exist_ok=True)
    normal_dir.mkdir(parents=True, exist_ok=True)
    
    # Create readme files
    with open(varicose_dir / "README.txt", "w") as f:
        f.write("Place varicose vein images here.\n")
        f.write("Supported formats: .jpg, .jpeg, .png\n")
        f.write("Recommended: 1000+ high-quality images\n")
    
    with open(normal_dir / "README.txt", "w") as f:
        f.write("Place normal leg images here.\n")
        f.write("Supported formats: .jpg, .jpeg, .png\n")
        f.write("Recommended: 1000+ high-quality images\n")
    
    print(f"✅ Created data structure:")
    print(f"  - {varicose_dir}/")
    print(f"  - {normal_dir}/")
    return True

def check_dataset():
    """Check if dataset is available"""
    print("\n🔍 Checking dataset...")
    
    varicose_dir = Path("data/varicose")
    normal_dir = Path("data/normal")
    
    if not varicose_dir.exists() or not normal_dir.exists():
        print("❌ Data directories not found")
        return False
    
    # Count images
    varicose_images = list(varicose_dir.glob("*.jpg")) + \
                     list(varicose_dir.glob("*.jpeg")) + \
                     list(varicose_dir.glob("*.png"))
    
    normal_images = list(normal_dir.glob("*.jpg")) + \
                   list(normal_dir.glob("*.jpeg")) + \
                   list(normal_dir.glob("*.png"))
    
    varicose_count = len(varicose_images)
    normal_count = len(normal_images)
    
    print(f"Varicose images: {varicose_count}")
    print(f"Normal images: {normal_count}")
    print(f"Total images: {varicose_count + normal_count}")
    
    if varicose_count == 0 and normal_count == 0:
        print("⚠️  No images found. Please add images to data directories.")
        return False
    elif min(varicose_count, normal_count) < 100:
        print("⚠️  Very small dataset. Need at least 100 images per class for basic training.")
        print("   Recommended: 1000+ images per class for optimal performance.")
        return True
    elif min(varicose_count, normal_count) < 1000:
        print("⚠️  Small dataset. Consider adding more images for better performance.")
        return True
    else:
        print("✅ Good dataset size for high-performance training!")
        return True

def create_sample_training_script():
    """Create a simple training script"""
    print("\n📝 Creating sample training script...")
    
    sample_script = """#!/usr/bin/env python3
# Sample training script - customize as needed

from train_varicose_classifier import VaricoseTrainer
import os

def main():
    # Training configuration - adjust as needed
    config = {
        'batch_size': 16,        # Reduce if GPU memory issues
        'learning_rate': 1e-4,   # Learning rate
        'epochs': 50,            # Start with fewer epochs for testing
        'dropout_rate': 0.5,     # Regularization
        'weight_decay': 0.01,    # L2 regularization
        'patience': 10,          # Early stopping patience
        'use_focal_loss': True,  # Handle class imbalance
    }
    
    # Check if data directory exists
    if not os.path.exists('data'):
        print("❌ Data directory not found!")
        print("Please run setup_classifier.py first to create the data structure.")
        return
    
    # Initialize and train
    print("🚀 Starting training with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    trainer = VaricoseTrainer(config)
    train_loader, val_loader, class_weights = trainer.prepare_data('data')
    
    print(f"\\n📊 Dataset loaded successfully!")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Start training
    trainer.train(train_loader, val_loader, class_weights)
    
    print("\\n🎉 Training completed!")
    print("Check the generated files:")
    print("  - best_varicose_model.pth: Best model during training")
    print("  - final_varicose_model.pth: Final model with optimal threshold")
    print("  - confusion_matrix.png: Visual evaluation results")
    print("  - threshold_analysis.csv: Threshold optimization results")

if __name__ == "__main__":
    main()
"""
    
    with open("quick_train.py", "w") as f:
        f.write(sample_script)
    
    print("✅ Created quick_train.py - A simplified training script")
    return True

def run_quick_test():
    """Run a quick test to ensure everything is working"""
    print("\n🧪 Running quick compatibility test...")
    
    try:
        # Test imports
        import torch
        import torchvision
        import timm
        import cv2
        import albumentations
        import sklearn
        
        print("✅ All core libraries imported successfully!")
        
        # Test GPU
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - will use CPU")
        
        # Test model creation
        model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=2)
        print("✅ EfficientNet model creation successful!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        return False

def print_next_steps():
    """Print instructions for next steps"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\n📋 NEXT STEPS:")
    print()
    print("1. 📁 ADD YOUR DATASET:")
    print("   - Place varicose vein images in: data/varicose/")
    print("   - Place normal leg images in: data/normal/")
    print("   - Aim for 1000+ images per class for best results")
    print()
    print("2. 🏋️ START TRAINING:")
    print("   - Quick start: python quick_train.py")
    print("   - Full training: python train_varicose_classifier.py")
    print("   - Custom config: Edit the config dict in the script")
    print()
    print("3. 📊 MONITOR PROGRESS:")
    print("   - Watch the training metrics in real-time")
    print("   - Training will stop automatically when optimal performance is reached")
    print("   - Check confusion_matrix.png for visual results")
    print()
    print("4. 🚀 DEPLOY MODEL:")
    print("   - Use optimized_ml_model.py for inference")
    print("   - Integrate with your existing FastAPI system")
    print("   - Replace the current ml_model detector")
    print()
    print("📖 For detailed instructions, see: VARICOSE_CLASSIFIER_README.md")
    print()
    print("🎯 TARGET PERFORMANCE:")
    print("   ✅ 95%+ Overall Accuracy")
    print("   ✅ 90%+ Varicose Recall")
    print("   ✅ 85%+ Confidence Scores")
    print()
    print("Good luck with your training! 🚀")

def main():
    """Main setup function"""
    print("🩺 High-Performance Varicose Vein Classifier Setup")
    print("=" * 60)
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Check GPU
    check_gpu_availability()
    
    # Step 3: Install requirements
    print("\n" + "="*60)
    response = input("📦 Install required packages? This may take a few minutes. (y/n): ")
    if response.lower().startswith('y'):
        if not install_requirements():
            print("❌ Package installation failed. Please check your Python environment.")
            sys.exit(1)
    else:
        print("⚠️  Skipping package installation. Make sure all dependencies are installed.")
    
    # Step 4: Create data structure
    create_data_structure()
    
    # Step 5: Check dataset
    check_dataset()
    
    # Step 6: Create sample scripts
    create_sample_training_script()
    
    # Step 7: Run compatibility test
    if not run_quick_test():
        print("❌ Compatibility test failed. Please check your installation.")
        return
    
    # Step 8: Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
