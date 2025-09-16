#!/usr/bin/env python3
"""
Pre-Training Setup and Validation for Medium Dataset
Ensures your system is ready for 2-4 hour training session
"""

import os
import sys
import subprocess
import psutil
from pathlib import Path
import time

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def check_system_requirements():
    """Check if system meets requirements for medium dataset training"""
    print("🔍 SYSTEM REQUIREMENTS CHECK")
    print("=" * 50)
    
    checks_passed = 0
    total_checks = 6
    
    # 1. Python version
    if sys.version_info >= (3, 8):
        print("✅ Python version: OK")
        checks_passed += 1
    else:
        print("❌ Python version: Need 3.8+")
    
    # 2. GPU availability
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        checks_passed += 1
    else:
        if TORCH_AVAILABLE:
            print("⚠️  GPU: Not available (CPU training will be slower)")
        else:
            print("❌ GPU: Cannot check (PyTorch not installed)")
    
    # 3. RAM check
    ram_gb = psutil.virtual_memory().total / 1024**3
    if ram_gb >= 8:
        print(f"✅ RAM: {ram_gb:.1f}GB (sufficient)")
        checks_passed += 1
    else:
        print(f"⚠️  RAM: {ram_gb:.1f}GB (8GB+ recommended)")
    
    # 4. Disk space
    disk_free = psutil.disk_usage('.').free / 1024**3
    if disk_free >= 10:
        print(f"✅ Disk Space: {disk_free:.1f}GB free")
        checks_passed += 1
    else:
        print(f"⚠️  Disk Space: {disk_free:.1f}GB (10GB+ recommended)")
    
    # 5. CPU cores
    cpu_count = os.cpu_count()
    if cpu_count >= 4:
        print(f"✅ CPU Cores: {cpu_count} cores")
        checks_passed += 1
    else:
        print(f"⚠️  CPU Cores: {cpu_count} (4+ recommended)")
    
    # 6. Check PyTorch installation
    if TORCH_AVAILABLE:
        try:
            import torchvision
            print(f"✅ PyTorch: {torch.__version__}")
            checks_passed += 1
        except ImportError:
            print("⚠️  PyTorch installed but torchvision missing")
    else:
        print("❌ PyTorch: Not installed")
    
    print(f"\nSystem Readiness: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed >= 4:
        print("🚀 System ready for training!")
        return True
    else:
        print("⚠️  System may have issues during training")
        return False

def install_dependencies():
    """Install required dependencies for medium training"""
    print("\n📦 DEPENDENCY INSTALLATION")
    print("=" * 50)
    
    required_packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "opencv-python>=4.8.0",
        "albumentations>=1.3.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0"
    ]
    
    print("Installing required packages...")
    for package in required_packages:
        try:
            print(f"  Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"  ⚠️  Failed to install {package}")
    
    print("✅ Dependencies installation completed")

def validate_data_structure():
    """Validate data directory structure and count images"""
    print("\n📁 DATA VALIDATION")
    print("=" * 50)
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found!")
        print("Creating data structure...")
        data_dir.mkdir(exist_ok=True)
        (data_dir / "varicose").mkdir(exist_ok=True)
        (data_dir / "normal").mkdir(exist_ok=True)
        
        # Create README files
        with open(data_dir / "varicose" / "README.txt", "w") as f:
            f.write("Place varicose vein images here (.jpg, .png, .jpeg)\n")
        with open(data_dir / "normal" / "README.txt", "w") as f:
            f.write("Place normal leg images here (.jpg, .png, .jpeg)\n")
            
        print("✅ Data directory structure created")
        return False
    
    # Count images
    varicose_images = len(list((data_dir / "varicose").glob("*.[jp][pn]g"))) + \
                     len(list((data_dir / "varicose").glob("*.jpeg")))
    normal_images = len(list((data_dir / "normal").glob("*.[jp][pn]g"))) + \
                   len(list((data_dir / "normal").glob("*.jpeg")))
    total_images = varicose_images + normal_images
    
    print(f"📊 Dataset Summary:")
    print(f"  Varicose images: {varicose_images:,}")
    print(f"  Normal images: {normal_images:,}")
    print(f"  Total images: {total_images:,}")
    
    # Validate dataset size
    if total_images == 0:
        print("❌ No images found!")
        print("Please add images to data/varicose/ and data/normal/ directories")
        return False
    elif total_images < 1000:
        print("⚠️  Small dataset - results may be limited")
        print("Recommended: 2,500+ images per class for optimal performance")
    elif total_images >= 5000:
        print("✅ Excellent dataset size for high-performance training!")
    else:
        print("✅ Good dataset size")
    
    # Check class balance
    if varicose_images > 0 and normal_images > 0:
        ratio = max(varicose_images, normal_images) / min(varicose_images, normal_images)
        if ratio <= 3:
            print(f"✅ Class balance: Good (ratio {ratio:.1f}:1)")
        else:
            print(f"⚠️  Class imbalance: {ratio:.1f}:1 (may affect performance)")
    
    return total_images > 0

def test_training_components():
    """Test that all training components work"""
    print("\n🧪 TRAINING COMPONENTS TEST")
    print("=" * 50)
    
    try:
        # Test imports
        from train_varicose_classifier import (
            EfficientNetVaricoseClassifier, FocalLoss, AdvancedVaricoseDataset
        )
        print("✅ Training components imported successfully")
        
        # Test model creation
        model = EfficientNetVaricoseClassifier(num_classes=2)
        print("✅ Model creation: OK")
        
        # Test loss function
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("✅ Loss function: OK")
        
        # Test GPU if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            model = model.cuda()
            print("✅ GPU model transfer: OK")
        
        print("✅ All training components working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Training component test failed: {e}")
        return False

def estimate_training_time():
    """Estimate training time based on system specs"""
    print("\n⏱️  TRAINING TIME ESTIMATION")
    print("=" * 50)
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        if "rtx" in gpu_name or "tesla" in gpu_name or "v100" in gpu_name:
            estimated_time = "2-3 hours"
            speed_rating = "🚀 Fast"
        elif "gtx" in gpu_name or "quadro" in gpu_name:
            estimated_time = "3-5 hours"
            speed_rating = "⚡ Good"
        else:
            estimated_time = "4-8 hours"
            speed_rating = "🐌 Slow"
    else:
        estimated_time = "8-16 hours"
        speed_rating = "🐌 Very Slow (CPU only)"
    
    print(f"Estimated training time: {estimated_time}")
    print(f"Performance rating: {speed_rating}")
    
    return estimated_time

def create_training_config():
    """Create optimized training configuration"""
    print("\n⚙️  TRAINING CONFIGURATION")
    print("=" * 50)
    
    # Base configuration
    config = {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 50,
        'dropout_rate': 0.5,
        'weight_decay': 0.01,
        'patience': 12,
        'use_focal_loss': True,
    }
    
    # Adjust based on system capabilities
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory >= 8:
            config['batch_size'] = 32
            print("✅ GPU memory sufficient for batch size 32")
        elif gpu_memory >= 4:
            config['batch_size'] = 16
            print("⚠️  Reduced batch size to 16 due to GPU memory")
        else:
            config['batch_size'] = 8
            print("⚠️  Reduced batch size to 8 due to limited GPU memory")
    else:
        config['batch_size'] = 8
        print("⚠️  CPU training - using small batch size")
    
    print(f"Final configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return config

def main():
    """Main setup function"""
    print("🏥 MEDIUM DATASET TRAINING SETUP")
    print("=" * 80)
    print("This script prepares your system for 2-4 hour training session")
    print("Target: 95%+ accuracy with 90%+ varicose recall")
    print("=" * 80)
    
    # Step 1: Check system requirements
    system_ready = check_system_requirements()
    
    # Step 2: Install dependencies if needed
    if TORCH_AVAILABLE:
        try:
            import torchvision
            import timm
            print("\n✅ Core dependencies already installed")
        except ImportError:
            print("\n📦 Installing missing dependencies...")
            install_dependencies()
    else:
        print("\n📦 Installing missing dependencies...")
        install_dependencies()
    
    # Step 3: Validate data
    data_ready = validate_data_structure()
    
    # Step 4: Test training components
    components_ready = test_training_components()
    
    # Step 5: Estimate training time
    estimated_time = estimate_training_time()
    
    # Step 6: Create configuration
    config = create_training_config()
    
    # Final readiness check
    print("\n" + "=" * 80)
    print("🎯 SETUP SUMMARY")
    print("=" * 80)
    
    ready_items = []
    if system_ready:
        ready_items.append("✅ System requirements")
    else:
        ready_items.append("⚠️  System requirements (may have issues)")
    
    if data_ready:
        ready_items.append("✅ Dataset available")
    else:
        ready_items.append("❌ Dataset missing")
    
    if components_ready:
        ready_items.append("✅ Training components")
    else:
        ready_items.append("❌ Training components failed")
    
    for item in ready_items:
        print(f"  {item}")
    
    print(f"\n⏱️  Estimated training time: {estimated_time}")
    
    if data_ready and components_ready:
        print(f"\n🚀 READY TO START TRAINING!")
        print(f"Run: python train_medium_dataset.py")
        print(f"\n💡 Pro tips:")
        print(f"  - Training will run for {estimated_time}")
        print(f"  - Model will save checkpoints every 5 epochs")
        print(f"  - Early stopping will prevent overfitting")
        print(f"  - You can interrupt with Ctrl+C and resume from checkpoints")
    else:
        print(f"\n❌ NOT READY FOR TRAINING")
        if not data_ready:
            print(f"  🔧 Add images to data/varicose/ and data/normal/")
        if not components_ready:
            print(f"  🔧 Fix component installation issues")
    
    print(f"\n📚 Next steps:")
    print(f"  1. Ensure you have 2,500+ images per class")
    print(f"  2. Run: python train_medium_dataset.py")
    print(f"  3. Monitor training progress")
    print(f"  4. Deploy final_medium_varicose_model.pth")

if __name__ == "__main__":
    main()
