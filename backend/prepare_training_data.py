#!/usr/bin/env python3
"""
Data Preparation and Training Readiness Guide
Helps organize your varicose vein dataset for training
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

def create_synthetic_samples(count=10):
    """Create synthetic sample images for testing (NOT for real training!)"""
    print(f"\n🔬 Creating {count} synthetic test samples...")
    
    data_dir = Path("data")
    
    for class_name in ["varicose", "normal"]:
        class_dir = data_dir / class_name
        
        for i in range(count // 2):  # Half in each class
            # Create synthetic image data
            if class_name == "varicose":
                # Reddish/purple tones for varicose
                img = np.random.randint(80, 180, (224, 224, 3), dtype=np.uint8)
                img[:, :, 0] = np.random.randint(120, 200, (224, 224))  # More red
                img[:, :, 2] = np.random.randint(100, 160, (224, 224))  # Some blue
            else:
                # More skin-toned for normal
                img = np.random.randint(150, 220, (224, 224, 3), dtype=np.uint8)
                img[:, :, 1] = np.random.randint(140, 200, (224, 224))  # More green/yellow
            
            # Add some texture/noise
            noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Save as PIL Image
            pil_img = Image.fromarray(img)
            filename = f"synthetic_{class_name}_{i:03d}.jpg"
            pil_img.save(class_dir / filename)
    
    print(f"✅ Created synthetic samples in data/varicose/ and data/normal/")
    print("⚠️  These are for TESTING ONLY - use real medical images for training!")

def check_image_validity(image_path):
    """Check if an image file is valid and readable"""
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify image integrity
        return True
    except Exception:
        return False

def organize_existing_images():
    """Help organize images from other directories"""
    print("\n📁 ORGANIZE EXISTING IMAGES")
    print("=" * 50)
    
    # Common image directories to check
    common_dirs = [
        Path.home() / "Downloads",
        Path.home() / "Pictures",
        Path.home() / "Desktop",
        Path(".") / "images",
        Path(".") / "dataset",
        Path(".") / "medical_images"
    ]
    
    found_images = []
    
    print("Scanning common directories for images...")
    for dir_path in common_dirs:
        if dir_path.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                found_images.extend(list(dir_path.glob(ext)))
    
    if found_images:
        print(f"Found {len(found_images)} images in common directories:")
        for img in found_images[:10]:  # Show first 10
            print(f"  {img}")
        if len(found_images) > 10:
            print(f"  ... and {len(found_images) - 10} more")
        
        print(f"\n💡 To organize these images:")
        print(f"  1. Copy varicose vein images to: data/varicose/")
        print(f"  2. Copy normal leg images to: data/normal/")
        print(f"  3. Aim for 2,500+ images per class for best results")
    else:
        print("No images found in common directories.")
        print("Please place your medical images in:")
        print("  - data/varicose/ (for varicose vein images)")
        print("  - data/normal/ (for normal leg images)")

def validate_medical_dataset():
    """Validate the medical dataset structure and quality"""
    print("\n🏥 MEDICAL DATASET VALIDATION")
    print("=" * 50)
    
    data_dir = Path("data")
    classes = ["varicose", "normal"]
    total_images = 0
    valid_images = 0
    
    for class_name in classes:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"❌ Missing directory: {class_dir}")
            continue
        
        # Count images
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_files.extend(list(class_dir.glob(ext)))
        
        class_count = len(image_files)
        total_images += class_count
        
        # Validate a sample of images
        sample_size = min(10, class_count)
        if sample_size > 0:
            valid_sample = 0
            for img_path in image_files[:sample_size]:
                if check_image_validity(img_path):
                    valid_sample += 1
            
            validity_rate = valid_sample / sample_size if sample_size > 0 else 0
            valid_images += int(class_count * validity_rate)
            
            print(f"📊 {class_name.capitalize()} class:")
            print(f"  Images: {class_count:,}")
            print(f"  Sample validity: {validity_rate:.1%}")
            
            if class_count < 100:
                print(f"  ⚠️  Very small dataset - need 1,000+ per class")
            elif class_count < 1000:
                print(f"  ⚠️  Small dataset - recommended 2,500+ per class")
            elif class_count >= 2500:
                print(f"  ✅ Excellent size!")
            else:
                print(f"  ✅ Good size")
        else:
            print(f"📊 {class_name.capitalize()} class: Empty")
    
    print(f"\n📈 Dataset Summary:")
    print(f"  Total images: {total_images:,}")
    print(f"  Estimated valid: {valid_images:,}")
    
    if total_images == 0:
        print(f"  Status: ❌ No dataset")
        return False
    elif total_images < 500:
        print(f"  Status: ⚠️  Very small - results will be limited")
        return False
    elif total_images < 2000:
        print(f"  Status: ⚠️  Small - may need more data for 95%+ accuracy")
        return True
    else:
        print(f"  Status: ✅ Good size for high accuracy training")
        return True

def estimate_training_requirements():
    """Estimate training time and requirements"""
    print("\n⏱️  TRAINING ESTIMATION")
    print("=" * 50)
    
    # Count actual images
    data_dir = Path("data")
    total_images = 0
    
    for class_name in ["varicose", "normal"]:
        class_dir = data_dir / class_name
        if class_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                total_images += len(list(class_dir.glob(ext)))
    
    if total_images == 0:
        print("No dataset to train on!")
        return
    
    print(f"Dataset size: {total_images:,} images")
    
    # Estimate training time (very rough estimates)
    if total_images < 1000:
        cpu_time = "2-4 hours"
        gpu_time = "30-60 minutes"
        expected_accuracy = "70-85%"
    elif total_images < 5000:
        cpu_time = "6-12 hours"
        gpu_time = "1-3 hours"
        expected_accuracy = "85-92%"
    else:
        cpu_time = "12-24 hours"
        gpu_time = "3-6 hours"
        expected_accuracy = "90-96%"
    
    print(f"Expected accuracy: {expected_accuracy}")
    print(f"Training time (CPU): {cpu_time}")
    print(f"Training time (GPU): {gpu_time}")
    
    # System recommendation
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ GPU available - training will be faster")
        else:
            print("⚠️  CPU only - training will be slower but still possible")
    except:
        print("❌ PyTorch not available")

def main():
    """Main data preparation function"""
    print("🏥 VARICOSE VEIN TRAINING DATA PREPARATION")
    print("=" * 80)
    print("This script helps prepare your medical image dataset for training")
    print("=" * 80)
    
    # Step 1: Check current dataset
    has_data = validate_medical_dataset()
    
    # Step 2: Offer to organize existing images
    organize_existing_images()
    
    # Step 3: Offer to create synthetic samples for testing
    if not has_data:
        print(f"\n🔬 TESTING OPTION")
        print("=" * 50)
        print("Since you don't have a dataset yet, I can create synthetic samples")
        print("for testing the training pipeline (NOT for real medical use!).")
        
        response = input("Create synthetic test samples? (y/N): ").lower().strip()
        if response in ['y', 'yes']:
            create_synthetic_samples(20)
            print("\n🎯 Next steps with synthetic data:")
            print("  1. Run: python train_medium_dataset.py")
            print("  2. This will test the training pipeline")
            print("  3. Replace with real medical images for production")
            
            # Re-validate after creating samples
            has_data = validate_medical_dataset()
    
    # Step 4: Training estimation
    if has_data:
        estimate_training_requirements()
        
        print(f"\n🚀 READY FOR TRAINING!")
        print("=" * 50)
        print("Your dataset is ready. To start training:")
        print("  1. Run: python train_medium_dataset.py")
        print("  2. Monitor training progress")
        print("  3. Best model will be saved as 'final_medium_varicose_model.pth'")
    else:
        print(f"\n📋 TODO: PREPARE DATASET")
        print("=" * 50)
        print("To prepare for training:")
        print("  1. Collect 2,500+ varicose vein images")
        print("  2. Collect 2,500+ normal leg images")  
        print("  3. Place in data/varicose/ and data/normal/ directories")
        print("  4. Run this script again to validate")
        print("  5. Then run: python train_medium_dataset.py")

if __name__ == "__main__":
    main()
