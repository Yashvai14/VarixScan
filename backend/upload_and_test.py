#!/usr/bin/env python3
"""
Script to help upload and test a varicose vein image
"""

import requests
import os
from pathlib import Path
import json

def upload_image_via_api(image_path):
    """Upload an image via the API and get analysis results"""
    
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return False
    
    # Check file size
    file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
    print(f"📁 Image file: {image_path}")
    print(f"📊 File size: {file_size:.2f} MB")
    
    if file_size > 10:
        print("⚠️  Warning: File is larger than 10MB limit")
        return False
    
    # Test the analysis
    url = "http://localhost:8000/analyze"
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            data = {
                "patient_id": 1,
                "language": "en"
            }
            
            print("🔄 Uploading and analyzing image...")
            response = requests.post(url, files=files, data=data, timeout=120)
            
            print(f"📊 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Analysis successful!")
                print(f"   📋 Diagnosis: {result.get('diagnosis', 'N/A')}")
                print(f"   🔍 Severity: {result.get('severity', 'N/A')}")
                print(f"   📈 Confidence: {result.get('confidence', 'N/A')}%")
                print(f"   🎯 Detection Count: {result.get('detection_count', 'N/A')}")
                print(f"   📐 Affected Area Ratio: {result.get('affected_area_ratio', 'N/A'):.3f}")
                
                # Show preprocessing info
                preprocessing = result.get('preprocessing_info', {})
                if preprocessing:
                    print(f"   🔬 Original Size: {preprocessing.get('original_shape', 'N/A')}")
                    print(f"   🔬 Skin Area Ratio: {preprocessing.get('skin_area_ratio', 'N/A'):.3f}")
                
                # Show recommendations
                recommendations = result.get('recommendations', [])
                if recommendations:
                    print("   💡 Recommendations:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"      {i}. {rec}")
                
                return True
            elif response.status_code == 422:
                print("❌ Validation error (422):")
                try:
                    error_data = response.json()
                    print(f"   📄 Error: {error_data}")
                except:
                    print(f"   📄 Raw response: {response.text}")
                return False
            else:
                print(f"❌ Analysis failed with status code: {response.status_code}")
                print(f"   📄 Response: {response.text}")
                return False
                
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure it's running: python main.py")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The image might be too large.")
        return False
    except Exception as e:
        print(f"❌ Upload failed with error: {e}")
        return False

def check_server_status():
    """Check if the server is running and responsive"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and responsive")
            return True
        else:
            print(f"⚠️  Server responding but with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server not running on port 8000")
        return False
    except Exception as e:
        print(f"❌ Server check failed: {e}")
        return False

def main():
    print("🩺 Varicose Vein Image Upload and Test Tool")
    print("=" * 60)
    
    # Check server
    if not check_server_status():
        print("\n💡 To start the server, run: python main.py")
        return
    
    print("\n📂 How to test your varicose vein image:")
    print("1. Save the image to your computer (e.g., as 'varicose_legs.jpg')")
    print("2. Put the image file in this folder or specify the full path")
    print("3. Run this script with the image filename")
    
    # Look for common image files
    current_dir = Path(".")
    common_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    found_images = []
    
    for ext in common_extensions:
        found_images.extend(list(current_dir.glob(ext)))
    
    if found_images:
        print(f"\n📁 Found images in current directory:")
        for i, img in enumerate(found_images, 1):
            size_mb = img.stat().st_size / (1024 * 1024)
            print(f"   {i}. {img.name} ({size_mb:.2f} MB)")
        
        # Test the first reasonable sized image
        for img in found_images:
            size_mb = img.stat().st_size / (1024 * 1024)
            if 0.1 < size_mb < 10:  # Between 0.1MB and 10MB
                print(f"\n🧪 Testing image: {img.name}")
                upload_image_via_api(str(img))
                break
        else:
            print("\n⚠️  No suitable images found (need 0.1-10 MB)")
    else:
        print("\n📂 No images found in current directory")
        print("💡 Please save your varicose vein image here and run the script again")
    
    print("\n" + "=" * 60)
    print("💡 Usage examples:")
    print("   python upload_and_test.py")
    print("   # Then manually specify image path in the code if needed")

if __name__ == "__main__":
    main()
