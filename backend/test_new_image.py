#!/usr/bin/env python3
"""
Test the new varicose vein image
"""

import requests
import os
from pathlib import Path

def test_new_image():
    """Test the new image with actual varicose veins"""
    
    # Check if server is running
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ Server not responding properly")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server not running on port 8000. Start it with: python main.py")
        return False
    
    # Look for the new image file
    # The image should be saved in uploads directory
    uploads_dir = Path("uploads")
    
    # List all available images
    if uploads_dir.exists():
        images = list(uploads_dir.glob("*.jpg")) + list(uploads_dir.glob("*.png"))
        print(f"📁 Available images in uploads: {[img.name for img in images]}")
        
        if not images:
            print("❌ No images found. Please save the varicose vein image to the uploads folder first.")
            return False
        
        # Use the most recent image or let user specify
        test_image = images[-1]  # Use the last (most recent) image
        
    else:
        print("❌ No uploads directory found")
        return False
    
    print(f"🧪 Testing analysis with image: {test_image}")
    
    # Test the analysis
    url = "http://localhost:8000/analyze"
    
    try:
        with open(test_image, "rb") as f:
            file_content = f.read()
        
        files = {"file": ("varicose_test.jpg", file_content, "image/jpeg")}
        data = {
            "patient_id": 1,
            "language": "en"
        }
        
        print("🔄 Sending request to analysis endpoint...")
        response = requests.post(url, files=files, data=data, timeout=60)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis successful!")
            print(f"   📋 Diagnosis: {result.get('diagnosis', 'N/A')}")
            print(f"   🔍 Severity: {result.get('severity', 'N/A')}")  
            print(f"   📈 Confidence: {result.get('confidence', 'N/A')}%")
            print(f"   🎯 Detection Count: {result.get('detection_count', 'N/A')}")
            print(f"   📐 Affected Area Ratio: {result.get('affected_area_ratio', 'N/A'):.3f}")
            
            # Show recommendations
            recommendations = result.get('recommendations', [])
            if recommendations:
                print("   💡 Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"      {i}. {rec}")
            
            return True
        else:
            print(f"❌ Analysis failed with status code: {response.status_code}")
            print(f"   📄 Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The image might be too large or the server is overloaded.")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_direct_analysis():
    """Test direct ML model analysis"""
    print("\n🧪 Testing direct ML model analysis...")
    
    try:
        from ml_model import VaricoseVeinDetector
        
        uploads_dir = Path("uploads")
        images = list(uploads_dir.glob("*.jpg")) + list(uploads_dir.glob("*.png"))
        
        if not images:
            print("❌ No images found")
            return False
        
        test_image = images[-1]
        print(f"📁 Using image: {test_image}")
        
        detector = VaricoseVeinDetector()
        result = detector.detect_veins(str(test_image))
        
        print("✅ Direct analysis successful!")
        print(f"   📋 Diagnosis: {result.get('diagnosis', 'N/A')}")
        print(f"   🔍 Severity: {result.get('severity', 'N/A')}")
        print(f"   📈 Confidence: {result.get('confidence', 'N/A')}%")
        print(f"   🎯 Detection Count: {result.get('detection_count', 'N/A')}")
        print(f"   📐 Skin Area Ratio: {result.get('preprocessing_info', {}).get('skin_area_ratio', 'N/A'):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🩺 Testing Varicose Vein Analysis with New Image")
    print("=" * 60)
    
    # Test direct model first
    direct_success = test_direct_analysis()
    
    print("\n" + "=" * 60)
    
    # Test API endpoint
    api_success = test_new_image()
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   Direct ML Model: {'✅ PASSED' if direct_success else '❌ FAILED'}")
    print(f"   API Endpoint: {'✅ PASSED' if api_success else '❌ FAILED'}")
    
    if direct_success and api_success:
        print("\n🎉 All tests passed! Your varicose vein analysis is working correctly.")
    elif direct_success:
        print("\n⚠️ ML model works but API failed. Check if server is running.")
    else:
        print("\n❌ Tests failed. Check error messages above.")
