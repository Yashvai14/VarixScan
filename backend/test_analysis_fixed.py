#!/usr/bin/env python3
"""
Test script to verify that the analysis functionality is working correctly
"""

import requests
import os
from pathlib import Path

def test_analysis_endpoint():
    """Test the /analyze endpoint with an existing image"""
    
    # Check if we have uploaded images to test with
    uploads_dir = Path("uploads")
    if not uploads_dir.exists():
        print("❌ No uploads directory found")
        return False
    
    image_files = list(uploads_dir.glob("*.jpg"))
    if not image_files:
        print("❌ No test images found in uploads directory")
        return False
    
    test_image = image_files[0]
    print(f"🧪 Testing analysis with image: {test_image}")
    
    # Test data
    url = "http://localhost:8000/analyze"
    
    try:
        # Read file content first to avoid handle conflicts
        with open(test_image, "rb") as f:
            file_content = f.read()
        
        files = {"file": ("test.jpg", file_content, "image/jpeg")}
        data = {
            "patient_id": 1,
            "language": "en"
        }
        
        response = requests.post(url, files=files, data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Analysis successful!")
            print(f"   Diagnosis: {result.get('diagnosis', 'N/A')}")
            print(f"   Severity: {result.get('severity', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 'N/A')}%")
            print(f"   Detection Count: {result.get('detection_count', 'N/A')}")
            return True
        else:
            print(f"❌ Analysis failed with status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
                
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the server is running on port 8000")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def test_direct_ml_model():
    """Test the ML model directly without the API"""
    print("\n🧪 Testing ML model directly...")
    
    try:
        from ml_model import VaricoseVeinDetector
        
        uploads_dir = Path("uploads")
        image_files = list(uploads_dir.glob("*.jpg"))
        
        if not image_files:
            print("❌ No test images found")
            return False
            
        detector = VaricoseVeinDetector()
        result = detector.detect_veins(str(image_files[0]))
        
        print("✅ Direct ML model test successful!")
        print(f"   Diagnosis: {result.get('diagnosis', 'N/A')}")
        print(f"   Severity: {result.get('severity', 'N/A')}")
        print(f"   Confidence: {result.get('confidence', 'N/A')}%")
        print(f"   Skin Area Ratio: {result.get('preprocessing_info', {}).get('skin_area_ratio', 'N/A'):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct ML model test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting analysis tests...\n")
    
    # Test direct ML model first
    ml_test_passed = test_direct_ml_model()
    
    print("\n" + "="*50)
    print("🏥 Testing API endpoint...")
    print("💡 Make sure the server is running: python main.py")
    print("="*50)
    
    # Test API endpoint
    api_test_passed = test_analysis_endpoint()
    
    print("\n" + "="*50)
    print("📊 Test Results Summary:")
    print(f"   Direct ML Model: {'✅ PASSED' if ml_test_passed else '❌ FAILED'}")
    print(f"   API Endpoint: {'✅ PASSED' if api_test_passed else '❌ FAILED'}")
    
    if ml_test_passed and api_test_passed:
        print("\n🎉 All tests passed! Analysis functionality is working correctly.")
    elif ml_test_passed:
        print("\n⚠️  ML model works but API test failed. Check if server is running.")
    else:
        print("\n❌ Tests failed. Check the error messages above.")
    print("="*50)
