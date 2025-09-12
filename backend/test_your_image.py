#!/usr/bin/env python3
"""
Simple script to test your varicose vein image
"""

import requests
import os

def test_your_image(image_path):
    """Test your specific varicose vein image"""
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("💡 Make sure to save your varicose vein image to this folder first")
        return
    
    print(f"🔄 Testing image: {image_path}")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            data = {
                'patient_id': 1,  # MUST be integer, not string
                'language': 'en'
            }
            
            response = requests.post('http://localhost:8000/analyze', files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ SUCCESS! Analysis completed:")
                print(f"   📋 Diagnosis: {result['diagnosis']}")
                print(f"   🔍 Severity: {result['severity']}")
                print(f"   📈 Confidence: {result['confidence']}%")
                print(f"   🎯 Detections: {result['detection_count']}")
                
                if result['detection_count'] > 0:
                    print(f"   📐 Affected Area: {result['affected_area_ratio']:.3f}")
                
                print("\n💡 Recommendations:")
                for i, rec in enumerate(result['recommendations'], 1):
                    print(f"   {i}. {rec}")
                    
            elif response.status_code == 422:
                print("❌ 422 Validation Error:")
                error = response.json()
                print(f"   Details: {error.get('details', [])}")
                
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    print("🩺 Varicose Vein Image Tester")
    print("=" * 40)
    
    # Common image names people might use
    possible_names = [
        "varicose_legs.jpg", "varicose.jpg", "legs.jpg", 
        "varicose_veins.jpg", "test_image.jpg", "image.jpg"
    ]
    
    found_image = None
    for name in possible_names:
        if os.path.exists(name):
            found_image = name
            break
    
    if found_image:
        print(f"📁 Found image: {found_image}")
        test_your_image(found_image)
    else:
        print("📂 No image found. Please:")
        print("1. Save your varicose vein image as 'varicose_legs.jpg'")
        print("2. Put it in this folder")
        print("3. Run this script again")
        print(f"\nCurrent folder: {os.getcwd()}")
