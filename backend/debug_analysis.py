#!/usr/bin/env python3
"""
Debug script to identify the 500 error in analysis
"""

import sys
import traceback
from fastapi.testclient import TestClient

def test_analysis_debug():
    try:
        print("🔍 Debugging analysis endpoint...")
        
        # Import the app
        from main import app
        client = TestClient(app)
        
        # Test with existing image
        with open('uploads/1_20250912_060242_test.jpg', 'rb') as f:
            response = client.post(
                '/analyze',
                files={'file': ('test.jpg', f, 'image/jpeg')},
                data={'patient_id': 1, 'language': 'en'}
            )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Text: {response.text}")
        
        if response.status_code == 200:
            print("✅ Analysis endpoint working correctly!")
            result = response.json()
            print(f"Diagnosis: {result.get('diagnosis', 'N/A')}")
            print(f"Confidence: {result.get('confidence', 'N/A')}%")
        else:
            print("❌ Analysis endpoint failed")
            
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        print("📊 Full traceback:")
        traceback.print_exc()

def test_direct_components():
    """Test individual components directly"""
    print("\n🔧 Testing individual components...")
    
    try:
        # Test database
        from database import db_manager, Patient
        db = next(db_manager.get_db())
        patient = db.query(Patient).filter(Patient.id == 1).first()
        print(f"✅ Database: Patient 1 exists - {patient.name if patient else 'No'}")
        db.close()
    except Exception as e:
        print(f"❌ Database error: {e}")
        
    try:
        # Test ML model
        from ml_model import VaricoseVeinDetector
        detector = VaricoseVeinDetector()
        result = detector.detect_veins('uploads/1_20250912_060242_test.jpg')
        print(f"✅ ML Model: {result['diagnosis']} ({result['confidence']}%)")
    except Exception as e:
        print(f"❌ ML Model error: {e}")
        
    try:
        # Test file handling
        import os
        test_file = 'uploads/1_20250912_060242_test.jpg'
        if os.path.exists(test_file):
            size = os.path.getsize(test_file)
            print(f"✅ File access: {test_file} ({size} bytes)")
        else:
            print(f"❌ File not found: {test_file}")
    except Exception as e:
        print(f"❌ File access error: {e}")

if __name__ == "__main__":
    test_direct_components()
    test_analysis_debug()
