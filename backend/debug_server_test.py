#!/usr/bin/env python3
"""
Debug script to test the ML processing pipeline exactly as the server does
"""

import tempfile
import cv2
import numpy as np
import traceback
import os

# Import the same modules as main.py
try:
    # Try to import the advanced medical-grade model first
    from advanced_ml_model import advanced_detector
    detector = advanced_detector
    ML_BASIC_AVAILABLE = True
    print("✅ Advanced ML model loaded successfully")
except Exception as e:
    try:
        # Fallback to basic model if advanced model fails
        from ml_model import VaricoseVeinDetector
        detector = VaricoseVeinDetector()
        ML_BASIC_AVAILABLE = True
        print("✅ Basic ML model loaded successfully (fallback)")
    except Exception as e2:
        ML_BASIC_AVAILABLE = False
        detector = None
        print(f"⚠️ No ML model available: Advanced: {e}, Basic: {e2}")

def test_ml_processing():
    """Test the ML processing exactly like the server does"""
    
    # Create a test image (similar to what might be uploaded)
    test_img = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
    
    # Add some patterns to make it more realistic
    # Add some lines that might be detected as veins
    cv2.line(test_img, (50, 100), (250, 120), (80, 40, 40), 3)
    cv2.line(test_img, (80, 150), (200, 180), (60, 30, 30), 2)
    
    # Save to temporary file (exactly like server does)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        cv2.imwrite(temp_file.name, test_img)
        temp_path = temp_file.name
    
    print(f"🖼️ Test image created at: {temp_path}")
    
    # ML analysis (exactly like server does)
    analysis_result = None
    if ML_BASIC_AVAILABLE and detector:
        try:
            print(f"🔍 Starting ML analysis with detector type: {type(detector).__name__}")
            # Check if it's the advanced detector
            if hasattr(detector, 'detect_varicose_veins'):
                print("Using advanced detection method")
                analysis_result = detector.detect_varicose_veins(temp_path)
                print(f"✅ Advanced detection completed: {analysis_result.get('diagnosis', 'Unknown')}")
            else:
                # Fallback to basic detector method
                print("Using basic detection method")
                analysis_result = detector.detect_veins(temp_path)
                print(f"✅ Basic detection completed: {analysis_result.get('diagnosis', 'Unknown')}")
        except Exception as e:
            print(f"🔴 ML Detection Error: {str(e)}")
            traceback.print_exc()
            analysis_result = None

    if not analysis_result:
        print("⚠️ Using fallback dummy data")
        analysis_result = {
            'diagnosis': 'Image processed - AI analysis temporarily unavailable',
            'severity': 'Normal',
            'confidence': 60.0,
            'detection_count': 0,
            'affected_area_ratio': 0.0,
            'recommendations': ['Consult with healthcare provider']
        }
    
    print("\n📊 Final Analysis Result:")
    for key, value in analysis_result.items():
        print(f"  {key}: {value}")
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)
    
    return analysis_result

if __name__ == "__main__":
    print("🧪 Testing ML Processing Pipeline")
    print("=" * 50)
    result = test_ml_processing()
    print("=" * 50)
    print("✅ Test completed")