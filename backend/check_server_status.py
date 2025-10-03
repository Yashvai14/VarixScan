#!/usr/bin/env python3
"""Quick server status check"""

from main import app, detector, ML_BASIC_AVAILABLE
import tempfile, cv2, numpy as np, os

print('🔍 Current Server Status:')
print(f'ML Available: {ML_BASIC_AVAILABLE}')
print(f'Detector Type: {type(detector).__name__ if detector else "None"}')

if detector:
    print('Testing detector...')
    img = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
    cv2.imwrite('quick_test.jpg', img)
    
    try:
        if hasattr(detector, 'detect_varicose_veins'):
            result = detector.detect_varicose_veins('quick_test.jpg')
            print('✅ Advanced detection works:', result.get('diagnosis'))
        else:
            result = detector.detect_veins('quick_test.jpg')  
            print('✅ Basic detection works:', result.get('diagnosis'))
    except Exception as e:
        print('❌ Detection failed:', str(e))
    
    if os.path.exists('quick_test.jpg'):
        os.remove('quick_test.jpg')
else:
    print('❌ No detector available')