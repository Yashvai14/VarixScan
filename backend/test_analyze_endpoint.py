import requests
import os
from io import BytesIO
from PIL import Image
import tempfile

def create_test_image():
    """Create a simple test image"""
    # Create a simple colored image for testing
    img = Image.new('RGB', (100, 100), color='red')
    
    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    img.save(temp_file.name, 'JPEG')
    return temp_file.name

def test_analyze_endpoint():
    """Test the /analyze endpoint to see the exact error"""
    url = "http://localhost:8000/analyze"
    
    # Create test image
    test_image_path = create_test_image()
    
    try:
        # Test 1: Send request without patient_id (should cause 422)
        print("Test 1: Request without patient_id")
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            data = {'language': 'en'}
            response = requests.post(url, files=files, data=data)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            print("-" * 50)
        
        # Test 2: Send request without file (should cause 422)
        print("Test 2: Request without file")
        data = {'patient_id': 1, 'language': 'en'}
        response = requests.post(url, data=data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
        print("-" * 50)
        
        # Test 3: Send request with invalid patient_id format (should cause 422)
        print("Test 3: Request with invalid patient_id format")
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            data = {'patient_id': 'invalid', 'language': 'en'}
            response = requests.post(url, files=files, data=data)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            print("-" * 50)
            
        # Test 4: Send valid request but patient doesn't exist
        print("Test 4: Request with non-existent patient_id")
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            data = {'patient_id': 999999, 'language': 'en'}
            response = requests.post(url, files=files, data=data)
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            print("-" * 50)
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the server is running on http://localhost:8000")
    finally:
        # Clean up test image
        if os.path.exists(test_image_path):
            os.unlink(test_image_path)

if __name__ == "__main__":
    test_analyze_endpoint()
