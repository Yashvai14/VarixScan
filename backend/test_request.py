import requests
import tempfile
import os

def create_test_image():
    """Create a simple test image file"""
    # Create a minimal JPEG file (just header bytes)
    jpeg_header = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00'
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    temp_file.write(jpeg_header)
    # Add some dummy data
    temp_file.write(b'\x00' * 1000)  
    temp_file.close()
    
    return temp_file.name

def test_analyze_request():
    """Test the /analyze endpoint with proper multipart form data"""
    print("🧪 Testing /analyze endpoint with proper request...")
    
    # Create test image
    image_path = create_test_image()
    print(f"📷 Created test image: {image_path}")
    
    try:
        # Make the request
        with open(image_path, 'rb') as img_file:
            files = {
                'file': ('test_image.jpg', img_file, 'image/jpeg')
            }
            data = {
                'patient_id': '1',
                'language': 'en'
            }
            
            print("📤 Sending request...")
            print(f"  - Files: {list(files.keys())}")
            print(f"  - Data: {data}")
            
            response = requests.post(
                'http://localhost:8000/analyze',
                files=files,
                data=data,
                timeout=30
            )
            
            print(f"📥 Response Status: {response.status_code}")
            
            if response.status_code == 422:
                print("❌ Validation Error:")
                try:
                    error_data = response.json()
                    print(f"  {error_data}")
                except:
                    print(f"  Raw response: {response.text}")
            elif response.status_code == 200:
                print("✅ Success!")
                print(f"  Response: {response.json()}")
            else:
                print(f"❓ Unexpected status: {response.status_code}")
                print(f"  Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
    
    finally:
        # Clean up
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"🗑️ Cleaned up test image")

if __name__ == "__main__":
    test_analyze_request()
