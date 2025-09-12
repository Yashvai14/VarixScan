import requests
import json
import tempfile
import os
from PIL import Image

def create_test_image(color='red', size=(100, 100)):
    """Create a test image"""
    img = Image.new('RGB', size, color=color)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    img.save(temp_file.name, 'JPEG')
    return temp_file.name

def create_large_test_image():
    """Create a large test image (>10MB)"""
    # Create a very large image
    img = Image.new('RGB', (5000, 5000), color='blue')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    img.save(temp_file.name, 'JPEG', quality=95)
    return temp_file.name

def test_api_help():
    """Test the API help endpoint"""
    print("🔍 Testing API help endpoint...")
    try:
        response = requests.get("http://localhost:8000/api/help")
        if response.status_code == 200:
            help_data = response.json()
            print("✅ API help endpoint working")
            print(f"   Available endpoints: {list(help_data['endpoints'].keys())}")
            return True
        else:
            print(f"❌ API help failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing API help: {str(e)}")
        return False

def test_validation_errors():
    """Test various validation error scenarios"""
    print("\n🧪 Testing validation error scenarios...")
    
    test_image_path = create_test_image()
    results = []
    
    try:
        # Test 1: Missing file
        print("Test 1: Missing file parameter")
        response = requests.post("http://localhost:8000/analyze", 
                               data={'patient_id': 1, 'language': 'en'})
        print(f"  Status: {response.status_code}")
        if response.status_code == 422:
            error_data = response.json()
            print(f"  Error message: {error_data.get('message', 'N/A')}")
            print(f"  Details: {error_data.get('details', [])}")
            results.append("✅ Missing file validation working")
        else:
            results.append("❌ Missing file validation failed")
        
        # Test 2: Missing patient_id
        print("\nTest 2: Missing patient_id parameter")
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            response = requests.post("http://localhost:8000/analyze", 
                                   files=files, data={'language': 'en'})
        print(f"  Status: {response.status_code}")
        if response.status_code == 422:
            error_data = response.json()
            print(f"  Error message: {error_data.get('message', 'N/A')}")
            print(f"  Details: {error_data.get('details', [])}")
            results.append("✅ Missing patient_id validation working")
        else:
            results.append("❌ Missing patient_id validation failed")
        
        # Test 3: Invalid patient_id format
        print("\nTest 3: Invalid patient_id format")
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            response = requests.post("http://localhost:8000/analyze", 
                                   files=files, data={'patient_id': 'invalid', 'language': 'en'})
        print(f"  Status: {response.status_code}")
        if response.status_code == 422:
            error_data = response.json()
            print(f"  Error message: {error_data.get('message', 'N/A')}")
            print(f"  Details: {error_data.get('details', [])}")
            results.append("✅ Invalid patient_id format validation working")
        else:
            results.append("❌ Invalid patient_id format validation failed")
        
        # Test 4: Non-existent patient_id (should now return 404)
        print("\nTest 4: Non-existent patient_id")
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            response = requests.post("http://localhost:8000/analyze", 
                                   files=files, data={'patient_id': 999999, 'language': 'en'})
        print(f"  Status: {response.status_code}")
        if response.status_code == 404:
            error_data = response.json()
            print(f"  Error message: {error_data.get('detail', 'N/A')}")
            results.append("✅ Non-existent patient_id validation working")
        else:
            results.append("❌ Non-existent patient_id validation failed")
            print(f"  Response: {response.text}")
        
        # Test 5: Invalid file type
        print("\nTest 5: Invalid file type")
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_txt:
            temp_txt.write(b"This is not an image")
            temp_txt_path = temp_txt.name
        
        try:
            with open(temp_txt_path, 'rb') as f:
                files = {'file': ('test.txt', f, 'text/plain')}
                response = requests.post("http://localhost:8000/analyze", 
                                       files=files, data={'patient_id': 1, 'language': 'en'})
            print(f"  Status: {response.status_code}")
            if response.status_code == 400:
                error_data = response.json()
                print(f"  Error: {error_data.get('detail', {}).get('error', 'N/A')}")
                results.append("✅ Invalid file type validation working")
            else:
                results.append("❌ Invalid file type validation failed")
        finally:
            os.unlink(temp_txt_path)
        
        # Test 6: Large file (if implemented)
        print("\nTest 6: Large file validation")
        large_image_path = None
        try:
            large_image_path = create_large_test_image()
            file_size = os.path.getsize(large_image_path) / (1024 * 1024)
            print(f"  Created large image: {file_size:.1f}MB")
            
            if file_size > 10:  # Only test if file is actually large
                with open(large_image_path, 'rb') as f:
                    files = {'file': ('large_test.jpg', f, 'image/jpeg')}
                    response = requests.post("http://localhost:8000/analyze", 
                                           files=files, data={'patient_id': 1, 'language': 'en'})
                print(f"  Status: {response.status_code}")
                if response.status_code == 400:
                    error_data = response.json()
                    print(f"  Error: {error_data.get('detail', {}).get('error', 'N/A')}")
                    results.append("✅ Large file validation working")
                else:
                    results.append("⚠️  Large file validation not triggered or not implemented")
            else:
                results.append("⚠️  Could not create large enough file for testing")
                
        except Exception as e:
            results.append(f"⚠️  Large file test failed: {str(e)}")
        finally:
            if large_image_path and os.path.exists(large_image_path):
                os.unlink(large_image_path)
        
        # Test 7: Valid request (should work)
        print("\nTest 7: Valid request")
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            response = requests.post("http://localhost:8000/analyze", 
                                   files=files, data={'patient_id': 1, 'language': 'en'})
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            result_data = response.json()
            print(f"  Analysis ID: {result_data.get('analysis_id', 'N/A')}")
            print(f"  Diagnosis: {result_data.get('diagnosis', 'N/A')}")
            results.append("✅ Valid request working correctly")
        else:
            results.append("❌ Valid request failed")
            print(f"  Response: {response.text}")
    
    finally:
        if os.path.exists(test_image_path):
            os.unlink(test_image_path)
    
    return results

def test_patient_creation():
    """Test patient creation endpoint"""
    print("\n👥 Testing patient creation...")
    
    patient_data = {
        "name": "API Test Patient",
        "age": 42,
        "gender": "Female",
        "phone": "+9876543210",
        "email": "api_test@example.com"
    }
    
    try:
        response = requests.post("http://localhost:8000/patients/", json=patient_data)
        if response.status_code == 200:
            result = response.json()
            patient_id = result.get('patient_id')
            print(f"✅ Patient created successfully with ID: {patient_id}")
            return patient_id
        else:
            print(f"❌ Patient creation failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error creating patient: {str(e)}")
        return None

def main():
    """Main test function"""
    print("🚀 Comprehensive 422 Error Fix Testing")
    print("=" * 60)
    
    # Test API availability
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code != 200:
            print("❌ API server not available. Please start the server first.")
            return
        print("✅ API server is running")
    except:
        print("❌ Cannot connect to API server. Please start the server first.")
        return
    
    # Test API help endpoint
    help_working = test_api_help()
    
    # Test patient creation
    new_patient_id = test_patient_creation()
    
    # Test validation errors
    validation_results = test_validation_errors()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print("=" * 60)
    
    if help_working:
        print("✅ API Help endpoint working")
    else:
        print("❌ API Help endpoint failed")
    
    if new_patient_id:
        print(f"✅ Patient creation working (ID: {new_patient_id})")
    else:
        print("❌ Patient creation failed")
    
    print("\n📋 Validation Tests:")
    for result in validation_results:
        print(f"  {result}")
    
    # Count successes
    success_count = sum(1 for result in validation_results if result.startswith("✅"))
    total_tests = len(validation_results)
    
    print(f"\n🎯 Overall Success Rate: {success_count}/{total_tests} ({success_count/total_tests*100:.1f}%)")
    
    if success_count == total_tests:
        print("🎉 All tests passed! The 422 errors have been fixed.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
