import requests
import os
from database import db_manager

def test_database():
    """Test database connectivity and patient existence"""
    print("🔍 Testing Database...")
    
    try:
        db = next(db_manager.get_db())
        
        # Check if any patients exist
        from database import Patient
        patients = db.query(Patient).all()
        print(f"📊 Found {len(patients)} patients in database")
        
        if patients:
            for patient in patients:
                print(f"  - Patient {patient.id}: {patient.name} (age: {patient.age})")
        else:
            print("❌ No patients found in database")
            
        db.close()
        return len(patients) > 0
        
    except Exception as e:
        print(f"❌ Database error: {str(e)}")
        return False

def test_endpoint():
    """Test the analyze endpoint with a simple request"""
    print("\n🔍 Testing /analyze endpoint...")
    
    # First check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        print(f"✅ Server is running: {response.json()}")
    except Exception as e:
        print(f"❌ Server not accessible: {str(e)}")
        return False
    
    # Test with missing parameters to see validation errors
    print("\n📝 Testing validation errors...")
    
    # Test 1: No parameters
    try:
        response = requests.post("http://localhost:8000/analyze", timeout=10)
        print(f"🔸 No params - Status: {response.status_code}")
        if response.status_code == 422:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
    
    # Test 2: Missing file
    try:
        response = requests.post(
            "http://localhost:8000/analyze",
            data={"patient_id": "1", "language": "en"},
            timeout=10
        )
        print(f"🔸 Missing file - Status: {response.status_code}")
        if response.status_code == 422:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
    
    # Test 3: Missing patient_id
    try:
        # Create a dummy image file for testing
        test_image_path = "test_image.jpg"
        with open(test_image_path, "wb") as f:
            f.write(b"fake image data")
        
        with open(test_image_path, "rb") as f:
            response = requests.post(
                "http://localhost:8000/analyze",
                files={"file": ("test.jpg", f, "image/jpeg")},
                data={"language": "en"},
                timeout=10
            )
        
        print(f"🔸 Missing patient_id - Status: {response.status_code}")
        if response.status_code == 422:
            print(f"   Response: {response.json()}")
        
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Debug Test...")
    
    # Test database first
    db_ok = test_database()
    
    # Test endpoint
    test_endpoint()
    
    print("\n📋 Summary:")
    print(f"  - Database: {'✅ OK' if db_ok else '❌ Issues'}")
    print("  - Check server logs for detailed validation errors")
