from database import db_manager, Patient
from sqlalchemy.orm import Session
import requests

def check_database_status():
    """Check if database is accessible and has data"""
    print("🔍 Checking database status...")
    
    try:
        # Get database session
        db = next(db_manager.get_db())
        
        # Count patients
        patient_count = db.query(Patient).count()
        print(f"📊 Found {patient_count} patients in database")
        
        # Show existing patients
        if patient_count > 0:
            patients = db.query(Patient).limit(5).all()
            print("\n👥 Existing patients:")
            for patient in patients:
                print(f"  ID: {patient.id}, Name: {patient.name}, Age: {patient.age}")
        
        db.close()
        return patient_count > 0
        
    except Exception as e:
        print(f"❌ Database error: {str(e)}")
        return False

def create_test_patient():
    """Create a test patient for API testing"""
    print("\n🧪 Creating test patient...")
    
    try:
        # Create patient via API
        url = "http://localhost:8000/patients/"
        patient_data = {
            "name": "Test Patient",
            "age": 35,
            "gender": "Male",
            "phone": "+1234567890",
            "email": "test@example.com"
        }
        
        response = requests.post(url, json=patient_data)
        
        if response.status_code == 200:
            result = response.json()
            patient_id = result.get('patient_id')
            print(f"✅ Test patient created with ID: {patient_id}")
            return patient_id
        else:
            print(f"❌ Failed to create patient: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Server might not be running.")
        return None
    except Exception as e:
        print(f"❌ Error creating patient: {str(e)}")
        return None

def test_valid_analyze_request():
    """Test a valid analyze request"""
    print("\n🧪 Testing valid /analyze request...")
    
    # First, ensure we have a patient
    has_patients = check_database_status()
    
    test_patient_id = 1
    if not has_patients:
        test_patient_id = create_test_patient()
        if not test_patient_id:
            print("❌ Cannot create test patient. Cannot test /analyze endpoint.")
            return
    
    # Create a simple test image
    from PIL import Image
    import tempfile
    import os
    
    # Create test image
    img = Image.new('RGB', (100, 100), color='red')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    img.save(temp_file.name, 'JPEG')
    
    try:
        url = "http://localhost:8000/analyze"
        with open(temp_file.name, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            data = {'patient_id': test_patient_id, 'language': 'en'}
            response = requests.post(url, files=files, data=data)
            
        print(f"Status: {response.status_code}")
        if response.status_code == 422:
            print(f"❌ 422 Error Response: {response.text}")
        elif response.status_code == 200:
            print(f"✅ Success: {response.json()}")
        else:
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        # Clean up
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

def main():
    print("🚀 Database and API Status Check")
    print("=" * 50)
    
    # Check database status
    check_database_status()
    
    # Test the analyze endpoint
    test_valid_analyze_request()

if __name__ == "__main__":
    main()
