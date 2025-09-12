#!/usr/bin/env python3
"""
Script to diagnose and fix 422 validation errors
"""

import requests
import json
from pathlib import Path

def test_validation_scenarios():
    """Test different scenarios that cause 422 errors"""
    
    base_url = "http://localhost:8000"
    
    print("🔍 Testing different validation scenarios that cause 422 errors...")
    print("="*70)
    
    # Scenario 1: Missing file
    print("\n1️⃣ Testing missing file field:")
    try:
        response = requests.post(f"{base_url}/analyze", data={"patient_id": 1, "language": "en"})
        print(f"   Status: {response.status_code}")
        if response.status_code == 422:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Scenario 2: Missing patient_id
    print("\n2️⃣ Testing missing patient_id field:")
    try:
        # Create a dummy file for testing
        files = {"file": ("test.txt", b"dummy content", "text/plain")}
        response = requests.post(f"{base_url}/analyze", files=files, data={"language": "en"})
        print(f"   Status: {response.status_code}")
        if response.status_code == 422:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Scenario 3: Invalid patient_id (string instead of int)
    print("\n3️⃣ Testing invalid patient_id (string):")
    try:
        files = {"file": ("test.txt", b"dummy content", "text/plain")}
        response = requests.post(f"{base_url}/analyze", files=files, data={"patient_id": "invalid", "language": "en"})
        print(f"   Status: {response.status_code}")
        if response.status_code == 422:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Scenario 4: Non-existent patient_id
    print("\n4️⃣ Testing non-existent patient_id:")
    try:
        files = {"file": ("test.jpg", b"dummy image content", "image/jpeg")}
        response = requests.post(f"{base_url}/analyze", files=files, data={"patient_id": 999999, "language": "en"})
        print(f"   Status: {response.status_code}")
        if response.status_code in [404, 422]:
            print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")

def create_correct_request():
    """Show the correct way to make the request"""
    print("\n" + "="*70)
    print("✅ CORRECT way to make the request:")
    print("="*70)
    
    # Check if we have a valid image
    uploads_dir = Path("uploads")
    if uploads_dir.exists():
        images = list(uploads_dir.glob("*.jpg"))
        if images:
            test_image = images[0]
            print(f"📁 Using test image: {test_image}")
            
            try:
                with open(test_image, "rb") as f:
                    files = {"file": (test_image.name, f, "image/jpeg")}
                    data = {"patient_id": 1, "language": "en"}
                    
                    print("🔄 Making correct request...")
                    response = requests.post("http://localhost:8000/analyze", files=files, data=data)
                    
                    print(f"📊 Status: {response.status_code}")
                    if response.status_code == 200:
                        result = response.json()
                        print("✅ Success!")
                        print(f"   Diagnosis: {result.get('diagnosis')}")
                        print(f"   Confidence: {result.get('confidence')}%")
                    else:
                        print(f"❌ Failed: {response.text}")
                        
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("❌ No test images found in uploads directory")
    else:
        print("❌ No uploads directory found")

def show_curl_examples():
    """Show correct curl command examples"""
    print("\n" + "="*70)
    print("💡 CORRECT CURL Examples:")
    print("="*70)
    
    print("\n🔹 Windows PowerShell:")
    print("   $image = 'path\\to\\your\\varicose_image.jpg'")
    print("   $uri = 'http://localhost:8000/analyze'")
    print("   $form = @{")
    print("       file = Get-Item $image")
    print("       patient_id = 1")
    print("       language = 'en'")
    print("   }")
    print("   Invoke-RestMethod -Uri $uri -Method Post -Form $form")
    
    print("\n🔹 Using requests in Python:")
    print("   import requests")
    print("   with open('your_image.jpg', 'rb') as f:")
    print("       files = {'file': ('image.jpg', f, 'image/jpeg')}")
    print("       data = {'patient_id': 1, 'language': 'en'}")
    print("       response = requests.post('http://localhost:8000/analyze', files=files, data=data)")

def check_patient_exists():
    """Check if patient ID 1 exists"""
    print("\n" + "="*70)
    print("👤 Checking if patient ID 1 exists:")
    print("="*70)
    
    try:
        from database import db_manager, Patient
        db = next(db_manager.get_db())
        patient = db.query(Patient).filter(Patient.id == 1).first()
        
        if patient:
            print("✅ Patient ID 1 exists:")
            print(f"   Name: {patient.name}")
            print(f"   Age: {patient.age}")
            print(f"   Gender: {patient.gender}")
        else:
            print("❌ Patient ID 1 does not exist!")
            print("💡 Creating default patient...")
            
            # Create a default patient
            default_patient = {
                "name": "Test Patient",
                "age": 35,
                "gender": "Male",
                "phone": "+1234567890",
                "email": "test@example.com"
            }
            new_patient = db_manager.create_patient(db, default_patient)
            print(f"✅ Created patient with ID: {new_patient.id}")
        
        db.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    print("🔧 422 Validation Error Diagnostic Tool")
    print("This script will help identify why you're getting 422 errors")
    
    # Check patient exists first
    check_patient_exists()
    
    # Test validation scenarios
    test_validation_scenarios()
    
    # Show correct examples
    create_correct_request()
    show_curl_examples()
    
    print("\n" + "="*70)
    print("🎯 MOST COMMON 422 ERROR CAUSES:")
    print("="*70)
    print("1️⃣ Missing 'file' field in the request")
    print("2️⃣ Missing 'patient_id' field in the request") 
    print("3️⃣ patient_id is not a number (must be integer)")
    print("4️⃣ patient_id doesn't exist in database")
    print("5️⃣ Wrong Content-Type (must be multipart/form-data)")
    print("6️⃣ File field is empty or corrupted")
    
    print("\n💡 TO FIX YOUR 422 ERROR:")
    print("   Make sure your request includes:")
    print("   - file: The image file")
    print("   - patient_id: 1 (or another valid patient ID)")
    print("   - language: 'en' (optional)")
    print("   - Content-Type: multipart/form-data")
