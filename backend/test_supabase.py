#!/usr/bin/env python3
"""
Test Supabase connection and database operations
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from supabase_client import get_supabase_client
    from database import DatabaseManager
    
    def test_supabase_connection():
        print("🧪 Testing Supabase Connection...")
        
        # Test 1: Basic connection
        try:
            supabase = get_supabase_client()
            print("✅ Supabase client created successfully")
        except Exception as e:
            print(f"❌ Failed to create Supabase client: {e}")
            return False
        
        # Test 2: List tables
        try:
            # Try to get table info
            result = supabase.table("patients").select("*").limit(1).execute()
            print(f"✅ Can access patients table. Result: {result}")
        except Exception as e:
            print(f"❌ Cannot access patients table: {e}")
            return False
        
        # Test 3: Test database manager
        try:
            db_manager = DatabaseManager()
            print("✅ DatabaseManager created successfully")
        except Exception as e:
            print(f"❌ Failed to create DatabaseManager: {e}")
            return False
        
        # Test 4: Try creating a test patient
        try:
            test_patient = {
                "name": "Test User",
                "age": 30,
                "gender": "Male",
                "phone": "+1234567890",
                "email": "test@example.com"
            }
            
            patient = db_manager.create_patient(None, test_patient)
            print(f"✅ Test patient created: {patient}")
            
            # Try to delete the test patient
            if patient and 'id' in patient:
                delete_result = supabase.table("patients").delete().eq("id", patient['id']).execute()
                print(f"✅ Test patient deleted: {delete_result}")
                
        except Exception as e:
            print(f"❌ Failed to create test patient: {e}")
            print(f"Error type: {type(e)}")
            return False
        
        return True
    
    def check_environment():
        print("🔍 Checking Environment Variables...")
        
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_ANON_KEY')
        
        print(f"SUPABASE_URL: {'✅ Set' if supabase_url else '❌ Missing'}")
        print(f"SUPABASE_ANON_KEY: {'✅ Set' if supabase_key else '❌ Missing'}")
        
        if supabase_url:
            print(f"URL starts with: {supabase_url[:20]}...")
        if supabase_key:
            print(f"Key starts with: {supabase_key[:20]}...")
    
    if __name__ == "__main__":
        print("=" * 50)
        print("  SUPABASE CONNECTION TEST")
        print("=" * 50)
        
        check_environment()
        print()
        
        if test_supabase_connection():
            print("\n🎉 All tests passed! Supabase is working correctly.")
        else:
            print("\n💥 Some tests failed. Check the errors above.")
            
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all dependencies are installed.")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
