from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import numpy as np
from supabase_client import get_supabase_client
from supabase import Client

def convert_numpy_types(obj):
    """Recursively convert numpy data types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class DatabaseManager:
    def __init__(self):
        self.supabase: Client = get_supabase_client()
    
    def get_db(self):
        """For compatibility with existing code - returns self"""
        return self
    
    def SessionLocal(self):
        """For compatibility with SQLAlchemy-style code - returns self"""
        return self
    
    def close(self):
        """For compatibility - Supabase client doesn't need explicit closing"""
        pass
    
    # Patient operations
    def create_patient(self, db, patient_data: dict):
        """Create a new patient record"""
        try:
            print(f"Inserting patient data: {patient_data}")
            result = self.supabase.table("patients").insert(patient_data).execute()
            print(f"Supabase result: {result}")
            
            if result.data and len(result.data) > 0:
                patient = result.data[0]
                print(f"Patient created: {patient}")
                return patient
            else:
                print(f"No data returned from Supabase. Full result: {result}")
                raise Exception(f"Failed to create patient. Result: {result}")
        except Exception as e:
            print(f"Exception in create_patient: {str(e)}")
            print(f"Exception type: {type(e)}")
            raise e
    
    def get_patient(self, db, patient_id: int):
        """Get patient by ID"""
        try:
            result = self.supabase.table("patients").select("*").eq("id", patient_id).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            print(f"Error getting patient: {str(e)}")
            return None
    
    def get_patient_by_email(self, db, email: str):
        """Get patient by email"""
        try:
            result = self.supabase.table("patients").select("*").eq("email", email).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            print(f"Error getting patient by email: {str(e)}")
            return None
    
    # Analysis operations
    def create_analysis(self, db, analysis_data: dict):
        """Create a new analysis record"""
        try:
            print(f"Creating analysis with data: {analysis_data}")
            
            # Convert numpy types to native Python types for JSON serialization
            analysis_data = convert_numpy_types(analysis_data)
            
            # Convert lists to JSON strings for JSONB columns
            if 'recommendations' in analysis_data and isinstance(analysis_data['recommendations'], list):
                analysis_data['recommendations'] = analysis_data['recommendations']  # Supabase handles JSONB automatically
            if 'preprocessing_info' in analysis_data and isinstance(analysis_data['preprocessing_info'], dict):
                analysis_data['preprocessing_info'] = analysis_data['preprocessing_info']  # Supabase handles JSONB automatically
            
            print(f"Inserting analysis data: {analysis_data}")
            result = self.supabase.table("analyses").insert(analysis_data).execute()
            print(f"Supabase analysis result: {result}")
            
            if result.data and len(result.data) > 0:
                analysis = result.data[0]
                print(f"Analysis created successfully: {analysis}")
                return analysis
            else:
                print(f"No analysis data returned. Full result: {result}")
                raise Exception(f"Failed to create analysis. Result: {result}")
        except Exception as e:
            print(f"Exception in create_analysis: {str(e)}")
            print(f"Exception type: {type(e)}")
            raise e
    
    def get_patient_analyses(self, db, patient_id: int):
        """Get all analyses for a patient"""
        try:
            result = self.supabase.table("analyses").select("*").eq("patient_id", patient_id).order("created_at", desc=True).execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error getting patient analyses: {str(e)}")
            return []
    
    def get_analysis_comparison(self, db, patient_id: int, limit: int = 5):
        """Get recent analyses for comparison"""
        try:
            result = self.supabase.table("analyses").select("*").eq("patient_id", patient_id).order("created_at", desc=True).limit(limit).execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error getting analysis comparison: {str(e)}")
            return []
    
    # Symptom record operations
    def create_symptom_record(self, db, symptom_data: dict):
        """Create a new symptom record"""
        try:
            result = self.supabase.table("symptom_records").insert(symptom_data).execute()
            if result.data:
                return result.data[0]
            else:
                raise Exception("Failed to create symptom record")
        except Exception as e:
            print(f"Error creating symptom record: {str(e)}")
            raise e
    
    def get_patient_symptoms(self, db, patient_id: int):
        """Get all symptom records for a patient"""
        try:
            result = self.supabase.table("symptom_records").select("*").eq("patient_id", patient_id).order("created_at", desc=True).execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error getting patient symptoms: {str(e)}")
            return []
    
    # Report operations
    def create_report(self, db, report_data: dict):
        """Create a new report"""
        try:
            result = self.supabase.table("reports").insert(report_data).execute()
            if result.data:
                return result.data[0]
            else:
                raise Exception("Failed to create report")
        except Exception as e:
            print(f"Error creating report: {str(e)}")
            raise e
    
    def get_patient_reports(self, db, patient_id: int):
        """Get all reports for a patient"""
        try:
            result = self.supabase.table("reports").select("*").eq("patient_id", patient_id).order("created_at", desc=True).execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error getting patient reports: {str(e)}")
            return []
    
    def get_report(self, db, report_id: int):
        """Get report by ID"""
        try:
            result = self.supabase.table("reports").select("*").eq("id", report_id).execute()
            if result.data:
                return result.data[0]
            return None
        except Exception as e:
            print(f"Error getting report: {str(e)}")
            return None
    
    # Appointment operations
    def create_appointment(self, db, appointment_data: dict):
        """Create a new appointment"""
        try:
            result = self.supabase.table("appointments").insert(appointment_data).execute()
            if result.data:
                return result.data[0]
            else:
                raise Exception("Failed to create appointment")
        except Exception as e:
            print(f"Error creating appointment: {str(e)}")
            raise e
    
    def get_patient_appointments(self, db, patient_id: int):
        """Get all appointments for a patient"""
        try:
            result = self.supabase.table("appointments").select("*").eq("patient_id", patient_id).order("scheduled_date", desc=True).execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error getting patient appointments: {str(e)}")
            return []
    
    # Reminder operations
    def create_reminder(self, db, reminder_data: dict):
        """Create a new reminder"""
        try:
            result = self.supabase.table("reminders").insert(reminder_data).execute()
            if result.data:
                return result.data[0]
            else:
                raise Exception("Failed to create reminder")
        except Exception as e:
            print(f"Error creating reminder: {str(e)}")
            raise e
    
    def get_pending_reminders(self, db):
        """Get all pending reminders"""
        try:
            result = self.supabase.table("reminders").select("*").eq("is_sent", False).lte("scheduled_date", datetime.utcnow().isoformat()).execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error getting pending reminders: {str(e)}")
            return []
    
    # Risk assessment operations
    def create_risk_assessment(self, db, assessment_data: dict):
        """Create a new risk assessment"""
        try:
            # Convert risk_factors and recommendations to JSON
            if 'risk_factors' in assessment_data and isinstance(assessment_data['risk_factors'], (list, dict)):
                assessment_data['risk_factors'] = json.dumps(assessment_data['risk_factors'])
            if 'recommendations' in assessment_data and isinstance(assessment_data['recommendations'], (list, dict)):
                assessment_data['recommendations'] = json.dumps(assessment_data['recommendations'])
            
            result = self.supabase.table("risk_assessments").insert(assessment_data).execute()
            if result.data:
                return result.data[0]
            else:
                raise Exception("Failed to create risk assessment")
        except Exception as e:
            print(f"Error creating risk assessment: {str(e)}")
            raise e
    
    def get_patient_risk_assessments(self, db, patient_id: int):
        """Get all risk assessments for a patient"""
        try:
            result = self.supabase.table("risk_assessments").select("*").eq("patient_id", patient_id).order("created_at", desc=True).execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error getting patient risk assessments: {str(e)}")
            return []
    
    # Chat message operations
    def save_chat_message(self, session_id: str, user_message: str, ai_response: str, language: str = "en"):
        """Save chat message to database"""
        try:
            message_data = {
                "session_id": session_id,
                "user_message": user_message,
                "ai_response": ai_response,
                "language": language
            }
            result = self.supabase.table("chat_messages").insert(message_data).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"Error saving chat message: {str(e)}")
            return None
    
    def get_chat_history(self, session_id: str, limit: int = 50):
        """Get chat history for a session"""
        try:
            result = self.supabase.table("chat_messages").select("*").eq("session_id", session_id).order("created_at", desc=True).limit(limit).execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error getting chat history: {str(e)}")
            return []
    
    # Wearable data operations
    def save_wearable_data(self, db, wearable_data: dict):
        """Save wearable device data"""
        try:
            result = self.supabase.table("wearable_data").insert(wearable_data).execute()
            if result.data:
                return result.data[0]
            else:
                raise Exception("Failed to save wearable data")
        except Exception as e:
            print(f"Error saving wearable data: {str(e)}")
            raise e
    
    def get_patient_wearable_data(self, db, patient_id: int, days: int = 7):
        """Get wearable data for a patient for the specified number of days"""
        try:
            from datetime import timedelta
            start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            result = self.supabase.table("wearable_data").select("*").eq("patient_id", patient_id).gte("recorded_at", start_date).order("recorded_at", desc=True).execute()
            return result.data if result.data else []
        except Exception as e:
            print(f"Error getting patient wearable data: {str(e)}")
            return []

# Initialize database manager
db_manager = DatabaseManager()

# Legacy classes for compatibility
class Patient:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class Analysis:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class SymptomRecord:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class Report:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class Appointment:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class Reminder:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
