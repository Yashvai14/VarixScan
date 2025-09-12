from datetime import datetime
from typing import Optional, List, Dict, Any
import json
from supabase_client import get_supabase_client
from supabase import Client

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=False)
    phone = Column(String(20), nullable=True)
    email = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    analyses = relationship("Analysis", back_populates="patient")
    symptoms = relationship("SymptomRecord", back_populates="patient")

class Analysis(Base):
    __tablename__ = "analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    image_path = Column(String(255), nullable=False)
    diagnosis = Column(String(100), nullable=False)
    severity = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    detection_count = Column(Integer, default=0)
    affected_area_ratio = Column(Float, default=0.0)
    recommendations = Column(JSON, nullable=True)
    preprocessing_info = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="analyses")

class SymptomRecord(Base):
    __tablename__ = "symptom_records"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    pain_level = Column(Integer, nullable=False)  # 0-10 scale
    swelling = Column(Boolean, default=False)
    cramping = Column(Boolean, default=False)
    itching = Column(Boolean, default=False)
    burning_sensation = Column(Boolean, default=False)
    leg_heaviness = Column(Boolean, default=False)
    skin_discoloration = Column(Boolean, default=False)
    ulcers = Column(Boolean, default=False)
    duration_symptoms = Column(String(50), nullable=True)  # weeks, months, years
    activity_impact = Column(Integer, nullable=True)  # 0-10 scale
    family_history = Column(Boolean, default=False)
    occupation_standing = Column(Boolean, default=False)
    pregnancy_history = Column(Boolean, default=False)
    previous_treatment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    patient = relationship("Patient", back_populates="symptoms")

class Report(Base):
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    analysis_id = Column(Integer, ForeignKey("analyses.id"))
    report_type = Column(String(50), default="standard")  # standard, comparative, follow-up
    content = Column(Text, nullable=True)
    pdf_path = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Appointment(Base):
    __tablename__ = "appointments"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    doctor_name = Column(String(100), nullable=True)
    appointment_type = Column(String(50), nullable=False)  # consultation, follow-up, teleconsultation
    scheduled_date = Column(DateTime, nullable=False)
    status = Column(String(20), default="scheduled")  # scheduled, completed, cancelled
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Reminder(Base):
    __tablename__ = "reminders"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    reminder_type = Column(String(50), nullable=False)  # scan, medication, exercise, appointment
    message = Column(Text, nullable=False)
    scheduled_date = Column(DateTime, nullable=False)
    is_sent = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    def __init__(self, database_url: str = "sqlite:///varicose_vein_app.db"):
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
    
    def get_db(self):
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def create_patient(self, db, patient_data: dict):
        """Create a new patient record"""
        patient = Patient(**patient_data)
        db.add(patient)
        db.commit()
        db.refresh(patient)
        return patient
    
    def get_patient(self, db, patient_id: int):
        """Get patient by ID"""
        return db.query(Patient).filter(Patient.id == patient_id).first()
    
    def get_patient_by_email(self, db, email: str):
        """Get patient by email"""
        return db.query(Patient).filter(Patient.email == email).first()
    
    def create_analysis(self, db, analysis_data: dict):
        """Create a new analysis record"""
        analysis = Analysis(**analysis_data)
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        return analysis
    
    def get_patient_analyses(self, db, patient_id: int):
        """Get all analyses for a patient"""
        return db.query(Analysis).filter(Analysis.patient_id == patient_id).order_by(Analysis.created_at.desc()).all()
    
    def create_symptom_record(self, db, symptom_data: dict):
        """Create a new symptom record"""
        symptom_record = SymptomRecord(**symptom_data)
        db.add(symptom_record)
        db.commit()
        db.refresh(symptom_record)
        return symptom_record
    
    def get_patient_symptoms(self, db, patient_id: int):
        """Get all symptom records for a patient"""
        return db.query(SymptomRecord).filter(SymptomRecord.patient_id == patient_id).order_by(SymptomRecord.created_at.desc()).all()
    
    def create_report(self, db, report_data: dict):
        """Create a new report"""
        report = Report(**report_data)
        db.add(report)
        db.commit()
        db.refresh(report)
        return report
    
    def get_analysis_comparison(self, db, patient_id: int, limit: int = 5):
        """Get recent analyses for comparison"""
        return db.query(Analysis).filter(Analysis.patient_id == patient_id).order_by(Analysis.created_at.desc()).limit(limit).all()
    
    def create_appointment(self, db, appointment_data: dict):
        """Create a new appointment"""
        appointment = Appointment(**appointment_data)
        db.add(appointment)
        db.commit()
        db.refresh(appointment)
        return appointment
    
    def create_reminder(self, db, reminder_data: dict):
        """Create a new reminder"""
        reminder = Reminder(**reminder_data)
        db.add(reminder)
        db.commit()
        db.refresh(reminder)
        return reminder
    
    def get_pending_reminders(self, db):
        """Get all pending reminders"""
        return db.query(Reminder).filter(
            Reminder.is_sent == False,
            Reminder.scheduled_date <= datetime.utcnow()
        ).all()

# Initialize database manager
db_manager = DatabaseManager()
