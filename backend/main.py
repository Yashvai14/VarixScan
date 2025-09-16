# backend/main.py
from fastapi import FastAPI, UploadFile, Form, Depends, HTTPException, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError
from typing import Optional, List
from contextlib import asynccontextmanager
import shutil, os
import uvicorn
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

# Import core modules with fallback handling
from database import db_manager, convert_numpy_types
from ai_chatbot import medical_chatbot

# Optional ML model imports
try:
    from ml_model import VaricoseVeinDetector
    detector = VaricoseVeinDetector()
    ML_BASIC_AVAILABLE = True
    print("✅ Basic ML model loaded successfully")
except ImportError as e:
    print(f"⚠️ Basic ML model not available: {e}")
    ML_BASIC_AVAILABLE = False
    detector = None

try:
    from advanced_ml_model import advanced_detector
    ML_ADVANCED_AVAILABLE = True
    print("✅ Advanced ML model loaded successfully")
except ImportError as e:
    print(f"⚠️ Advanced ML model not available: {e}")
    ML_ADVANCED_AVAILABLE = False
    advanced_detector = None

# Optional PDF report generation
try:
    from report_generator import report_generator
    REPORTS_AVAILABLE = True
    print("✅ PDF report generator loaded successfully")
except ImportError as e:
    print(f"⚠️ PDF reports not available: {e}")
    try:
        # Try alternative PDF generator
        from fpdf_report_generator import fpdf_report_generator as report_generator
        REPORTS_AVAILABLE = True
        print("✅ Alternative PDF generator loaded successfully")
    except ImportError:
        print("⚠️ No PDF generators available")
        REPORTS_AVAILABLE = False
        report_generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    try:
        # Test database connection
        db = db_manager.get_db()
        print("✅ Connected to Supabase database successfully")
    except Exception as e:
        print(f"⚠️  Warning: Could not connect to database: {str(e)}")
    
    yield  # App runs here
    
    # Shutdown (if needed)
    print("🔄 Application shutting down...")

app = FastAPI(
    title="Varicose Vein Detection API", 
    version="2.0.0",
    lifespan=lifespan
)

# Custom exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle 422 validation errors with user-friendly messages"""
    # Debug logging for validation errors
    print(f"🔴 VALIDATION ERROR on {request.method} {request.url}:")
    print(f"  Raw errors: {exc.errors()}")
    
    # Log request details
    try:
        content_type = request.headers.get('content-type', 'unknown')
        print(f"  Content-Type: {content_type}")
        
        # Try to log form data if it's multipart
        if 'multipart/form-data' in content_type:
            print(f"  📎 This is a multipart form request")
        else:
            print(f"  ⚠️ This is NOT a multipart form request")
            
    except Exception as e:
        print(f"  Error logging request details: {e}")
    
    errors = []
    for error in exc.errors():
        field = error['loc'][-1] if error['loc'] else 'unknown'
        error_type = error['type']
        
        if error_type == 'missing':
            if field == 'file':
                errors.append("Image file is required. Please upload an image.")
            elif field == 'patient_id':
                errors.append("Patient ID is required. Please provide a valid patient ID.")
            else:
                errors.append(f"Field '{field}' is required.")
        elif error_type == 'int_parsing':
            if field == 'patient_id':
                errors.append("Patient ID must be a valid number.")
            else:
                errors.append(f"Field '{field}' must be a valid number.")
        else:
            errors.append(f"Invalid value for field '{field}': {error['msg']}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Please check your request parameters.",
            "details": errors,
            "help": {
                "required_fields": {
                    "file": "An image file (JPG, JPEG, PNG)",
                    "patient_id": "A valid patient ID (integer)",
                    "language": "Language code (optional, defaults to 'en')"
                },
                "example_curl": "curl -X POST http://localhost:8000/analyze -F 'file=@image.jpg' -F 'patient_id=1' -F 'language=en'"
            }
        }
    )

# CORS (to allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ML model is now initialized above with error handling

# Ensure upload directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# Pydantic models for request/response
class PatientCreate(BaseModel):
    name: str
    age: int
    gender: str
    phone: Optional[str] = None
    email: Optional[str] = None

class SymptomData(BaseModel):
    patient_id: int
    pain_level: int  # 0-10
    swelling: bool = False
    cramping: bool = False
    itching: bool = False
    burning_sensation: bool = False
    leg_heaviness: bool = False
    skin_discoloration: bool = False
    ulcers: bool = False
    duration_symptoms: Optional[str] = None
    activity_impact: Optional[int] = None  # 0-10
    family_history: bool = False
    occupation_standing: bool = False
    pregnancy_history: bool = False
    previous_treatment: Optional[str] = None

class AppointmentCreate(BaseModel):
    patient_id: int
    doctor_name: Optional[str] = None
    appointment_type: str
    scheduled_date: datetime
    notes: Optional[str] = None

class ReminderCreate(BaseModel):
    patient_id: int
    reminder_type: str
    message: str
    scheduled_date: datetime

# Multilingual support
TRANSLATIONS = {
    "en": {
        "diagnosis_varicose": "Varicose Veins Detected",
        "diagnosis_normal": "No Varicose Veins Detected",
        "severity_mild": "Mild",
        "severity_moderate": "Moderate",
        "severity_severe": "Severe",
        "severity_normal": "Normal"
    },
    "hi": {
        "diagnosis_varicose": "वैरिकाज़ वेन्स का पता चला",
        "diagnosis_normal": "कोई वैरिकाज़ वेन्स नहीं मिली",
        "severity_mild": "हल्का",
        "severity_moderate": "मध्यम",
        "severity_severe": "गंभीर",
        "severity_normal": "सामान्य"
    },
    "mr": {
        "diagnosis_varicose": "व्हॅरिकोज व्हेन्स आढळल्या",
        "diagnosis_normal": "व्हॅरिकोज व्हेन्स नाहीत",
        "severity_mild": "सौम्य",
        "severity_moderate": "मध्यम",
        "severity_severe": "गंभीर",
        "severity_normal": "सामान्य"
    }
}

def get_db():
    """Database dependency"""
    db = db_manager.get_db()
    try:
        yield db
    finally:
        if hasattr(db, 'close'):
            db.close()

def translate_text(text: str, language: str = "en") -> str:
    """Translate text to specified language"""
    if language in TRANSLATIONS and text in TRANSLATIONS[language]:
        return TRANSLATIONS[language][text]
    return text

# Health check endpoint for Render deployment
@app.get("/")
async def root():
    """Root endpoint for health checks"""
    return {
        "message": "VarixScan AI API is running",
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    try:
        # Test database connection
        db = db_manager.get_db()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "database": db_status,
        "ai_model": "loaded" if detector else "not loaded",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/patients/")
async def create_patient(patient: PatientCreate, db: Session = Depends(get_db)):
    """Create a new patient record"""
    try:
        # Fix Pydantic deprecation warning
        patient_data = patient.model_dump() if hasattr(patient, 'model_dump') else patient.dict()
        print(f"Creating patient with data: {patient_data}")
        
        db_patient = db_manager.create_patient(db, patient_data)
        print(f"Patient created successfully: {db_patient}")
        
        return {"message": "Patient created successfully", "patient_id": db_patient['id']}
    except Exception as e:
        print(f"Error creating patient: {str(e)}")
        print(f"Error type: {type(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    patient_id: int = Form(...),
    language: str = Form("en"),
    db: Session = Depends(get_db)
):
    """Analyze uploaded image for varicose veins"""
    # Debug logging
    print(f"🔍 DEBUG: Received analyze request:")
    print(f"  - File: {file.filename if file else 'None'} (content_type: {getattr(file, 'content_type', 'None')})")
    print(f"  - Patient ID: {patient_id} (type: {type(patient_id)})")
    print(f"  - Language: {language}")
    
    try:
        # Validate patient exists
        patient = db_manager.get_patient(db, patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient with ID {patient_id} not found")
        
        # Validate file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp']
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "Invalid file type",
                    "message": f"File must be an image. Received: {file.content_type or 'unknown'}",
                    "allowed_types": allowed_types,
                    "help": "Please upload a valid image file (JPEG, PNG, GIF, or BMP)"
                }
            )
        
        # Validate file size (max 10MB)
        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "File too large",
                    "message": f"File size ({file.size / 1024 / 1024:.1f}MB) exceeds the maximum limit of 10MB",
                    "help": "Please upload a smaller image file"
                }
            )
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{patient_id}_{timestamp}_{file.filename}"
        file_path = f"uploads/{filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run AI analysis with fallback handling
        analysis_result = None
        
        if ML_BASIC_AVAILABLE and detector:
            try:
                print(f"Running basic AI analysis on: {file_path}")
                analysis_result = detector.detect_veins(file_path)
                print(f"Basic AI analysis result: {analysis_result}")
            except Exception as e:
                print(f"Basic AI analysis failed: {str(e)}")
                analysis_result = None
        
        # If basic model failed or unavailable, provide fallback
        if not analysis_result:
            print("Using fallback analysis (ML models unavailable)")
            analysis_result = {
                'diagnosis': 'Image processed - AI analysis temporarily unavailable',
                'severity': 'Normal',
                'confidence': 60.0,
                'detection_count': 0,
                'affected_area_ratio': 0.0,
                'recommendations': [
                    'Consult with healthcare provider for detailed analysis',
                    'AI models temporarily unavailable - manual review recommended'
                ],
                'preprocessing_info': {'note': 'Fallback analysis - AI models unavailable'}
            }
            
        # Try advanced model for enhancement if available and needed
        if (ML_ADVANCED_AVAILABLE and advanced_detector and analysis_result and 
            analysis_result.get('confidence', 0) < 70 and analysis_result.get('detection_count', 0) == 0):
            print(f"Low confidence result, trying advanced analysis for verification...")
            try:
                advanced_result = advanced_detector.detect_varicose_veins(file_path)
                print(f"Advanced verification result: {advanced_result}")
                
                # If advanced model finds something significant, use it
                if advanced_result.get('severity') != 'Normal' and advanced_result.get('confidence', 0) > 80:
                    print("Advanced model found significant findings, using advanced result")
                    analysis_result = advanced_result
                else:
                    print("Both models agree on normal result, using original model")
                    # Keep original result but boost confidence if both agree
                    if analysis_result.get('severity') == 'Normal' and advanced_result.get('severity') == 'Normal':
                        analysis_result['confidence'] = min(90.0, analysis_result.get('confidence', 70) + 10)
            except Exception as adv_error:
                print(f"Advanced model verification failed: {adv_error}")
                # Keep original result
                pass
        
        # Translate results if needed
        if language != "en":
            diagnosis_key = "diagnosis_varicose" if "Detected" in analysis_result.get('diagnosis', '') else "diagnosis_normal"
            severity_key = f"severity_{analysis_result.get('severity', 'normal').lower()}"
            
            analysis_result['diagnosis'] = translate_text(diagnosis_key, language)
            analysis_result['severity'] = translate_text(severity_key, language)
        
        # Save analysis to database
        analysis_data = {
            "patient_id": patient_id,
            "image_path": file_path,
            "diagnosis": analysis_result['diagnosis'],
            "severity": analysis_result['severity'],
            "confidence": analysis_result['confidence'],
            "detection_count": analysis_result.get('detection_count', 0),
            "affected_area_ratio": analysis_result.get('affected_area_ratio', 0.0),
            "recommendations": analysis_result.get('recommendations', []),
            "preprocessing_info": analysis_result.get('preprocessing_info', {})
        }
        
        print(f"Saving analysis data: {analysis_data}")
        try:
            db_analysis = db_manager.create_analysis(db, analysis_data)
            print(f"Analysis saved successfully: {db_analysis}")
        except Exception as e:
            print(f"Failed to save analysis: {str(e)}")
            raise Exception(f"Database save failed: {str(e)}")
        
        # Convert numpy types for JSON response
        response_data = {
            "analysis_id": db_analysis['id'],
            **analysis_result
        }
        return convert_numpy_types(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/symptoms/")
async def record_symptoms(symptoms: SymptomData, db: Session = Depends(get_db)):
    """Record patient symptoms"""
    try:
        db_symptoms = db_manager.create_symptom_record(db, symptoms.dict())
        return {"message": "Symptoms recorded successfully", "record_id": db_symptoms.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/patients/{patient_id}/analyses")
async def get_patient_analyses(patient_id: int, db: Session = Depends(get_db)):
    """Get all analyses for a patient"""
    analyses = db_manager.get_patient_analyses(db, patient_id)
    return [{
        "id": analysis.id,
        "diagnosis": analysis.diagnosis,
        "severity": analysis.severity,
        "confidence": analysis.confidence,
        "detection_count": analysis.detection_count,
        "affected_area_ratio": analysis.affected_area_ratio,
        "created_at": analysis.created_at,
        "recommendations": analysis.recommendations
    } for analysis in analyses]

@app.get("/patients/{patient_id}/comparison")
async def get_comparison_data(patient_id: int, limit: int = 5, db: Session = Depends(get_db)):
    """Get comparison data for patient analyses"""
    analyses = db_manager.get_analysis_comparison(db, patient_id, limit)
    
    if len(analyses) < 2:
        raise HTTPException(status_code=400, detail="Insufficient data for comparison")
    
    comparison_data = []
    for analysis in analyses:
        comparison_data.append({
            "date": analysis.created_at,
            "severity": analysis.severity,
            "confidence": analysis.confidence,
            "detection_count": analysis.detection_count,
            "affected_area_ratio": analysis.affected_area_ratio
        })
    
    return {
        "comparison_data": comparison_data,
        "trend_analysis": analyze_trend(comparison_data)
    }

def analyze_trend(data: List[dict]) -> dict:
    """Analyze trend in patient data"""
    if len(data) < 2:
        return {"trend": "insufficient_data"}
    
    severity_map = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
    recent_severity = severity_map.get(data[0]['severity'], 0)
    previous_severity = severity_map.get(data[1]['severity'], 0)
    
    if recent_severity > previous_severity:
        trend = "worsening"
    elif recent_severity < previous_severity:
        trend = "improving"
    else:
        trend = "stable"
    
    return {
        "trend": trend,
        "severity_change": recent_severity - previous_severity,
        "confidence_change": data[0]['confidence'] - data[1]['confidence']
    }

@app.post("/generate-report/{patient_id}")
async def generate_report(
    patient_id: int,
    analysis_id: int,
    report_type: str = "standard",
    db: Session = Depends(get_db)
):
    """Generate PDF report for patient analysis"""
    if not REPORTS_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="PDF report generation is temporarily unavailable. ReportLab not installed."
        )
    
    try:
        # Get patient data
        patient = db_manager.get_patient(db, patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        # Get analysis data from Supabase
        analyses = db_manager.get_patient_analyses(db, patient_id)
        analysis = None
        for a in analyses:
            if a.get('id') == analysis_id:
                analysis = a
                break
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Prepare data for report generation
        patient_data = {
            "id": patient["id"],
            "name": patient["name"],
            "age": patient["age"],
            "gender": patient["gender"]
        }
        
        analysis_data = {
            "diagnosis": analysis["diagnosis"],
            "severity": analysis["severity"],
            "confidence": analysis["confidence"],
            "detection_count": analysis["detection_count"],
            "affected_area_ratio": analysis["affected_area_ratio"],
            "recommendations": analysis["recommendations"],
            "created_at": analysis["created_at"]
        }
        
        # Get symptoms if available
        symptoms = db_manager.get_patient_symptoms(db, patient_id)
        symptoms_data = symptoms[0] if symptoms else None
        
        # Generate report
        if report_type == "comparison":
            analyses_data = [{
                "diagnosis": a["diagnosis"],
                "severity": a["severity"],
                "confidence": a["confidence"],
                "created_at": a["created_at"]
            } for a in analyses]
            report_path = report_generator.generate_comparison_report(patient_data, analyses_data)
        else:
            report_path = report_generator.generate_standard_report(patient_data, analysis_data, symptoms_data)
        
        # Save report record to database
        report_data = {
            "patient_id": patient_id,
            "analysis_id": analysis_id,
            "report_type": report_type,
            "pdf_path": report_path
        }
        db_report = db_manager.create_report(db, report_data)
        
        return {"message": "Report generated successfully", "report_id": db_report["id"], "file_path": report_path}
        
    except Exception as e:
        print(f"Report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/download-report/{report_id}")
async def download_report(report_id: int, db: Session = Depends(get_db)):
    """Download generated PDF report"""
    if not REPORTS_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="PDF report downloads are temporarily unavailable. ReportLab not installed."
        )
    
    try:
        report = db_manager.get_report(db, report_id)
        if not report or not os.path.exists(report["pdf_path"]):
            raise HTTPException(status_code=404, detail="Report not found")
        
        filename = f"varicose_report_{report['patient_id']}.pdf"
        return FileResponse(
            path=report["pdf_path"],
            media_type="application/pdf",
            filename=filename,
            headers={
                "Content-Disposition": f"attachment; filename=\"{filename}\"",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Content-Type-Options": "nosniff"
            }
        )
    except Exception as e:
        print(f"Download report error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download report: {str(e)}")

@app.post("/appointments/")
async def create_appointment(appointment: AppointmentCreate, db: Session = Depends(get_db)):
    """Schedule an appointment"""
    try:
        db_appointment = db_manager.create_appointment(db, appointment.dict())
        
        # Create reminder 24 hours before appointment
        reminder_data = {
            "patient_id": appointment.patient_id,
            "reminder_type": "appointment",
            "message": f"Reminder: You have a {appointment.appointment_type} appointment scheduled for tomorrow.",
            "scheduled_date": appointment.scheduled_date - timedelta(days=1)
        }
        db_manager.create_reminder(db, reminder_data)
        
        return {"message": "Appointment scheduled successfully", "appointment_id": db_appointment.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/reminders/")
async def create_reminder(reminder: ReminderCreate, db: Session = Depends(get_db)):
    """Create a reminder"""
    try:
        db_reminder = db_manager.create_reminder(db, reminder.dict())
        return {"message": "Reminder created successfully", "reminder_id": db_reminder.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/recommendations/{severity}")
async def get_recommendations(severity: str, language: str = "en"):
    """Get recommendations based on severity"""
    recommendations = {
        "Normal": [
            "Continue regular exercise and maintain a healthy lifestyle",
            "Monitor your legs for any changes",
            "Consider wearing compression socks during long periods of standing"
        ],
        "Mild": [
            "Wear compression stockings daily",
            "Elevate your legs when resting",
            "Avoid prolonged standing or sitting",
            "Exercise regularly, especially walking and swimming",
            "Maintain a healthy weight"
        ],
        "Moderate": [
            "Consult with a vascular specialist for proper evaluation",
            "Use medical-grade compression stockings",
            "Consider sclerotherapy or other minimally invasive treatments",
            "Elevate legs above heart level when possible",
            "Avoid high heels and tight clothing"
        ],
        "Severe": [
            "Seek immediate consultation with a vascular surgeon",
            "Consider surgical intervention (vein stripping, endovenous ablation)",
            "Use high-compression medical stockings",
            "Monitor for complications like ulcers or blood clots",
            "Follow strict lifestyle modifications"
        ]
    }
    
    return {"recommendations": recommendations.get(severity, [])}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

# Chat endpoint for AI assistant
class ChatMessage(BaseModel):
    message: str
    language: str = "en"
    session_id: str

@app.post("/chat")
async def chat_with_ai(chat_data: ChatMessage):
    """Chat with AI medical assistant"""
    try:
        response = await medical_chatbot.get_response(
            message=chat_data.message,
            language=chat_data.language,
            session_id=chat_data.session_id
        )
        return {
            "response": response,
            "language": chat_data.language,
            "session_id": chat_data.session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str, limit: int = 50):
    """Get chat history for a session"""
    try:
        history = db_manager.get_chat_history(session_id, limit)
        return {"history": history, "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting chat history: {str(e)}")

# Risk assessment endpoint
class RiskAssessmentData(BaseModel):
    patient_id: int
    age: int
    bmi: float
    risk_factors: list

@app.post("/risk-assessment")
async def create_risk_assessment(assessment: RiskAssessmentData, db = Depends(get_db)):
    """Create a risk assessment"""
    try:
        # Calculate risk score
        total_score = 0
        
        # Age factor
        if assessment.age > 50:
            total_score += 2
        if assessment.age > 65:
            total_score += 1
        
        # BMI factor
        if assessment.bmi > 30:
            total_score += 2.5
        elif assessment.bmi > 25:
            total_score += 1
        
        # Risk factors
        for factor in assessment.risk_factors:
            total_score += factor.get('weight', 0)
        
        # Determine risk level
        if total_score <= 2:
            risk_level = "Low"
        elif total_score <= 5:
            risk_level = "Moderate"
        elif total_score <= 8:
            risk_level = "High"
        else:
            risk_level = "Very High"
        
        # Prepare data
        assessment_data = {
            "patient_id": assessment.patient_id,
            "age_factor": 2 if assessment.age > 50 else 0,
            "bmi_factor": 2.5 if assessment.bmi > 30 else (1 if assessment.bmi > 25 else 0),
            "risk_factors": assessment.risk_factors,
            "total_score": total_score,
            "risk_level": risk_level,
            "recommendations": []
        }
        
        result = db_manager.create_risk_assessment(db, assessment_data)
        return {"assessment": result, "risk_level": risk_level, "score": total_score}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {str(e)}")

@app.get("/patients/{patient_id}/risk-assessments")
async def get_patient_risk_assessments(patient_id: int, db = Depends(get_db)):
    """Get risk assessments for a patient"""
    try:
        assessments = db_manager.get_patient_risk_assessments(db, patient_id)
        return {"assessments": assessments}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting risk assessments: {str(e)}")

@app.get("/api/help")
async def api_help():
    """Get API usage help and examples"""
    return {
        "message": "Varicose Vein Detection API - Usage Guide",
        "endpoints": {
            "/patients/": {
                "method": "POST",
                "description": "Create a new patient record",
                "required_fields": ["name", "age", "gender"],
                "example": {
                    "name": "John Doe",
                    "age": 45,
                    "gender": "Male",
                    "phone": "+1234567890",
                    "email": "john@example.com"
                }
            },
            "/analyze": {
                "method": "POST",
                "description": "Analyze an image for varicose veins",
                "content_type": "multipart/form-data",
                "required_fields": ["file", "patient_id"],
                "optional_fields": ["language"],
                "example_curl": "curl -X POST http://localhost:8000/analyze -F 'file=@leg_image.jpg' -F 'patient_id=1' -F 'language=en'"
            },
            "/chat": {
                "method": "POST",
                "description": "Chat with AI medical assistant",
                "required_fields": ["message", "session_id"],
                "optional_fields": ["language"],
                "example": {
                    "message": "What are varicose veins?",
                    "language": "en",
                    "session_id": "user123"
                }
            },
            "/risk-assessment": {
                "method": "POST",
                "description": "Create a risk assessment",
                "required_fields": ["patient_id", "age", "bmi", "risk_factors"]
            }
        },
        "common_errors": {
            "422": "Validation Error - Missing required fields or invalid data types",
            "404": "Patient not found - Make sure the patient_id exists",
            "400": "Bad Request - Invalid file type (only images allowed)"
        },
        "supported_languages": ["en", "hi", "mr"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

