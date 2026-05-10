# backend/main.py
from fastapi import FastAPI, UploadFile, Form, Depends, HTTPException, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import shutil, os, traceback, tempfile, uuid
import uvicorn
from datetime import datetime
from sqlalchemy.orm import Session

import redis.asyncio as redis_async
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.gzip import GZipMiddleware

from config import settings

# --- Database import ---
try:
    from database import db_manager, Patient, Analysis, Report
    DATABASE_AVAILABLE = True
    print("[OK] Database module loaded successfully")
except Exception as e:
    print(f"[ERROR] CRITICAL: Database not available: {e}")
    DATABASE_AVAILABLE = False
    # In production, we might want to raise here
    # raise e

# --- ML model import ---
try:
    # Use Roboflow API for real-time detection
    from roboflow_client import roboflow_detector
    detector = roboflow_detector
    ML_BASIC_AVAILABLE = True
    print("[OK] Roboflow ML model loaded successfully")
except Exception as e:
    try:
        # Fallback to advanced local model if Roboflow fails
        from advanced_ml_model import advanced_detector
        detector = advanced_detector
        ML_BASIC_AVAILABLE = True
        print("[OK] Advanced ML model loaded successfully (fallback)")
    except Exception as e2:
        try:
            # Final fallback to basic model
            from ml_model import VaricoseVeinDetector
            detector = VaricoseVeinDetector()
            ML_BASIC_AVAILABLE = True
            print("[OK] Basic ML model loaded successfully (final fallback)")
        except Exception as e3:
            ML_BASIC_AVAILABLE = False
            detector = None
            print(f"[WARNING] No ML model available: Roboflow: {e}, Advanced: {e2}, Basic: {e3}")

# --- PDF generator ---
try:
    from report_generator import report_generator
    REPORTS_AVAILABLE = True
except ImportError:
    try:
        from fpdf_report_generator import fpdf_report_generator as report_generator
        REPORTS_AVAILABLE = True
    except ImportError:
        REPORTS_AVAILABLE = False
        report_generator = None

# --- Rate Limiting Config ---
limiter = Limiter(key_func=get_remote_address, storage_uri=settings.REDIS_URL)

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Redis and FastAPI Cache
    redis_client = redis_async.from_url(settings.REDIS_URL, encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis_client), prefix="varixscan-cache")
    print(f"[OK] Redis cache initialized at {settings.REDIS_URL}")
    
    try:
        db = db_manager.get_db()
        print("[OK] Connected to database")
    except Exception as e:
        print(f"[WARNING] Could not connect to database: {e}")
    yield
    print("[INFO] Application shutting down...")

app = FastAPI(title="VarixScan AI API", version="2.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Exception handler ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"[ERROR] VALIDATION ERROR on {request.url}: {exc.errors()}")
    errors = [f"{error['loc'][-1]}: {error['msg']}" for error in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "Validation Error", "details": errors})

import numpy as np

def convert_numpy_types(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
    else: return obj

# --- CORS & Middlewares ---
origins = [
    settings.FRONTEND_URL,
    "http://localhost:3000",
    "http://localhost:3001"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.middleware("http")
async def add_request_id_tracing(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# --- Ensure directories exist ---
os.makedirs("uploads", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# --- Pydantic models ---
class PatientCreate(BaseModel):
    name: str
    age: int
    gender: str
    phone: Optional[str] = None
    email: Optional[str] = None

# --- Translations ---
TRANSLATIONS = {
    "en": {"diagnosis_varicose": "Varicose Veins Detected", "diagnosis_normal": "No Varicose Veins Detected"},
    "hi": {"diagnosis_varicose": "वैरिकाज़ वेन्स का पता चला", "diagnosis_normal": "कोई वैरिकाज़ वेन्स नहीं मिली"},
    "mr": {"diagnosis_varicose": "व्हॅरिकोज व्हेन्स आढळल्या", "diagnosis_normal": "व्हॅरिकोज व्हेन्स नाहीत"}
}

def get_db():
    if not DATABASE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database connection is currently unavailable. Please try again later.")
    
    db = db_manager.SessionLocal()
    try:
        yield db
    finally:
        db.close()

def translate_text(text: str, language: str = "en") -> str:
    if language in TRANSLATIONS and text in TRANSLATIONS[language]:
        return TRANSLATIONS[language][text]
    return text

# --- Endpoints ---
@app.get("/")
async def root():
    return {"message": "VarixScan AI API running", "status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    redis_status = "unknown"
    try:
        from fastapi_cache import FastAPICache
        backend = FastAPICache.get_backend()
        if backend:
            # Simple ping to check if alive
            await backend.redis.ping()
            redis_status = "connected"
    except Exception as e:
        redis_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "database": "connected" if DATABASE_AVAILABLE else "mock",
        "redis": redis_status,
        "features": {"basic_ml_model": ML_BASIC_AVAILABLE, "pdf_reports": REPORTS_AVAILABLE}
    }

@app.post("/predict")
async def predict_varicose_veins(file: UploadFile = File(...)):
    """
    Standalone prediction endpoint using Roboflow API
    Accepts an image and returns predictions without storing in database
    """
    temp_path = None
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Save uploaded file temporarily
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name
        
        # Run prediction using Roboflow
        if ML_BASIC_AVAILABLE and detector:
            try:
                print(f"[INFO] Running Roboflow prediction")
                if hasattr(detector, 'detect_varicose_veins'):
                    result = detector.detect_varicose_veins(temp_path)
                    print(f"[OK] Prediction completed: {result.get('diagnosis', 'Unknown')}")
                    return convert_numpy_types(result)
                else:
                    raise HTTPException(status_code=500, detail="Detector not available")
            except Exception as e:
                print(f"[ERROR] Prediction failed: {str(e)}")
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        else:
            raise HTTPException(status_code=503, detail="ML model not available")
            
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

@app.post("/patients/")
async def create_patient(patient: PatientCreate, db: Session = Depends(get_db)):
    try:
        patient_data = patient.model_dump() if hasattr(patient, 'model_dump') else patient.dict()
        db_patient = db_manager.create_patient(db, patient_data)
        patient_id = db_patient.id if hasattr(db_patient, 'id') else db_patient['id']
        return {"message": "Patient created", "patient_id": patient_id}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/patients/")
async def get_patients(db: Session = Depends(get_db)):
    patients = db.query(Patient).order_by(Patient.created_at.desc()).all()
    return [{"id": p.id, "name": p.name, "age": p.age, "gender": p.gender, "created_at": str(p.created_at)} for p in patients]

@app.get("/api/dashboard/stats")
@cache(expire=60)
async def get_dashboard_stats(db: Session = Depends(get_db)):
    total_patients = db.query(Patient).count()
    total_analyses = db.query(Analysis).count()
    
    from datetime import datetime, timedelta
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    recent_analyses = db.query(Analysis).filter(Analysis.created_at >= seven_days_ago).count()
    
    confidence_data = db.query(Analysis.confidence).all()
    avg_confidence = round(sum([c[0] for c in confidence_data]) / len(confidence_data), 1) if confidence_data else 95.2
    
    return {
        "totalPatients": total_patients,
        "totalAnalyses": total_analyses,
        "recentAnalyses": recent_analyses,
        "avgConfidence": avg_confidence
    }

@app.get("/api/dashboard/recent-analyses")
@cache(expire=60)
async def get_recent_analyses(limit: int = 10, db: Session = Depends(get_db)):
    analyses = db.query(Analysis).order_by(Analysis.created_at.desc()).limit(limit).all()
    result = []
    for a in analyses:
        patient = db.query(Patient).filter(Patient.id == a.patient_id).first()
        result.append({
            "id": a.id,
            "patient_id": a.patient_id,
            "diagnosis": a.diagnosis,
            "severity": a.severity,
            "confidence": a.confidence,
            "created_at": str(a.created_at),
            "patients": {"id": patient.id, "name": patient.name} if patient else None
        })
    return result

@app.get("/api/analyses")
@cache(expire=60)
async def get_all_analyses(db: Session = Depends(get_db)):
    analyses = db.query(Analysis).order_by(Analysis.created_at.desc()).all()
    result = []
    for a in analyses:
        patient = db_manager.get_patient(db, a.patient_id)
        result.append({
            "id": a.id,
            "patient_id": a.patient_id,
            "diagnosis": a.diagnosis,
            "severity": a.severity,
            "confidence": a.confidence,
            "created_at": str(a.created_at),
            "patients": {"id": patient.id, "name": patient.name} if patient else None
        })
    return result

@app.get("/api/reports")
@cache(expire=60)
async def get_all_reports(db: Session = Depends(get_db)):
    reports = db.query(Report).order_by(Report.created_at.desc()).all()
    result = []
    for r in reports:
        patient = db_manager.get_patient(db, r.patient_id)
        analysis = db.query(Analysis).filter(Analysis.id == r.analysis_id).first()
        result.append({
            "id": r.id,
            "patient_id": r.patient_id,
            "analysis_id": r.analysis_id,
            "report_type": r.report_type,
            "pdf_path": r.pdf_path,
            "created_at": str(r.created_at),
            "patients": {"id": patient.id, "name": patient.name} if patient else None,
            "analyses": {"id": analysis.id, "diagnosis": analysis.diagnosis, "severity": analysis.severity} if analysis else None
        })
    return result

@app.get("/api/patients/{patient_id}/analyses")
async def get_patient_analyses(patient_id: int, db: Session = Depends(get_db)):
    analyses = db_manager.get_patient_analyses(db, patient_id)
    return [{
        "id": a.id,
        "diagnosis": a.diagnosis,
        "severity": a.severity,
        "confidence": a.confidence,
        "created_at": str(a.created_at)
    } for a in analyses]
    
@app.get("/api/patients/{patient_id}/reports")
async def get_patient_reports(patient_id: int, db: Session = Depends(get_db)):
    reports = db.query(Report).filter(Report.patient_id == patient_id).order_by(Report.created_at.desc()).all()
    result = []
    for r in reports:
        analysis = db.query(Analysis).filter(Analysis.id == r.analysis_id).first()
        result.append({
            "id": r.id,
            "pdf_path": r.pdf_path,
            "created_at": str(r.created_at),
            "analyses": {"id": analysis.id, "diagnosis": analysis.diagnosis, "severity": analysis.severity} if analysis else None
        })
    return result

@app.post("/analyze")
@limiter.limit("5/minute")
async def analyze_image(
    request: Request,
    file: UploadFile = File(...),
    patient_id: int = Form(...),
    language: str = Form("en"),
    db: Session = Depends(get_db)
):
    temp_path = None
    try:
        patient = db_manager.get_patient(db, patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        # Save uploaded file
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        # ML analysis with robust retry mechanism
        import time
        max_retries = 3
        analysis_result = None
        last_error = None

        if ML_BASIC_AVAILABLE and detector:
            for attempt in range(max_retries):
                try:
                    print(f"[INFO] ML Analysis attempt {attempt + 1}/{max_retries} with detector type: {type(detector).__name__}")
                    # Check if it's the advanced detector
                    if hasattr(detector, 'detect_varicose_veins'):
                        result = detector.detect_varicose_veins(temp_path)
                    else:
                        result = detector.detect_veins(temp_path)
                        
                    # Validate output
                    if result and isinstance(result, dict) and 'diagnosis' in result:
                        analysis_result = result
                        print(f"[OK] Detection completed: {analysis_result.get('diagnosis', 'Unknown')}")
                        break
                    else:
                        raise ValueError("Empty or invalid response from AI model")
                        
                except Exception as e:
                    last_error = e
                    print(f"[WARNING] ML Detection Attempt {attempt + 1} Failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(1) # Brief pause before retry
                    else:
                        traceback.print_exc()

        if not analysis_result:
            print(f"[ERROR] All ML detection attempts failed. Using safe fallback.")
            analysis_result = {
                'diagnosis': 'AI Analysis Failed. Please retry.',
                'severity': 'Unknown',
                'confidence': 0.0,
                'detection_count': 0,
                'affected_area_ratio': 0.0,
                'recommendations': ['System experienced an error during analysis.', 'Please try uploading the image again in a few moments.', f'Error details: {str(last_error) if last_error else "Model unavailable"}']
            }

        # Translation
        if language != "en":
            key = "diagnosis_varicose" if "Detected" in analysis_result['diagnosis'] else "diagnosis_normal"
            analysis_result['diagnosis'] = translate_text(key, language)

        # Save analysis in DB (filter fields to match database schema)
        # Only include fields that exist in the analyses table
        filtered_result = {
            'diagnosis': analysis_result.get('diagnosis', ''),
            'severity': analysis_result.get('severity', 'Normal'),
            'confidence': analysis_result.get('confidence', 0.0),
            'detection_count': analysis_result.get('detection_count', 0),
            'affected_area_ratio': analysis_result.get('affected_area_ratio', 0.0),
            'recommendations': analysis_result.get('recommendations', []),
            'preprocessing_info': analysis_result.get('preprocessing_info', {})
        }
        
        analysis_data = {"patient_id": patient_id, "image_path": temp_path, **filtered_result}
        db_analysis = db_manager.create_analysis(db, analysis_data)

        return convert_numpy_types({"analysis_id": db_analysis.id, **analysis_result})

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/generate-report/{patient_id}")
@limiter.limit("10/minute")
async def generate_report(
    request: Request,
    patient_id: int,
    analysis_id: Optional[int] = None,
    report_type: str = "standard",
    db: Session = Depends(get_db)
):
    """Generate report - accepts analysis_id from query params, form data, or JSON body"""
    if not REPORTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Report generation unavailable")
    try:
        # Try to get analysis_id from multiple sources
        if analysis_id is None and request:
            # Try query params
            analysis_id = request.query_params.get('analysis_id')
            if analysis_id:
                analysis_id = int(analysis_id)
        
        if analysis_id is None:
            raise HTTPException(status_code=400, detail="analysis_id is required (in query params, form data, or JSON body)")
        
        patient = db_manager.get_patient(db, patient_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        analyses = db_manager.get_patient_analyses(db, patient_id)
        analysis = next((a for a in analyses if (a.id if hasattr(a, 'id') else a.get('id')) == analysis_id), None)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        # Generate report
        patient_dict = {
            "id": patient.id if hasattr(patient, 'id') else patient.get("id"),
            "name": patient.name if hasattr(patient, 'name') else patient.get("name"),
            "age": patient.age if hasattr(patient, 'age') else patient.get("age"),
            "gender": patient.gender if hasattr(patient, 'gender') else patient.get("gender")
        }
        patient_data = patient_dict
        analysis_data = {
            "diagnosis": analysis.diagnosis if hasattr(analysis, 'diagnosis') else analysis.get('diagnosis', ''),
            "severity": analysis.severity if hasattr(analysis, 'severity') else analysis.get('severity', 'Normal'),
            "confidence": analysis.confidence if hasattr(analysis, 'confidence') else analysis.get('confidence', 0)
        }

        report_path = report_generator.generate_standard_report(patient_data, analysis_data, None)
        db_report = db_manager.create_report(db, {"patient_id": patient_id, "pdf_path": report_path, "analysis_id": analysis_id})

        return {"message": "Report generated", "report_path": report_path, "report_id": db_report.id}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/download-report/{report_id}")
async def download_report(report_id: int, db: Session = Depends(get_db)):
    """Download a generated report by report ID"""
    try:
        report = db_manager.get_report(db, report_id)
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        pdf_path = report.pdf_path if hasattr(report, 'pdf_path') else report.get('pdf_path')
        if not pdf_path or not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="Report file not found")
        
        # Get patient info for filename
        patient_id = report.patient_id if hasattr(report, 'patient_id') else report.get('patient_id')
        patient = db_manager.get_patient(db, patient_id) if patient_id else None
        
        # Create a nice filename
        if patient:
            patient_name = patient.name if hasattr(patient, 'name') else patient.get('name', 'Patient')
            filename = f"VarixScan_Report_{patient_name}_{report_id}.pdf"
        else:
            filename = f"VarixScan_Report_{report_id}.pdf"
        
        return FileResponse(
            path=pdf_path,
            filename=filename,
            media_type="application/pdf"
        )
        
    except Exception as e:
        print(f"Download error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

# --- Run Uvicorn ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
