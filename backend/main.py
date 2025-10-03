# backend/main.py
from fastapi import FastAPI, UploadFile, Form, Depends, HTTPException, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
import shutil, os, traceback, tempfile
import uvicorn
from datetime import datetime
from sqlalchemy.orm import Session

# --- Database import with fallback ---
try:
    from database import db_manager, convert_numpy_types
    DATABASE_AVAILABLE = True
    print("[OK] Database module loaded successfully")
except Exception as e:
    print(f"[WARNING] Database not available: {e}")
    DATABASE_AVAILABLE = False

    class MockDBManager:
        def get_db(self): return None
        def create_patient(self, db, data): return {'id': 1}
        def get_patient(self, db, patient_id): return {'id': patient_id, 'name': 'Test Patient', 'age': 30, 'gender': 'Male'}
        def create_analysis(self, db, data): return {'id': 1}
        def get_patient_analyses(self, db, patient_id): return []
        def create_report(self, db, data): return {'id': 1}
        def get_report(self, db, report_id): return {'pdf_path': 'reports/sample.pdf', 'patient_id': 1}

    db_manager = MockDBManager()
    convert_numpy_types = lambda x: x

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

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        db = db_manager.get_db()
        print("[OK] Connected to database")
    except Exception as e:
        print(f"[WARNING] Could not connect to database: {e}")
    yield
    print("[INFO] Application shutting down...")

app = FastAPI(title="VarixScan AI API", version="2.0.0", lifespan=lifespan)

# --- Exception handler ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"[ERROR] VALIDATION ERROR on {request.url}: {exc.errors()}")
    errors = [f"{error['loc'][-1]}: {error['msg']}" for error in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "Validation Error", "details": errors})

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    db = db_manager.get_db()
    try:
        yield db
    finally:
        if hasattr(db, 'close') and db:
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
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "database": "connected" if DATABASE_AVAILABLE else "mock",
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
        return {"message": "Patient created", "patient_id": db_patient['id']}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze")
async def analyze_image(
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

        # ML analysis
        analysis_result = None
        if ML_BASIC_AVAILABLE and detector:
            try:
                print(f"[INFO] Starting ML analysis with detector type: {type(detector).__name__}")
                # Check if it's the advanced detector
                if hasattr(detector, 'detect_varicose_veins'):
                    print("Using advanced detection method")
                    analysis_result = detector.detect_varicose_veins(temp_path)
                    print(f"[OK] Advanced detection completed: {analysis_result.get('diagnosis', 'Unknown')}")
                else:
                    # Fallback to basic detector method
                    print("Using basic detection method")
                    analysis_result = detector.detect_veins(temp_path)
                    print(f"[OK] Basic detection completed: {analysis_result.get('diagnosis', 'Unknown')}")
            except Exception as e:
                print(f"[ERROR] ML Detection Error: {str(e)}")
                traceback.print_exc()
                analysis_result = None

        if not analysis_result:
            analysis_result = {
                'diagnosis': 'Image processed - AI analysis temporarily unavailable',
                'severity': 'Normal',
                'confidence': 60.0,
                'detection_count': 0,
                'affected_area_ratio': 0.0,
                'recommendations': ['Consult with healthcare provider']
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

        return convert_numpy_types({"analysis_id": db_analysis['id'], **analysis_result})

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/generate-report/{patient_id}")
async def generate_report(
    patient_id: int,
    analysis_id: Optional[int] = None,
    report_type: str = "standard",
    request: Request = None,
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
        analysis = next((a for a in analyses if getattr(a, 'id', a.get('id', None)) == analysis_id), None)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        # Generate report
        patient_data = {"id": patient["id"], "name": patient["name"], "age": patient["age"], "gender": patient["gender"]}
        analysis_data = {
            "diagnosis": getattr(analysis, 'diagnosis', analysis.get('diagnosis', '')),
            "severity": getattr(analysis, 'severity', analysis.get('severity', 'Normal')),
            "confidence": getattr(analysis, 'confidence', 0)
        }

        report_path = report_generator.generate_standard_report(patient_data, analysis_data, None)
        db_report = db_manager.create_report(db, {"patient_id": patient_id, "pdf_path": report_path, "analysis_id": analysis_id})

        return {"message": "Report generated", "report_path": report_path, "report_id": db_report['id']}

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
        
        pdf_path = report.get('pdf_path')
        if not pdf_path or not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="Report file not found")
        
        # Get patient info for filename
        patient_id = report.get('patient_id')
        patient = db_manager.get_patient(db, patient_id) if patient_id else None
        
        # Create a nice filename
        if patient:
            filename = f"VarixScan_Report_{patient.get('name', 'Patient')}_{report_id}.pdf"
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
