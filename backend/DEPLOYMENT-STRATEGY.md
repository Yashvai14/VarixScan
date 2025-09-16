# VarixScan Deployment Strategy

## Two-Tier Deployment Approach

Due to deployment platform limitations with heavy ML libraries, VarixScan uses a **smart two-tier approach**:

### 🚀 **Production Deployment (Render)**
**File:** `requirements.txt` (minimal, bulletproof)

**Features Enabled:**
- ✅ Core FastAPI application
- ✅ Image upload and basic processing
- ✅ Patient management system
- ✅ Database operations (Supabase)
- ✅ AI chatbot (OpenAI GPT)
- ✅ PDF report generation (FPDF2)
- ✅ Basic image analysis with OpenCV-headless
- ✅ Authentication and security

**Features with Graceful Fallbacks:**
- ⚠️ Advanced ML models (PyTorch, scikit-learn) - Optional
- ⚠️ Complex image processing - Falls back to basic analysis

### 🔬 **Local Development (Full Features)**
**File:** `requirements-full.txt` (complete ML stack)

**Additional Features:**
- ✅ Complete PyTorch integration
- ✅ Advanced scikit-learn models
- ✅ Full OpenCV with GUI support
- ✅ YOLO object detection
- ✅ Complex image processing
- ✅ Multiple PDF generation options

## How It Works

### 1. **Graceful Fallbacks**
The application automatically detects which modules are available and adjusts functionality:

```python
# ML models with fallback
if ML_BASIC_AVAILABLE and detector:
    analysis_result = detector.detect_veins(file_path)
else:
    # Provide fallback analysis
    analysis_result = fallback_analysis()
```

### 2. **Feature Detection**
The `/health` endpoint shows exactly which features are loaded:

```json
{
  "status": "healthy",
  "features": {
    "basic_ml_model": false,
    "pdf_reports": true,
    "ai_chatbot": true
  }
}
```

### 3. **Progressive Enhancement**
- **Core features always work** (image upload, patient management, chatbot)
- **ML features enhance the experience** when available
- **No crashes or errors** when ML libraries are missing

## Deployment Commands

### For Render (Production)
```bash
# Uses requirements.txt (minimal)
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### For Local Development
```bash
# Uses requirements-full.txt (complete)
pip install -r requirements-full.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Benefits

1. **✅ Reliable Deployment** - Minimal dependencies = fewer conflicts
2. **✅ Full Feature Parity** - All original features work locally
3. **✅ Production Ready** - Core functionality never breaks
4. **✅ Scalable** - Easy to upgrade deployment tier when needed
5. **✅ No Maintenance Overhead** - Automatic fallback handling

## Migration Path

When deployment platforms improve ML support:
1. Simply switch to `requirements-full.txt`
2. All features automatically activate
3. No code changes needed

This approach ensures **VarixScan works everywhere** while maintaining **full functionality** where possible.
