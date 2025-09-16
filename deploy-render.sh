#!/bin/bash

# VarixScan Render Deployment Script
# This script helps deploy the backend to Render

echo "🚀 VarixScan Render Deployment Helper"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "backend/main.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

echo "📋 Pre-deployment Checklist:"
echo ""

# Check Python version
python_version=$(python --version 2>&1)
echo "✅ Python version: $python_version"

# Check if backend dependencies are specified
if [ -f "backend/requirements.txt" ]; then
    dep_count=$(wc -l < backend/requirements.txt)
    echo "✅ Requirements.txt found with $dep_count dependencies"
else
    echo "❌ Error: backend/requirements.txt not found"
    exit 1
fi

# Check if main.py exists
if [ -f "backend/main.py" ]; then
    echo "✅ Backend main.py found"
else
    echo "❌ Error: backend/main.py not found"
    exit 1
fi

# Check for essential environment variables template
if [ -f ".env.example" ]; then
    echo "✅ Environment variables template found"
else
    echo "⚠️  Warning: .env.example not found"
fi

echo ""
echo "🔧 Render Configuration:"
echo "- Runtime: Python 3.11.10"
echo "- Build Command: cd backend && pip install --upgrade pip setuptools wheel && pip install --no-cache-dir -r requirements.txt"
echo "- Start Command: cd backend && uvicorn main:app --host 0.0.0.0 --port \$PORT"
echo "- Health Check Path: /"

echo ""
echo "📝 Required Environment Variables for Render:"
echo "- SUPABASE_URL"
echo "- SUPABASE_KEY" 
echo "- OPENAI_API_KEY"
echo "- ENVIRONMENT=production"
echo "- DEBUG=false"

echo ""
echo "🌐 Next Steps:"
echo "1. Go to https://render.com/dashboard"
echo "2. Create New -> Web Service"
echo "3. Connect your GitHub repository"
echo "4. Select Python runtime"
echo "5. Copy the build/start commands above"
echo "6. Add all environment variables"
echo "7. Deploy!"

echo ""
echo "✅ Deployment preparation complete!"
echo "Your VarixScan backend is ready for Render deployment."
