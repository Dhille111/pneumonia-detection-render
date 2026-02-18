@echo off
REM Quick deployment test script for Windows

echo 🔍 Checking deployment readiness...

REM Check Python version
python --version

REM Check if model file exists
if exist pneumonia_model.pth (
    echo ✅ Model file found
) else (
    echo ❌ Model file not found
)

REM Install requirements
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Test Flask app import
echo 🧪 Testing app import...
python -c "from app import app; print('✅ App imports successfully')" || echo ❌ App import failed

REM Start app
echo 🚀 Starting application on http://localhost:5000
python app.py
