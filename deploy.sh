#!/bin/bash
# Quick deployment test script

echo "🔍 Checking deployment readiness..."

# Check Python version
python --version

# Check if model file exists
if [ -f "pneumonia_model.pth" ]; then
    echo "✅ Model file found"
else
    echo "❌ Model file not found"
fi

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Test Flask app import
echo "🧪 Testing app import..."
python -c "from app import app; print('✅ App imports successfully')" || echo "❌ App import failed"

# Start app
echo "🚀 Starting application on http://localhost:5000"
python app.py
