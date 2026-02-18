# Pneumonia Detection from Chest X-rays - Deployment Ready

## Overview
Production-ready Flask application for pneumonia detection using chest X-ray images with a pre-trained ResNet18 model.

## Features
- ✅ Secure file upload with validation
- ✅ Production logging
- ✅ Error handling and health checks
- ✅ WSGI-compatible for production servers
- ✅ Docker support
- ✅ Easy deployment to Render, Heroku, or other platforms

## Installation

### Local Development
```bash
# Clone the repository
git clone <your-repo-url>
cd Pneumonia_detection_from_chest_xrays

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Docker Deployment
```bash
# Build image
docker build -t pneumonia-detection .

# Run container
docker run -p 5000:5000 pneumonia-detection

# Or use docker-compose
docker-compose up -d
```

## Project Structure
```
├── app.py                 # Main Flask application
├── main.py                # Model training script
├── pneumonia_model.pth    # Pre-trained model weights
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose configuration
├── render.yaml            # Render.com deployment config
├── Procfile               # Heroku deployment config
├── wsgi.py                # WSGI entry point
├── .env.example           # Environment variables template
├── templates/
│   └── index.html         # Web interface
└── static/
    └── uploads/           # User uploaded X-rays
```

## Environment Variables
```
PORT=5000                    # Application port (default: 5000)
FLASK_ENV=production         # Flask environment
FLASK_DEBUG=False            # Disable debug mode in production
```

## API Endpoints

### GET /
- **Description**: Main web interface
- **Returns**: HTML page with upload form

### POST /predict
- **Description**: Predict pneumonia from chest X-ray
- **Parameters**: 
  - `file`: Image file (PNG, JPG, GIF, BMP, WebP)
- **Returns**: HTML page with prediction result and confidence score
- **Max file size**: 16MB

### GET /health
- **Description**: Health check endpoint (for monitoring/load balancers)
- **Returns**: JSON with status
- **Example response**: `{"status": "ok", "model_loaded": true}`

## Deployment

### Render.com (Recommended)
```bash
# Push to GitHub
git push origin main

# Connect repository to Render
# Use render_updated.yaml for configuration
```

### Heroku
```bash
# Login to Heroku
heroku login

# Create Heroku app
heroku create your-app-name

# Deploy
git push heroku main
```

### Docker/VPS
```bash
# Build and push image
docker build -t yourusername/pneumonia-detection .
docker push yourusername/pneumonia-detection

# Run on server
docker pull yourusername/pneumonia-detection
docker run -d -p 80:5000 yourusername/pneumonia-detection
```

## Model Training
To retrain the model with new data:

```bash
# Ensure your data structure matches:
# chest_xray/
#   ├── train/
#   │   ├── NORMAL/
#   │   └── PNEUMONIA/
#   ├── val/
#   │   ├── NORMAL/
#   │   └── PNEUMONIA/
#   └── test/
#       ├── NORMAL/
#       └── PNEUMONIA/

python main.py
```

## Security Features
- File type validation (only image formats allowed)
- Filename sanitization to prevent path traversal
- File size limits (16MB max)
- Error handling without revealing sensitive info
- Secure headers in responses
- No debug mode in production

## Performance Notes
- **Model**: ResNet18 (lightweight)
- **Framework**: PyTorch (CPU only for deployment)
- **Inference time**: ~200-500ms per image
- **Memory usage**: ~500MB

## Troubleshooting

### Model file not found
- Ensure `pneumonia_model.pth` is in the root directory
- Check file permissions

### Port already in use
- Change PORT environment variable
- Or kill process using the port

### Import errors
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`
- Check Python version (3.9+)

### Deployment issues
- Check logs: `docker logs <container-id>`
- Verify health endpoint: `curl http://localhost:5000/health`

## API Response Examples

### Success Response
```json
{
  "prediction": "NORMAL",
  "confidence": "95.23%"
}
```

### Error Response
```json
{
  "error": "Invalid file format"
}
```

## License
MIT License

## Support
For issues and questions, please create a GitHub issue.
