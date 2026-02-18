import os
import logging
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, render_template, request, redirect, jsonify
from PIL import Image
from werkzeug.utils import secure_filename
import io

# -------------------- Logging setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------- App setup --------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

# -------------------- Device (CPU only for Render) --------------------
device = torch.device("cpu")

# -------------------- Load model --------------------
MODEL_PATH = os.path.join(BASE_DIR, "pneumonia_model.pth")

model = None
try:
    model = models.resnet18(weights=None)  # Updated: use weights parameter
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# -------------------- Image transforms --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def allowed_file(filename):
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html", prediction=None)

@app.route("/health")
def health():
    """Health check endpoint for deployment monitoring"""
    return jsonify({"status": "ok", "model_loaded": model is not None}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Process image and make prediction"""
    try:
        if model is None:
            logger.error("Model not loaded")
            return render_template(
                "index.html",
                prediction="ERROR",
                error="Model failed to load. Please try again later."
            ), 500

        if "file" not in request.files:
            logger.warning("No file provided in request")
            return redirect("/")

        file = request.files["file"]
        if file.filename == "":
            logger.warning("Empty filename provided")
            return redirect("/")

        # Validate file type
        if not allowed_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            return render_template(
                "index.html",
                prediction="ERROR",
                error="Invalid file type. Please upload an image (PNG, JPG, GIF, BMP, WebP)."
            ), 400

        # Secure filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        # Create unique filename to avoid overwrites
        import uuid
        base, ext = os.path.splitext(filename)
        filename = f"{base}_{uuid.uuid4().hex[:8]}{ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        file.save(filepath)
        logger.info(f"File uploaded: {filename}")

        # Load and process image
        image = Image.open(filepath).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            confidence = probabilities[0][preds.item()].item() * 100
            
            classes = ["NORMAL", "PNEUMONIA"]
            result = classes[preds.item()]

        logger.info(f"Prediction: {result} (Confidence: {confidence:.2f}%)")
        
        return render_template(
            "index.html",
            prediction=result,
            confidence=f"{confidence:.2f}%",
            img_path=f"static/uploads/{filename}"
        )
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return render_template(
            "index.html",
            prediction="ERROR",
            error="An error occurred during prediction. Please try again."
        ), 500

@app.errorhandler(400)
def bad_request(error):
    return render_template("index.html", error="Bad request"), 400

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return render_template("index.html", error="Internal server error"), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
