import os
import logging
import numpy as np
import onnxruntime as ort
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

# -------------------- Load ONNX model --------------------
MODEL_PATH = os.path.join(BASE_DIR, "pneumonia_model.onnx")

ort_session = None
try:
    if not os.path.exists(MODEL_PATH):
        logger.error(f"ONNX Model file not found at {MODEL_PATH}")
        raise FileNotFoundError(f"ONNX Model file not found: {MODEL_PATH}")
    
    # Load the ONNX model using CPU execution provider
    ort_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    logger.info("ONNX Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading ONNX model: {str(e)}")
    ort_session = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(filepath):
    """Load and preprocess image using numpy and PIL (no PyTorch dependencies)"""
    # 1. Load image and convert to RGB
    image = Image.open(filepath).convert("RGB")
    # 2. Resize to 224x224 using Bilinear interpolation matching torchvision transforms.Resize
    image = image.resize((224, 224), Image.BILINEAR)
    # 3. Convert to np array and scale to [0, 1]
    img_data = np.array(image).astype(np.float32) / 255.0
    # 4. Transpose from HWC to CHW
    img_data = np.transpose(img_data, (2, 0, 1))
    # 5. Normalize using ImageNet mean & std values
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img_data = (img_data - mean) / std
    # 6. Add batch dimension [1, 3, 224, 224]
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html", prediction=None)

@app.route("/health")
def health():
    """Health check endpoint for deployment monitoring"""
    return jsonify({"status": "ok", "model_loaded": ort_session is not None}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """Process image and make prediction using ONNX Runtime"""
    try:
        if ort_session is None:
            logger.error("ONNX model session not loaded")
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

        # Load and preprocess image using pure NumPy and Pillow
        img_np = preprocess_image(filepath)

        # Make prediction using ONNX runtime
        inputs = {ort_session.get_inputs()[0].name: img_np}
        outputs = ort_session.run(None, inputs)
        logits = outputs[0]

        # Calculate softmax probabilities and prediction in NumPy
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        pred_idx = np.argmax(probabilities[0])
        confidence = probabilities[0][pred_idx] * 100
        
        classes = ["NORMAL", "PNEUMONIA"]
        result = classes[pred_idx]

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
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
