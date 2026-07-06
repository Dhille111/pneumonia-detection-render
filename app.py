import os
import logging
import numpy as np
import onnxruntime as ort
import uuid
import shutil
from datetime import datetime
from flask import Flask, render_template, request, redirect, jsonify, send_file
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

        # Check if running a preset sample
        sample_name = request.form.get("sample_name")
        if sample_name in ["normal", "pneumonia"]:
            sample_file = os.path.join(BASE_DIR, "static", "samples", f"{sample_name}.jpeg")
            if not os.path.exists(sample_file):
                logger.error(f"Sample file not found: {sample_file}")
                return render_template(
                    "index.html",
                    prediction="ERROR",
                    error=f"Sample file '{sample_name}.jpeg' not found on server."
                ), 404
            
            filename = f"sample_{sample_name}_{uuid.uuid4().hex[:8]}.jpeg"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            shutil.copy(sample_file, filepath)
            logger.info(f"Loaded sample image: {sample_name} -> {filename}")
        else:
            # Handle regular file upload
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

@app.route("/generate_report", methods=["POST"])
def generate_report():
    """Generates a downloadable or previewable clinical PDF report for the patient case"""
    try:
        patient_name = request.form.get("patient_name", "Anonymous Patient").strip()
        patient_age = request.form.get("patient_age", "N/A").strip()
        notes = request.form.get("notes", "No clinical findings reported.").strip()
        prediction = request.form.get("prediction", "UNKNOWN")
        confidence = request.form.get("confidence", "0.00%")
        img_path = request.form.get("img_path", "")
        preview_mode = request.form.get("preview") == "true"

        # Format patient info safely
        if not patient_name:
            patient_name = "Anonymous Patient"
            
        # Create PDF using fpdf2
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        
        # Corporate Navy Banner
        pdf.set_fill_color(24, 43, 73)
        pdf.rect(0, 0, 210, 30, 'F')
        
        pdf.set_xy(10, 8)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 15)
        pdf.cell(0, 8, "CLINICAL DIAGNOSTIC SCREENING REPORT", align="C")
        pdf.ln(8)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 5, "Decision Support System for Pneumonia Detection", align="C")
        
        # Demographic Table
        pdf.set_xy(10, 38)
        pdf.set_text_color(44, 62, 80)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "1. Patient Demographics & Identification Details")
        pdf.ln(8)
        
        pdf.set_fill_color(240, 244, 248)
        pdf.set_text_color(60, 60, 60)
        pdf.set_font("Helvetica", "", 9.5)
        
        # Demographics Table cells
        pdf.cell(40, 7, "Patient Name:", border=1, fill=True)
        pdf.cell(55, 7, patient_name, border=1)
        pdf.cell(40, 7, "Patient Age:", border=1, fill=True)
        pdf.cell(55, 7, patient_age, border=1)
        pdf.ln()
        
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        case_id = f"PX-{uuid.uuid4().hex[:8].upper()}"
        
        pdf.cell(40, 7, "Case ID:", border=1, fill=True)
        pdf.cell(55, 7, case_id, border=1)
        pdf.cell(40, 7, "Report Date/Time:", border=1, fill=True)
        pdf.cell(55, 7, current_date, border=1)
        pdf.ln(12)
        
        # Diagnostic Outcome Section
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(24, 43, 73)
        pdf.cell(0, 8, "2. Diagnostic Screening Analysis Result")
        pdf.ln(8)
        
        is_pneumonia = (prediction == "PNEUMONIA")
        bg_color = (253, 242, 242) if is_pneumonia else (240, 253, 244)
        border_color = (252, 165, 165) if is_pneumonia else (187, 247, 208)
        text_color = (153, 27, 27) if is_pneumonia else (22, 101, 52)
        
        pdf.set_fill_color(*bg_color)
        pdf.set_draw_color(*border_color)
        pdf.set_line_width(0.4)
        
        pdf.cell(0, 16, "", border=1, fill=True)
        current_y = pdf.get_y()
        pdf.set_xy(15, current_y - 14)
        
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(*text_color)
        pdf.cell(0, 6, f"SCREENING OUTCOME: {prediction}")
        pdf.ln(5.5)
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.cell(0, 5, f"Neural Network Confidence: {confidence} (ResNet-18 Model Backend)")
        
        pdf.set_xy(10, current_y + 6)
        pdf.ln(5)
        
        # Ingested Image block
        if img_path:
            full_img_path = os.path.join(BASE_DIR, img_path)
            if os.path.exists(full_img_path):
                pdf.set_font("Helvetica", "B", 11)
                pdf.set_text_color(24, 43, 73)
                pdf.cell(0, 8, "3. Ingested Chest Radiograph (X-Ray)")
                pdf.ln(8)
                
                # Draw a soft border around image
                pdf.image(full_img_path, x=45, y=pdf.get_y(), w=120)
                pdf.set_y(pdf.get_y() + 100) # push past image space
                pdf.ln(5)
            
        # Clinical Notes Section
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(24, 43, 73)
        pdf.cell(0, 8, "4. Clinician Observations & Notes")
        pdf.ln(8)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(60, 60, 60)
        pdf.multi_cell(0, 5, notes)
        pdf.ln(10)
        
        # Signatures
        pdf.set_line_width(0.1)
        pdf.set_draw_color(180, 180, 180)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(4)
        
        pdf.set_font("Helvetica", "I", 7.5)
        pdf.set_text_color(128, 128, 128)
        disclaimer = (
            "Medical Disclaimer: This report is generated dynamically by an automated AI diagnostic support system. "
            "It is not a final certified diagnosis. The results should be evaluated in context with other clinical findings "
            "and laboratory parameters, and must be reviewed and signed off by a certified medical radiologist."
        )
        pdf.multi_cell(0, 4, disclaimer)
        
        # Save temp file
        import tempfile
        temp_dir = tempfile.gettempdir()
        report_filename = f"Clinical_Report_{case_id}.pdf"
        report_filepath = os.path.join(temp_dir, report_filename)
        pdf.output(report_filepath)
        
        return send_file(
            report_filepath,
            as_attachment=(not preview_mode),
            download_name=report_filename,
            mimetype="application/pdf"
        )
    except Exception as e:
        logger.error(f"Error compiling PDF report: {str(e)}", exc_info=True)
        return "Internal server error compilation failed", 500

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
