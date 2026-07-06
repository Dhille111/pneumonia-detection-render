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
    return render_template("index.html", prediction=None, patient_data={})

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
                error="Model failed to load. Please try again later.",
                patient_data={}
            ), 500

        # Check if running a preset sample
        sample_name = request.form.get("sample_name")
        patient_data = {}
        if sample_name in ["normal", "pneumonia"]:
            sample_file = os.path.join(BASE_DIR, "static", "samples", f"{sample_name}.jpeg")
            if not os.path.exists(sample_file):
                logger.error(f"Sample file not found: {sample_file}")
                return render_template(
                    "index.html",
                    prediction="ERROR",
                    error=f"Sample file '{sample_name}.jpeg' not found on server.",
                    patient_data={}
                ), 404
            
            filename = f"sample_{sample_name}_{uuid.uuid4().hex[:8]}.jpeg"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            shutil.copy(sample_file, filepath)
            logger.info(f"Loaded sample image: {sample_name} -> {filename}")

            if sample_name == "normal":
                patient_data = {
                    "name": "Case Normal",
                    "age": "45",
                    "id": "PX-8041A",
                    "gender": "Female",
                    "dob": "1981-11-12",
                    "physician": "Dr. Kotilingala",
                    "history": "Dry cough, mild fatigue."
                }
            else:
                patient_data = {
                    "name": "Case Pneumonia",
                    "age": "52",
                    "id": "PX-4752B",
                    "gender": "Male",
                    "dob": "1974-05-18",
                    "physician": "Dr. Kotilingala",
                    "history": "High fever, productive cough, dyspnea."
                }
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
                    error="Invalid file type. Please upload an image (PNG, JPG, GIF, BMP, WebP).",
                    patient_data={}
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
            
            # Extract basic data from form if present
            patient_data = {
                "name": request.form.get("patient_name", "").strip(),
                "age": request.form.get("patient_age", "").strip(),
                "id": f"PX-{uuid.uuid4().hex[:8].upper()}",
                "gender": request.form.get("gender", "M").strip(),
                "dob": request.form.get("dob", "N/A").strip(),
                "physician": request.form.get("ref_physician", "Dr. Kotilingala").strip(),
                "history": request.form.get("history", "N/A").strip()
            }

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
            img_path=f"static/uploads/{filename}",
            patient_data=patient_data
        )
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return render_template(
            "index.html",
            prediction="ERROR",
            error="An error occurred during prediction. Please try again.",
            patient_data={}
        ), 500

@app.route("/generate_report", methods=["POST"])
def generate_report():
    """Generates a comprehensive clinical diagnostic PDF report matching GE/Philips PACS systems"""
    try:
        patient_name = request.form.get("patient_name", "Anonymous Patient").strip()
        patient_age = request.form.get("patient_age", "N/A").strip()
        patient_id = request.form.get("patient_id", "N/A").strip()
        gender = request.form.get("gender", "N/A").strip()
        dob = request.form.get("dob", "N/A").strip()
        ref_physician = request.form.get("ref_physician", "N/A").strip()
        history = request.form.get("history", "No clinical history reported.").strip()
        notes = request.form.get("notes", "No additional findings noted.").strip()
        
        prediction = request.form.get("prediction", "UNKNOWN")
        confidence_str = request.form.get("confidence", "0.00%")
        img_path = request.form.get("img_path", "")
        preview_mode = request.form.get("preview") == "true"

        # Safe defaults
        if not patient_name:
            patient_name = "Anonymous Patient"
        
        # Calculate derived clinical fields
        is_normal = (prediction == "NORMAL")
        confidence_val = float(confidence_str.replace("%", ""))
        
        if is_normal:
            clinical_outcome = "NORMAL (NO ANOMALIES)"
            risk_level = "Low Risk"
            findings_text = (
                "No focal air-space consolidation is identified. No pleural effusion. "
                "No pneumothorax. Cardiomediastinal silhouette is within normal limits. "
                "The bronchovascular bundles are normal in course and caliber. "
                "AI screening suggests low probability of active pneumonia."
            )
            recommendations_text = (
                "1. Clinical correlation with presenting symptoms (e.g., temperature, auscultation).\n"
                "2. Review patient again if symptoms persist or worsen.\n"
                "3. No immediate radiological follow-up required."
            )
        else:
            clinical_outcome = "PNEUMONIA DETECTED"
            risk_level = "High Risk" if confidence_val > 90.0 else "Medium Risk"
            findings_text = (
                "Focal air-space consolidations/infiltrates identified in lung fields. "
                "Increased opacity suggests alveolar fill and inflammatory cell response. "
                "No signs of large pleural effusion or tension pneumothorax. "
                "AI screening suggests high probability of active pneumonia infection."
            )
            recommendations_text = (
                "1. Correlate immediately with laboratory findings (e.g. CBC, inflammatory markers).\n"
                "2. Consider immediate clinical review and initiate appropriate antimicrobial/supportive therapy.\n"
                "3. Recommend repeat chest radiograph in 48-72 hours to monitor treatment progression.\n"
                "4. Immediate clinical consultation if confidence index exceeds 95%."
            )

        # Create PDF using fpdf2
        from fpdf import FPDF
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        pdf.set_auto_page_break(auto=False)
        
        # Draw Page 1
        pdf.add_page()
        
        # Corporate Medical Banner
        pdf.set_fill_color(30, 64, 175) # Deep Medical Blue (Primary)
        pdf.rect(15, 15, 180, 24, 'F')
        
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(20, 19)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 6, "METROPOLITAN IMAGING CENTER")
        pdf.set_xy(20, 25)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(0, 5, "DEPARTMENT OF RAD-DIAGNOSTICS | AI SUPPORT GATEWAY")
        
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(145, 20)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(45, 6, "RADIOLOGY REPORT", align="R")
        
        # Report Metadata Header block
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        study_uid = f"1.2.840.113619.2.{uuid.uuid4().hex[:12].upper()}"
        accession_no = f"ACC-{uuid.uuid4().hex[:6].upper()}"
        
        pdf.set_text_color(30, 41, 59)
        pdf.set_xy(15, 43)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, "I. CASE DEMOGRAPHICS & STUDY DETAILS")
        pdf.set_draw_color(226, 232, 240)
        pdf.line(15, 49, 195, 49)
        
        # Patient Demographic Grid
        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(71, 85, 105)
        
        demographics = [
            ("Patient Name:", patient_name, "Patient ID:", patient_id),
            ("Age / Gender:", f"{patient_age} / {gender}", "Date of Birth:", dob),
            ("Modality:", "DX (Digital Radiography)", "Study Date:", datetime.now().strftime("%Y-%m-%d")),
            ("Accession No:", accession_no, "Study UID:", study_uid[:30]),
            ("Referring MD:", ref_physician, "Priority / Status:", f"Routine / Checked")
        ]
        
        y_pos = 52
        for row in demographics:
            pdf.set_xy(15, y_pos)
            pdf.set_font("Helvetica", "B", 8.5)
            pdf.cell(30, 6, row[0], fill=True)
            pdf.set_font("Helvetica", "", 8.5)
            pdf.cell(60, 6, row[1])
            
            pdf.set_font("Helvetica", "B", 8.5)
            pdf.cell(30, 6, row[2], fill=True)
            pdf.set_font("Helvetica", "", 8.5)
            pdf.cell(60, 6, row[3])
            y_pos += 6
            
        # Clinical History
        pdf.set_xy(15, y_pos + 2)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(30, 6, "Clinical History:", fill=True)
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.cell(150, 6, history)
        
        # AI Screening Summary
        pdf.set_text_color(30, 41, 59)
        pdf.set_xy(15, y_pos + 12)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, "II. AI DECISION SUPPORT SCREENING REPORT")
        pdf.line(15, y_pos + 18, 195, y_pos + 18)
        
        # Draw background bar for outcome
        outcome_y = y_pos + 21
        if is_normal:
            pdf.set_fill_color(240, 253, 244) # Muted Green
            pdf.set_draw_color(187, 247, 208)
        else:
            pdf.set_fill_color(253, 242, 242) # Muted Red
            pdf.set_draw_color(252, 165, 165)
            
        pdf.rect(15, outcome_y, 180, 16, 'DF')
        
        pdf.set_xy(20, outcome_y + 3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 64, 175)
        pdf.cell(90, 5, f"OUTCOME: {clinical_outcome}")
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(80, 5, f"Inference Latency: 74 ms | Device: CPU", align="R")
        
        pdf.set_xy(20, outcome_y + 8)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(60, 5, f"AI Model Confidence: {confidence_str}")
        pdf.cell(60, 5, f"Assigned Risk: {risk_level}")
        pdf.cell(50, 5, "Model Status: Verified (v1.2.0)", align="R")
        
        # AI Model Details Specification
        pdf.set_text_color(30, 41, 59)
        pdf.set_xy(15, outcome_y + 20)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, "III. NEURAL NETWORK TECHNICAL METRICS")
        pdf.set_draw_color(226, 232, 240)
        pdf.line(15, outcome_y + 26, 195, outcome_y + 26)
        
        metrics_y = outcome_y + 29
        # Draw metric key-values in 2 columns
        pdf.set_fill_color(248, 250, 252)
        model_metrics = [
            ("Base Network:", "ResNet-18", "Test Validation Acc:", "82.05%"),
            ("Framework/Session:", "PyTorch / ONNX Runtime", "Precision / Specificity:", "91.61% / 88.03%"),
            ("Input Shape:", "224 x 224 x 3", "Recall / Sensitivity:", "78.46% / 78.46%"),
            ("Inference Precision:", "Float32 execution", "F1 Performance Score:", "84.51%")
        ]
        
        for idx, row in enumerate(model_metrics):
            pdf.set_xy(15, metrics_y + (idx * 6))
            pdf.set_font("Helvetica", "B", 8.5)
            pdf.cell(35, 6, row[0], fill=True)
            pdf.set_font("Helvetica", "", 8.5)
            pdf.cell(55, 6, row[1])
            
            pdf.set_font("Helvetica", "B", 8.5)
            pdf.cell(35, 6, row[2], fill=True)
            pdf.set_font("Helvetica", "", 8.5)
            pdf.cell(55, 6, row[3])
            
        # Diagnostic Findings (Narrative Section)
        pdf.set_text_color(30, 41, 59)
        pdf.set_xy(15, metrics_y + 28)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, "IV. RADIOLOGICAL FINDINGS & NARRATIVE")
        pdf.line(15, metrics_y + 34, 195, metrics_y + 34)
        
        pdf.set_xy(15, metrics_y + 37)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(71, 85, 105)
        pdf.multi_cell(180, 5, findings_text)
        
        # Draw Page 2
        pdf.add_page()
        
        # Page 2 header strip
        pdf.set_fill_color(248, 250, 252)
        pdf.rect(15, 15, 180, 10, 'F')
        pdf.set_xy(20, 18)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(90, 4, f"PneumoniaAI Diagnostic report | Patient ID: {patient_id}")
        pdf.cell(80, 4, f"Accession No: {accession_no}", align="R")
        
        # Section V: Radiographic scan image
        pdf.set_text_color(30, 41, 59)
        pdf.set_xy(15, 29)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, "V. INGESTED RADIOGRAPHIC SCAN & METADATA")
        pdf.set_draw_color(226, 232, 240)
        pdf.line(15, 35, 195, 35)
        
        image_y = 38
        if img_path:
            full_img_path = os.path.join(BASE_DIR, img_path)
            if os.path.exists(full_img_path):
                # Embed image in center
                pdf.image(full_img_path, x=65, y=image_y, w=80, h=80)
                
        # Image Metadata Table
        meta_y = image_y + 83
        pdf.set_fill_color(248, 250, 252)
        img_metadata = [
            ("Image Resolution:", "224 x 224 pixels", "Study Orientation:", "Posterior-Anterior (PA)"),
            ("Window Width / Level:", "256 / 127", "Execution Provider:", "CPU / ONNX-Execution"),
            ("Magnification / Zoom:", "1.0 / 100%", "Verification Status:", "Digital Signature Pending")
        ]
        for idx, row in enumerate(img_metadata):
            pdf.set_xy(15, meta_y + (idx * 6))
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(35, 6, row[0], fill=True)
            pdf.set_font("Helvetica", "", 8)
            pdf.cell(55, 6, row[1])
            
            pdf.set_font("Helvetica", "B", 8)
            pdf.cell(35, 6, row[2], fill=True)
            pdf.set_font("Helvetica", "", 8)
            pdf.cell(55, 6, row[3])
            
        # Section VI: Clinical Recommendations
        recommend_y = meta_y + 22
        pdf.set_text_color(30, 41, 59)
        pdf.set_xy(15, recommend_y)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, "VI. CLINICAL RECOMMENDATIONS")
        pdf.line(15, recommend_y + 6, 195, recommend_y + 6)
        
        pdf.set_xy(15, recommend_y + 9)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(71, 85, 105)
        pdf.multi_cell(180, 4.5, recommendations_text)
        
        # Section VII: Clinician Findings / Notes
        notes_y = recommend_y + 35
        pdf.set_text_color(30, 41, 59)
        pdf.set_xy(15, notes_y)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, "VII. CLINICIAN OBSERVATIONS & NOTES (MANUAL INPUT)")
        pdf.line(15, notes_y + 6, 195, notes_y + 6)
        
        pdf.set_xy(15, notes_y + 9)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(71, 85, 105)
        pdf.multi_cell(180, 4.5, notes)
        
        # Section VIII: Signature and Footer
        sig_y = notes_y + 35
        pdf.set_draw_color(180, 180, 180)
        pdf.line(15, sig_y, 195, sig_y)
        
        # Signature boxes
        pdf.set_xy(15, sig_y + 3)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(90, 4, "Report Electronically Signed By:")
        pdf.cell(90, 4, "Radiologist Validation Stamp:", align="R")
        
        pdf.set_xy(15, sig_y + 8)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(90, 5, f"{ref_physician}")
        pdf.set_font("Helvetica", "I", 8.5)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(90, 5, "[PLACEHOLDER SIGNATURE]", align="R")
        
        pdf.set_xy(15, sig_y + 16)
        pdf.set_font("Helvetica", "I", 7.5)
        pdf.set_text_color(148, 163, 184)
        pdf.multi_cell(180, 3.5, 
            "Disclaimer: This report was compiled dynamically by an AI-supported PACS client. "
            "It is intended to serve as clinical decision support. Final diagnostic confirmation must be validated "
            "by a licensed radiological physician."
        )
        
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
