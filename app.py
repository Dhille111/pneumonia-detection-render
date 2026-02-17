import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, render_template, request, redirect
from PIL import Image

# -------------------- App setup --------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------- Device (CPU only for Render) --------------------
device = torch.device("cpu")

# -------------------- Load model --------------------
MODEL_PATH = os.path.join(BASE_DIR, "pneumonia_model.pth")

model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.to(device)
model.eval()

# -------------------- Image transforms --------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect("/")

    file = request.files["file"]
    if file.filename == "":
        return redirect("/")

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    image = Image.open(filepath).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        classes = ["NORMAL", "PNEUMONIA"]
        result = classes[preds.item()]

    return render_template(
        "index.html",
        prediction=result,
        img_path=f"static/uploads/{file.filename}"
    )
