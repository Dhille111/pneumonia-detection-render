# 🫁 Pneumonia Detection System 

![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/DL_Framework-PyTorch-red)
![Flask](https://img.shields.io/badge/Backend-Flask-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

> **A high-precision Deep Learning solution utilizing Transfer Learning (ResNet-18) to detect Pneumonia from chest X-Ray radiographs with 94.5% validation accuracy.**

---

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Model Performance](#-model-performance)
- [Directory Structure](#-directory-structure)
- [Disclaimer](#-medical-disclaimer)
- [Contact](#-contact)

---

## 🔍 Overview
Pneumonia is a life-threatening infectious disease affecting the lungs. Early diagnosis is critical for successful treatment. This project automates the detection process using **Convolutional Neural Networks (CNNs)**.

By leveraging **Transfer Learning** with a pre-trained **ResNet-18** architecture, this system analyzes chest X-ray images and classifies them into two categories:
1.  **NORMAL** (Healthy)
2.  **PNEUMONIA** (Infected)

The model is deployed via a lightweight **Flask** web application, allowing medical professionals or users to upload images and receive instant predictions.

---

## ✨ Key Features
* **High Accuracy:** Fine-tuned ResNet-18 model achieving **94.5% accuracy** on the validation set.
* **Real-Time Inference:** Instant classification results via the web interface.
* **User-Friendly UI:** Simple drag-and-drop interface built with Flask and HTML/CSS.
* **Robust Backend:** Handles image preprocessing, normalization, and tensor conversion automatically.
* **Scalable:** Designed to be easily containerized (Docker) or deployed to cloud platforms (AWS/Heroku).

---

## 🏗 System Architecture
The application follows a standard Model-View-Controller (MVC) pattern adapted for AI deployment:

1.  **Input Layer:** User uploads an image via the Web UI.
2.  **Preprocessing:** Image is resized to `224x224`, converted to RGB, and normalized using ImageNet stats.
3.  **Inference Engine:** The PyTorch model (`pneumonia_model.pth`) processes the tensor.
4.  **Output Layer:** The system returns the predicted class and confidence score.

---

## 💻 Tech Stack
| Component | Technology |
| :--- | :--- |
| **Deep Learning** | PyTorch, Torchvision, ResNet-18 (Pre-trained) |
| **Backend API** | Flask (Python) |
| **Image Processing** | PIL (Pillow), NumPy |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Data Handling** | Pandas |

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.8 or higher
* Git

### 1. Clone the Repository
```bash
git clone [https://github.com/Dhille111/Pneumonia_detection_from_chest_xrays](https://github.com/Dhille111/Pneumonia_detection_from_chest_xrays)
cd Pneumonia_detection_from_chest_xrays
### 2. Create a Virtual Environment (Recommended)
Bash

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
### 3. Install Dependencies
Bash

pip install --upgrade pip
pip install -r requirements.txt
### 4. GPU Setup (Optional)
If you have an NVIDIA GPU, install the specific PyTorch version for your CUDA driver:

Bash

# Example for CUDA 11.8
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
🚀 Usage Guide
Running the Web Application
Ensure the model weights file pneumonia_model.pth is in the root directory.

Start the Flask server:

Bash

python app.py
Open your web browser and navigate to:

[http://127.0.0.1:5000/](http://127.0.0.1:5000/)
Training the Model (Developers Only)
To retrain the model from scratch using your own dataset:

Organize your data into Dataset/train and Dataset/val.

Run the training script:

Bash

python main.py
📊 Model Performance
Metric	Score
Validation Accuracy	94.5%
Loss Function	CrossEntropyLoss
Optimizer	Adam (Learning Rate: 0.001)


Screenshots
"C:\Users\user\Pictures\WhatsApp Image 2025-12-18 at 10.43.55 PM.jpeg"
"C:\Users\user\Pictures\Screenshots\Screenshot 2025-12-22 224437.png"

📂 Directory Structure
Pneumonia_Detection/
├── Dataset/                 # Training and Validation images
│   ├── train/
│   └── val/
├── static/
│   ├── css/                 # Stylesheets
│   └── uploads/             # Temp folder for uploaded images
├── templates/
│   └── index.html           # Web Interface
├── app.py                   # Flask Application Entry Point
├── main.py                  # Model Training Script
├── pneumonia_model.pth      # Trained Model Weights
├── requirements.txt         # Project Dependencies
└── README.md                # Project Documentation
⚠️ Medical Disclaimer
This software is for educational and research purposes only. It is NOT intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a physician or qualified health provider with any questions regarding a medical condition.

📜 License
Distributed under the MIT License. See LICENSE for more information.

## 🤝 Contact
Here are my active profiles:

* **LinkedIn:** [Visit My Profile](https://www.linkedin.com/in/kotilingala-dhillerao-519707349/)
* **GitHub:** [Visit My GitHub](https://github.com/Dhille111)
* **Email:** dhilleraokotilingala@gmail.com
