# 🩺 Clinician Operator Manual & Guide

## PneumoniaAI Web-PACS Decision Support Module
*For Clinical Operators, Radiologists, and General Practitioners*

---

## 1. System Overview
PneumoniaAI is an automated Clinical Decision Support System (CDSS) designed to assist medical practitioners in screening chest X-ray radiographs (DX modality) for indications of pneumonia. Using a fine-tuned **ResNet-18** model optimized via **ONNX Runtime**, the gateway provides binary classification recommendations with confidence indices in under 100ms.

> **⚠️ Regulatory Disclaimer:** This system is a decision support tool and does NOT replace diagnostic review by a certified radiologist. All compiled cases must be signed off by a licensed radiological physician before initiating treatment protocols.

---

## 2. Workspace Interface & Layout

The interface is structured to mirror hospital Picture Archiving and Communication Systems (PACS):

```
+-------------------------------------------------------------------------+
|                                  HEADER                                 |
+------------+--------------------+---------------------+-----------------+
| Navigation | Patient Worklist   | Diagnostic Viewport | AI Report       |
| Sidebar    |                    |                     | & Document      |
|            | Ingest Study       | PACS Toolbar        |                 |
|            | Patient Cards      | Image Canvas        | Screening Outcome|
|            | Engine Specs       | DICOM HUD Text      | PDF Generator   |
+------------+--------------------+---------------------+-----------------+
```

### Key Modules:
1. **Sidebar Navigation**: Switch between the active *Study Worklist* (Viewport), *Dashboard* (KPI metrics), *AI Screening* (accuracy specs), and *Reports* (archived database logs).
2. **Patient Worklist**: Access active patient case files. Clicking any case reloads the viewport and pre-fills patient demographics.
3. **Diagnostic Viewport**: Render the grayscale radiograph scan surrounded by corner DICOM metadata (Patient ID, DOB, facility details, and acquisition tags).
4. **AI Report Panel**: View classification outcomes and pre-fill clinician notes to preview or export signed PDF laboratory reports.

---

## 3. Radiographic Viewport Manipulation Tools
To enhance visual inspection, the toolbar above the image canvas provides interactive filters:

### 1. Invert Contrast (`Invert` Button)
* **Clinical Utility**: Radiologists frequently invert the grayscale contrast of X-ray films. In negative contrast, high-density structures (like fluid, bone, or consolidations) show as dark shades instead of white, helping to trace soft-tissue borders and detect subtle infiltrative patches.
* **Operation**: Applies a dynamic `filter: invert(1)` style directly in the browser viewport.

### 2. Flip Horizontal (`Flip` Button)
* **Clinical Utility**: Used to inspect lung bilateral symmetry. Comparing the left and right lung lobes side-by-side helps identify unilateral anomalies.
* **Operation**: Horizontally mirrors the image using `scaleX(-1)`.

### 3. Rotate CW (`Rotate` Button)
* **Clinical Utility**: Corrects the orientation of ingested radiographs that were scanned or stored sideways.
* **Operation**: Rotates the image canvas 90 degrees sequentially.

---

## 4. Identifying Pneumonia Anomalies on Chest X-Rays
Clinicians should visually cross-reference AI outcomes with the following anatomical markers:

| Anomaly Type | Radiographic Presentation | Clinical Indication |
| :--- | :--- | :--- |
| **Lung Consolidation** | Fuzzy white patches or opaque densities within the black lung voids. | Alveoli filled with inflammatory fluid/pus instead of air (typical in bacterial pneumonia). |
| **Interstitial Infiltrates** | Diffuse, hazy, "ground-glass" or web-like markings spread bilaterally. | Fluid accumulation in the connective tissues (typical in viral pneumonia). |
| **Blunted Costophrenic Angles** | Cloudy or rounded bottom sharp corners of the lung fields. | Pleural effusion (excess fluid buildup in the pleural cavity). |

---

## 5. Model Validation & Performance Calibration

The saved ONNX model weights have been validated against standard chest X-ray databases:

* **Generalization Accuracy**: **82.05%** (512 out of 624 clinical test cases correctly classified).
* **Positive Predictive Value (Precision)**: **91.61%** (low rate of false positives on healthy patients).
* **Model Sensitivity (Recall)**: **78.46%** (rate of correctly identifying pneumonia cases).
* **Specificity**: **88.03%** (rate of correctly identifying normal cases).

---

## 6. Clinical Verification Protocol (Step-by-Step)

1. **Ingest scan**: Ingest the patient chest X-ray file via the upload zone or choose a preset case from the worklist.
2. **Review Demographics**: Verify that Patient ID, Age, Gender, and Clinical History are correctly prefilled in the reporting form.
3. **Inspect Image**: Apply **Invert Contrast** or **Zoom** to visual boundaries to inspect for focal consolidations.
4. **Compare AI Recommendation**: Review the neural network classification (`NORMAL` vs `PNEUMONIA DETECTED`) and confidence percentage.
5. **Document Observations**: Type clinician notes in the findings textarea.
6. **Generate Report**: 
   * Click **Preview Report** to open and review the compiled report in a new tab.
   * Click **Generate PDF Report** to download the signed PDF diagnostic sheet for the patient's record.
