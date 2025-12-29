---

## 🧠 Project Overview

This repository implements a complete **end-to-end medical imaging pipeline** for brain tumor analysis using the BraTS2020 MRI dataset.  
The project integrates multiple core skills expected in biomedical AI roles:

- Multimodal **3D MRI preprocessing** (T1, T1CE, T2, FLAIR)
- Tumor **segmentation using a 3D U-Net** (PyTorch + MONAI)
- **Radiomic feature extraction** (tumor subregion volumes & ratios)
- Integration of **clinical metadata** (age, resection status)
- **1-year survival risk prediction** using machine learning

This mirrors the real workflows used in clinical AI labs (e.g., UTSW, MD Anderson).

---

## 🎯 Problem Statement

Gliomas are heterogeneous and aggressive brain tumors.  
Clinicians routinely use MRI for diagnosis, but:

1. Tumor boundaries are difficult to delineate manually  
2. Tumor shape and volume correlate with **patient prognosis**  
3. Combining imaging + clinical variables improves risk assessment  

This project builds a system that:

> **Segments brain tumors from MRI and predicts whether a patient is high-risk (<1-year survival).**

---

## 🧩 Dataset

- **BraTS2020** (Brain Tumor Segmentation Challenge)
- Modalities: T1, T1CE, T2, FLAIR
- Labels include:
  - 0 = background  
  - 1 = necrotic / non-enhancing core  
  - 2 = edema  
  - 4 = enhancing tumor (mapped to 3)  

Clinical metadata includes:

- Age  
- Survival days  
- Extent of Resection (GTR, STR, NA)

---

## 🧱 Pipeline Summary

### **1️⃣ Preprocessing & Splits**
- Loaded NIfTI volumes with `nibabel`
- Used **MONAI transforms** for orientation (RAS), normalization, and patch sampling
- Automatic train/validation CSV split generation
- Visual inspection utilities for quality control

---

### **2️⃣ 3D Tumor Segmentation**
Model:  
- **3D U-Net (4-channel input)**  
- Loss: **Dice + Cross Entropy**  
- Training via patch-based sampling  
- Sliding-window inference for full volumes  

Results after 2 CPU epochs (sanity-check training):  
- **Validation Dice ≈ 0.38**  
(The score increases substantially with GPU training.)

Outputs:  
- `.nii.gz` predicted segmentation  
- Side-by-side PNG visualization  

---

### **3️⃣ Radiomics Feature Extraction**

From each segmentation mask we computed:

- Voxel & volume of each tumor subregion  
- Whole tumor volume  
- Ratios:
  - core / whole  
  - edema / whole  

Stored in:  
`data/processed/brats_radiomics_gt.csv`

These features form the basis of the outcome model.

---

### **4️⃣ Outcome Prediction Model**

Merged radiomics + clinical metadata:  
`data/processed/brats_outcome_dataset.csv`

Target:  
**High-Risk (1-year survival)**  
```text
1 if Survival_days < 365
0 otherwise
⚠️ Survival_days was NOT used as a model feature (to avoid label leakage).

### **Model:**

RandomForestClassifier

One-hot encoding of clinical categories

Stratified train/test split

### **✔ Evaluation (held-out test set)**
Metric	            Score
Accuracy	          0.611
ROC AUC	            0.741
High-Risk Recall	  0.765

Interpretation:
The model captures meaningful morphology-prognosis relationships, performing comparably to early radiomics literature.

### **📦 Tech Stack**

Python

PyTorch, MONAI, scikit-learn

Nibabel for NIfTI IO

Pandas / NumPy

Joblib for model serialization

### **🚀 Future Work**

Train segmentation model longer on GPU for higher Dice

Use predicted masks (not GT) for outcome model

Add feature importance + SHAP explanations

Train survival models (Cox, DeepSurv)

Evaluate calibration & confidence intervals

### **👨‍⚕️ Clinical Relevance**

This pipeline demonstrates how MRI tumor morphology, combined with minimal clinical variables, can contribute to risk stratification—a key step toward personalized treatment planning.