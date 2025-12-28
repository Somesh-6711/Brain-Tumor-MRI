# Brain Tumor MRI: Multimodal Segmentation & Outcome Prediction

End-to-end deep learning pipeline on the **BraTS** brain tumor MRI dataset:
1. 3D tumor **segmentation** using MONAI-based U-Net.
2. **Outcome / risk prediction** using radiomic features + clinical data.

This project is designed to mirror real-world work in medical AI and neuroimaging labs
(e.g., high-performance scientific computing, multimodal datasets, and reproducible pipelines).

---

## 1. Project Goals

- Build a **reproducible** PyTorch + MONAI pipeline for:
  - Loading and preprocessing multi-modal brain MRI (NIfTI).
  - Training and evaluating a 3D U-Net tumor segmentation model.
- Extract **radiomic + clinical features** from segmentation outputs.
- Train a downstream **risk prediction model** (e.g., 1-year progression or risk group).
- Keep everything **config-driven** via YAML files and simple shell scripts (ready for SLURM/HPC).

---

## 2. Tech Stack

- **Python**
- **PyTorch**, **MONAI**
- **NiBabel** for NIfTI I/O
- **scikit-learn**, **lifelines** for outcome modeling
- **matplotlib**, **seaborn** for visualization

---

## 3. Repository Structure

```text
.
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ template.py
├─ configs/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ splits/
├─ src/
│  ├─ datasets/
│  ├─ models/
│  ├─ training/
│  ├─ utils/
│  └─ inference/
├─ notebooks/
└─ scripts/
