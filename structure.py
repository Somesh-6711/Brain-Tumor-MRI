"""
structure.py
Creates the full folder structure for the Brain Tumor MRI Project.
Run this once after cloning the repo:
    python structure.py
"""

import os

# ---------------------------
# Directory Structure
# ---------------------------

DIRS = [
    "configs",
    "data/raw",
    "data/processed",
    "data/splits",
    "src/datasets",
    "src/models",
    "src/training",
    "src/utils",
    "src/inference",
    "notebooks",
    "scripts",
]

FILES = {
    "README.md": """# Brain Tumor MRI Project\nProject initialized. Full README will be filled later.""",

    ".gitignore": """# Python cache
__pycache__/
*.pyc
*.pyo

# Virtual env
.env/
.venv/
env/
venv/

# VSCode
.vscode/

# Jupyter
.ipynb_checkpoints/

# Data
data/raw/
data/processed/
data/splits/

# Model weights
*.pth
*.pt
*.ckpt

# Logs
logs/
""",

    "requirements.txt": """numpy
scipy
pandas
scikit-learn
matplotlib
seaborn
tqdm
pyyaml
torch
torchvision
monai
nibabel
lifelines
torchmetrics
""",

    "template.py": """# Main template file for Brain Tumor MRI project.
# This file contains skeleton code; we will fill it step-by-step.

print("Template placeholder – will be filled during development.")""",

    "src/__init__.py": "",
    "src/datasets/__init__.py": "",
    "src/models/__init__.py": "",
    "src/training/__init__.py": "",
    "src/utils/__init__.py": "",
    "src/inference/__init__.py": "",

    # Example empty placeholders
    "scripts/train_segmentation.sh": "#!/bin/bash\n# SLURM/Training script placeholder",
}


# ---------------------------
# Create directories & files
# ---------------------------

def create_dirs():
    for d in DIRS:
        os.makedirs(d, exist_ok=True)
        print(f"[DIR] Created: {d}")

def create_files():
    for filename, content in FILES.items():
        # Ensure directory exists
        dir_name = os.path.dirname(filename)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # Create file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[FILE] Created: {filename}")

def main():
    print("Generating project structure...")
    create_dirs()
    create_files()
    print("\n✔ Project structure created successfully!")

if __name__ == "__main__":
    main()
