# Samsung PRISM Appliance Error Code OCR Pipeline

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![YOLO](https://img.shields.io/badge/YOLO-ROI%20Detection-red)
![EasyOCR](https://img.shields.io/badge/EasyOCR-Text%20Engine-orange)

## Overview
This repository contains a Samsung PRISM-focused Appliance Error Code OCR system designed for difficult real-world conditions, including dim LEDs, heavy glare, and skewed camera angles.

The project is organized around production-style OCR modules (`src/`), strict evaluation tooling (`eval/`), and dataset/versioning controls for ML workflows.

## Architecture
### Two-Stage Detection + Recognition
The pipeline has transitioned to a robust two-stage architecture:

1. **YOLO Detector (Stage 1)**
   Detects and crops the display ROI from noisy appliance panel images.
2. **Custom CRNN Recognizer (Stage 2)**
   Performs sequence recognition on the cropped ROI to predict the full error code.
3. **Code Validation Layer**
   Validates recognized output against the source-of-truth mapping in `data/washing_machine/error_code_mapping.json`.

### Intelligent Vision Preprocessing
To rescue dim or distorted LED text before recognition, the preprocessing stack applies:

- **HSV Color-Space Masking** for LED-region isolation.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** for local contrast recovery.
- **Top-Hat Morphological Transforms** to emphasize bright segments on uneven backgrounds.
- **Bridge Snapper** to break thin accidental merges between adjacent characters.

### Strict Evaluation
Automated benchmarking is provided via:

- `eval/batch_evaluate.py`

The evaluator performs strict, label-driven validation against:

- `data/washing_machine/error_code_mapping.json`

## Directory Structure (Validated)
```text
.
|-- config/
|   |-- assets/
|   |   `-- models/
|   `-- settings.py
|-- data/                          # Core datasets
|   |-- raw/
|   |   `-- .gitkeep
|   |-- processed/
|   |   `-- .gitkeep
|   |-- washing_machine/
|   |   |-- Washing Machines/
|   |   |-- error_code_mapping.json
|   |   `-- error_codes.json
|   |-- microwave/
|   `-- refrigerators/
|-- src/                           # Core OCR pipeline modules
|   |-- __init__.py
|   |-- pipeline.py
|   |-- preprocessing.py
|   `-- postprocess.py
|-- eval/                          # Evaluation and benchmarking
|   |-- __init__.py
|   `-- batch_evaluate.py
|-- ocr_pipeline/
|   |-- __init__.py
|   |-- ocr_engine.py
|   `-- schemas.py
|-- scripts/
|   |-- run_ocr.py
|   `-- run_eval_capture.py
|-- synthetic_generator/
|-- TroubleShootHelperDataset/
|-- utils/
|-- requirements.txt
|-- .gitignore
`-- README.md
```

> Note: Per current repository policy, there is **no `configs/` folder**. Error-code JSON remains under `data/washing_machine/`.

## Installation
### Step 1: Clone
```bash
git clone https://github.com/adityagangwani30/TroubleshootHelperSamsungPrism.git
cd 25ST32MS_Troubleshoot_Helper_Revolutionizing_Device_Diagnostics
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Environment Configuration
Create a local `.env` file (never commit this file):

```dotenv
# .env
LOG_LEVEL=INFO
EASYOCR_MODEL_DIR=config/assets/models
```

## Usage Guide
### Run OCR on One Image
```bash
python scripts/run_ocr.py --image "data/washing_machine/Washing Machines/dC.jpg" --debug
```

### Run Strict Batch Evaluation
```bash
python eval/batch_evaluate.py --input "data/washing_machine/Washing Machines"
```

### Capture Evaluation Report
```bash
python scripts/run_eval_capture.py
```

## Evaluation Protocol
- Ground truth is parsed from file names.
- Predictions are normalized and matched with deterministic + fuzzy logic.
- Final pass/fail is measured against `data/washing_machine/error_code_mapping.json`.

## Roadmap
- Replace/extend OCR backends with optimized CRNN checkpoints.
- Add domain adaptation for new appliance families.
- Improve deployment packaging for edge-device inference.
