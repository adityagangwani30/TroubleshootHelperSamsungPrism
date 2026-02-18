# 7-Segment Appliance OCR Pipeline

![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![EasyOCR](https://img.shields.io/badge/EasyOCR-Deep%20Learning-orange.svg)


> **A robust, production-grade Computer Vision pipeline designed to read error codes (e.g., "5Ud", "dC", "UE") from appliance LED displays.**

This system employs a hybrid approach, combining **Traditional Computer Vision (OpenCV)** for precise localization and filtering and **Deep Learning (EasyOCR)** for robust text extraction.

---

## ✨ Key Features

### � Intelligent Auto-Crop
- **Dynamic Isolation:** Uses HSV color masking and density clustering to distinguish glowing LED segments from the control panel background.
- **Resolution Independent:** Adapts to any image size using relative height filtering (rejecting noise < 50% of max digit height) and horizontal grouping loops.

### 🧠 Adaptive Morphological Processing
- **Safe-Limit Directional Closing:** specialized algorithms bridge the gaps in 7-segment displays without merging adjacent digits.
- **Bridge Snapper:** A "micro-weld" post-processing step that intentionally erodes hairline connections between characters (e.g., "U" touching "d").

### 🤖 Recognition Engine
- **Primary Engine:** Local, fast neural network inference using **EasyOCR**.
- **Performance:** Optimized for 7-segment display fonts and low-contrast conditions.

### ✅ Database-Driven Validation
- **Smart Validation:** Extracted tax is validated against a structured JSON dataset (`error_code_mapping.json`).
- **Fuzzy Matching:** Implements length-constrained fuzzy logic to correct common OCR errors (e.g., reading "5" as "S" or "O" as "0") while rejecting false positives.

### 📺 Real-Time Debug Mode
- **Visual Pipeline:** See exactly what the computer sees. Terminal windows visualize every stage: *Original -> Crop -> Threshold -> Morph -> Final*.

---

## 📂 Directory Structure

```text
.
├── config/                 # Global configuration settings
│   └── assets/             # Fonts, icons, and model storage
├── data/
│   ├── real/               # Real world test images
│   └── washing_machine/    # Error code database (JSON)
├── ocr_pipeline/           # Core Logic Modules
│   ├── preprocessing.py    # Auto-crop, HSV masking, Morphology
│   ├── ocr_engine.py       # EasyOCR wrapper & config
│   ├── postprocess.py      # Validation & Fuzzy matching
│   └── pipeline.py         # Main orchestration & Debug UI
├── scripts/                # Executable Scripts
│   └── run_ocr.py          # Primary entry point
├── utils/                  # Helper utilities (I/O)
├── .env                    # Environment variables (API Keys)
├── requirements.txt        # Python dependencies
└── README.md               # Project Documentation
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/appliance-ocr.git
cd appliance-ocr
```

### 2. Install Dependencies
Ensure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```



---

## 💻 Usage Guide

Run the OCR pipeline on any image using the `run_ocr.py` script:

```bash
python scripts/run_ocr.py --image data/washing_machine/Washing_Machines/dC.jpg --debug
```

### Command Flags
- `--image`: Path to the input image file.
- `--debug`: Enable visual debug windows (Press any key to step through).

### Terminal Output
The system provides a detailed engineering report for every run:

```text
============================================================
       OCR PIPELINE — ENGINEERING REPORT
============================================================
  Image       : data/washing_machine/dC.jpg
------------------------------------------------------------
  📋 RESULTS
     Raw Text     : dC
     Clean Text   : dC
     Valid Code   : Yes
     Match Type   : ✅ DIRECT_MATCH
     Confidence   : 62.10%
------------------------------------------------------------
  🔧 ERROR DETAILS
     Name            : Unbalanced Load
     Description     : Unbalance or cabinet bump detected during final spin.
     Troubleshooting : Redistribute laundry evenly, reduce load size, retry spin cycle.
------------------------------------------------------------
```

---

## ⚙️ Technical Pipeline Breakdown

1.  **Image Input**: Loads image and applies a regex filter to valid validation scope.
2.  **Preprocessing (Auto-Crop)**:
    *   **HSV Masking**: Isolates LED colors.
    *   **Cluster localization**: Identifies dominant character groups based on density and aspect ratio.
    *   **Cropping**: Zooms in on the display area with 20% padding.
3.  **Morphological Welding**:
    *   **Vertical Snap**: Connects top/bottom segments of characters.
    *   **Horizontal Snap**: Connects side segments (corners) but stops short of merging digits.
    *   **Bridge Snapper**: Erodes weak inter-digit connections.
4.  **OCR Inference**: Feeds the "welded" binary image into EasyOCR for text extraction.
5.  **Post-Processing**:
    *   **Validation**: Checks if text matches known codes in `error_code_mapping.json`.
    *   **Fuzzy Logic**: Attempts to correct minor misreadings if direct match fails.

---

## 🔮 Future Improvements

- **Real-Time Video Support**: Optimize pipeline for >30 FPS inference on edge devices.
- **Edge Deployment**: Quantize models for Raspberry Pi / Android deployment.
- **Multi-Appliance Support**: Expand database to include Refrigerators and Ovens.
