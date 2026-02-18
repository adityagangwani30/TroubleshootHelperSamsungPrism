# Appliance Troubleshooting OCR System

This repository contains the skeleton for an OCR-based system designed to read error codes from appliance displays (specifically washing machines).

## 📂 Project Structure

```
├── config/                 # Global configuration settings
│   ├── assets/             # Fonts, templates for generation
│   └── settings.py         # Main settings file
├── data/
│   ├── real/               # Real captured images for testing
│   └── synthetic/          # Generated images for training/verification
├── ocr_pipeline/           # Core OCR logic
│   ├── preprocessing.py    # Image cleanup (grayscale, threshold)
│   ├── ocr_engine.py       # Tesseract/OCR wrapper
│   ├── postprocess.py      # Text validation and cleanup
│   └── pipeline.py         # Main pipeline orchestration
├── synthetic_generator/    # Synthetic data generation tools
│   └── generator.py        # Logic to create fake error code images
├── scripts/                # Executable scripts
│   ├── generate_synthetic.py
│   └── run_ocr.py
├── utils/                  # Helper functions
│   └── io_utils.py         
└── docs/                   # Documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- (Later) Tesseract OCR, OpenCV, Pillow

### Configuration
All settings are managed in `config/settings.py`. check this file to adjust paths, image resolution, and generation parameters.

### Running Scripts (Boilerplate)

**Generate Synthetic Data:**
```bash
python scripts/generate_synthetic.py
```

**Run OCR Pipeline:**
```bash
python scripts/run_ocr.py --image path/to/image.png
```

## 🛠 modules

- **OCR Pipeline**: Handles the end-to-end process of reading an image.
- **Synthetic Generator**: Creates training data to robustify the OCR against various display types and lighting conditions.

## 📝 Status
🚧 **Skeleton Phase**: This project currently contains the architectural skeleton and boilerplate code. No real logic has been implemented yet.
