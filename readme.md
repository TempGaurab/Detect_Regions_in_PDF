# DETECT_REGIONS_IN_PDF

This project focuses on **document region detection** and **annotation**, specifically targeting PDF-to-image workflows using _OpenCV_ and _Donut_ models.

---

## 📂 Project Structure

```text
DETECT_REGIONS_IN_PDF/
├── annotations/                  # True Labels (JSON files)
├── clicking_mechanism/           # Logic for interaction
│   └── model_output/             # OPENCV output (JSON coordinates)
├── data/                         # Raw .png images (Input files)
├── opencv_donut outputs/         # Scripts for detection and saving
│   └── detect_and_save.py        # Core OpenCV detection script
├── poppler-25.07.0/              # PDF rendering dependencies
├── DONUT.ipynb                   # Donut model implementation walkthrough
├── OpenCV+DONUT.ipynb            # Hybrid approach (Shows DONUT is redundant)
├── PDF-Filing-OPENCV.ipynb       # Performance evaluation of OpenCV
├── basic_code.ipynb              # Legacy/Iteration 1 (Non-functional)
├── clicking_mechanism.ipynb      # Legacy/Iteration 1 (Non-functional)
├── annotate.py                   # Main annotation tool
├── form_fields.json              # Extracted field metadata
└── windows.txt                   # Environment configuration

Folder & File Details

Data & Annotations
data/: Contains the raw .png files used as the primary input.

clicking_mechanism/model_output/: Contains the output of the OPENCV logic, stored as .json files containing specific coordinates.

annotations/: Stores .json files representing the true labels (ground truth) of the input image forms.

Notebooks & Logic
DONUT.ipynb: Demonstrates how the DONUT model works.

OPENCV_Donut.ipynb: Shows how these two models work together. Note: Analysis shows that DONUT is actually not required for this workflow.

PDF-Filing-OpenCV: Demonstrates the performance of the OPENCV logic used by detect_and_save.py.

BASIC-CODE.IPYNB & clicking_mechanism.ipynb: Legacy Code. These are the first iteration attempts; they are not functional but kept for reference.

Major Files

detect_and_save.py -> OPENCV code that detects regions and saves them as JSON files.
annotate.py -> The main code used to fill and annotate the PDF files.

Outline:
506 pdf files related to forms collected
99 empty file(s) deleted as they did not have any fillable elements
407 files tested.


Initial OPENCV:
============================================================
TOLERANCE: 5%
============================================================

Checkboxes:
  Precision: 0.817 | Recall: 0.683 | F1: 0.744 | Accuracy: 0.592

Lines:
  Precision: 0.215 | Recall: 0.453 | F1: 0.291 | Accuracy: 0.170

Boxes:
  Precision: 0.273 | Recall: 0.590 | F1: 0.374 | Accuracy: 0.230

============================================================
TOLERANCE: 10%
============================================================

Checkboxes:
  Precision: 0.819 | Recall: 0.685 | F1: 0.746 | Accuracy: 0.595

Lines:
  Precision: 0.228 | Recall: 0.480 | F1: 0.309 | Accuracy: 0.183

Boxes:
  Precision: 0.295 | Recall: 0.638 | F1: 0.404 | Accuracy: 0.253

============================================================
TOLERANCE: 20%
============================================================

Checkboxes:
  Precision: 0.822 | Recall: 0.687 | F1: 0.749 | Accuracy: 0.598

Lines:
  Precision: 0.251 | Recall: 0.530 | F1: 0.341 | Accuracy: 0.205

Boxes:
  Precision: 0.347 | Recall: 0.750 | F1: 0.474 | Accuracy: 0.311


OpenCV Advanced
```
