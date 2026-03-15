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
============================================================
TOLERANCE: 5%
============================================================

Checkboxes:
  Precision: 0.679 | Recall: 0.707 | F1: 0.693 | Accuracy: 0.530

Lines:
  Precision: 0.515 | Recall: 0.352 | F1: 0.418 | Accuracy: 0.264

Boxes:
  Precision: 0.320 | Recall: 0.571 | F1: 0.410 | Accuracy: 0.258

============================================================
TOLERANCE: 10%
============================================================

Checkboxes:
  Precision: 0.682 | Recall: 0.710 | F1: 0.696 | Accuracy: 0.533

Lines:
  Precision: 0.522 | Recall: 0.356 | F1: 0.423 | Accuracy: 0.268

Boxes:
  Precision: 0.345 | Recall: 0.615 | F1: 0.442 | Accuracy: 0.284

============================================================
TOLERANCE: 20%
============================================================

Checkboxes:
  Precision: 0.685 | Recall: 0.713 | F1: 0.699 | Accuracy: 0.537

Lines:
  Precision: 0.559 | Recall: 0.382 | F1: 0.454 | Accuracy: 0.294

Boxes:
  Precision: 0.406 | Recall: 0.724 | F1: 0.520 | Accuracy: 0.351
```

'''YOLO-v8'''

# V8

# TOLERANCE: 5%

checkboxes:
Precision: 0.789 | Recall: 0.794 | F1: 0.791 | Accuracy (Jaccard): 0.655

Lines:
Precision: 0.715 | Recall: 0.824 | F1: 0.766 | Accuracy (Jaccard): 0.620

Boxes:
Precision: 0.540 | Recall: 0.642 | F1: 0.587 | Accuracy (Jaccard): 0.415

============================================================
TOLERANCE: 10%
============================================================
Checkboxes:
Precision: 0.808 | Recall: 0.813 | F1: 0.811 | Accuracy (Jaccard): 0.682

Lines:
Precision: 0.736 | Recall: 0.847 | F1: 0.788 | Accuracy (Jaccard): 0.650

Boxes:
Precision: 0.574 | Recall: 0.682 | F1: 0.623 | Accuracy (Jaccard): 0.453

============================================================
TOLERANCE: 20%
============================================================
Checkboxes:
Precision: 0.820 | Recall: 0.826 | F1: 0.823 | Accuracy (Jaccard): 0.699

Lines:
Precision: 0.755 | Recall: 0.869 | F1: 0.808 | Accuracy (Jaccard): 0.677

Boxes:
Precision: 0.667 | Recall: 0.793 | F1: 0.724 | Accuracy (Jaccard): 0.568

# '''YOLO-v11'''

# TOLERANCE: 5%

Checkboxes:
Precision: 0.827 | Recall: 0.774 | F1: 0.800 | Accuracy (Jaccard): 0.666

Lines:
Precision: 0.752 | Recall: 0.829 | F1: 0.789 | Accuracy (Jaccard): 0.652

Boxes:
Precision: 0.587 | Recall: 0.645 | F1: 0.615 | Accuracy (Jaccard): 0.444

============================================================
TOLERANCE: 10%
============================================================

Checkboxes:
Precision: 0.845 | Recall: 0.790 | F1: 0.817 | Accuracy (Jaccard): 0.690

Lines:
Precision: 0.777 | Recall: 0.857 | F1: 0.815 | Accuracy (Jaccard): 0.688

Boxes:
Precision: 0.628 | Recall: 0.690 | F1: 0.658 | Accuracy (Jaccard): 0.490

============================================================
TOLERANCE: 20%
============================================================

Checkboxes:
Precision: 0.855 | Recall: 0.800 | F1: 0.827 | Accuracy (Jaccard): 0.705

Lines:
Precision: 0.797 | Recall: 0.878 | F1: 0.836 | Accuracy (Jaccard): 0.718

Boxes:
Precision: 0.726 | Recall: 0.798 | F1: 0.760 | Accuracy (Jaccard): 0.613

# '''YOLO-v26-s'''

============================================================
TOLERANCE: 5%
============================================================
Checkboxes:
Precision: 0.912 | Recall: 0.601 | F1: 0.725 | Accuracy (Jaccard): 0.569

Lines:
Precision: 0.793 | Recall: 0.697 | F1: 0.742 | Accuracy (Jaccard): 0.590

Boxes:
Precision: 0.571 | Recall: 0.588 | F1: 0.579 | Accuracy (Jaccard): 0.408

============================================================
TOLERANCE: 10%
============================================================
Checkboxes:
Precision: 0.933 | Recall: 0.615 | F1: 0.741 | Accuracy (Jaccard): 0.589

Lines:
Precision: 0.817 | Recall: 0.717 | F1: 0.764 | Accuracy (Jaccard): 0.618

Boxes:
Precision: 0.615 | Recall: 0.633 | F1: 0.624 | Accuracy (Jaccard): 0.454

============================================================
TOLERANCE: 20%
============================================================
Checkboxes:
Precision: 0.944 | Recall: 0.622 | F1: 0.750 | Accuracy (Jaccard): 0.600

Lines:
Precision: 0.839 | Recall: 0.737 | F1: 0.785 | Accuracy (Jaccard): 0.646

Boxes:
Precision: 0.713 | Recall: 0.734 | F1: 0.724 | Accuracy (Jaccard): 0.567

# '''YOLO-v26-l'''

============================================================
TOLERANCE: 5%
============================================================

Checkboxes:
Precision: 0.958 | Recall: 0.562 | F1: 0.708 | Accuracy (Jaccard): 0.549

Lines:
Precision: 0.790 | Recall: 0.784 | F1: 0.787 | Accuracy (Jaccard): 0.649

Boxes:
Precision: 0.624 | Recall: 0.634 | F1: 0.629 | Accuracy (Jaccard): 0.459

============================================================
TOLERANCE: 10%
============================================================

Checkboxes:
Precision: 0.973 | Recall: 0.571 | F1: 0.719 | Accuracy (Jaccard): 0.562

Lines:
Precision: 0.806 | Recall: 0.799 | F1: 0.802 | Accuracy (Jaccard): 0.670

Boxes:
Precision: 0.663 | Recall: 0.673 | F1: 0.668 | Accuracy (Jaccard): 0.502

============================================================
TOLERANCE: 20%
============================================================
Checkboxes:
Precision: 0.981 | Recall: 0.575 | F1: 0.725 | Accuracy (Jaccard): 0.569

Lines:
Precision: 0.825 | Recall: 0.818 | F1: 0.821 | Accuracy (Jaccard): 0.697

Boxes:
Precision: 0.761 | Recall: 0.773 | F1: 0.767 | Accuracy (Jaccard): 0.622
