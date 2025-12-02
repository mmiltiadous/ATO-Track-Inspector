# Rail Surface Defect Detection using ATO Cameras

This repository contains the code and pipeline for the thesis project **"Rail Surface Defect Detection using Cameras Installed for Automatic Train Operation (ATO)"**. The system leverages cameras already installed for train automation to perform continuous rail condition monitoring, aiming to provide an early-warning mechanism for rail surface degradation.

---

##  Project Overview

This research investigates whether standard RGB cameras installed for ATO can be repurposed for rail defect detection. The pipeline includes:
- **Data preprocessing** to handle variable lighting and operational conditions
- **Deep learning-based defect classification** using transformer and CNN architectures
- **Sequence-aware post-processing** to reduce false positives
- **Explainability analysis** to validate model decisions

The system achieves high precision on held-out test data while maintaining interpretability for operational deployment.

---

##  Repository Structure
├── AnalyzeAndPreprocessOSDAR23_1.ipynb # Dataset analysis, masking, patch extraction & preprocessing
├── Create_labels_masks_2.ipynb # Labeling functions, mask creation, dataset organization
├── splitData_inspectResults_3.ipynb # Proposed splitting algorithm, data leakage checks
├── Experiment1_Experiment2.ipynb # Preprocessing experiments & cross-validation analysis
├── Experiment3.ipynb # Model comparison (DeiT, ResNet, PatchCore)
├── Experiment4.ipynb # Sequential refinement methods
├── Experiment5.ipynb # Hyperparameter tuning & final test on held-out set
├── Experiment6.ipynb # Explainability analysis (XAI)
├── requirements.txt # Python dependencies
└── README.md # This file

## Dataset

The project uses the **OSDAR23** dataset, which contains rail track images from Hamburg stations.

### Setup Instructions:
1. Download the dataset from: [https://data.fid-move.de/dataset/osdar23](https://data.fid-move.de/dataset/osdar23)
2. Place the extracted data in: `./rail_data/DB/`
3. Run the notebooks in numerical order (1→8) to reproduce the full pipeline.

---

##  Pipeline Workflow

### 1. **Data Preparation & Preprocessing**
**`AnalyzeAndPreprocessOSDAR23_1.ipynb`**  
- Analyzes dataset structure and filters irrelevant data
- Corrects metadata inconsistencies
- Masks rail tracks and crops patches with black background
- Extracts patches per frame/track (optionally multiple patches per track for larger datasets)
- Applies all 6 preprocessing pipelines from the paper

### 2. **Labeling & Mask Creation**
**`Create_labels_masks_2.ipynb`**  
- Provides functions to connect patches with source images using filename patterns
- Implements labeling functions for defect annotation
- Creates binary masks for defect regions
- Organizes labeled data into structured folders

### 3. **Data Splitting & Validation**
**`splitData_inspectResults_3.ipynb`**  
- Implements the proposed group-aware splitting algorithm (by station/sequence)
- Prevents data leakage between train/validation/test sets
- Validates split integrity and checks for potential leakage

### 4. **Experiments 1 & 2: Preprocessing Analysis**
**`Experiment1_Experiment2.ipynb`**  
- **Experiment 1**: Compares 6 preprocessing pipelines using AUPRC metrics
- **Experiment 2**: Performs cross-validation with different random seeds
- Saves performance metrics and generates publication-ready plots
- Analyzes sensitivity to data splits

### 5. **Experiment 3: Model Comparison**
**`Experiment3.ipynb`**  
- Implements DeiT-Small (both distilled and standard variants)
- Compares against ResNet-18 (results imported from Experiment 2)
- Integrates PatchCore from original repository for unsupervised baseline
- Analyzes results from all three models and creates comparison plots
- Demonstrates transformer superiority for global context understanding

### 6. **Experiment 4: Sequential Refinement**
**`Experiment4.ipynb`**  
- Applies sequence-aware post-processing methods:
  - Confidence-weighted smoothing
  - Isolated false positive removal
  - Bidirectional LSTM refinement
- Compares heuristic vs. learned refinement approaches
- Generates thesis plots for sequential analysis

### 7. **Experiment 5: Final Evaluation**
**`Experiment5.ipynb`**  
- Performs hyperparameter tuning on validation set
- Evaluates final pipeline on completely held-out test set
- Reports precision, recall, F1-score, and AUPRC
- Demonstrates near-perfect performance on unseen data

### 8. **Experiment 6: Explainability Analysis**
**`Experiment6.ipynb`**  
- Applies gradient-based attribution methods (Grad-CAM, Saliency Maps)
- Analyzes True Positives, True Negatives, and False Positives
- Identifies failure modes and model attention patterns
- Provides visual justifications for model decisions

