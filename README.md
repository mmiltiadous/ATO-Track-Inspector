# Rail Surface Defect Detection using ATO Cameras

This repository contains the code and pipeline for the MSc thesis project **"Leveraging Autonomous Train Operation Cameras
for Rail Track Monitoring"**. The designed system leverages cameras already installed for train automation to perform continuous rail condition monitoring, aiming to provide an early-warning mechanism for rail surface degradation.

---
## Dataset

The project uses the **OSDAR23** dataset, which contains rail track images from Hamburg stations.

---

## Setup Instructions:
### Prerequisites
- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- For GPU acceleration: NVIDIA GPU with CUDA 11.8 compatible drivers
### Steps
1. Clone the repository and set the working directory
```bash
git clone https://github.com/mmiltiadous/MSc-thesis-ATO
cd MSc-thesis-ATO
```
2. Download the OSDAR23 dataset from: [https://data.fid-move.de/dataset/osdar23](https://data.fid-move.de/dataset/osdar23) 
3. Extract the downloaded data and place each folder (e.g., 1_calibration_1.1, 1_calibration_1.2, etc.) in: `./rail_data/DB/`
4. Set up the virtual environment using conda:
```bash 
conda env create -f environment.yml
conda activate thesis_env
pip install -r requirements.txt
```
5. Run the notebooks in the numerical order presented in the following section (1.→8.) to reproduce the project.

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

