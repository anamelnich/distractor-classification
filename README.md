# EEG Distractor Decoder: Classification of Distractor Positivty (Pd)

This project implements an LDA classification for detecting Pd component from EEG data. 
It is designed for use in neurofeedback and brain-computer interface (BCI) systems.

## Repository Structure
```text
distractor-classification/
├── code/
│   ├── functions/              # Utility functions
│   └── model/                  
│       ├── main.m              # Main script to run training + evaluation 
│       ├── computeModel.m      
│       ├── computeDecoderLeft.m 
│       ├── computeDecoderRight.m 
│       ├── singleClassificationLeft.m 
│       └── singleClassificationRight.m 
├── data/                       # Sample data
└── README.md
```
---

## How It Works

### 1. **Load and Preprocess Data**
- Removes non-EEG channels (EOG, FP1/2, M1/2)
- Applies bandpass filter (configurable)
- Epochs trials around event markers

### 2. **Feature Extraction**
- Uses posterior lateral electrodes (e.g., PO7/PO8)
- Extracts ERP amplitudes and difference waves (left vs. right)
- Applies CCA-based spatial filtering
- Feature Reduction

### 3. **Model Training**
- Trains two separate LDA classifiers:
  - Left decoder: classifies right-distractor vs. others
  - Right decoder: classifies left-distractor vs. others
- Uses leave-one-run-out cross-validation
- Stores model parameters 

### 4. **Model Evaluation**
- Computes ROC and optimal threshold
- Outputs confusion matrix, AUC, TPR, TNR, and Accuracy
