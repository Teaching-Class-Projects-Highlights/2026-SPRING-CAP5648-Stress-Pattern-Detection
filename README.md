# 2026-SPRING-CAP5648-Stress-Pattern-Detection
This project focuses on finding global stress patterns using physiological data from wearable devices.

**Authors:** Amelia McCormack and Lillian Malik <br>
**Emails:** alm20h@fsu.edu , lam22g@fsu.edu <br>
**Institution:** Department of Computer Science, Florida State University <br>

## 🚀 Project Overview
This project is not intended for medical diagnosis, but rather as a tool to capture **subconscious physiological indicators of stress**. The primary objective is to determine whether stress-related patterns are distinguishable from baseline physiological signals.

While many existing studies focus on maximizing predictive accuracy, this project emphasizes the **interpretation of learned patterns**. Understanding these patterns is important because their separability in a controlled setting can provide a foundation for future research.

## 🔗 Dataset
**WESAD (Wearable Stress and Affect Detection Dataset)**
- Multimodal physiological signals from wearable devices
- Includes ECG, EDA, temperature, and motion data
- Contains labeled conditions such as baseline and stress

Download From: 
```
https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset?utm_source=chatgpt.com
```
After downloading the dataset, place the it inside the ```/data``` folder.

## ⚙️ Setup
### 1. Data Preprocessing 
``` 
python main.py
```
- Loads raw .pkl files from the dataset
- Filters data to baseline (1) and stress (2) labels
- Downsamples signals for efficiency
- **Outputs:** processed_features.csv

### 2. Feature Extraction
``` 
python featureExtraction.py
```
- Segments data into fixed-size time windows
- Computes statistical features (mean, std, min, max)
- **Outputs:** windowed_features.csv

### 3. Modeling & Analysis
``` 
python models.py
```
- Trains and evaluates:
  - Support Vector Machines (linear and RBF)
  - K-Nearest Neighbors
  - Naïve Bayes
- Compares different feature sets (ECG, EDA, combined)
- **Outputs:** classification metrics and PCA visualizations

### All Libraries & Imports
```
import os
import pickle
import warnings

import pandas as pd
from src.parse.load import load_all_subjects

from sklearn.naive_bayes import GaussianNB            
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
```

## 🖼️ Results
<img width="611" height="418" alt="Screenshot 2026-05-05 at 4 38 05 PM" src="https://github.com/user-attachments/assets/8dbd6e0b-aafc-4508-9857-9aa3860728c6" />

- Stress and baseline signals show significant overlap
- Models achieve moderate accuracy (~65%) but fail to detect stress reliably
- Most models default to predicting the baseline state
- Stress detection recall is very low, indicating missed stress cases
  
### Takeaways
- The limitation is not just the models, but also how the data is represented.
- Simple statistical features do not fully capture the complexity of stress responses.

### Future Work
- Explore subject-specific models
- Incorporate temporal and frequency-based features (e.g., HRV)
- Improve feature representation for better separability

## ⚠️ Disclaimer
This project is for research and educational purposes only.
It is not intended for medical or clinical use

