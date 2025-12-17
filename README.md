# 🪐 Exoplanet Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Supervised-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![Model](https://img.shields.io/badge/Model-Random%20Forest-green.svg)
![Domain](https://img.shields.io/badge/Domain-Astroinformatics-purple.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blueviolet.svg)
![Course](https://img.shields.io/badge/Course-CYT180-blueviolet.svg)

This project implements a **machine learning–based exoplanet detection system** using stellar light curve data collected by the **Kepler Space Telescope**.  

The objective is to identify the presence of exoplanets from **high-dimensional, highly imbalanced time-series data** while following correct and reproducible ML practices.

---

## 📌 Project Overview

Detecting exoplanets is a challenging machine learning problem due to:

- Extremely **high-dimensional time-series data**
- Severe **class imbalance** (very few confirmed exoplanets)
- Subtle brightness variations that can be hidden by noise
- Risk of overfitting when data is limited

This project focuses on building a **clean, end-to-end ML pipeline** that emphasizes **methodology, correctness, and transparency**, rather than inflated performance claims.

---

## 📂 Dataset

- **Source:** Kaggle  
- **Dataset:** Kepler Labelled Time Series Data  
- **Link:** https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data

### Dataset Description

- Each row represents flux measurements of a star over time
- Columns correspond to brightness values at different timestamps
- Target column `LABEL`:
  - `0` → No exoplanet detected
  - `1` → Exoplanet detected
- The dataset is **extremely imbalanced**, with very few positive samples

Due to dataset access constraints, a **single labeled dataset** is used in this project.

---

## 🔁 Workflow

### **Stage 1 — Data Preparation and Preprocessing**

1. **Dataset Loading & Cleaning**
   - Load the dataset using Pandas
   - Handle missing values to ensure numerical stability
   - Convert labels into binary format for classification

2. **Train–Test Split**
   - Perform an **internal stratified split (80/20)**
   - Ensures minority class representation in both sets
   - Fixed random state for reproducibility

3. **Normalization**
   - Normalize flux values to prevent magnitude dominance
   - Important for distance- and kernel-based models

4. **Gaussian Smoothing**
   - Apply Gaussian filtering to reduce high-frequency noise
   - Smooths stellar brightness curves while preserving patterns

5. **Feature Scaling**
   - Apply standardization (zero mean, unit variance)
   - Scaler is fit **only on training data** to prevent data leakage

---

### **Stage 2 — Dimensionality Reduction**

6. **Principal Component Analysis (PCA)**
   - Original data contains thousands of features
   - PCA reduces dimensionality and mitigates overfitting
   - Number of components selected to retain **90% variance**
   - PCA is trained on training data and applied to test data

---

### **Stage 3 — Model Training and Evaluation**

7. **Class Imbalance Handling (Without SMOTE)**
   - Minority class contains very few samples
   - Oversampling methods (e.g., SMOTE) were intentionally avoided
   - **Class-weighted models** are used instead to reduce bias safely

8. **Model Training**
   - Two supervised models are trained and compared:
     - Support Vector Machine (SVM)
     - Random Forest Classifier

9. **Model Evaluation**
   - Metrics used:
     - Accuracy
     - Precision
     - Recall
     - F1-score
   - Confusion matrix for error analysis
   - 5-fold cross-validation to reduce evaluation bias

---

## 🤖 Models Used

### 🔹 Support Vector Machine (SVM)
- RBF kernel to capture non-linear patterns
- Class weighting to handle imbalance
- Well-suited for high-dimensional PCA-transformed data

### 🔹 Random Forest Classifier
- Ensemble learning approach using multiple decision trees
- Robust to noise and non-linear relationships
- Performs reliably under severe class imbalance
- Chosen as the final saved model

---

## 📊 Key Results

- Stable classification behavior despite extreme imbalance
- High precision for the minority (exoplanet) class
- Reduced overfitting due to PCA and class weighting
- Consistent results across cross-validation folds

> Reported metrics are **indicative**, not production-level, due to dataset limitations.

---

## 💾 Model Persistence

- The trained Random Forest model is saved using `pickle`
- Allows reuse without retraining
- Any new input must follow the same preprocessing pipeline:
- normalize → gaussian_filter → scaler → PCA

---

## 📚 References

- Kaggle – Kepler Labelled Time Series Data  
  https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data

- NASA Exoplanet Exploration Program  
  https://exoplanets.nasa.gov/

---

## 🚀 Key Takeaways

- Built a complete ML pipeline with limited real-world data
- Avoided data leakage and synthetic overfitting
- Applied principled dimensionality reduction
- Used class-weighted models for extreme imbalance
- Prioritized reproducibility and transparency

---

## 🔮 Future Improvements

- Validation on an external unseen dataset
- Time-series–specific deep learning models (CNN/LSTM)
- Hyperparameter tuning
- Domain-specific feature engineering

---

## 👤 Author

**Rupesh Vanneldas**  
Machine Learning & Cybersecurity Student  
CYT180 – Fall 2025

---

