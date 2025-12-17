# Exoplanet Detection Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![Status](https://img.shields.io/badge/Status-Academic%20Project-success.svg)
![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blueviolet.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

## Overview

This project demonstrates an end-to-end machine learning pipeline for detecting the presence of exoplanets using stellar light curve (flux) data. The primary goal of this project is to showcase data preprocessing, dimensionality reduction, class imbalance handling, and model comparison techniques in a constrained real-world scenario where only a single labeled dataset is available.

Rather than claiming production-level performance, this project focuses on **methodology, correctness, and interpretability**.

---

## Dataset

**Source:** Kaggle  
**Link:** https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data

### Dataset Description
- The dataset contains time-series flux values recorded from the Kepler Space Telescope.
- Each row represents observations of a star over time.
- The target column `LABEL` indicates:
  - `0` → No exoplanet detected
  - `1` → Exoplanet detected
- The dataset is **extremely imbalanced**, with very few confirmed exoplanet samples.

Due to dataset access limitations, only a single labeled dataset is used in this project.

---

## Project Approach

Because only one labeled dataset is available, an **internal stratified train–test split** is used.  
The focus is on building a **clean, reproducible, and technically correct ML pipeline**, not on inflating performance metrics.

---

## Libraries and Tools Used

- **Python**
- **NumPy & Pandas** – numerical computation and data handling
- **Scikit-learn** – preprocessing, PCA, model training, and evaluation
- **SciPy** – Gaussian filtering for noise reduction
- **Matplotlib & Seaborn** – visualization
- **Pickle** – model persistence

---

## Step-by-Step Methodology

### 1. Data Loading and Cleaning
- The dataset is loaded using Pandas.
- Missing values are filled with zeros to ensure numerical stability.
- Target labels are converted to binary format for classification.

---

### 2. Train–Test Split
- An internal **stratified split** is used due to single-dataset availability.
- Stratification ensures rare exoplanet samples appear in both sets.
- A fixed random state ensures reproducibility.

---

### 3. Data Preprocessing

#### a. Normalization
- Normalization ensures features are on a comparable scale.
- Prevents magnitude dominance in distance-based learning.

#### b. Gaussian Smoothing
- A Gaussian filter is applied to reduce high-frequency noise.
- Helps smooth flux variations while preserving meaningful patterns.

#### c. Feature Scaling
- Standardization is applied after smoothing.
- The scaler is fit **only on training data** to prevent data leakage.

---

### 4. Dimensionality Reduction (PCA)
- The dataset contains thousands of features, increasing overfitting risk.
- **Principal Component Analysis (PCA)** reduces dimensionality.
- The number of components is chosen to retain **90% of variance**.
- PCA is fit on training data and applied to test data.

---

### 5. Class Imbalance Handling (Without SMOTE)
- The minority class contains very few samples.
- Oversampling techniques such as SMOTE were intentionally avoided to prevent synthetic overfitting.
- **Class-weighted models** are used instead to handle imbalance safely.

This approach reflects real-world ML best practices for extremely imbalanced datasets.

---

### 6. Model Selection

#### Support Vector Machine (SVM)
- RBF kernel captures non-linear relationships.
- Class weighting compensates for imbalance.
- Suitable for high-dimensional data after PCA.

#### Random Forest Classifier
- Ensemble-based method combining multiple decision trees.
- Robust to noise and non-linear patterns.
- Uses class weighting instead of oversampling.

---

### 7. Model Evaluation

Models are evaluated using:
- **5-fold cross-validation**
- **Accuracy score**
- **Precision, recall, and F1-score**
- **Confusion matrix visualization**

Cross-validation helps mitigate bias from limited data availability.

---

### 8. Model Persistence
- The trained Random Forest model is saved using `pickle`.
- Enables reuse without retraining.
- Future predictions must follow the same preprocessing pipeline:
  
  `normalize → gaussian_filter → scaler → PCA`

---

## Important Notes

- This project is **not production-ready**.
- Results are **indicative**, not generalizable.
- Emphasis is placed on correct ML practices and transparency.

---

## Key Takeaways

- Built a complete ML pipeline under realistic data constraints
- Avoided data leakage and synthetic oversampling
- Applied principled dimensionality reduction
- Used class-weighted models for severe imbalance
- Prioritized reproducibility and clarity

---

## Future Improvements

- Validation on an external unseen dataset
- Time-series–specific deep learning models
- Hyperparameter optimization
- Domain-driven feature engineering

---

## Author

**Rupesh Vanneldas**  
Machine Learning & Cybersecurity Student

---

