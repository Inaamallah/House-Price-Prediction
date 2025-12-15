# Ames Housing Price Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![XGBoost](https://img.shields.io/badge/Library-XGBoost-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Project Overview
This project implements an end-to-end machine learning pipeline to predict residential housing prices in Ames, Iowa. The goal was to build a production-ready system that handles real-world data challenges—including missing values, outliers, and high dimensionality—while minimizing the Root Mean Squared Error (RMSE).

The solution features a **Dual-Pipeline Architecture** that treats linear and tree-based models differently to maximize performance.

## Key Technical Features

### 1. Custom Transformers
I developed custom Scikit-Learn estimators to integrate complex logic directly into the pipeline:
- **`GroupImputer`**: A context-aware imputer that fills missing `LotFrontage` values using the median frontage of the specific `Neighborhood` rather than a global average.
- **`OutlierCapper`**: A transformer that dynamically clips numerical outliers based on the Interquartile Range (IQR), preventing extreme values from skewing the model during cross-validation.

### 2. Advanced Feature Engineering
- **Ordinal Encoding**: Mapped 21 categorical features (e.g., `KitchenQual`, `BsmtExposure`) to explicit numerical hierarchies (e.g., `Poor`=0 → `Excellent`=5) to preserve their mathematical magnitude.
- **Nominal Encoding**: Applied One-Hot Encoding with rare category handling for nominal variables.
- **Feature Selection**: Implemented a "Tournament" strategy:
  - **Lasso Regression** for linear model feature selection.
  - **Tree Model Importance** for tree-based feature selection.

### 3. Dual-Pipeline Architecture
Recognizing that different algorithms require different data treatments, I built two distinct preprocessing pipelines:
- **Scaled Pipeline**: For Linear/Ridge/Lasso models (Requires Standardization).
- **Unscaled Pipeline**: For XGBoost/Random Forest (Preserves original distributions).

## Model Performance

The models were evaluated using **RMSE (Root Mean Squared Error)** on a hold-out test set.

| Model | Preprocessing | Feature Selection | Test RMSE | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | Scaled | Lasso | ~9 Trillion | Failed (Overfitting) |
| **Ridge Regression** | Scaled | Lasso | $28,857 | Stable Baseline |
| **Random Forest** | Unscaled | Tree-Based | $26,027 | Good Accuracy |
| **XGBoost (Tuned)** | Unscaled | Tree-Based | **~$24,500** | **Best Performer** |

*Note: The initial XGBoost model suffered from high variance (Overfitting gap of ~$22k). Extensive Hyperparameter tuning (Regularization & Subsampling) was performed to close this gap.*


