# Human Activity Recognition from Raw Accelerometer Data

Predicting human physical activity states (standing, walking, stairs up, stairs down) from tri-axial smartphone accelerometer time-series data using engineered data features, and supervised machine learning models.

## Overview

Raw accelerometer signals (x, y, z) capture body motion over time but are noisy and not directly usable for classification. To make them useful, the signal is broken into small overlapping windows, and derived features are extracted from each window. These features are based on a short recent segment of data rather than a single timestamp, so the model can capture patterns like repeated movement during walking or low variation during standing.

hese features are fed into classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost) where each model uses GridSearchCV to find the rolling window size and hyperparameters that result in the highest accuracy.

**Key steps:**
- Labels from `train_labels.csv` are merged into `train_time_series.csv` and forward-filled to expand label coverage across all timesteps
- Train/test split uses `shuffle=False` to preserve time order
- Rolling statistical and frequency domain features are computed after the split to prevent data leakage
- Four models compared: Logistic Regression, Decision Tree, Random Forest, XGBoost
- `GridSearchCV` with `StratifiedKFold(n_splits=3)` is used for hyperparameter tuning on each model

## Output

| Model | Best Accuracy | Best rolv | Best Params |
|---|---|---|---|
| Logistic Regression | 0.6537 | 40 | `C=40, max_iter=400` |
| Decision Tree | 0.7017 | 37 | `max_depth=None, min_samples_split=2` |
| XGBoost | 0.7240 | 47 | `learning_rate=0.1, max_depth=3, n_estimators=200` |
| **Random Forest** | **0.7438** | **34** | `max_depth=None, min_samples_split=4, n_estimators=50` |

## Requirements
Open in prompt and run:
```bash
pip install pandas numpy matplotlib scikit-learn xgboost
```

## Data

Download the following files and place them in the same directory as the notebook before running:
- `train_labels.csv`
- `train_time_series.csv`
- `test_labels.csv`
- `test_time_series.csv`

## How to Run

1. Download the notebook and the CSV files above
2. Install dependencies (see Requirements)
3. Open `HARprediction.ipynb` and run all cells top to bottom


