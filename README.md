# GitHub Repository Description (one-liner)
> Smartphone accelerometer activity classifier (standing, walking, stairs) using rolling time-series features and GridSearchCV-tuned ML models — best accuracy 74.38% with Random Forest.

---

# README.md

# Activity Recognition from Accelerometer Data

Classifies physical activities (standing, walking, stairs down, stairs up) from tri-axial smartphone accelerometer data using machine learning.

## Problem
Given raw x, y, z accelerometer readings from a smartphone, predict which of 4 activities the user is performing at each timestep.

## Approach
- Labels are provided every 10th observation and forward-filled across the full time series
- Train/test split uses `shuffle=False` to preserve temporal order, with `test_size=0.3`
- Rolling statistical and frequency domain features are computed **after** the split to prevent data leakage
- `GridSearchCV` with `StratifiedKFold(n_splits=3)` is used for hyperparameter tuning on each model

## Features
Rolling window features computed per axis (x, y, z):
- Statistical: mean, std, var, min, max, median, range, IQR
- Motion: jerk (rate of change), jerk std, signal magnitude area (SMA)
- Cross-axis: xy, xz, yz correlations
- Frequency domain (FFT): dominant frequency, FFT energy, FFT mean
- Magnitude: resultant vector mean and std

## Models & Results

| Model | Best Accuracy | Best rolv | Best Params |
|---|---|---|---|
| Logistic Regression | 0.6537 | 40 | `C=40, max_iter=400` |
| Decision Tree | 0.7017 | 37 | `max_depth=None, min_samples_split=2` |
| XGBoost | 0.7240 | 47 | `learning_rate=0.1, max_depth=3, n_estimators=200` |
| **Random Forest** | **0.7438** | **34** | `max_depth=None, min_samples_split=4, n_estimators=50` |

## Key Design Decisions
- `shuffle=False` preserves time order so rolling features remain meaningful
- Rolling applied separately on train and test to avoid data leakage
- `StratifiedKFold` inside GridSearchCV ensures all 4 classes appear in every CV fold
- FFT features added to capture rhythmic movement patterns across activities

## Stack
Python, scikit-learn, XGBoost, pandas, numpy, matplotlib

---

# CV Entry

**Activity Recognition from Smartphone Accelerometer Data** | Python, scikit-learn, XGBoost
- Built a 4-class activity classifier (standing, walking, stairs down, stairs up) from raw tri-axial accelerometer time-series data
- Engineered 40+ rolling features including statistical, jerk, cross-axis correlation, and FFT frequency-domain features
- Implemented leak-free pipeline by applying rolling windows after train/test split to preserve temporal integrity
- Tuned Logistic Regression, Decision Tree, Random Forest, and XGBoost using GridSearchCV with StratifiedKFold cross-validation
- Achieved best test accuracy of **74.38%** with Random Forest (rolv=34, n_estimators=50, min_samples_split=4)
