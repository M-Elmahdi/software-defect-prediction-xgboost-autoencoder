# Software Defect Prediction Pipeline Implementation Guide

## Overview

This document describes the implementation of the thesis methodology comparing two software defect prediction pipelines:

- **Pipeline A (Baseline SPAM-XAI):** SMOTE + PCA + MLP + LIME
- **Pipeline B (Proposed):** SMOTE + Autoencoder + XGBoost + LIME

The implementation follows Chapter 4 of the thesis methodology and evaluates both approaches on NASA PROMISE datasets (CM1, PC1, PC2) using 10-fold stratified cross-validation.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Implementation Details](#implementation-details)
   - [Environment Setup](#1-environment-setup)
   - [Data Loading](#2-data-loading)
   - [Preprocessing Pipeline](#3-preprocessing-pipeline)
   - [Baseline Pipeline (PCA + MLP)](#4-baseline-pipeline-pca--mlp)
   - [Proposed Pipeline (Autoencoder + XGBoost)](#5-proposed-pipeline-autoencoder--xgboost)
   - [Evaluation Metrics](#6-evaluation-metrics)
   - [Explainability (LIME)](#7-explainability-lime)
   - [Main Runner](#8-main-experiment-runner)
3. [Experiment Results](#experiment-results)
4. [Results Discussion](#results-discussion)
5. [How to Run](#how-to-run)

---

## Project Structure

```
thesis_implementation/
+-- main.py                    # Main experiment runner
+-- requirements.txt           # Python dependencies
+-- readme.md                  # This document
+-- promise_datasets/          # NASA PROMISE ARFF files
|   +-- CM1.arff
|   +-- PC1.arff
|   +-- PC2.arff
+-- results/                   # Experiment outputs
|   +-- results_summary_*.csv
|   +-- results_detailed_*.json
+-- src/
    +-- __init__.py
    +-- config.py              # All hyperparameters
    +-- data_loader.py         # ARFF file parsing
    +-- preprocessing.py       # Imputation, scaling, SMOTE
    +-- pipeline_baseline.py   # PCA + MLP implementation
    +-- pipeline_proposed.py   # Autoencoder + XGBoost implementation
    +-- autoencoder.py         # PyTorch autoencoder model
    +-- evaluation.py          # Metrics and statistical tests
    +-- explainability.py      # LIME wrapper
```

---

## Implementation Details

### 1. Environment Setup

**File:** `requirements.txt`

The implementation uses Python 3.8+ with the following key dependencies:

| Library | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.21 | Numerical computing |
| pandas | >= 1.3 | Data manipulation |
| scikit-learn | >= 1.0 | ML utilities, PCA, MLP |
| xgboost | >= 1.5 | XGBoost classifier |
| torch | >= 2.0 | Autoencoder (PyTorch) |
| lime | >= 0.2 | Explainability |
| scipy | >= 1.7 | Wilcoxon test |
| liac-arff | >= 2.5 | ARFF file parsing |

**Note:** A custom SMOTE implementation was created in `preprocessing.py` to avoid compatibility issues between `imbalanced-learn` and scikit-learn 1.8+ on Python 3.14.

---

### 2. Data Loading

**File:** `src/data_loader.py`

The data loader handles NASA PROMISE ARFF files:

- Parses ARFF format using the `liac-arff` library
- Extracts 21 McCabe/Halstead software metrics (CM1, PC1) or 36 metrics (PC2)
- Converts class labels (`{false,true}` or `{N,Y}`) to binary (0=clean, 1=defective)

**Datasets Summary:**

| Dataset | Samples | Features | Defective | Clean | Defect Rate |
|---------|---------|----------|-----------|-------|-------------|
| CM1 | 498 | 21 | 49 | 449 | 9.84% |
| PC1 | 1,109 | 21 | 77 | 1,032 | 6.94% |
| PC2 | 745 | 36 | 16 | 729 | 2.15% |

---

### 3. Preprocessing Pipeline

**File:** `src/preprocessing.py`

The preprocessing pipeline applies (in order):

1. **Missing Value Imputation:** `SimpleImputer(strategy='median')`
2. **Feature Scaling:** `StandardScaler()` (z-score normalization)
3. **Class Balancing:** Custom SMOTE implementation (`k_neighbors=5`)
4. **Cross-Validation:** `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`

**Key Design Decisions:**
- Imputer and scaler are fitted **only on training data** to prevent data leakage
- SMOTE is applied **only to training data** within each fold
- Stratification preserves class distribution across folds

---

### 4. Baseline Pipeline (PCA + MLP)

**File:** `src/pipeline_baseline.py`

Implements the SPAM-XAI baseline:

**PCA Configuration:**
```python
PCA(n_components=0.95, random_state=42)  # Retain 95% variance
```
Typically reduces 21 features to 7-12 principal components.

**MLP Configuration:**
```python
MLPClassifier(
    hidden_layer_sizes=(100, 50),    # Two hidden layers
    activation='relu',
    solver='adam',
    alpha=0.0001,                     # L2 regularization
    learning_rate='adaptive',
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
```

---

### 5. Proposed Pipeline (Autoencoder + XGBoost)

**Files:** `src/autoencoder.py`, `src/pipeline_proposed.py`

**Autoencoder Architecture (PyTorch):**

```
Encoder: Input(21) --> Dense(64, ReLU) --> Dense(32, ReLU) --> Latent(10, ReLU)
Decoder: Latent(10) --> Dense(32, ReLU) --> Dense(64, ReLU) --> Output(21, Linear)
```

**Training Parameters:**
- Optimizer: Adam (lr=0.001)
- Loss: Mean Squared Error (MSE)
- Epochs: 100 (early stopping, patience=10)
- Batch size: 32
- Validation split: 10%

**XGBoost Configuration:**
```python
XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    random_state=42
)
```

---

### 6. Evaluation Metrics

**File:** `src/evaluation.py`

**Classification Metrics:**
- **Precision:** TP / (TP + FP)
- **Recall (Sensitivity):** TP / (TP + FN)
- **F1-Score:** Harmonic mean of Precision and Recall
- **AUC-ROC:** Area under the ROC curve

**Statistical Testing:**
- **Wilcoxon Signed-Rank Test:** Non-parametric paired comparison
- **Significance Level:** alpha = 0.05

The `MetricsAggregator` class collects per-fold metrics and computes mean +/- std across all folds.

---

### 7. Explainability (LIME)

**File:** `src/explainability.py`

**LIME Configuration:**
```python
LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=['Clean', 'Defective'],
    mode='classification',
    discretize_continuous=True
)

explainer.explain_instance(
    data_row=instance,
    predict_fn=model.predict_proba,
    num_features=10,
    num_samples=5000
)
```

Explanations are generated for:
- True Positives (correctly identified defects)
- True Negatives (correctly identified clean modules)
- Misclassified instances (FP/FN)

---

### 8. Main Experiment Runner

**File:** `main.py`

The main runner orchestrates the complete experiment:

1. Loads all specified datasets
2. For each dataset, runs 10-fold cross-validation
3. For each fold:
   - Preprocesses data (impute, scale, SMOTE)
   - Trains and evaluates both pipelines
   - Collects metrics
4. Aggregates results and performs Wilcoxon tests
5. Generates LIME explanations on the final fold
6. Saves results to CSV and JSON files

---

## Experiment Results

### Dataset: CM1 (498 samples, 9.84% defective)

| Metric | Baseline (PCA+MLP) | Proposed (AE+XGB) | p-value | Significant |
|--------|-------------------|-------------------|---------|-------------|
| Precision | 0.213 +/- 0.045 | 0.157 +/- 0.113 | 0.232 | No |
| Recall | 0.675 +/- 0.238 | 0.285 +/- 0.203 | 0.004 | Yes |
| F1-Score | 0.319 +/- 0.074 | 0.200 +/- 0.143 | 0.064 | No |
| AUC-ROC | 0.778 +/- 0.072 | 0.672 +/- 0.099 | 0.027 | Yes |

### Dataset: PC1 (1,109 samples, 6.94% defective)

| Metric | Baseline (PCA+MLP) | Proposed (AE+XGB) | p-value | Significant |
|--------|-------------------|-------------------|---------|-------------|
| Precision | 0.221 +/- 0.050 | 0.333 +/- 0.124 | 0.049 | Yes |
| Recall | 0.655 +/- 0.162 | 0.513 +/- 0.169 | 0.031 | Yes |
| F1-Score | 0.324 +/- 0.064 | 0.388 +/- 0.109 | 0.232 | No |
| AUC-ROC | 0.817 +/- 0.063 | 0.851 +/- 0.078 | 0.160 | No |

### Dataset: PC2 (745 samples, 2.15% defective)

| Metric | Baseline (PCA+MLP) | Proposed (AE+XGB) | p-value | Significant |
|--------|-------------------|-------------------|---------|-------------|
| Precision | 0.048 +/- 0.076 | 0.165 +/- 0.295 | 0.313 | No |
| Recall | 0.250 +/- 0.403 | 0.250 +/- 0.335 | 1.000 | No |
| F1-Score | 0.078 +/- 0.123 | 0.162 +/- 0.220 | 0.375 | No |
| AUC-ROC | 0.734 +/- 0.271 | 0.837 +/- 0.123 | 0.193 | No |

---

## Results Discussion

### Key Findings

#### 1. Dataset-Dependent Performance

The relative performance of the two pipelines varies significantly by dataset:

- **CM1:** Baseline (PCA+MLP) significantly outperforms proposed (AE+XGB) in Recall and AUC-ROC
- **PC1:** Mixed results - Proposed has better Precision, Baseline has better Recall
- **PC2:** No significant differences due to extreme class imbalance (only 16 defective samples)

#### 2. Precision-Recall Trade-off

A consistent pattern emerges across datasets:

- **Baseline (MLP):** Higher Recall (0.655-0.675), Lower Precision (0.21-0.22)
- **Proposed (XGBoost):** Higher Precision (0.33 on PC1), Lower Recall (0.29-0.51)

This suggests:
- MLP tends to be more aggressive in predicting defects (more true positives but also more false positives)
- XGBoost is more conservative, resulting in fewer false alarms but missing some defects

#### 3. Impact of Class Imbalance

Performance degrades dramatically with increasing imbalance:

| Dataset | Defect Rate | Best F1-Score |
|---------|-------------|---------------|
| CM1 | 9.84% | 0.319 (Baseline) |
| PC1 | 6.94% | 0.388 (Proposed) |
| PC2 | 2.15% | 0.162 (Proposed) |

PC2's severe imbalance (only 16 defects out of 745 modules) makes reliable prediction extremely challenging.

#### 4. Dimensionality Reduction Comparison

- **PCA:** Linear transformation retaining 95% variance (typically 7-12 components)
- **Autoencoder:** Non-linear transformation to 10 latent dimensions

The autoencoder's non-linear representations did not consistently outperform PCA, suggesting that:
- Linear relationships may be sufficient for these static code metrics
- The autoencoder may require more training data or hyperparameter tuning

#### 5. LIME Feature Importance

Top features identified by LIME across both pipelines:

| Rank | CM1 | PC1 | PC2 |
|------|-----|-----|-----|
| 1 | locCodeAndComment | I (content) | DECISION_DENSITY |
| 2 | lOComment | uniq_Opnd | PERCENT_COMMENTS |
| 3 | uniq_Op | locCodeAndComment | DESIGN_DENSITY |

This aligns with domain knowledge:
- Comment-related metrics indicate documentation practices
- Complexity metrics (unique operators/operands) relate to code understandability
- Halstead metrics capture program difficulty

### Limitations

1. **Small Sample Sizes:** Particularly for PC2, the extreme imbalance limits statistical power
2. **High Variance:** Large standard deviations indicate fold-to-fold variability
3. **No Hyperparameter Tuning:** Fixed configurations from the methodology may not be optimal for all datasets
4. **Single Random Seed:** Results depend on the specific train/test splits

### Recommendations for Future Work

1. **Ensemble Methods:** Combine both pipelines for better precision-recall balance
2. **Advanced Sampling:** Try SMOTE variants (Borderline-SMOTE, ADASYN) for severe imbalance
3. **Hyperparameter Optimization:** Use grid search or Bayesian optimization
4. **Cross-Project Validation:** Test generalization across different datasets

---

## How to Run

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full experiment (all datasets, 10-fold CV)
python main.py

# Run quick test
python main.py --datasets CM1 --folds 3
```

### Command-Line Options

```bash
python main.py [OPTIONS]

Options:
  --datasets    Datasets to evaluate (default: CM1 PC1 PC2)
  --folds       Number of CV folds (default: 10)
  --verbose     Print detailed progress
  --quiet       Minimal output
  --no-save     Don't save results to files
```

### Running Individual Modules

```bash
# Test data loading
python -m src.data_loader

# Test preprocessing
python -m src.preprocessing

# Test baseline pipeline
python -m src.pipeline_baseline

# Test proposed pipeline
python -m src.pipeline_proposed

# Test evaluation metrics
python -m src.evaluation

# Test LIME explainability
python -m src.explainability
```

### Output Files

Results are saved to the `results/` directory with timestamps:

- `results_summary_YYYYMMDD_HHMMSS.csv` - Aggregated metrics per dataset
- `results_detailed_YYYYMMDD_HHMMSS.json` - Full results including per-fold data

---

## Conclusion

This implementation successfully reproduces the SPAM-XAI methodology and compares it against the proposed Autoencoder + XGBoost approach. The results demonstrate that:

1. Neither approach universally dominates the other
2. The baseline shows advantages in recall (defect detection)
3. The proposed approach shows advantages in precision (fewer false alarms)
4. Extreme class imbalance remains a significant challenge

The implementation provides a solid foundation for further experimentation with different configurations, datasets, and evaluation scenarios.

