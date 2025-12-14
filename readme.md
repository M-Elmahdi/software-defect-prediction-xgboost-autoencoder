# Software Defect Prediction Pipeline Implementation Guide

## Overview

This document describes the implementation of the thesis methodology comparing two software defect prediction pipelines:

- **Pipeline A (Baseline SPAM-XAI):** SMOTE + PCA + MLP + LIME
- **Pipeline B (Proposed):** SMOTE + Autoencoder + XGBoost + LIME

The implementation follows Chapter 4 of the thesis methodology and evaluates both approaches on NASA PROMISE datasets (CM1, PC1, PC2) using 10 repetitions × 10-fold stratified cross-validation (100 paired observations per dataset) for statistically robust results.

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
- **MCC (Matthews Correlation Coefficient):** Balanced metric for imbalanced data: (TP×TN - FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
- **Accuracy:** (TP + TN) / Total

**Statistical Testing:**
- **Wilcoxon Signed-Rank Test:** Non-parametric paired comparison
- **Significance Level:** alpha = 0.05
- **Multiple Repetitions:** 10 repetitions with different seeds (42, 142, 242, ..., 942) for 100 paired observations

The `MetricsAggregator` class collects per-fold metrics across all repetitions and computes mean +/- std.

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

The main runner orchestrates the complete experiment with multiple repetitions for statistical robustness:

1. Loads all specified datasets
2. For each dataset, runs **10 repetitions** with different random seeds (42, 142, 242, ..., 942)
3. For each repetition, runs **10-fold stratified cross-validation**
4. For each fold:
   - Preprocesses data (impute, scale, SMOTE)
   - Trains and evaluates both pipelines
   - Collects all metrics including MCC
5. Aggregates results across all 100 observations (10 reps × 10 folds)
6. Performs Wilcoxon signed-rank tests with high statistical power
7. Generates LIME explanations on the final repetition
8. Saves results to CSV and JSON files

---

## Experiment Results

> **Methodology:** 10 repetitions × 10-fold stratified CV = 100 paired observations per dataset  
> **Date:** December 14, 2025

### Dataset: CM1 (498 samples, 9.84% defective)

| Metric | Baseline (PCA+MLP) | Proposed (AE+XGB) | p-value | Significant | Winner |
|--------|-------------------|-------------------|---------|-------------|--------|
| Precision | 0.2213 ± 0.0740 | 0.2143 ± 0.1341 | 0.5956 | No | — |
| Recall | 0.6790 ± 0.2167 | 0.3710 ± 0.2265 | 0.0000 | **Yes** | Baseline |
| F1-Score | 0.3300 ± 0.1018 | 0.2646 ± 0.1553 | 0.0008 | **Yes** | Baseline |
| AUC-ROC | 0.7714 ± 0.0916 | 0.7309 ± 0.0946 | 0.0015 | **Yes** | Baseline |
| **MCC** | **0.2694 ± 0.1398** | **0.1754 ± 0.1826** | **0.0002** | **Yes** | **Baseline** |
| Accuracy | 0.7281 ± 0.0658 | 0.8044 ± 0.0507 | 0.0000 | **Yes** | Proposed |

### Dataset: PC1 (1,109 samples, 6.94% defective)

| Metric | Baseline (PCA+MLP) | Proposed (AE+XGB) | p-value | Significant | Winner |
|--------|-------------------|-------------------|---------|-------------|--------|
| Precision | 0.2220 ± 0.0648 | 0.3282 ± 0.1031 | 0.0000 | **Yes** | Proposed |
| Recall | 0.6552 ± 0.1695 | 0.5180 ± 0.1682 | 0.0000 | **Yes** | Baseline |
| F1-Score | 0.3271 ± 0.0847 | 0.3942 ± 0.1123 | 0.0000 | **Yes** | Proposed |
| AUC-ROC | 0.8154 ± 0.0743 | 0.8565 ± 0.0669 | 0.0000 | **Yes** | Proposed |
| **MCC** | **0.3007 ± 0.1062** | **0.3533 ± 0.1262** | **0.0002** | **Yes** | **Proposed** |
| Accuracy | 0.8092 ± 0.0475 | 0.8905 ± 0.0253 | 0.0000 | **Yes** | Proposed |

### Dataset: PC2 (745 samples, 2.15% defective)

| Metric | Baseline (PCA+MLP) | Proposed (AE+XGB) | p-value | Significant | Winner |
|--------|-------------------|-------------------|---------|-------------|--------|
| Precision | 0.0817 ± 0.1289 | 0.0865 ± 0.1929 | 0.4269 | No | — |
| Recall | 0.2800 ± 0.3894 | 0.1850 ± 0.3291 | 0.0237 | **Yes** | Baseline |
| F1-Score | 0.1176 ± 0.1715 | 0.1020 ± 0.1824 | 0.4306 | No | — |
| AUC-ROC | 0.7519 ± 0.2227 | 0.7669 ± 0.1847 | 0.5296 | No | — |
| **MCC** | **0.1101 ± 0.2083** | **0.0873 ± 0.2130** | **0.6667** | No | — |
| Accuracy | 0.9110 ± 0.0396 | 0.9314 ± 0.0255 | 0.0000 | **Yes** | Proposed |

### Summary Table (Key Metrics)

| Dataset | Metric | Baseline | Proposed | p-value | Winner |
|---------|--------|----------|----------|---------|--------|
| CM1 | MCC | 0.2694 ± 0.14 | 0.1754 ± 0.18 | 0.0002* | Baseline |
| CM1 | F1 | 0.3300 ± 0.10 | 0.2646 ± 0.16 | 0.0008* | Baseline |
| CM1 | AUC-ROC | 0.7714 ± 0.09 | 0.7309 ± 0.09 | 0.0015* | Baseline |
| **PC1** | **MCC** | **0.3007 ± 0.11** | **0.3533 ± 0.13** | **0.0002*** | **Proposed** |
| **PC1** | **F1** | **0.3271 ± 0.08** | **0.3942 ± 0.11** | **0.0000*** | **Proposed** |
| **PC1** | **AUC-ROC** | **0.8154 ± 0.07** | **0.8565 ± 0.07** | **0.0000*** | **Proposed** |
| PC2 | MCC | 0.1101 ± 0.21 | 0.0873 ± 0.21 | 0.6667 | No diff |
| PC2 | F1 | 0.1176 ± 0.17 | 0.1020 ± 0.18 | 0.4306 | No diff |
| PC2 | AUC-ROC | 0.7519 ± 0.22 | 0.7669 ± 0.18 | 0.5296 | No diff |

\* statistically significant (p < 0.05)

---

## Results Discussion

### Key Findings

#### 1. Dataset-Dependent Performance

The relative performance of the two pipelines varies significantly by dataset, with **statistically significant differences** (p < 0.05) confirmed by Wilcoxon signed-rank tests over 100 paired observations:

- **CM1:** Baseline (PCA+MLP) **significantly outperforms** proposed (AE+XGB) across all key metrics (MCC: +0.09, F1: +0.07, AUC-ROC: +0.04)
- **PC1:** Proposed (AE+XGB) **significantly outperforms** baseline across all key metrics (MCC: +0.05, F1: +0.07, AUC-ROC: +0.04)
- **PC2:** No significant differences in MCC, F1, or AUC-ROC due to extreme class imbalance (only 16 defective samples)

#### 2. Matthews Correlation Coefficient (MCC) Analysis

MCC is the most reliable metric for imbalanced datasets as it accounts for all four confusion matrix values:

| Dataset | Baseline MCC | Proposed MCC | Difference | Interpretation |
|---------|-------------|--------------|------------|----------------|
| CM1 | 0.269 | 0.175 | -0.094 | Baseline superior |
| PC1 | 0.301 | 0.353 | +0.052 | **Proposed superior** |
| PC2 | 0.110 | 0.087 | -0.023 | No significant difference |

**Insight:** MCC values in the 0.25-0.35 range indicate moderate but meaningful predictive ability. Both pipelines struggle with PC2's extreme imbalance (MCC near 0.1).

#### 3. Precision-Recall Trade-off

A consistent pattern emerges across datasets:

- **Baseline (MLP):** Higher Recall (0.52-0.68), Lower Precision (0.22)
- **Proposed (XGBoost):** Higher Precision (0.21-0.33), Lower Recall (0.19-0.52)

This suggests:
- MLP is more aggressive in predicting defects (catches more bugs but generates more false alarms)
- XGBoost is more conservative, resulting in fewer false alarms but missing some defects

**For PC1, the proposed approach achieves a better balance**, with higher F1-score (0.394 vs 0.327).

#### 4. Impact of Class Imbalance

Performance degrades with increasing imbalance, but the pattern differs between pipelines:

| Dataset | Defect Rate | Best MCC | Best F1 | Winner |
|---------|-------------|----------|---------|--------|
| CM1 | 9.84% | 0.269 | 0.330 | Baseline |
| PC1 | 6.94% | 0.353 | 0.394 | **Proposed** |
| PC2 | 2.15% | 0.110 | 0.118 | Tie |

**Key Insight:** On the larger PC1 dataset (1,109 samples), the proposed Autoencoder+XGBoost pipeline shows its strength. The autoencoder may require more training data to learn effective representations.

#### 5. Accuracy vs MCC: The Imbalance Trap

Accuracy can be misleading for imbalanced datasets:

| Dataset | Baseline Acc | Proposed Acc | Baseline MCC | Proposed MCC |
|---------|-------------|--------------|--------------|--------------|
| PC2 | 91.1% | 93.1% | 0.110 | 0.087 |

Despite 93% accuracy, the proposed model has poor MCC (0.087) on PC2 — it's mostly just predicting "clean" for everything. **Always prioritize MCC over accuracy for SDP.**

#### 6. LIME Feature Importance

Top features identified by LIME across both pipelines:

| Dataset | Pipeline | Top Features |
|---------|----------|--------------|
| CM1 | Baseline | lOComment, locCodeAndComment, lOBlank |
| CM1 | Proposed | lOComment, d (difficulty), lOCode |
| PC1 | Baseline | I (content), ev(g), uniq_Opnd |
| PC1 | Proposed | locCodeAndComment, I, total_Op |
| PC2 | Baseline | PERCENT_COMMENTS, DESIGN_DENSITY, NORMALIZED_CYCLOMATIC_COMPLEXITY |
| PC2 | Proposed | PERCENT_COMMENTS, NORMALIZED_CYCLOMATIC_COMPLEXITY, HALSTEAD_DIFFICULTY |

**Alignment with domain knowledge:**
- Comment-related metrics (documentation quality) appear frequently
- Complexity metrics (cyclomatic, Halstead difficulty) are strong predictors
- Both pipelines agree on many top features, suggesting genuine signal

### Statistical Robustness

With 10 repetitions × 10-fold CV = 100 paired observations:
- Wilcoxon signed-rank tests have high statistical power
- Small p-values (< 0.001) for most significant findings
- Results are not dependent on a single random seed

### Limitations

1. **Extreme Imbalance:** PC2 (2.15% defective) remains challenging for both approaches
2. **Dataset Size Matters:** Proposed pipeline performs better on larger dataset (PC1)
3. **No Hyperparameter Tuning:** Fixed configurations may favor one pipeline on certain datasets
4. **Dataset-Specific Results:** No single pipeline dominates across all datasets

### Recommendations

1. **Use PC1-style datasets:** The proposed Autoencoder+XGBoost pipeline shows promise on larger, moderately imbalanced datasets
2. **Prioritize MCC:** Always evaluate with MCC, not accuracy, for imbalanced SDP
3. **Consider ensemble:** Combine both approaches for robust predictions
4. **Address extreme imbalance:** Explore cost-sensitive learning or advanced sampling for PC2-like scenarios

---

## How to Run

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full experiment (all datasets, 10 repetitions × 10-fold CV)
python main.py --cv

# Run quick test (1 repetition, single dataset)
python main.py --datasets CM1 --repetitions 1

# Run with holdout split (70/30)
python main.py --holdout
```

### Command-Line Options

```bash
python main.py [OPTIONS]

Options:
  --datasets      Datasets to evaluate (default: CM1 PC1 PC2)
  --folds         Number of CV folds (default: 10)
  --repetitions   Number of experimental repetitions (default: 10)
  --cv            Use k-fold cross-validation
  --holdout       Use 70/30 holdout split
  --verbose       Print detailed progress
  --quiet         Minimal output
  --no-save       Don't save results to files
```

### Example Runs

```bash
# Full robust evaluation (as in thesis)
python main.py --datasets CM1 PC1 PC2 --cv --repetitions 10

# Quick sanity check
python main.py --datasets CM1 --cv --repetitions 1

# Single holdout split (faster but less robust)
python main.py --holdout --repetitions 5
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

This implementation provides a rigorous comparison between the baseline PCA+MLP pipeline and the proposed Autoencoder+XGBoost approach using 100 paired observations (10 repetitions × 10-fold CV) for statistically robust results.

### Key Conclusions

| Finding | Evidence |
|---------|----------|
| **Dataset-dependent winner** | Baseline wins on CM1; Proposed wins on PC1 |
| **Proposed pipeline excels on larger datasets** | PC1 (1,109 samples): MCC +0.05, F1 +0.07 improvement |
| **MCC is essential for imbalanced SDP** | Accuracy misleading (93% with poor MCC on PC2) |
| **Extreme imbalance unsolved** | Both struggle with PC2 (2.15% defective) |
| **Statistical significance confirmed** | p < 0.001 for most key comparisons |

### Practical Recommendations

1. **For larger datasets (1000+ samples):** Prefer Autoencoder + XGBoost
2. **For smaller datasets (<500 samples):** Prefer PCA + MLP
3. **Always evaluate with MCC:** Not accuracy, for imbalanced classification
4. **Use multiple repetitions:** Single runs can be misleading

The implementation provides a solid foundation for further experimentation with different configurations, datasets, and evaluation scenarios. All results are fully reproducible with the provided random seeds.

