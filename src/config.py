"""
Configuration file for Software Defect Prediction Pipeline.
Contains all hyperparameters, constants, and settings from the methodology.
"""

import os

# =============================================================================
# Random Seed for Reproducibility
# =============================================================================
RANDOM_SEED = 42

# =============================================================================
# Dataset Configuration
# =============================================================================
DATASETS = ['CM1', 'PC1', 'PC2']
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'promise_datasets')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

# =============================================================================
# Cross-Validation Configuration
# =============================================================================
CV_FOLDS = 10
CV_SHUFFLE = True

# =============================================================================
# SMOTE Configuration
# =============================================================================
SMOTE_K_NEIGHBORS = 5
SMOTE_SAMPLING_STRATEGY = 'minority'

# =============================================================================
# PCA Configuration (Baseline Pipeline)
# =============================================================================
PCA_N_COMPONENTS = 0.95  # Retain 95% of variance

# =============================================================================
# MLP Configuration (Baseline Pipeline)
# =============================================================================
MLP_HIDDEN_LAYERS = (100, 50)
MLP_ACTIVATION = 'relu'
MLP_SOLVER = 'adam'
MLP_ALPHA = 0.0001
MLP_LEARNING_RATE = 'adaptive'
MLP_MAX_ITER = 500
MLP_EARLY_STOPPING = True
MLP_VALIDATION_FRACTION = 0.1

# =============================================================================
# Autoencoder Configuration (Proposed Pipeline)
# =============================================================================
AE_ENCODER_LAYERS = [64, 32]
AE_LATENT_DIM = 10
AE_DECODER_LAYERS = [32, 64]
AE_ACTIVATION = 'relu'
AE_OUTPUT_ACTIVATION = 'linear'
AE_OPTIMIZER_LR = 0.001
AE_LOSS = 'mse'
AE_EPOCHS = 100
AE_BATCH_SIZE = 32
AE_VALIDATION_SPLIT = 0.1
AE_EARLY_STOPPING_PATIENCE = 10

# =============================================================================
# XGBoost Configuration (Proposed Pipeline)
# =============================================================================
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_OBJECTIVE = 'binary:logistic'
XGB_EVAL_METRIC = 'logloss'

# =============================================================================
# LIME Configuration
# =============================================================================
LIME_NUM_FEATURES = 10
LIME_NUM_SAMPLES = 5000
LIME_DISCRETIZE_CONTINUOUS = True

# =============================================================================
# Statistical Testing
# =============================================================================
SIGNIFICANCE_LEVEL = 0.05

