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
# Train/Test Split Configuration (Paper uses 70/30 holdout)
# =============================================================================
CV_FOLDS = 10  # Keep for optional CV mode
CV_SHUFFLE = True
TEST_SIZE = 0.30  # 70% train, 30% test as per paper
USE_HOLDOUT_SPLIT = True  # True = paper's 70/30 split, False = 10-fold CV

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
# MLP Configuration (Baseline Pipeline) - Matching SPAM-XAI paper parameters
# =============================================================================
MLP_HIDDEN_LAYERS = (100, 50)
MLP_ACTIVATION = 'relu'  # Paper uses relu (though mentions sigmoid for output)
MLP_SOLVER = 'adam'
MLP_ALPHA = 0.001  # Paper specifies 0.001 (was 0.0001)
MLP_LEARNING_RATE = 'constant'  # Paper uses constant learning rate
MLP_LEARNING_RATE_INIT = 0.001  # Paper specifies 0.001
MLP_MAX_ITER = 200  # Paper specifies 200 (was 500)
MLP_MOMENTUM = 0.9  # Paper specifies 0.9
MLP_EPSILON = 1e-08  # Paper specifies 1e-08
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
# Evaluation Metrics
# =============================================================================
METRICS = ['precision', 'recall', 'f1', 'auc_roc', 'mcc', 'accuracy']

# =============================================================================
# Experimental Runs (Multiple repetitions for statistical robustness)
# =============================================================================
N_REPETITIONS = 10  # Run entire CV procedure 10 times with different seeds

# =============================================================================
# Statistical Testing
# =============================================================================
SIGNIFICANCE_LEVEL = 0.05

