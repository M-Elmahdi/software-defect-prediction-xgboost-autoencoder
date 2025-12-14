"""
Pipeline B: Proposed Pipeline
Implements SMOTE + Autoencoder + XGBoost + LIME for software defect prediction.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import xgboost as xgb

from .autoencoder import Autoencoder
from .config import (
    RANDOM_SEED,
    AE_LATENT_DIM,
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
    XGB_SUBSAMPLE,
    XGB_COLSAMPLE_BYTREE,
    XGB_OBJECTIVE,
    XGB_EVAL_METRIC
)


class ProposedPipeline:
    """
    Proposed Pipeline: Autoencoder + XGBoost
    
    This pipeline performs:
    1. Autoencoder for non-linear dimensionality reduction
    2. XGBoost for classification
    
    Note: SMOTE should be applied before this pipeline in the preprocessing step.
    """
    
    def __init__(self, 
                 latent_dim: int = AE_LATENT_DIM,
                 random_state: int = RANDOM_SEED,
                 verbose: bool = False):
        """
        Initialize the proposed pipeline.
        
        Args:
            latent_dim: Dimensionality of autoencoder latent space
            random_state: Random seed for reproducibility
            verbose: Whether to print training progress
        """
        self.latent_dim = latent_dim
        self.random_state = random_state
        self.verbose = verbose
        
        # Autoencoder for dimensionality reduction
        self.autoencoder = Autoencoder(
            latent_dim=latent_dim,
            random_state=random_state,
            verbose=verbose
        )
        
        # XGBoost classifier
        self.xgb_classifier = xgb.XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE_BYTREE,
            objective=XGB_OBJECTIVE,
            eval_metric=XGB_EVAL_METRIC,
            use_label_encoder=False,
            random_state=random_state,
            verbosity=0
        )
        
        self._is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ProposedPipeline':
        """
        Fit the pipeline on training data.
        
        Args:
            X: Training features (already preprocessed/SMOTE applied)
            y: Training labels
            
        Returns:
            self
        """
        # Fit autoencoder and transform data
        X_encoded = self.autoencoder.fit_transform(X)
        
        if self.verbose:
            print(f"Autoencoder: {X.shape[1]} -> {X_encoded.shape[1]} dimensions")
            print(f"Reconstruction error: {self.autoencoder.get_reconstruction_error(X):.6f}")
        
        # Fit XGBoost on encoded features
        self.xgb_classifier.fit(X_encoded, y)
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted class labels
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before prediction")
        
        X_encoded = self.autoencoder.transform(X)
        return self.xgb_classifier.predict(X_encoded)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted probabilities for each class
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before prediction")
        
        X_encoded = self.autoencoder.transform(X)
        return self.xgb_classifier.predict_proba(X_encoded)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply autoencoder transformation (for LIME compatibility).
        
        Args:
            X: Features to transform
            
        Returns:
            Encoded features
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        
        return self.autoencoder.transform(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get XGBoost feature importance scores.
        
        Returns:
            Feature importance array (for latent features)
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted first")
        return self.xgb_classifier.feature_importances_
    
    def get_info(self) -> Dict:
        """Get pipeline information."""
        info = {
            'name': 'Proposed (Autoencoder + XGBoost)',
            'dim_reduction': 'Autoencoder',
            'classifier': 'XGBoost',
            'latent_dim': self.latent_dim,
            'xgb_n_estimators': XGB_N_ESTIMATORS,
            'xgb_max_depth': XGB_MAX_DEPTH,
        }
        
        if self._is_fitted:
            info['reconstruction_error'] = self.autoencoder.get_reconstruction_error
        
        return info


def create_proposed_pipeline(latent_dim: int = AE_LATENT_DIM,
                             random_state: int = RANDOM_SEED,
                             verbose: bool = False) -> ProposedPipeline:
    """
    Factory function to create a proposed pipeline.
    
    Args:
        latent_dim: Dimensionality of autoencoder latent space
        random_state: Random seed for reproducibility
        verbose: Whether to print training progress
        
    Returns:
        Configured ProposedPipeline instance
    """
    return ProposedPipeline(
        latent_dim=latent_dim,
        random_state=random_state,
        verbose=verbose
    )


if __name__ == "__main__":
    # Test proposed pipeline
    from .data_loader import load_dataset
    from .preprocessing import preprocess_fold, get_cv_splits
    
    # Load dataset
    X, y, feature_names = load_dataset('CM1')
    print(f"Loaded CM1: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test with first fold
    for fold_idx, (train_idx, test_idx) in enumerate(get_cv_splits(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Preprocess
        X_train_proc, X_test_proc, y_train_proc = preprocess_fold(
            X_train, X_test, y_train, apply_smote=True
        )
        
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Training samples (after SMOTE): {len(y_train_proc)}")
        
        # Create and fit pipeline
        pipeline = create_proposed_pipeline(verbose=True)
        pipeline.fit(X_train_proc, y_train_proc)
        
        # Get predictions
        y_pred = pipeline.predict(X_test_proc)
        y_pred_proba = pipeline.predict_proba(X_test_proc)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        
        info = pipeline.get_info()
        print(f"  Latent dim: {info['latent_dim']}")
        print(f"  Test accuracy: {accuracy:.4f}")
        print(f"  Predictions shape: {y_pred.shape}")
        print(f"  Probabilities shape: {y_pred_proba.shape}")
        print(f"  Feature importance shape: {pipeline.get_feature_importance().shape}")
        
        # Only test first fold
        break
    
    print("\nProposed pipeline test completed successfully!")

