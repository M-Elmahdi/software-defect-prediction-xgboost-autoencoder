"""
Pipeline A: Baseline SPAM-XAI Pipeline
Implements SMOTE + PCA + MLP + LIME for software defect prediction.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

from .config import (
    RANDOM_SEED,
    PCA_N_COMPONENTS,
    MLP_HIDDEN_LAYERS,
    MLP_ACTIVATION,
    MLP_SOLVER,
    MLP_ALPHA,
    MLP_LEARNING_RATE,
    MLP_LEARNING_RATE_INIT,
    MLP_MAX_ITER,
    MLP_MOMENTUM,
    MLP_EPSILON,
    MLP_EARLY_STOPPING,
    MLP_VALIDATION_FRACTION
)


class BaselinePipeline:
    """
    Baseline SPAM-XAI Pipeline: PCA + MLP
    
    This pipeline performs:
    1. PCA for dimensionality reduction (retaining 95% variance)
    2. MLP for classification
    
    Note: SMOTE should be applied before this pipeline in the preprocessing step.
    """
    
    def __init__(self, random_state: int = RANDOM_SEED):
        """
        Initialize the baseline pipeline.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        
        # PCA for dimensionality reduction
        self.pca = PCA(n_components=PCA_N_COMPONENTS, random_state=random_state)
        
        # MLP classifier - configured to match SPAM-XAI paper parameters
        self.mlp = MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN_LAYERS,
            activation=MLP_ACTIVATION,
            solver=MLP_SOLVER,
            alpha=MLP_ALPHA,
            learning_rate=MLP_LEARNING_RATE,
            learning_rate_init=MLP_LEARNING_RATE_INIT,
            max_iter=MLP_MAX_ITER,
            momentum=MLP_MOMENTUM,
            epsilon=MLP_EPSILON,
            early_stopping=MLP_EARLY_STOPPING,
            validation_fraction=MLP_VALIDATION_FRACTION,
            random_state=random_state
        )
        
        self._is_fitted = False
        self.n_components_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaselinePipeline':
        """
        Fit the pipeline on training data.
        
        Args:
            X: Training features (already preprocessed/SMOTE applied)
            y: Training labels
            
        Returns:
            self
        """
        # Apply PCA
        X_pca = self.pca.fit_transform(X)
        self.n_components_ = X_pca.shape[1]
        
        # Fit MLP
        self.mlp.fit(X_pca, y)
        
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
        
        X_pca = self.pca.transform(X)
        return self.mlp.predict(X_pca)
    
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
        
        X_pca = self.pca.transform(X)
        return self.mlp.predict_proba(X_pca)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply PCA transformation (for LIME compatibility).
        
        Args:
            X: Features to transform
            
        Returns:
            PCA-transformed features
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")
        
        return self.pca.transform(X)
    
    def get_pca_components(self) -> np.ndarray:
        """Get PCA component weights for interpretability."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted first")
        return self.pca.components_
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get variance explained by each PCA component."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted first")
        return self.pca.explained_variance_ratio_
    
    def get_info(self) -> Dict:
        """Get pipeline information."""
        info = {
            'name': 'Baseline (SPAM-XAI)',
            'dim_reduction': 'PCA',
            'classifier': 'MLP',
            'pca_variance_threshold': PCA_N_COMPONENTS,
            'mlp_layers': MLP_HIDDEN_LAYERS,
        }
        
        if self._is_fitted:
            info['n_pca_components'] = self.n_components_
            info['total_variance_explained'] = sum(self.pca.explained_variance_ratio_)
        
        return info


def create_baseline_pipeline(random_state: int = RANDOM_SEED) -> BaselinePipeline:
    """
    Factory function to create a baseline pipeline.
    
    Args:
        random_state: Random seed for reproducibility
        
    Returns:
        Configured BaselinePipeline instance
    """
    return BaselinePipeline(random_state=random_state)


if __name__ == "__main__":
    # Test baseline pipeline
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
        pipeline = create_baseline_pipeline()
        pipeline.fit(X_train_proc, y_train_proc)
        
        # Get predictions
        y_pred = pipeline.predict(X_test_proc)
        y_pred_proba = pipeline.predict_proba(X_test_proc)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        
        info = pipeline.get_info()
        print(f"  PCA components: {info['n_pca_components']}")
        print(f"  Variance explained: {info['total_variance_explained']:.4f}")
        print(f"  Test accuracy: {accuracy:.4f}")
        print(f"  Predictions shape: {y_pred.shape}")
        print(f"  Probabilities shape: {y_pred_proba.shape}")
        
        # Only test first fold
        break
    
    print("\nBaseline pipeline test completed successfully!")

