"""
Preprocessing pipeline for Software Defect Prediction.
Includes missing value imputation, feature scaling, SMOTE, and cross-validation.
"""

import numpy as np
from typing import Tuple, Generator, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from .config import (
    RANDOM_SEED, CV_FOLDS, CV_SHUFFLE,
    SMOTE_K_NEIGHBORS, SMOTE_SAMPLING_STRATEGY
)


class SMOTE:
    """
    Synthetic Minority Over-sampling Technique (SMOTE) implementation.
    
    Creates synthetic samples for the minority class by interpolating
    between existing minority samples and their nearest neighbors.
    
    This is a custom implementation to avoid dependency issues with 
    imbalanced-learn on newer Python/sklearn versions.
    """
    
    def __init__(self, 
                 k_neighbors: int = 5, 
                 sampling_strategy: str = 'minority',
                 random_state: Optional[int] = None):
        """
        Initialize SMOTE.
        
        Args:
            k_neighbors: Number of nearest neighbors to use for synthesis
            sampling_strategy: 'minority' to balance classes, or float ratio
            random_state: Random seed for reproducibility
        """
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample the dataset by over-sampling the minority class.
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)
            
        Returns:
            X_resampled, y_resampled: Resampled arrays
        """
        np.random.seed(self.random_state)
        
        # Identify minority and majority classes
        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]
        
        n_minority = counts.min()
        n_majority = counts.max()
        
        # Get minority samples
        minority_mask = y == minority_class
        X_minority = X[minority_mask]
        
        # Calculate how many synthetic samples to generate
        n_synthetic = n_majority - n_minority
        
        if n_synthetic <= 0:
            return X.copy(), y.copy()
        
        # Fit nearest neighbors on minority samples
        k = min(self.k_neighbors, len(X_minority) - 1)
        if k < 1:
            return X.copy(), y.copy()
            
        nn = NearestNeighbors(n_neighbors=k + 1)  # +1 because sample is its own neighbor
        nn.fit(X_minority)
        
        # Generate synthetic samples
        synthetic_samples = []
        
        for _ in range(n_synthetic):
            # Randomly select a minority sample
            idx = np.random.randint(0, len(X_minority))
            sample = X_minority[idx]
            
            # Find k nearest neighbors
            distances, indices = nn.kneighbors([sample])
            # Skip first neighbor (the sample itself)
            neighbor_indices = indices[0][1:]
            
            # Randomly select one of the neighbors
            nn_idx = np.random.choice(neighbor_indices)
            neighbor = X_minority[nn_idx]
            
            # Generate synthetic sample by interpolation
            alpha = np.random.random()
            synthetic = sample + alpha * (neighbor - sample)
            synthetic_samples.append(synthetic)
        
        # Combine original and synthetic samples
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.full(n_synthetic, minority_class)
        
        X_resampled = np.vstack([X, X_synthetic])
        y_resampled = np.concatenate([y, y_synthetic])
        
        # Shuffle the result
        shuffle_idx = np.random.permutation(len(X_resampled))
        
        return X_resampled[shuffle_idx], y_resampled[shuffle_idx]


class Preprocessor:
    """
    Complete preprocessing pipeline for software defect prediction.
    Handles imputation, scaling, and provides train-test splitting.
    """
    
    def __init__(self, random_state: int = RANDOM_SEED):
        """Initialize the preprocessor with configured settings."""
        self.random_state = random_state
        self.imputer = None
        self.scaler = None
        self._is_fitted = False
        
    def fit(self, X: np.ndarray) -> 'Preprocessor':
        """
        Fit the imputer and scaler on training data.
        
        Args:
            X: Training feature array
            
        Returns:
            self
        """
        # Median imputation for missing values
        self.imputer = SimpleImputer(strategy='median')
        X_imputed = self.imputer.fit_transform(X)
        
        # Standard scaling (z-score normalization)
        self.scaler = StandardScaler()
        self.scaler.fit(X_imputed)
        
        self._is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply imputation and scaling to data.
        
        Args:
            X: Feature array to transform
            
        Returns:
            Transformed feature array
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        return X_scaled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Feature array
            
        Returns:
            Transformed feature array
        """
        self.fit(X)
        return self.transform(X)


def get_cv_splits(X: np.ndarray, 
                  y: np.ndarray, 
                  n_splits: int = CV_FOLDS,
                  shuffle: bool = CV_SHUFFLE,
                  random_state: int = RANDOM_SEED) -> Generator:
    """
    Generate stratified k-fold cross-validation splits.
    
    Args:
        X: Feature array
        y: Label array
        n_splits: Number of folds
        shuffle: Whether to shuffle before splitting
        random_state: Random seed
        
    Yields:
        (train_idx, test_idx) tuples for each fold
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    for train_idx, test_idx in cv.split(X, y):
        yield train_idx, test_idx


def preprocess_fold(X_train: np.ndarray, 
                    X_test: np.ndarray, 
                    y_train: np.ndarray,
                    apply_smote: bool = True,
                    random_state: int = RANDOM_SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess a single fold: impute, scale, and optionally apply SMOTE.
    
    Important: Imputer and scaler are fit ONLY on training data.
    SMOTE is applied ONLY to training data.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        apply_smote: Whether to apply SMOTE to balance classes
        random_state: Random seed
        
    Returns:
        X_train_processed, X_test_processed, y_train_processed
    """
    # Initialize and fit preprocessor on training data
    preprocessor = Preprocessor(random_state=random_state)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    y_train_processed = y_train.copy()
    
    # Apply SMOTE to balance training data
    if apply_smote:
        smote = SMOTE(
            k_neighbors=SMOTE_K_NEIGHBORS,
            sampling_strategy=SMOTE_SAMPLING_STRATEGY,
            random_state=random_state
        )
        X_train_processed, y_train_processed = smote.fit_resample(
            X_train_processed, y_train
        )
    
    return X_train_processed, X_test_processed, y_train_processed


if __name__ == "__main__":
    # Test preprocessing pipeline
    from .data_loader import load_dataset, get_dataset_info
    
    # Load CM1 dataset for testing
    X, y, feature_names = load_dataset('CM1')
    info = get_dataset_info(X, y, feature_names, 'CM1')
    
    print("Original dataset:")
    print(f"  Samples: {info['n_samples']}, Features: {info['n_features']}")
    print(f"  Defective: {info['n_defective']}, Clean: {info['n_clean']}")
    print(f"  Missing values: {info['n_missing']}")
    
    # Test with one fold
    for fold_idx, (train_idx, test_idx) in enumerate(get_cv_splits(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train: {len(train_idx)} samples, Defective: {sum(y_train)}")
        print(f"  Test: {len(test_idx)} samples, Defective: {sum(y_test)}")
        
        # Apply preprocessing with SMOTE
        X_train_proc, X_test_proc, y_train_proc = preprocess_fold(
            X_train, X_test, y_train, apply_smote=True
        )
        
        print(f"  After SMOTE: {len(y_train_proc)} samples, Defective: {sum(y_train_proc)}")
        print(f"  Feature range check - Train mean: {X_train_proc.mean():.4f}, "
              f"std: {X_train_proc.std():.4f}")
        
        # Only test first fold
        break
    
    print("\nPreprocessing pipeline test completed successfully!")

