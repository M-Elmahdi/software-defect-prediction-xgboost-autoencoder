"""
Data loader for NASA PROMISE repository ARFF files.
Handles parsing of ARFF format and extraction of features/labels.
"""

import os
import numpy as np
import pandas as pd
import arff
from typing import Tuple, Dict, List, Optional

from .config import DATA_DIR, DATASETS


def load_arff_file(filepath: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load an ARFF file and return features, labels, and feature names.
    
    Args:
        filepath: Path to the ARFF file
        
    Returns:
        Tuple of (X, y, feature_names) where:
        - X: numpy array of features (n_samples, n_features)
        - y: numpy array of binary labels (n_samples,) where 1=defective, 0=clean
        - feature_names: list of feature names
    """
    with open(filepath, 'r') as f:
        dataset = arff.load(f)
    
    # Extract data and attributes
    data = dataset['data']
    attributes = dataset['attributes']
    
    # Get feature names (all except the last 'class' attribute)
    feature_names = [attr[0] for attr in attributes[:-1]]
    
    # Convert to numpy arrays
    n_samples = len(data)
    n_features = len(feature_names)
    
    X = np.zeros((n_samples, n_features), dtype=np.float64)
    y = np.zeros(n_samples, dtype=np.int32)
    
    for i, row in enumerate(data):
        # Features are all columns except the last one
        for j in range(n_features):
            val = row[j]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                X[i, j] = np.nan
            else:
                X[i, j] = float(val)
        
        # Label is the last column
        label = row[-1]
        # Handle different label formats: {false,true} or {N,Y}
        if isinstance(label, str):
            label_lower = label.lower()
            if label_lower in ['true', 'y', 'yes', '1', 'defective']:
                y[i] = 1
            else:
                y[i] = 0
        elif isinstance(label, bool):
            y[i] = 1 if label else 0
        else:
            y[i] = int(label) if label else 0
    
    return X, y, feature_names


def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load a dataset by name from the promise_datasets directory.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'CM1', 'PC1', 'PC2')
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    filepath = os.path.join(DATA_DIR, f"{dataset_name}.arff")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    return load_arff_file(filepath)


def load_all_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    Load all configured datasets.
    
    Returns:
        Dictionary mapping dataset names to (X, y, feature_names) tuples
    """
    datasets = {}
    for name in DATASETS:
        try:
            datasets[name] = load_dataset(name)
            print(f"Loaded {name}: {datasets[name][0].shape[0]} samples, "
                  f"{datasets[name][0].shape[1]} features")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
    return datasets


def get_dataset_info(X: np.ndarray, y: np.ndarray, 
                     feature_names: List[str], 
                     dataset_name: str = "Dataset") -> Dict:
    """
    Get summary statistics for a dataset.
    
    Args:
        X: Feature array
        y: Label array  
        feature_names: List of feature names
        dataset_name: Name for display
        
    Returns:
        Dictionary with dataset statistics
    """
    n_samples, n_features = X.shape
    n_defective = np.sum(y == 1)
    n_clean = np.sum(y == 0)
    defect_rate = n_defective / n_samples * 100
    n_missing = np.sum(np.isnan(X))
    
    info = {
        'name': dataset_name,
        'n_samples': n_samples,
        'n_features': n_features,
        'n_defective': n_defective,
        'n_clean': n_clean,
        'defect_rate': defect_rate,
        'n_missing': n_missing,
        'feature_names': feature_names
    }
    
    return info


def print_dataset_summary(datasets: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]):
    """Print a summary table of all loaded datasets."""
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"{'Dataset':<10} {'Samples':>10} {'Features':>10} {'Defective':>10} "
          f"{'Clean':>10} {'Defect %':>10}")
    print("-"*70)
    
    for name, (X, y, feature_names) in datasets.items():
        info = get_dataset_info(X, y, feature_names, name)
        print(f"{info['name']:<10} {info['n_samples']:>10} {info['n_features']:>10} "
              f"{info['n_defective']:>10} {info['n_clean']:>10} "
              f"{info['defect_rate']:>9.2f}%")
    
    print("="*70)


if __name__ == "__main__":
    # Test data loading
    datasets = load_all_datasets()
    print_dataset_summary(datasets)

