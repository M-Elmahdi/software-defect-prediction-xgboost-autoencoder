"""
Explainability module using LIME for Software Defect Prediction.
Provides local interpretable explanations for both pipelines.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from lime.lime_tabular import LimeTabularExplainer

from .config import (
    LIME_NUM_FEATURES,
    LIME_NUM_SAMPLES,
    LIME_DISCRETIZE_CONTINUOUS
)


class LIMEExplainer:
    """
    LIME explainer wrapper for software defect prediction.
    
    Provides consistent interface for explaining predictions from
    both baseline and proposed pipelines.
    """
    
    def __init__(self,
                 training_data: np.ndarray,
                 feature_names: List[str],
                 class_names: List[str] = None,
                 num_features: int = LIME_NUM_FEATURES,
                 num_samples: int = LIME_NUM_SAMPLES,
                 discretize_continuous: bool = LIME_DISCRETIZE_CONTINUOUS,
                 random_state: int = None):
        """
        Initialize the LIME explainer.
        
        Args:
            training_data: Training data for computing statistics
            feature_names: List of feature names
            class_names: List of class names (default: ['Clean', 'Defective'])
            num_features: Number of top features to include in explanations
            num_samples: Number of perturbation samples
            discretize_continuous: Whether to discretize continuous features
            random_state: Random seed for reproducibility
        """
        self.feature_names = feature_names
        self.class_names = class_names or ['Clean', 'Defective']
        self.num_features = num_features
        self.num_samples = num_samples
        
        # Create LIME explainer
        self.explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=self.class_names,
            mode='classification',
            discretize_continuous=discretize_continuous,
            random_state=random_state
        )
    
    def explain_instance(self,
                        instance: np.ndarray,
                        predict_fn: Callable,
                        num_features: int = None) -> Dict:
        """
        Generate explanation for a single instance.
        
        Args:
            instance: Single data instance to explain
            predict_fn: Model's predict_proba function
            num_features: Number of features in explanation (overrides default)
            
        Returns:
            Dictionary containing explanation details
        """
        num_features = num_features or self.num_features
        
        # Get LIME explanation
        exp = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=self.num_samples
        )
        
        # Extract explanation components
        explanation = {
            'feature_weights': exp.as_list(),
            'prediction': exp.predict_proba,
            'local_prediction': exp.local_pred,
            'score': exp.score if hasattr(exp, 'score') else None
        }
        
        return explanation
    
    def explain_batch(self,
                     instances: np.ndarray,
                     predict_fn: Callable,
                     num_features: int = None,
                     verbose: bool = False) -> List[Dict]:
        """
        Generate explanations for multiple instances.
        
        Args:
            instances: Multiple data instances to explain
            predict_fn: Model's predict_proba function
            num_features: Number of features in explanation
            verbose: Whether to print progress
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        
        for i, instance in enumerate(instances):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Explained {i + 1}/{len(instances)} instances")
            
            exp = self.explain_instance(instance, predict_fn, num_features)
            explanations.append(exp)
        
        return explanations
    
    def get_feature_importance_summary(self,
                                       explanations: List[Dict]) -> Dict[str, Dict]:
        """
        Aggregate feature importance across multiple explanations.
        
        Args:
            explanations: List of explanation dictionaries
            
        Returns:
            Dictionary mapping feature names to importance statistics
        """
        # Collect all feature weights
        feature_weights = {}
        
        for exp in explanations:
            for feature, weight in exp['feature_weights']:
                # Extract base feature name (remove discretization info)
                base_feature = feature.split()[0]
                
                if base_feature not in feature_weights:
                    feature_weights[base_feature] = []
                feature_weights[base_feature].append(abs(weight))
        
        # Compute statistics
        summary = {}
        for feature, weights in feature_weights.items():
            summary[feature] = {
                'mean_importance': np.mean(weights),
                'std_importance': np.std(weights),
                'frequency': len(weights) / len(explanations),
                'max_importance': np.max(weights),
                'min_importance': np.min(weights)
            }
        
        # Sort by mean importance
        summary = dict(sorted(
            summary.items(), 
            key=lambda x: x[1]['mean_importance'], 
            reverse=True
        ))
        
        return summary


def explain_predictions(pipeline,
                        X_train: np.ndarray,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        feature_names: List[str],
                        n_samples: int = 10,
                        random_state: int = None) -> Dict:
    """
    Generate LIME explanations for sample predictions.
    
    Args:
        pipeline: Fitted pipeline with predict_proba method
        X_train: Training data for LIME statistics
        X_test: Test data to explain
        y_test: True labels for test data
        feature_names: List of feature names
        n_samples: Number of samples to explain
        random_state: Random seed
        
    Returns:
        Dictionary containing explanations and summary
    """
    np.random.seed(random_state)
    
    # Get predictions
    y_pred = pipeline.predict(X_test)
    
    # Select samples to explain
    # Include correct defective, correct clean, and misclassified samples
    correct_defective = np.where((y_test == 1) & (y_pred == 1))[0]
    correct_clean = np.where((y_test == 0) & (y_pred == 0))[0]
    misclassified = np.where(y_test != y_pred)[0]
    
    # Select up to n_samples/3 from each category
    n_each = max(1, n_samples // 3)
    
    selected = []
    categories = []
    
    if len(correct_defective) > 0:
        idx = np.random.choice(correct_defective, 
                               min(n_each, len(correct_defective)), 
                               replace=False)
        selected.extend(idx)
        categories.extend(['TP'] * len(idx))
    
    if len(correct_clean) > 0:
        idx = np.random.choice(correct_clean,
                               min(n_each, len(correct_clean)),
                               replace=False)
        selected.extend(idx)
        categories.extend(['TN'] * len(idx))
    
    if len(misclassified) > 0:
        idx = np.random.choice(misclassified,
                               min(n_each, len(misclassified)),
                               replace=False)
        selected.extend(idx)
        cats = ['FP' if y_test[i] == 0 else 'FN' for i in idx]
        categories.extend(cats)
    
    if not selected:
        # Fallback: random selection
        selected = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
        categories = ['random'] * len(selected)
    
    # Create explainer
    explainer = LIMEExplainer(
        training_data=X_train,
        feature_names=feature_names,
        random_state=random_state
    )
    
    # Generate explanations
    explanations = []
    for idx, cat in zip(selected, categories):
        exp = explainer.explain_instance(X_test[idx], pipeline.predict_proba)
        exp['category'] = cat
        exp['true_label'] = int(y_test[idx])
        exp['predicted_label'] = int(y_pred[idx])
        exp['instance_idx'] = int(idx)
        explanations.append(exp)
    
    # Get feature importance summary
    importance_summary = explainer.get_feature_importance_summary(explanations)
    
    return {
        'explanations': explanations,
        'importance_summary': importance_summary,
        'n_explained': len(explanations)
    }


def format_explanation(exp: Dict, feature_names: List[str] = None) -> str:
    """
    Format a single explanation as readable text.
    
    Args:
        exp: Explanation dictionary
        feature_names: Original feature names
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append(f"Instance {exp.get('instance_idx', '?')} "
                 f"[{exp.get('category', '?')}]")
    lines.append(f"  True: {'Defective' if exp['true_label'] == 1 else 'Clean'}, "
                 f"Predicted: {'Defective' if exp['predicted_label'] == 1 else 'Clean'}")
    lines.append("  Top contributing features:")
    
    for feature, weight in exp['feature_weights'][:5]:
        direction = "+" if weight > 0 else "-"
        lines.append(f"    {direction} {feature}: {weight:.4f}")
    
    return "\n".join(lines)


def format_importance_summary(summary: Dict[str, Dict], top_n: int = 10) -> str:
    """
    Format feature importance summary as readable text.
    
    Args:
        summary: Feature importance summary dictionary
        top_n: Number of top features to display
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("Feature Importance Summary (Top {} Features)".format(top_n))
    lines.append("-" * 60)
    lines.append(f"{'Feature':<25} {'Mean Imp.':>12} {'Frequency':>12}")
    lines.append("-" * 60)
    
    for i, (feature, stats) in enumerate(list(summary.items())[:top_n]):
        lines.append(
            f"{feature:<25} {stats['mean_importance']:>12.4f} "
            f"{stats['frequency']*100:>11.1f}%"
        )
    
    lines.append("-" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    # Test explainability module
    from .data_loader import load_dataset
    from .preprocessing import preprocess_fold, get_cv_splits
    from .pipeline_baseline import create_baseline_pipeline
    from .pipeline_proposed import create_proposed_pipeline
    
    # Load dataset
    X, y, feature_names = load_dataset('CM1')
    print(f"Loaded CM1: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Get first fold
    for train_idx, test_idx in get_cv_splits(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Preprocess
        X_train_proc, X_test_proc, y_train_proc = preprocess_fold(
            X_train, X_test, y_train, apply_smote=True
        )
        
        # Test with baseline pipeline
        print("\nTesting LIME with Baseline Pipeline...")
        baseline = create_baseline_pipeline()
        baseline.fit(X_train_proc, y_train_proc)
        
        results_baseline = explain_predictions(
            pipeline=baseline,
            X_train=X_train_proc,
            X_test=X_test_proc,
            y_test=y_test,
            feature_names=feature_names,
            n_samples=6,
            random_state=42
        )
        
        print(f"  Generated {results_baseline['n_explained']} explanations")
        
        # Print sample explanation
        if results_baseline['explanations']:
            print("\nSample explanation (Baseline):")
            print(format_explanation(results_baseline['explanations'][0]))
        
        # Print importance summary
        print("\n" + format_importance_summary(results_baseline['importance_summary']))
        
        # Test with proposed pipeline
        print("\n\nTesting LIME with Proposed Pipeline...")
        proposed = create_proposed_pipeline(verbose=False)
        proposed.fit(X_train_proc, y_train_proc)
        
        results_proposed = explain_predictions(
            pipeline=proposed,
            X_train=X_train_proc,
            X_test=X_test_proc,
            y_test=y_test,
            feature_names=feature_names,
            n_samples=6,
            random_state=42
        )
        
        print(f"  Generated {results_proposed['n_explained']} explanations")
        
        # Print sample explanation
        if results_proposed['explanations']:
            print("\nSample explanation (Proposed):")
            print(format_explanation(results_proposed['explanations'][0]))
        
        # Print importance summary
        print("\n" + format_importance_summary(results_proposed['importance_summary']))
        
        # Only test first fold
        break
    
    print("\nExplainability module test completed successfully!")

