"""
Evaluation metrics and statistical testing for Software Defect Prediction.
Includes Precision, Recall, F1-Score, AUC-ROC, and Wilcoxon signed-rank test.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from scipy.stats import wilcoxon

from .config import SIGNIFICANCE_LEVEL


def compute_metrics(y_true: np.ndarray, 
                    y_pred: np.ndarray, 
                    y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for AUC-ROC)
        
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    
    # Precision
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    
    # Recall (Sensitivity)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    
    # F1-Score
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC-ROC (requires probability scores)
    if y_pred_proba is not None:
        try:
            # Get probability of positive class
            if y_pred_proba.ndim == 2:
                proba_positive = y_pred_proba[:, 1]
            else:
                proba_positive = y_pred_proba
            
            # Check if we have both classes in y_true
            if len(np.unique(y_true)) > 1:
                metrics['auc_roc'] = roc_auc_score(y_true, proba_positive)
            else:
                metrics['auc_roc'] = np.nan
        except ValueError:
            metrics['auc_roc'] = np.nan
    else:
        metrics['auc_roc'] = np.nan
    
    # Accuracy
    metrics['accuracy'] = np.mean(y_true == y_pred)
    
    return metrics


def compute_confusion_matrix(y_true: np.ndarray, 
                             y_pred: np.ndarray) -> Dict[str, int]:
    """
    Compute confusion matrix values.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with TP, TN, FP, FN counts
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge cases
        tn, fp, fn, tp = 0, 0, 0, 0
        if cm.shape[0] >= 1:
            if np.unique(y_true)[0] == 0:
                tn = cm[0, 0]
            else:
                tp = cm[0, 0]
    
    return {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }


class MetricsAggregator:
    """
    Aggregates metrics across multiple folds/runs.
    """
    
    def __init__(self):
        """Initialize the aggregator."""
        self.fold_metrics: List[Dict[str, float]] = []
        
    def add_fold(self, metrics: Dict[str, float]):
        """
        Add metrics from a fold.
        
        Args:
            metrics: Dictionary of metric names to values
        """
        self.fold_metrics.append(metrics)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics across all folds.
        
        Returns:
            Dictionary of metric names to {mean, std, min, max}
        """
        if not self.fold_metrics:
            return {}
        
        summary = {}
        metric_names = self.fold_metrics[0].keys()
        
        for name in metric_names:
            values = [m[name] for m in self.fold_metrics if not np.isnan(m.get(name, np.nan))]
            
            if values:
                summary[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
            else:
                summary[name] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'values': []
                }
        
        return summary
    
    def get_fold_values(self, metric_name: str) -> List[float]:
        """
        Get all fold values for a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of values across folds
        """
        return [m.get(metric_name, np.nan) for m in self.fold_metrics]
    
    def reset(self):
        """Reset the aggregator."""
        self.fold_metrics = []


def wilcoxon_test(scores_a: List[float], 
                  scores_b: List[float],
                  alpha: float = SIGNIFICANCE_LEVEL) -> Dict:
    """
    Perform Wilcoxon signed-rank test to compare two methods.
    
    Args:
        scores_a: Scores from method A (e.g., baseline)
        scores_b: Scores from method B (e.g., proposed)
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)
    
    # Check if we have enough samples
    if len(scores_a) < 2 or len(scores_b) < 2:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'better': 'inconclusive',
            'error': 'Not enough samples'
        }
    
    # Check if differences exist
    differences = scores_a - scores_b
    if np.all(differences == 0):
        return {
            'statistic': np.nan,
            'p_value': 1.0,
            'significant': False,
            'better': 'equal',
            'error': 'No differences between methods'
        }
    
    try:
        statistic, p_value = wilcoxon(scores_a, scores_b, alternative='two-sided')
        
        significant = p_value < alpha
        
        # Determine which method is better
        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)
        
        if significant:
            if mean_b > mean_a:
                better = 'B (Proposed)'
            elif mean_a > mean_b:
                better = 'A (Baseline)'
            else:
                better = 'equal'
        else:
            better = 'no significant difference'
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': significant,
            'better': better,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'difference': mean_b - mean_a
        }
        
    except Exception as e:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'better': 'error',
            'error': str(e)
        }


def compare_pipelines(metrics_a: MetricsAggregator, 
                      metrics_b: MetricsAggregator,
                      metric_names: List[str] = None) -> Dict[str, Dict]:
    """
    Compare two pipelines across multiple metrics.
    
    Args:
        metrics_a: Metrics aggregator for pipeline A (baseline)
        metrics_b: Metrics aggregator for pipeline B (proposed)
        metric_names: List of metrics to compare (default: all)
        
    Returns:
        Dictionary of metric names to comparison results
    """
    if metric_names is None:
        metric_names = ['precision', 'recall', 'f1', 'auc_roc']
    
    comparison = {}
    
    for metric in metric_names:
        scores_a = metrics_a.get_fold_values(metric)
        scores_b = metrics_b.get_fold_values(metric)
        
        # Remove NaN values
        valid_pairs = [(a, b) for a, b in zip(scores_a, scores_b) 
                       if not np.isnan(a) and not np.isnan(b)]
        
        if valid_pairs:
            valid_a, valid_b = zip(*valid_pairs)
            comparison[metric] = wilcoxon_test(list(valid_a), list(valid_b))
        else:
            comparison[metric] = {
                'statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'better': 'no valid data'
            }
    
    return comparison


def format_results_table(summary_a: Dict, 
                         summary_b: Dict,
                         comparison: Dict,
                         pipeline_a_name: str = "Baseline",
                         pipeline_b_name: str = "Proposed") -> str:
    """
    Format results as a readable table.
    
    Args:
        summary_a: Summary from pipeline A
        summary_b: Summary from pipeline B
        comparison: Comparison results
        pipeline_a_name: Name of pipeline A
        pipeline_b_name: Name of pipeline B
        
    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"{'Metric':<15} {pipeline_a_name:^25} {pipeline_b_name:^25} {'p-value':>10}")
    lines.append("-" * 80)
    
    for metric in ['precision', 'recall', 'f1', 'auc_roc']:
        if metric in summary_a and metric in summary_b:
            mean_a = summary_a[metric]['mean']
            std_a = summary_a[metric]['std']
            mean_b = summary_b[metric]['mean']
            std_b = summary_b[metric]['std']
            
            p_value = comparison.get(metric, {}).get('p_value', np.nan)
            sig = "*" if comparison.get(metric, {}).get('significant', False) else ""
            
            lines.append(
                f"{metric:<15} {mean_a:.4f} +/- {std_a:.4f}       "
                f"{mean_b:.4f} +/- {std_b:.4f}       {p_value:.4f}{sig}"
            )
    
    lines.append("=" * 80)
    lines.append("* indicates statistically significant difference (p < 0.05)")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test evaluation metrics
    np.random.seed(42)
    
    # Simulate fold results
    print("Testing evaluation metrics...")
    
    # Create sample predictions
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 1])
    y_proba = np.random.random((10, 2))
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_proba)
    print("\nSingle fold metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Test aggregator
    print("\nTesting metrics aggregation...")
    agg_a = MetricsAggregator()
    agg_b = MetricsAggregator()
    
    # Simulate 10 folds
    for i in range(10):
        # Simulate baseline metrics (slightly worse)
        agg_a.add_fold({
            'precision': 0.65 + np.random.random() * 0.1,
            'recall': 0.60 + np.random.random() * 0.1,
            'f1': 0.62 + np.random.random() * 0.1,
            'auc_roc': 0.70 + np.random.random() * 0.1
        })
        
        # Simulate proposed metrics (slightly better)
        agg_b.add_fold({
            'precision': 0.72 + np.random.random() * 0.1,
            'recall': 0.68 + np.random.random() * 0.1,
            'f1': 0.70 + np.random.random() * 0.1,
            'auc_roc': 0.78 + np.random.random() * 0.1
        })
    
    summary_a = agg_a.get_summary()
    summary_b = agg_b.get_summary()
    
    print("\nBaseline summary:")
    for metric, stats in summary_a.items():
        print(f"  {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
    
    print("\nProposed summary:")
    for metric, stats in summary_b.items():
        print(f"  {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
    
    # Test comparison
    print("\nStatistical comparison:")
    comparison = compare_pipelines(agg_a, agg_b)
    for metric, result in comparison.items():
        print(f"  {metric}: p={result['p_value']:.4f}, "
              f"significant={result['significant']}, better={result['better']}")
    
    # Format table
    print("\n" + format_results_table(summary_a, summary_b, comparison))
    
    print("\nEvaluation module test completed successfully!")

