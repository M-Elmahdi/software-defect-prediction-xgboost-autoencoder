"""
Main experiment runner for Software Defect Prediction Thesis Implementation.

Compares Pipeline A (Baseline SPAM-XAI: SMOTE + PCA + MLP + LIME) with
Pipeline B (Proposed: SMOTE + Autoencoder + XGBoost + LIME).

Usage:
    python main.py [--datasets CM1 PC1 PC2] [--folds 10] [--verbose]
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Set random seeds before any other imports
SEED = 42
np.random.seed(SEED)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from src.config import (
    RANDOM_SEED, DATASETS, CV_FOLDS, RESULTS_DIR
)
from src.data_loader import load_dataset, get_dataset_info, print_dataset_summary
from src.preprocessing import preprocess_fold, get_cv_splits
from src.pipeline_baseline import create_baseline_pipeline
from src.pipeline_proposed import create_proposed_pipeline
from src.evaluation import (
    compute_metrics, MetricsAggregator, 
    compare_pipelines, format_results_table
)
from src.explainability import explain_predictions, format_importance_summary


def set_all_seeds(seed: int = RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_experiment(dataset_name: str,
                   n_folds: int = CV_FOLDS,
                   verbose: bool = True,
                   generate_explanations: bool = True) -> Dict:
    """
    Run experiment on a single dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'CM1')
        n_folds: Number of cross-validation folds
        verbose: Whether to print progress
        generate_explanations: Whether to generate LIME explanations
        
    Returns:
        Dictionary containing all results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset_name}")
        print(f"{'='*70}")
    
    # Load dataset
    X, y, feature_names = load_dataset(dataset_name)
    info = get_dataset_info(X, y, feature_names, dataset_name)
    
    if verbose:
        print(f"Samples: {info['n_samples']}, Features: {info['n_features']}")
        print(f"Defective: {info['n_defective']} ({info['defect_rate']:.2f}%), "
              f"Clean: {info['n_clean']}")
    
    # Initialize metrics aggregators
    metrics_baseline = MetricsAggregator()
    metrics_proposed = MetricsAggregator()
    
    # Store per-fold predictions for analysis
    fold_results = []
    
    # Cross-validation loop
    for fold_idx, (train_idx, test_idx) in enumerate(get_cv_splits(X, y, n_splits=n_folds)):
        if verbose:
            print(f"\nFold {fold_idx + 1}/{n_folds}...", end=" ")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Preprocess with SMOTE
        X_train_proc, X_test_proc, y_train_proc = preprocess_fold(
            X_train, X_test, y_train, apply_smote=True, random_state=RANDOM_SEED + fold_idx
        )
        
        # === Pipeline A: Baseline (PCA + MLP) ===
        baseline = create_baseline_pipeline(random_state=RANDOM_SEED + fold_idx)
        baseline.fit(X_train_proc, y_train_proc)
        
        y_pred_baseline = baseline.predict(X_test_proc)
        y_proba_baseline = baseline.predict_proba(X_test_proc)
        
        baseline_metrics = compute_metrics(y_test, y_pred_baseline, y_proba_baseline)
        metrics_baseline.add_fold(baseline_metrics)
        
        # === Pipeline B: Proposed (Autoencoder + XGBoost) ===
        proposed = create_proposed_pipeline(random_state=RANDOM_SEED + fold_idx, verbose=False)
        proposed.fit(X_train_proc, y_train_proc)
        
        y_pred_proposed = proposed.predict(X_test_proc)
        y_proba_proposed = proposed.predict_proba(X_test_proc)
        
        proposed_metrics = compute_metrics(y_test, y_pred_proposed, y_proba_proposed)
        metrics_proposed.add_fold(proposed_metrics)
        
        if verbose:
            print(f"Baseline F1: {baseline_metrics['f1']:.4f}, "
                  f"Proposed F1: {proposed_metrics['f1']:.4f}")
        
        # Store fold results
        fold_results.append({
            'fold': fold_idx + 1,
            'n_train': len(y_train_proc),
            'n_test': len(y_test),
            'baseline': baseline_metrics,
            'proposed': proposed_metrics
        })
    
    # Aggregate results
    summary_baseline = metrics_baseline.get_summary()
    summary_proposed = metrics_proposed.get_summary()
    
    # Statistical comparison
    comparison = compare_pipelines(metrics_baseline, metrics_proposed)
    
    if verbose:
        print("\n" + format_results_table(
            summary_baseline, summary_proposed, comparison,
            "Baseline (PCA+MLP)", "Proposed (AE+XGB)"
        ))
    
    # Generate explanations on last fold (sample)
    explanations = None
    if generate_explanations:
        if verbose:
            print("\nGenerating LIME explanations...")
        
        explanations = {
            'baseline': explain_predictions(
                baseline, X_train_proc, X_test_proc, y_test,
                feature_names, n_samples=10, random_state=RANDOM_SEED
            ),
            'proposed': explain_predictions(
                proposed, X_train_proc, X_test_proc, y_test,
                feature_names, n_samples=10, random_state=RANDOM_SEED
            )
        }
        
        if verbose:
            print("\nBaseline Pipeline - Feature Importance:")
            print(format_importance_summary(explanations['baseline']['importance_summary'], top_n=5))
            print("\nProposed Pipeline - Feature Importance:")
            print(format_importance_summary(explanations['proposed']['importance_summary'], top_n=5))
    
    return {
        'dataset': dataset_name,
        'dataset_info': info,
        'fold_results': fold_results,
        'summary_baseline': summary_baseline,
        'summary_proposed': summary_proposed,
        'comparison': comparison,
        'explanations': explanations
    }


def run_all_experiments(datasets: List[str] = None,
                        n_folds: int = CV_FOLDS,
                        verbose: bool = True,
                        save_results: bool = True) -> Dict:
    """
    Run experiments on all specified datasets.
    
    Args:
        datasets: List of dataset names (default: from config)
        n_folds: Number of cross-validation folds
        verbose: Whether to print progress
        save_results: Whether to save results to files
        
    Returns:
        Dictionary containing all results
    """
    datasets = datasets or DATASETS
    set_all_seeds(RANDOM_SEED)
    
    print("="*70)
    print("SOFTWARE DEFECT PREDICTION: PIPELINE COMPARISON")
    print("="*70)
    print(f"Baseline: SMOTE + PCA + MLP (SPAM-XAI)")
    print(f"Proposed: SMOTE + Autoencoder + XGBoost")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Cross-validation: {n_folds}-fold stratified")
    print(f"Random seed: {RANDOM_SEED}")
    print("="*70)
    
    all_results = {}
    
    for dataset in datasets:
        try:
            results = run_experiment(
                dataset, n_folds, verbose, generate_explanations=True
            )
            all_results[dataset] = results
        except Exception as e:
            print(f"\nError processing {dataset}: {e}")
            continue
    
    # Print overall summary
    if verbose and all_results:
        print("\n" + "="*70)
        print("OVERALL SUMMARY")
        print("="*70)
        
        print(f"\n{'Dataset':<10} {'Metric':<12} {'Baseline':>15} {'Proposed':>15} {'Winner':<15}")
        print("-"*70)
        
        for dataset, results in all_results.items():
            for metric in ['f1', 'auc_roc']:
                baseline_mean = results['summary_baseline'][metric]['mean']
                proposed_mean = results['summary_proposed'][metric]['mean']
                
                comp = results['comparison'].get(metric, {})
                winner = comp.get('better', 'N/A')
                if winner == 'B (Proposed)':
                    winner = 'Proposed*' if comp.get('significant') else 'Proposed'
                elif winner == 'A (Baseline)':
                    winner = 'Baseline*' if comp.get('significant') else 'Baseline'
                else:
                    winner = 'No diff'
                
                print(f"{dataset:<10} {metric:<12} {baseline_mean:>15.4f} "
                      f"{proposed_mean:>15.4f} {winner:<15}")
        
        print("-"*70)
        print("* indicates statistically significant difference (p < 0.05)")
    
    # Save results
    if save_results:
        save_experiment_results(all_results)
    
    return all_results


def save_experiment_results(results: Dict):
    """
    Save experiment results to files.
    
    Args:
        results: Dictionary of all results
    """
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary as CSV
    summary_rows = []
    for dataset, data in results.items():
        for metric in ['precision', 'recall', 'f1', 'auc_roc', 'accuracy']:
            row = {
                'dataset': dataset,
                'metric': metric,
                'baseline_mean': data['summary_baseline'].get(metric, {}).get('mean', np.nan),
                'baseline_std': data['summary_baseline'].get(metric, {}).get('std', np.nan),
                'proposed_mean': data['summary_proposed'].get(metric, {}).get('mean', np.nan),
                'proposed_std': data['summary_proposed'].get(metric, {}).get('std', np.nan),
                'p_value': data['comparison'].get(metric, {}).get('p_value', np.nan),
                'significant': data['comparison'].get(metric, {}).get('significant', False)
            }
            summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(RESULTS_DIR, f"results_summary_{timestamp}.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save detailed results as JSON
    json_results = {}
    for dataset, data in results.items():
        json_results[dataset] = {
            'dataset_info': {
                'name': data['dataset_info']['name'],
                'n_samples': int(data['dataset_info']['n_samples']),
                'n_features': int(data['dataset_info']['n_features']),
                'n_defective': int(data['dataset_info']['n_defective']),
                'defect_rate': float(data['dataset_info']['defect_rate'])
            },
            'summary_baseline': {
                k: {sk: float(sv) if isinstance(sv, (int, float, np.floating)) else sv 
                    for sk, sv in v.items() if sk != 'values'}
                for k, v in data['summary_baseline'].items()
            },
            'summary_proposed': {
                k: {sk: float(sv) if isinstance(sv, (int, float, np.floating)) else sv 
                    for sk, sv in v.items() if sk != 'values'}
                for k, v in data['summary_proposed'].items()
            },
            'comparison': {
                k: {sk: bool(sv) if isinstance(sv, (bool, np.bool_)) 
                    else (float(sv) if isinstance(sv, (int, float, np.floating, np.integer)) else sv)
                    for sk, sv in v.items()}
                for k, v in data['comparison'].items()
            }
        }
    
    json_path = os.path.join(RESULTS_DIR, f"results_detailed_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Detailed results saved to: {json_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Software Defect Prediction experiments"
    )
    parser.add_argument(
        '--datasets', nargs='+', default=DATASETS,
        help=f"Datasets to evaluate (default: {DATASETS})"
    )
    parser.add_argument(
        '--folds', type=int, default=CV_FOLDS,
        help=f"Number of CV folds (default: {CV_FOLDS})"
    )
    parser.add_argument(
        '--verbose', action='store_true', default=True,
        help="Print detailed progress"
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help="Minimal output"
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help="Don't save results to files"
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    save_results = not args.no_save
    
    # Run experiments
    results = run_all_experiments(
        datasets=args.datasets,
        n_folds=args.folds,
        verbose=verbose,
        save_results=save_results
    )
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()

