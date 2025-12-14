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
    RANDOM_SEED, DATASETS, CV_FOLDS, RESULTS_DIR, USE_HOLDOUT_SPLIT, 
    N_REPETITIONS, METRICS
)
from src.data_loader import load_dataset, get_dataset_info, print_dataset_summary
from src.preprocessing import preprocess_fold, get_cv_splits, get_data_splits
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


def run_single_repetition(dataset_name: str,
                          X: np.ndarray,
                          y: np.ndarray,
                          feature_names: List[str],
                          n_folds: int,
                          use_holdout: bool,
                          base_seed: int,
                          verbose: bool) -> Tuple[List[Dict], List[Dict]]:
    """
    Run a single repetition of the experiment (one full CV or holdout split).
    
    Args:
        dataset_name: Name of the dataset
        X: Feature matrix
        y: Labels
        feature_names: Feature names
        n_folds: Number of CV folds
        use_holdout: Whether to use holdout split
        base_seed: Base random seed for this repetition
        verbose: Whether to print progress
        
    Returns:
        Tuple of (baseline_fold_metrics, proposed_fold_metrics)
    """
    baseline_fold_metrics = []
    proposed_fold_metrics = []
    
    # Data splitting loop (either single holdout or k-fold CV)
    for fold_idx, (train_idx, test_idx) in enumerate(get_data_splits(X, y, use_holdout=use_holdout, random_state=base_seed)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        fold_seed = base_seed + fold_idx
        
        # Preprocess with SMOTE
        X_train_proc, X_test_proc, y_train_proc = preprocess_fold(
            X_train, X_test, y_train, apply_smote=True, random_state=fold_seed
        )
        
        # === Pipeline A: Baseline (PCA + MLP) ===
        baseline = create_baseline_pipeline(random_state=fold_seed)
        baseline.fit(X_train_proc, y_train_proc)
        
        y_pred_baseline = baseline.predict(X_test_proc)
        y_proba_baseline = baseline.predict_proba(X_test_proc)
        
        baseline_metrics = compute_metrics(y_test, y_pred_baseline, y_proba_baseline)
        baseline_fold_metrics.append(baseline_metrics)
        
        # === Pipeline B: Proposed (Autoencoder + XGBoost) ===
        proposed = create_proposed_pipeline(random_state=fold_seed, verbose=False)
        proposed.fit(X_train_proc, y_train_proc)
        
        y_pred_proposed = proposed.predict(X_test_proc)
        y_proba_proposed = proposed.predict_proba(X_test_proc)
        
        proposed_metrics = compute_metrics(y_test, y_pred_proposed, y_proba_proposed)
        proposed_fold_metrics.append(proposed_metrics)
    
    return baseline_fold_metrics, proposed_fold_metrics


def run_experiment(dataset_name: str,
                   n_folds: int = CV_FOLDS,
                   n_repetitions: int = N_REPETITIONS,
                   use_holdout: bool = USE_HOLDOUT_SPLIT,
                   verbose: bool = True,
                   generate_explanations: bool = True) -> Dict:
    """
    Run experiment on a single dataset with multiple repetitions for statistical robustness.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'CM1')
        n_folds: Number of cross-validation folds (only used if use_holdout=False)
        n_repetitions: Number of experimental repetitions with different seeds
        use_holdout: Whether to use 70/30 holdout split or k-fold CV
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
    
    # Initialize metrics aggregators for all observations across repetitions
    metrics_baseline = MetricsAggregator()
    metrics_proposed = MetricsAggregator()
    
    # Store all observations for analysis
    all_fold_results = []
    
    # Determine evaluation method description
    if use_holdout:
        n_obs_per_rep = 1
        split_method = "70/30 holdout split"
    else:
        n_obs_per_rep = n_folds
        split_method = f"{n_folds}-fold CV"
    
    total_observations = n_repetitions * n_obs_per_rep
    
    if verbose:
        print(f"Evaluation: {n_repetitions} repetitions × {split_method} = {total_observations} observations")
    
    # Multiple repetitions with different base seeds
    for rep_idx in range(n_repetitions):
        base_seed = RANDOM_SEED + (rep_idx * 100)  # Different seed per repetition
        
        if verbose:
            print(f"\n  Repetition {rep_idx + 1}/{n_repetitions} (seed={base_seed})...", end=" ")
        
        baseline_fold_metrics, proposed_fold_metrics = run_single_repetition(
            dataset_name, X, y, feature_names, n_folds, use_holdout, base_seed, verbose=False
        )
        
        # Add all fold metrics to aggregators
        for baseline_m, proposed_m in zip(baseline_fold_metrics, proposed_fold_metrics):
            metrics_baseline.add_fold(baseline_m)
            metrics_proposed.add_fold(proposed_m)
            all_fold_results.append({
                'repetition': rep_idx + 1,
                'baseline': baseline_m,
                'proposed': proposed_m
            })
        
        if verbose:
            # Report mean F1 for this repetition
            rep_baseline_f1 = np.mean([m['f1'] for m in baseline_fold_metrics])
            rep_proposed_f1 = np.mean([m['f1'] for m in proposed_fold_metrics])
            print(f"F1: {rep_baseline_f1:.4f} vs {rep_proposed_f1:.4f}")
    
    # Aggregate results across all repetitions
    summary_baseline = metrics_baseline.get_summary()
    summary_proposed = metrics_proposed.get_summary()
    
    # Statistical comparison with all paired observations
    comparison = compare_pipelines(metrics_baseline, metrics_proposed)
    
    if verbose:
        print(f"\n  Total observations: {len(metrics_baseline.fold_metrics)}")
        print("\n" + format_results_table(
            summary_baseline, summary_proposed, comparison,
            "Baseline (PCA+MLP)", "Proposed (AE+XGB)"
        ))
    
    # Generate explanations using last repetition's last fold
    explanations = None
    if generate_explanations:
        if verbose:
            print("\nGenerating LIME explanations (sample from last repetition)...")
        
        # Use last repetition seed for consistency
        last_seed = RANDOM_SEED + ((n_repetitions - 1) * 100)
        
        # Get a split for explanations
        for train_idx, test_idx in get_data_splits(X, y, use_holdout=use_holdout, random_state=last_seed):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            X_train_proc, X_test_proc, y_train_proc = preprocess_fold(
                X_train, X_test, y_train, apply_smote=True, random_state=last_seed
            )
            
            baseline = create_baseline_pipeline(random_state=last_seed)
            baseline.fit(X_train_proc, y_train_proc)
            
            proposed = create_proposed_pipeline(random_state=last_seed, verbose=False)
            proposed.fit(X_train_proc, y_train_proc)
            break  # Only need one split for explanations
        
        explanations = {
            'baseline': explain_predictions(
                baseline, X_train_proc, X_test_proc, y_test,
                feature_names, n_samples=10, random_state=last_seed
            ),
            'proposed': explain_predictions(
                proposed, X_train_proc, X_test_proc, y_test,
                feature_names, n_samples=10, random_state=last_seed
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
        'n_repetitions': n_repetitions,
        'n_folds': n_folds if not use_holdout else 1,
        'total_observations': len(metrics_baseline.fold_metrics),
        'fold_results': all_fold_results,
        'summary_baseline': summary_baseline,
        'summary_proposed': summary_proposed,
        'comparison': comparison,
        'explanations': explanations
    }


def run_all_experiments(datasets: List[str] = None,
                        n_folds: int = CV_FOLDS,
                        n_repetitions: int = N_REPETITIONS,
                        use_holdout: bool = USE_HOLDOUT_SPLIT,
                        verbose: bool = True,
                        save_results: bool = True) -> Dict:
    """
    Run experiments on all specified datasets with multiple repetitions.
    
    Args:
        datasets: List of dataset names (default: from config)
        n_folds: Number of cross-validation folds (only used if use_holdout=False)
        n_repetitions: Number of experimental repetitions with different seeds
        use_holdout: Whether to use 70/30 holdout split or k-fold CV
        verbose: Whether to print progress
        save_results: Whether to save results to files
        
    Returns:
        Dictionary containing all results
    """
    datasets = datasets or DATASETS
    set_all_seeds(RANDOM_SEED)
    
    if use_holdout:
        n_obs_per_rep = 1
        split_desc = "holdout"
    else:
        n_obs_per_rep = n_folds
        split_desc = f"{n_folds}-fold CV"
    
    total_obs = n_repetitions * n_obs_per_rep
    eval_method = f"{n_repetitions} repetitions × {split_desc} = {total_obs} paired observations"
    
    print("="*70)
    print("SOFTWARE DEFECT PREDICTION: PIPELINE COMPARISON")
    print("="*70)
    print(f"Baseline: SMOTE + PCA + MLP")
    print(f"Proposed: SMOTE + Autoencoder + XGBoost")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Evaluation: {eval_method}")
    print(f"Base random seed: {RANDOM_SEED}")
    print("="*70)
    
    all_results = {}
    
    for dataset in datasets:
        try:
            results = run_experiment(
                dataset, n_folds, n_repetitions=n_repetitions,
                use_holdout=use_holdout, verbose=verbose, generate_explanations=True
            )
            all_results[dataset] = results
        except Exception as e:
            print(f"\nError processing {dataset}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print overall summary
    if verbose and all_results:
        print("\n" + "="*80)
        print("OVERALL SUMMARY (mean ± std over all repetitions)")
        print("="*80)
        
        print(f"\n{'Dataset':<10} {'Metric':<12} {'Baseline':>20} {'Proposed':>20} {'p-value':>10}")
        print("-"*80)
        
        for dataset, results in all_results.items():
            for metric in ['mcc', 'f1', 'auc_roc']:
                baseline_mean = results['summary_baseline'][metric]['mean']
                baseline_std = results['summary_baseline'][metric]['std']
                proposed_mean = results['summary_proposed'][metric]['mean']
                proposed_std = results['summary_proposed'][metric]['std']
                
                comp = results['comparison'].get(metric, {})
                p_value = comp.get('p_value', np.nan)
                sig = "*" if comp.get('significant', False) else ""
                
                print(f"{dataset:<10} {metric:<12} {baseline_mean:>8.4f} ± {baseline_std:<8.4f} "
                      f"{proposed_mean:>8.4f} ± {proposed_std:<8.4f} {p_value:>8.4f}{sig}")
        
        print("-"*80)
        print(f"* indicates statistically significant difference (p < 0.05)")
        print(f"Total observations per dataset: {total_obs}")
    
    # Save results
    if save_results:
        save_experiment_results(all_results, n_repetitions)
    
    return all_results


def save_experiment_results(results: Dict, n_repetitions: int = N_REPETITIONS):
    """
    Save experiment results to files.
    
    Args:
        results: Dictionary of all results
        n_repetitions: Number of repetitions used in the experiment
    """
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary as CSV
    summary_rows = []
    for dataset, data in results.items():
        for metric in METRICS:
            row = {
                'dataset': dataset,
                'metric': metric,
                'n_observations': data.get('total_observations', n_repetitions),
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
        '--repetitions', type=int, default=N_REPETITIONS,
        help=f"Number of experimental repetitions (default: {N_REPETITIONS})"
    )
    parser.add_argument(
        '--holdout', action='store_true', default=USE_HOLDOUT_SPLIT,
        help="Use 70/30 holdout split instead of k-fold CV"
    )
    parser.add_argument(
        '--cv', action='store_true',
        help="Use k-fold cross-validation instead of holdout split"
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
    
    # Determine evaluation method
    use_holdout = args.holdout if not args.cv else False
    
    # Run experiments
    results = run_all_experiments(
        datasets=args.datasets,
        n_folds=args.folds,
        n_repetitions=args.repetitions,
        use_holdout=use_holdout,
        verbose=verbose,
        save_results=save_results
    )
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("="*70)
    
    return results


if __name__ == "__main__":
    main()

