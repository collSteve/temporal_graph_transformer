"""
Cross-Validation Framework for Temporal Graph Transformer

Comprehensive cross-validation system for statistical validation and comparison
of baseline methods. Includes stratified k-fold, temporal split validation,
and cross-chain validation strategies.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from collections import defaultdict
import json
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from scipy import stats
import warnings

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from baselines.base_interface import BaselineMethodInterface
from utils.metrics import BinaryClassificationMetrics, CrossValidationMetrics


@dataclass
class CrossValidationResult:
    """Container for cross-validation results."""
    method_name: str
    fold_results: List[Dict[str, float]]
    summary_stats: Dict[str, Dict[str, float]]
    confidence_intervals: Dict[str, Tuple[float, float]]
    statistical_tests: Dict[str, Dict[str, float]]
    
    def get_mean_f1(self) -> float:
        """Get mean F1 score across folds."""
        return self.summary_stats.get('f1', {}).get('mean', 0.0)
    
    def get_f1_confidence_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """Get confidence interval for F1 score."""
        return self.confidence_intervals.get('f1', (0.0, 0.0))


class CrossValidationStrategy:
    """Base class for cross-validation strategies."""
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
    
    def split(self, dataset: torch.utils.data.Dataset, labels: List[int]) -> List[Tuple[List[int], List[int]]]:
        """Split dataset into train/test folds."""
        raise NotImplementedError("Subclasses must implement split()")


class StratifiedCrossValidation(CrossValidationStrategy):
    """Stratified k-fold cross-validation ensuring balanced class distribution."""
    
    def split(self, dataset: torch.utils.data.Dataset, labels: List[int]) -> List[Tuple[List[int], List[int]]]:
        """Split dataset into stratified folds."""
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        splits = []
        indices = np.arange(len(dataset))
        
        for train_idx, test_idx in skf.split(indices, labels):
            splits.append((train_idx.tolist(), test_idx.tolist()))
        
        return splits


class TemporalCrossValidation(CrossValidationStrategy):
    """Temporal cross-validation for time-series data."""
    
    def split(self, dataset: torch.utils.data.Dataset, timestamps: List[float]) -> List[Tuple[List[int], List[int]]]:
        """Split dataset temporally."""
        tscv = TimeSeriesSplit(n_splits=self.n_folds)
        
        splits = []
        indices = np.arange(len(dataset))
        
        # Sort by timestamp for temporal splitting
        sorted_indices = np.argsort(timestamps)
        
        for train_idx, test_idx in tscv.split(indices):
            # Map back to original indices
            train_original = sorted_indices[train_idx].tolist()
            test_original = sorted_indices[test_idx].tolist()
            splits.append((train_original, test_original))
        
        return splits


class ChainCrossValidation(CrossValidationStrategy):
    """Cross-chain validation for testing generalization across blockchains."""
    
    def __init__(self, chain_info: Dict[int, str], n_folds: int = None, random_state: int = 42):
        self.chain_info = chain_info  # Maps indices to chain names
        self.unique_chains = list(set(chain_info.values()))
        super().__init__(n_folds or len(self.unique_chains), random_state)
    
    def split(self, dataset: torch.utils.data.Dataset, chain_labels: List[str]) -> List[Tuple[List[int], List[int]]]:
        """Split by leaving one chain out for testing."""
        splits = []
        
        for test_chain in self.unique_chains:
            train_indices = [i for i, chain in enumerate(chain_labels) if chain != test_chain]
            test_indices = [i for i, chain in enumerate(chain_labels) if chain == test_chain]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
        
        return splits


class CrossValidationFramework:
    """
    Comprehensive cross-validation framework for baseline comparison.
    
    Supports multiple validation strategies and statistical analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Validation strategies
        self.strategies = {
            'stratified': StratifiedCrossValidation(
                n_folds=config.get('stratified_folds', 5),
                random_state=config.get('random_state', 42)
            ),
            'temporal': TemporalCrossValidation(
                n_folds=config.get('temporal_folds', 5),
                random_state=config.get('random_state', 42)
            ),
            'chain': ChainCrossValidation(
                chain_info={},  # Will be populated during validation
                random_state=config.get('random_state', 42)
            )
        }
    
    def validate_method(self, method: BaselineMethodInterface, dataset: torch.utils.data.Dataset,
                       strategy: str = 'stratified', **kwargs) -> CrossValidationResult:
        """
        Perform cross-validation for a single method.
        
        Args:
            method: Baseline method to validate
            dataset: Dataset for validation
            strategy: Validation strategy ('stratified', 'temporal', 'chain')
            **kwargs: Additional arguments for validation strategy
            
        Returns:
            CrossValidationResult with comprehensive statistics
        """
        print(f"\n🔬 Cross-validating {method.name} using {strategy} strategy")
        
        # Extract labels and metadata for splitting
        labels, metadata = self._extract_dataset_info(dataset, strategy)
        
        # Get splits from strategy
        cv_strategy = self.strategies[strategy]
        if strategy == 'temporal':
            splits = cv_strategy.split(dataset, metadata)
        elif strategy == 'chain':
            cv_strategy.chain_info = {i: chain for i, chain in enumerate(metadata)}
            splits = cv_strategy.split(dataset, metadata)
        else:
            splits = cv_strategy.split(dataset, labels)
        
        print(f"Performing {len(splits)} folds")
        
        # Perform cross-validation
        fold_results = []
        
        for fold_idx, (train_indices, test_indices) in enumerate(splits):
            print(f"  Fold {fold_idx + 1}/{len(splits)} - Train: {len(train_indices)}, Test: {len(test_indices)}")
            
            try:
                # Create fold datasets
                train_subset = Subset(dataset, train_indices)
                test_subset = Subset(dataset, test_indices)
                
                # Create data loaders
                train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
                test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
                val_loader = test_loader  # Use test as validation for simplicity
                
                # Clone method for this fold (to avoid state contamination)
                fold_method = self._clone_method(method)
                
                # Train and evaluate
                train_result = fold_method.train(train_loader, val_loader, self.device)
                test_result = fold_method.evaluate(test_loader, self.device)
                
                # Combine results
                fold_result = {**train_result, **{f'test_{k}': v for k, v in test_result.items()}}
                fold_results.append(fold_result)
                
                print(f"    Test F1: {test_result.get('f1', 0.0):.4f}")
                
            except Exception as e:
                print(f"    ❌ Fold {fold_idx + 1} failed: {e}")
                # Add empty result to maintain fold count
                fold_results.append({
                    'test_f1': 0.0, 'test_accuracy': 0.0, 
                    'test_precision': 0.0, 'test_recall': 0.0
                })
        
        # Compute summary statistics
        summary_stats = self._compute_summary_statistics(fold_results)
        confidence_intervals = self._compute_confidence_intervals(fold_results)
        
        print(f"✅ {method.name} CV completed - Mean F1: {summary_stats['test_f1']['mean']:.4f} ± {summary_stats['test_f1']['std']:.4f}")
        
        return CrossValidationResult(
            method_name=method.name,
            fold_results=fold_results,
            summary_stats=summary_stats,
            confidence_intervals=confidence_intervals,
            statistical_tests={}
        )
    
    def compare_methods(self, methods: List[BaselineMethodInterface], dataset: torch.utils.data.Dataset,
                       strategy: str = 'stratified') -> Dict[str, CrossValidationResult]:
        """
        Compare multiple methods using cross-validation.
        
        Args:
            methods: List of baseline methods to compare
            dataset: Dataset for validation
            strategy: Validation strategy
            
        Returns:
            Dictionary mapping method names to validation results
        """
        print(f"\n🔬 Cross-Validation Comparison: {len(methods)} methods")
        print(f"Strategy: {strategy}")
        print(f"Dataset size: {len(dataset)}")
        
        results = {}
        
        # Validate each method
        for method in methods:
            result = self.validate_method(method, dataset, strategy)
            results[method.name] = result
        
        # Perform statistical comparisons
        self._perform_statistical_tests(results)
        
        return results
    
    def _extract_dataset_info(self, dataset: torch.utils.data.Dataset, strategy: str) -> Tuple[List[int], List]:
        """Extract labels and metadata from dataset."""
        labels = []
        metadata = []
        
        # Sample a few items to understand structure
        sample_size = min(len(dataset), 100)
        for i in range(sample_size):
            try:
                item = dataset[i]
                if isinstance(item, dict):
                    label = item.get('label', item.get('labels', 0))
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    labels.append(int(label))
                    
                    # Extract metadata based on strategy
                    if strategy == 'temporal':
                        timestamp = item.get('timestamp', i * 3600)  # Default timestamp
                        if isinstance(timestamp, torch.Tensor):
                            timestamp = timestamp.item()
                        metadata.append(float(timestamp))
                    elif strategy == 'chain':
                        chain = item.get('chain_id', 'unknown')
                        if isinstance(chain, torch.Tensor):
                            chain = str(chain.item())
                        metadata.append(str(chain))
                    else:
                        metadata.append(0)  # Placeholder for stratified
                else:
                    # Assume second element is label for tuple datasets
                    labels.append(0)
                    metadata.append(0 if strategy != 'temporal' else i * 3600)
            except:
                labels.append(0)
                metadata.append(0 if strategy != 'temporal' else i * 3600)
        
        # Extend to full dataset size (for demonstration)
        full_labels = labels * (len(dataset) // len(labels) + 1)
        full_metadata = metadata * (len(dataset) // len(metadata) + 1)
        
        return full_labels[:len(dataset)], full_metadata[:len(dataset)]
    
    def _clone_method(self, method: BaselineMethodInterface) -> BaselineMethodInterface:
        """Create a clean copy of the method for cross-validation."""
        # Create new instance with same config
        method_class = type(method)
        return method_class(method.config)
    
    def _compute_summary_statistics(self, fold_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Compute summary statistics across folds."""
        if not fold_results:
            return {}
        
        # Get all metric names
        all_metrics = set()
        for result in fold_results:
            all_metrics.update(result.keys())
        
        summary = {}
        for metric in all_metrics:
            values = [result.get(metric, 0.0) for result in fold_results]
            values = np.array(values)
            
            summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'n_folds': len(values)
            }
        
        return summary
    
    def _compute_confidence_intervals(self, fold_results: List[Dict[str, float]], 
                                    alpha: float = 0.05) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals for metrics."""
        if not fold_results:
            return {}
        
        intervals = {}
        key_metrics = ['test_f1', 'test_accuracy', 'test_precision', 'test_recall']
        
        for metric in key_metrics:
            values = [result.get(metric, 0.0) for result in fold_results]
            values = np.array(values)
            
            if len(values) > 1:
                mean = np.mean(values)
                std_err = stats.sem(values)
                dof = len(values) - 1
                t_critical = stats.t.ppf(1 - alpha/2, dof)
                
                margin_error = t_critical * std_err
                intervals[metric] = (float(mean - margin_error), float(mean + margin_error))
            else:
                intervals[metric] = (float(values[0]), float(values[0]))
        
        return intervals
    
    def _perform_statistical_tests(self, results: Dict[str, CrossValidationResult]):
        """Perform statistical significance tests between methods."""
        method_names = list(results.keys())
        
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names):
                if i >= j:
                    continue
                
                # Get F1 scores for both methods
                f1_scores_1 = [fold.get('test_f1', 0.0) for fold in results[method1].fold_results]
                f1_scores_2 = [fold.get('test_f1', 0.0) for fold in results[method2].fold_results]
                
                # Perform paired t-test
                if len(f1_scores_1) == len(f1_scores_2) and len(f1_scores_1) > 1:
                    try:
                        t_stat, p_value = stats.ttest_rel(f1_scores_1, f1_scores_2)
                        
                        test_result = {
                            'compared_with': method2,
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'effect_size': float(np.mean(f1_scores_1) - np.mean(f1_scores_2))
                        }
                        
                        # Add to both methods' results
                        if 'statistical_tests' not in results[method1].statistical_tests:
                            results[method1].statistical_tests = {}
                        results[method1].statistical_tests[method2] = test_result
                        
                    except Exception as e:
                        print(f"Statistical test failed for {method1} vs {method2}: {e}")
    
    def save_results(self, results: Dict[str, CrossValidationResult], output_path: str):
        """Save cross-validation results to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = {}
        
        for method_name, result in results.items():
            serializable_results[method_name] = {
                'method_name': result.method_name,
                'fold_results': result.fold_results,
                'summary_stats': result.summary_stats,
                'confidence_intervals': {k: list(v) for k, v in result.confidence_intervals.items()},
                'statistical_tests': result.statistical_tests,
                'mean_f1': result.get_mean_f1()
            }
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✅ Cross-validation results saved to: {output_path}")
    
    def print_comparison_summary(self, results: Dict[str, CrossValidationResult]):
        """Print a summary of cross-validation comparison."""
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        # Sort methods by mean F1 score
        sorted_methods = sorted(results.items(), key=lambda x: x[1].get_mean_f1(), reverse=True)
        
        print(f"{'Method':<25} {'Mean F1':<12} {'95% CI':<20} {'Accuracy':<12} {'Significance':<15}")
        print(f"{'-'*80}")
        
        for method_name, result in sorted_methods:
            mean_f1 = result.get_mean_f1()
            f1_ci = result.get_f1_confidence_interval()
            mean_acc = result.summary_stats.get('test_accuracy', {}).get('mean', 0.0)
            
            # Count significant differences
            significant_wins = sum(1 for test in result.statistical_tests.values() 
                                 if test.get('significant', False) and test.get('effect_size', 0) > 0)
            
            print(f"{method_name:<25} {mean_f1:<12.4f} [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]"
                  f"{'':<5} {mean_acc:<12.4f} {significant_wins} wins")
        
        print(f"\n🏆 Best performing method: {sorted_methods[0][0]} (F1: {sorted_methods[0][1].get_mean_f1():.4f})")
        
        # Statistical significance summary
        print(f"\n📊 Statistical Significance Tests:")
        for method_name, result in sorted_methods:
            significant_tests = [test for test in result.statistical_tests.values() 
                               if test.get('significant', False)]
            if significant_tests:
                print(f"  {method_name}: {len(significant_tests)} significant differences")