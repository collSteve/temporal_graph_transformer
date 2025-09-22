"""
Binary Classification Metrics Module

Comprehensive metrics for binary classification tasks including airdrop hunter detection.
Provides precision, recall, F1, AUC, and statistical significance testing.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
from scipy import stats
import warnings


class BinaryClassificationMetrics:
    """
    Comprehensive binary classification metrics tracker.
    
    Accumulates predictions and labels across batches and computes
    various performance metrics including statistical significance testing.
    """
    
    def __init__(self, threshold: float = 0.5, device: Optional[torch.device] = None):
        """
        Initialize metrics tracker.
        
        Args:
            threshold: Decision threshold for binary classification
            device: Device for tensor operations
        """
        self.threshold = threshold
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.predictions = []
        self.labels = []
        self.probabilities = []
    
    def update(self, predictions: Union[torch.Tensor, np.ndarray], 
               labels: Union[torch.Tensor, np.ndarray]):
        """
        Update metrics with new batch of predictions and labels.
        
        Args:
            predictions: Model predictions (probabilities or logits)
            labels: Ground truth binary labels (0/1)
        """
        # Convert to numpy arrays
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Handle different prediction formats
        if predictions.ndim > 1:
            # If predictions are 2D (batch_size, num_classes), take the positive class
            if predictions.shape[1] == 2:
                predictions = predictions[:, 1]  # Positive class probability
            else:
                predictions = predictions.squeeze()
        
        # Store probabilities and convert to binary predictions
        self.probabilities.extend(predictions.flatten())
        self.predictions.extend((predictions >= self.threshold).astype(int).flatten())
        self.labels.extend(labels.flatten())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all binary classification metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        if not self.labels:
            return self._empty_metrics()
        
        y_true = np.array(self.labels)
        y_pred = np.array(self.predictions)
        y_prob = np.array(self.probabilities)
        
        # Basic metrics
        metrics = {}
        
        # Classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0.0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0.0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0.0)
        metrics['specificity'] = self._compute_specificity(y_true, y_pred)
        
        # Probability-based metrics
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc_roc'] = 0.0
            
        try:
            metrics['auc_pr'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['auc_pr'] = 0.0
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics['true_positive'] = int(tp)
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        
        # Additional derived metrics
        metrics['positive_rate'] = np.mean(y_true)
        metrics['predicted_positive_rate'] = np.mean(y_pred)
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        # Matthews Correlation Coefficient
        metrics['mcc'] = self._compute_mcc(tp, tn, fp, fn)
        
        return metrics
    
    def compute_confidence_intervals(self, alpha: float = 0.05, 
                                   n_bootstrap: int = 1000) -> Dict[str, Tuple[float, float]]:
        """
        Compute confidence intervals for key metrics using bootstrap sampling.
        
        Args:
            alpha: Significance level (0.05 for 95% CI)
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary mapping metric names to (lower, upper) confidence intervals
        """
        if not self.labels:
            return {}
        
        y_true = np.array(self.labels)
        y_prob = np.array(self.probabilities)
        
        # Bootstrap sampling
        n_samples = len(y_true)
        metrics_bootstrap = {'f1': [], 'auc_roc': [], 'accuracy': [], 'precision': [], 'recall': []}
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_prob_boot = y_prob[indices]
            y_pred_boot = (y_prob_boot >= self.threshold).astype(int)
            
            # Compute metrics for this bootstrap sample
            try:
                metrics_bootstrap['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
                metrics_bootstrap['precision'].append(precision_score(y_true_boot, y_pred_boot, zero_division=0.0))
                metrics_bootstrap['recall'].append(recall_score(y_true_boot, y_pred_boot, zero_division=0.0))
                metrics_bootstrap['f1'].append(f1_score(y_true_boot, y_pred_boot, zero_division=0.0))
                metrics_bootstrap['auc_roc'].append(roc_auc_score(y_true_boot, y_prob_boot))
            except ValueError:
                # Skip this bootstrap sample if there's an issue
                continue
        
        # Compute confidence intervals
        confidence_intervals = {}
        for metric, values in metrics_bootstrap.items():
            if values:
                lower = np.percentile(values, 100 * alpha / 2)
                upper = np.percentile(values, 100 * (1 - alpha / 2))
                confidence_intervals[metric] = (lower, upper)
        
        return confidence_intervals
    
    def statistical_test(self, other_metrics: 'BinaryClassificationMetrics', 
                        metric: str = 'f1') -> Dict[str, float]:
        """
        Perform statistical significance test between two sets of metrics.
        
        Args:
            other_metrics: Another BinaryClassificationMetrics instance to compare against
            metric: Which metric to test ('f1', 'accuracy', 'auc_roc')
            
        Returns:
            Dictionary with test statistics and p-value
        """
        if not self.labels or not other_metrics.labels:
            return {'p_value': 1.0, 'statistic': 0.0, 'significant': False}
        
        # Get predictions and labels for both sets
        y_true_1 = np.array(self.labels)
        y_prob_1 = np.array(self.probabilities)
        y_pred_1 = np.array(self.predictions)
        
        y_true_2 = np.array(other_metrics.labels)
        y_prob_2 = np.array(other_metrics.probabilities)
        y_pred_2 = np.array(other_metrics.predictions)
        
        # Compute metric values for bootstrap test
        n_bootstrap = 1000
        metric_diffs = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample for both methods
            n1, n2 = len(y_true_1), len(y_true_2)
            idx1 = np.random.choice(n1, size=n1, replace=True)
            idx2 = np.random.choice(n2, size=n2, replace=True)
            
            try:
                if metric == 'f1':
                    m1 = f1_score(y_true_1[idx1], y_pred_1[idx1], zero_division=0.0)
                    m2 = f1_score(y_true_2[idx2], y_pred_2[idx2], zero_division=0.0)
                elif metric == 'accuracy':
                    m1 = accuracy_score(y_true_1[idx1], y_pred_1[idx1])
                    m2 = accuracy_score(y_true_2[idx2], y_pred_2[idx2])
                elif metric == 'auc_roc':
                    m1 = roc_auc_score(y_true_1[idx1], y_prob_1[idx1])
                    m2 = roc_auc_score(y_true_2[idx2], y_prob_2[idx2])
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
                
                metric_diffs.append(m1 - m2)
            except ValueError:
                continue
        
        if not metric_diffs:
            return {'p_value': 1.0, 'statistic': 0.0, 'significant': False}
        
        # Two-sided test: H0: difference = 0
        metric_diffs = np.array(metric_diffs)
        t_stat, p_value = stats.ttest_1samp(metric_diffs, 0)
        
        return {
            'p_value': float(p_value),
            'statistic': float(t_stat),
            'significant': p_value < 0.05,
            'mean_difference': float(np.mean(metric_diffs)),
            'std_difference': float(np.std(metric_diffs))
        }
    
    def _compute_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _compute_mcc(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """Compute Matthews Correlation Coefficient."""
        numerator = tp * tn - fp * fn
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return numerator / denominator if denominator > 0 else 0.0
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary."""
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'specificity': 0.0,
            'auc_roc': 0.0,
            'auc_pr': 0.0,
            'balanced_accuracy': 0.0,
            'mcc': 0.0,
            'positive_rate': 0.0,
            'predicted_positive_rate': 0.0,
            'true_positive': 0,
            'true_negative': 0,
            'false_positive': 0,
            'false_negative': 0
        }
    
    def get_roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ROC curve data for plotting."""
        if not self.labels:
            return np.array([]), np.array([]), np.array([])
        
        y_true = np.array(self.labels)
        y_prob = np.array(self.probabilities)
        
        return roc_curve(y_true, y_prob)
    
    def get_precision_recall_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get precision-recall curve data for plotting."""
        if not self.labels:
            return np.array([]), np.array([]), np.array([])
        
        y_true = np.array(self.labels)
        y_prob = np.array(self.probabilities)
        
        return precision_recall_curve(y_true, y_prob)


class MultiClassMetrics:
    """
    Metrics tracker for multi-class classification tasks.
    
    Can be used for multi-blockchain classification or other categorical tasks.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize multi-class metrics tracker.
        
        Args:
            num_classes: Number of classes
            class_names: Optional list of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.predictions = []
        self.labels = []
        self.probabilities = []
    
    def update(self, predictions: Union[torch.Tensor, np.ndarray], 
               labels: Union[torch.Tensor, np.ndarray]):
        """Update metrics with new batch."""
        # Convert to numpy arrays
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        
        # Handle different prediction formats
        if predictions.ndim > 1:
            # Take argmax for class predictions
            class_predictions = np.argmax(predictions, axis=1)
            self.probabilities.extend(predictions)
        else:
            class_predictions = predictions.astype(int)
            # Create one-hot for probabilities if not provided
            probs = np.zeros((len(class_predictions), self.num_classes))
            probs[np.arange(len(class_predictions)), class_predictions] = 1.0
            self.probabilities.extend(probs)
        
        self.predictions.extend(class_predictions.flatten())
        self.labels.extend(labels.flatten())
    
    def compute(self) -> Dict[str, float]:
        """Compute multi-class metrics."""
        if not self.labels:
            return {}
        
        y_true = np.array(self.labels)
        y_pred = np.array(self.predictions)
        
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Per-class and averaged metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0.0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0.0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0.0)
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics[f'precision_{class_name}'] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # Averaged metrics
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0.0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0.0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0.0)
        
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0.0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0.0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0.0)
        
        return metrics


class CrossValidationMetrics:
    """
    Metrics aggregator for cross-validation experiments.
    
    Collects metrics across multiple folds and computes summary statistics.
    """
    
    def __init__(self):
        """Initialize cross-validation metrics aggregator."""
        self.fold_metrics = []
    
    def add_fold(self, metrics: Dict[str, float]):
        """Add metrics from a single fold."""
        self.fold_metrics.append(metrics)
    
    def compute_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics across all folds.
        
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        if not self.fold_metrics:
            return {}
        
        # Get all metric names
        all_metrics = set()
        for fold in self.fold_metrics:
            all_metrics.update(fold.keys())
        
        summary = {}
        for metric in all_metrics:
            values = [fold.get(metric, 0.0) for fold in self.fold_metrics]
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
    
    def get_confidence_interval(self, metric: str, alpha: float = 0.05) -> Tuple[float, float]:
        """Get confidence interval for a specific metric."""
        if not self.fold_metrics:
            return (0.0, 0.0)
        
        values = [fold.get(metric, 0.0) for fold in self.fold_metrics]
        values = np.array(values)
        
        if len(values) < 2:
            return (float(values[0]), float(values[0]))
        
        # Use t-distribution for small samples
        mean = np.mean(values)
        std_err = stats.sem(values)
        dof = len(values) - 1
        t_critical = stats.t.ppf(1 - alpha/2, dof)
        
        margin_error = t_critical * std_err
        return (float(mean - margin_error), float(mean + margin_error))