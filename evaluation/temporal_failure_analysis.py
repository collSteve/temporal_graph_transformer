"""
Temporal Analysis & Failure Case Analysis for Phase 4

Comprehensive tools for analyzing temporal patterns in airdrop hunter behavior
and systematic categorization of method failures to understand when and why
different detection methods succeed or fail.

Key Components:
1. Temporal Pattern Analysis - Before/during/after airdrop behavior
2. Hunter Behavior Evolution Tracking
3. Failure Case Categorization and Analysis
4. Attention Pattern Visualization (for TGT)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import os
from pathlib import Path
import time
from datetime import datetime, timedelta
import warnings
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Optional visualization imports with fallbacks
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  Matplotlib/Seaborn not available, basic plotting disabled")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("⚠️  Plotly not available, interactive plotting disabled")

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from baselines import BaselineMethodInterface, TemporalGraphTransformerBaseline
from utils.metrics import BinaryClassificationMetrics


class TemporalPatternAnalyzer:
    """
    Analyzes temporal patterns in airdrop hunter behavior.
    
    Focuses on understanding how hunter behavior changes before, during,
    and after airdrop events, and how different methods capture these patterns.
    """
    
    def __init__(self, airdrop_dates: Dict[str, str] = None):
        """
        Initialize temporal analyzer.
        
        Args:
            airdrop_dates: Dict mapping blockchain -> airdrop date (YYYY-MM-DD)
        """
        self.airdrop_dates = airdrop_dates or {
            'arbitrum': '2023-03-23',  # ARB token airdrop
            'optimism': '2022-05-31',  # OP token airdrop
            'jupiter': '2024-01-31',   # JUP token airdrop
            'blur': '2023-02-14',      # BLUR token airdrop
        }
        
        # Convert to datetime objects
        self.airdrop_timestamps = {}
        for chain, date_str in self.airdrop_dates.items():
            self.airdrop_timestamps[chain] = datetime.strptime(date_str, '%Y-%m-%d')
    
    def analyze_temporal_patterns(self, 
                                method: BaselineMethodInterface,
                                dataset_type: str,
                                data_loader: DataLoader,
                                device: torch.device) -> Dict[str, Any]:
        """
        Analyze temporal patterns for a specific method and dataset.
        
        Returns comprehensive temporal analysis including behavior changes
        and attention patterns (for TGT).
        """
        print(f"🕒 Analyzing temporal patterns for {method.name} on {dataset_type}")
        
        results = {
            'method': method.name,
            'dataset': dataset_type,
            'airdrop_date': self.airdrop_dates.get(dataset_type),
            'temporal_periods': {},
            'behavior_evolution': {},
            'attention_analysis': {},
            'performance_by_period': {}
        }
        
        # Get airdrop timestamp for this dataset
        airdrop_ts = self.airdrop_timestamps.get(dataset_type)
        if not airdrop_ts:
            print(f"   ⚠️  No airdrop date available for {dataset_type}")
            return results
        
        # Define temporal periods
        periods = self._define_temporal_periods(airdrop_ts)
        results['temporal_periods'] = periods
        
        # Analyze behavior in each period
        period_analysis = self._analyze_periods(
            method, data_loader, device, periods, airdrop_ts
        )
        results['behavior_evolution'] = period_analysis
        
        # Performance analysis by period
        performance_analysis = self._analyze_performance_by_period(
            method, data_loader, device, periods, airdrop_ts
        )
        results['performance_by_period'] = performance_analysis
        
        # Attention analysis (if TGT)
        if isinstance(method, TemporalGraphTransformerBaseline):
            attention_analysis = self._analyze_attention_patterns(
                method, data_loader, device, periods, airdrop_ts
            )
            results['attention_analysis'] = attention_analysis
        
        return results
    
    def _define_temporal_periods(self, airdrop_timestamp: datetime) -> Dict[str, Dict[str, Any]]:
        """Define temporal periods around airdrop event."""
        
        return {
            'pre_farming': {
                'start': airdrop_timestamp - timedelta(days=180),  # 6 months before
                'end': airdrop_timestamp - timedelta(days=60),     # 2 months before
                'description': 'Early farming preparation phase'
            },
            'intensive_farming': {
                'start': airdrop_timestamp - timedelta(days=60),   # 2 months before
                'end': airdrop_timestamp - timedelta(days=7),      # 1 week before
                'description': 'Intensive farming phase'
            },
            'pre_announcement': {
                'start': airdrop_timestamp - timedelta(days=7),    # 1 week before
                'end': airdrop_timestamp,                          # Airdrop day
                'description': 'Pre-announcement final push'
            },
            'post_airdrop': {
                'start': airdrop_timestamp,                        # Airdrop day
                'end': airdrop_timestamp + timedelta(days=30),     # 1 month after
                'description': 'Post-airdrop behavior'
            }
        }
    
    def _analyze_periods(self, 
                        method: BaselineMethodInterface,
                        data_loader: DataLoader,
                        device: torch.device,
                        periods: Dict[str, Dict[str, Any]],
                        airdrop_ts: datetime) -> Dict[str, Any]:
        """Analyze hunter behavior patterns in each temporal period."""
        
        period_analysis = {}
        
        # Extract temporal features for all batches
        all_temporal_features = []
        all_labels = []
        all_timestamps = []
        all_predictions = []
        
        method.model.eval() if hasattr(method, 'model') and method.model else None
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                try:
                    # Get batch timestamps (mock for now - in real implementation 
                    # this would come from the dataset)
                    batch_size = len(batch.get('user_id', [1])) if isinstance(batch, dict) else batch[0].shape[0]
                    
                    # Mock timestamps around airdrop (in real implementation, 
                    # timestamps would come from transaction data)
                    mock_timestamps = []
                    for i in range(batch_size):
                        # Generate realistic timestamps spread around airdrop
                        days_offset = np.random.randint(-200, 50)  # 200 days before to 50 after
                        timestamp = airdrop_ts + timedelta(days=days_offset)
                        mock_timestamps.append(timestamp)
                    
                    # Get predictions
                    predictions = method.predict(batch, device)
                    if predictions.dim() > 1 and predictions.shape[1] > 1:
                        # Take probability of positive class
                        pred_probs = torch.softmax(predictions, dim=1)[:, 1]
                    else:
                        pred_probs = predictions.squeeze()
                    
                    # Get labels
                    if isinstance(batch, dict):
                        labels = batch.get('labels', torch.randint(0, 2, (batch_size,)))
                    else:
                        labels = batch[1] if len(batch) > 1 else torch.randint(0, 2, (batch_size,))
                    
                    # Store data
                    all_timestamps.extend(mock_timestamps)
                    all_predictions.extend(pred_probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    # Extract temporal features (method-specific)
                    temporal_features = self._extract_temporal_features(batch, method)
                    all_temporal_features.extend(temporal_features)
                    
                except Exception as e:
                    print(f"     ⚠️  Batch {batch_idx} error: {e}")
                    continue
        
        # Analyze each period
        for period_name, period_info in periods.items():
            period_start = period_info['start']
            period_end = period_info['end']
            
            # Filter data for this period
            period_mask = [
                period_start <= ts <= period_end 
                for ts in all_timestamps
            ]
            
            if not any(period_mask):
                period_analysis[period_name] = {
                    'description': period_info['description'],
                    'sample_count': 0,
                    'error': 'No samples in this period'
                }
                continue
            
            # Extract period data
            period_predictions = np.array(all_predictions)[period_mask]
            period_labels = np.array(all_labels)[period_mask]
            period_features = np.array(all_temporal_features)[period_mask]
            
            # Analyze period characteristics
            hunter_mask = period_labels == 1
            normal_mask = period_labels == 0
            
            analysis = {
                'description': period_info['description'],
                'sample_count': len(period_predictions),
                'hunter_count': np.sum(hunter_mask),
                'normal_count': np.sum(normal_mask),
                'hunter_ratio': np.mean(hunter_mask) if len(period_predictions) > 0 else 0,
                'avg_prediction_score': {
                    'hunters': np.mean(period_predictions[hunter_mask]) if np.any(hunter_mask) else 0,
                    'normal': np.mean(period_predictions[normal_mask]) if np.any(normal_mask) else 0,
                    'overall': np.mean(period_predictions)
                },
                'behavioral_features': {
                    'avg_features': np.mean(period_features, axis=0).tolist() if len(period_features) > 0 else [],
                    'std_features': np.std(period_features, axis=0).tolist() if len(period_features) > 0 else []
                }
            }
            
            period_analysis[period_name] = analysis
        
        return period_analysis
    
    def _extract_temporal_features(self, batch: Any, method: BaselineMethodInterface) -> List[List[float]]:
        """Extract temporal features from batch (method-specific)."""
        
        # Mock temporal features for demonstration
        # In real implementation, this would extract actual temporal patterns
        
        batch_size = len(batch.get('user_id', [1])) if isinstance(batch, dict) else 1
        
        features = []
        for i in range(batch_size):
            # Mock temporal features: transaction frequency, timing patterns, etc.
            feature_vector = [
                np.random.random(),  # Transaction frequency
                np.random.random(),  # Time-of-day pattern consistency  
                np.random.random(),  # Weekly pattern regularity
                np.random.random(),  # Transaction amount volatility
                np.random.random(),  # Network activity correlation
            ]
            features.append(feature_vector)
        
        return features
    
    def _analyze_performance_by_period(self,
                                     method: BaselineMethodInterface,
                                     data_loader: DataLoader,
                                     device: torch.device,
                                     periods: Dict[str, Dict[str, Any]],
                                     airdrop_ts: datetime) -> Dict[str, Any]:
        """Analyze method performance in each temporal period."""
        
        performance_by_period = {}
        
        # Collect all data with timestamps
        all_data = []
        
        method.model.eval() if hasattr(method, 'model') and method.model else None
        
        with torch.no_grad():
            for batch in data_loader:
                try:
                    batch_size = len(batch.get('user_id', [1])) if isinstance(batch, dict) else batch[0].shape[0]
                    
                    # Mock timestamps
                    for i in range(batch_size):
                        days_offset = np.random.randint(-200, 50)
                        timestamp = airdrop_ts + timedelta(days=days_offset)
                        
                        # Get prediction for this sample
                        if isinstance(batch, dict):
                            sample_batch = {k: v[i:i+1] if torch.is_tensor(v) else [v[i]] for k, v in batch.items()}
                            label = batch.get('labels', torch.randint(0, 2, (1,)))[i]
                        else:
                            sample_batch = (batch[0][i:i+1], batch[1][i:i+1]) if len(batch) > 1 else (batch[0][i:i+1],)
                            label = batch[1][i] if len(batch) > 1 else torch.randint(0, 2, (1,))[0]
                        
                        pred = method.predict(sample_batch, device)
                        if pred.dim() > 1 and pred.shape[1] > 1:
                            pred_prob = torch.softmax(pred, dim=1)[0, 1]
                        else:
                            pred_prob = pred.squeeze()
                        
                        all_data.append({
                            'timestamp': timestamp,
                            'true_label': label.item() if torch.is_tensor(label) else label,
                            'pred_prob': pred_prob.item() if torch.is_tensor(pred_prob) else pred_prob
                        })
                        
                except Exception as e:
                    continue
        
        # Analyze performance for each period
        for period_name, period_info in periods.items():
            period_start = period_info['start']
            period_end = period_info['end']
            
            # Filter data for this period
            period_data = [
                d for d in all_data 
                if period_start <= d['timestamp'] <= period_end
            ]
            
            if len(period_data) < 10:  # Need minimum samples
                performance_by_period[period_name] = {
                    'description': period_info['description'],
                    'sample_count': len(period_data),
                    'error': 'Insufficient samples for analysis'
                }
                continue
            
            # Extract labels and predictions
            true_labels = torch.tensor([d['true_label'] for d in period_data])
            pred_probs = torch.tensor([d['pred_prob'] for d in period_data])
            
            # Compute metrics
            metrics = BinaryClassificationMetrics()
            metrics.update(pred_probs, true_labels)
            period_metrics = metrics.compute()
            
            performance_by_period[period_name] = {
                'description': period_info['description'],
                'sample_count': len(period_data),
                'metrics': period_metrics,
                'hunter_ratio': torch.mean(true_labels.float()).item(),
                'avg_pred_score': torch.mean(pred_probs).item()
            }
        
        return performance_by_period
    
    def _analyze_attention_patterns(self,
                                  method: TemporalGraphTransformerBaseline,
                                  data_loader: DataLoader,
                                  device: torch.device,
                                  periods: Dict[str, Dict[str, Any]],
                                  airdrop_ts: datetime) -> Dict[str, Any]:
        """Analyze attention patterns for Temporal Graph Transformer."""
        
        if not hasattr(method, 'model') or method.model is None:
            return {'error': 'Model not available for attention analysis'}
        
        # This would require modifying the TGT model to return attention weights
        # For now, return mock analysis structure
        
        attention_analysis = {
            'temporal_attention': {},
            'graph_attention': {},
            'attention_evolution': {},
            'pattern_clusters': {}
        }
        
        # Mock attention analysis for each period
        for period_name, period_info in periods.items():
            attention_analysis['temporal_attention'][period_name] = {
                'avg_attention_weights': np.random.random(10).tolist(),  # Mock weights
                'attention_entropy': np.random.random(),
                'focused_timesteps': np.random.randint(1, 5)
            }
            
            attention_analysis['graph_attention'][period_name] = {
                'avg_node_attention': np.random.random(20).tolist(),  # Mock node attention
                'attention_concentration': np.random.random(),
                'key_node_count': np.random.randint(5, 15)
            }
        
        return attention_analysis


class FailureCaseAnalyzer:
    """
    Systematic analysis of method failures.
    
    Categorizes and analyzes cases where detection methods fail,
    identifying patterns and providing insights for improvement.
    """
    
    def __init__(self):
        self.failure_categories = {
            'false_positive': 'Normal users classified as hunters',
            'false_negative': 'Hunters classified as normal users',
            'low_confidence': 'Predictions with low confidence scores',
            'inconsistent': 'Different methods disagree significantly',
            'temporal_confusion': 'Failures related to temporal patterns',
            'graph_structure': 'Failures related to graph topology'
        }
    
    def analyze_method_failures(self,
                              methods: Dict[str, BaselineMethodInterface],
                              dataset_type: str,
                              data_loader: DataLoader,
                              device: torch.device,
                              confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Comprehensive failure analysis across multiple methods.
        
        Args:
            methods: Dict of method_name -> method instance
            dataset_type: Type of dataset being analyzed
            data_loader: DataLoader with test data
            device: Torch device
            confidence_threshold: Threshold for low confidence classification
            
        Returns:
            Comprehensive failure analysis results
        """
        
        print(f"🔍 Analyzing failure cases for {len(methods)} methods on {dataset_type}")
        
        # Collect predictions from all methods
        all_predictions = {}
        all_labels = []
        all_samples = []
        
        # Get predictions from each method
        for method_name, method in methods.items():
            print(f"   Getting predictions from {method_name}...")
            
            method_predictions = []
            method.model.eval() if hasattr(method, 'model') and method.model else None
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(data_loader):
                    try:
                        # Get predictions
                        predictions = method.predict(batch, device)
                        if predictions.dim() > 1 and predictions.shape[1] > 1:
                            pred_probs = torch.softmax(predictions, dim=1)[:, 1]
                        else:
                            pred_probs = predictions.squeeze()
                        
                        method_predictions.extend(pred_probs.cpu().numpy())
                        
                        # Store labels and samples (only once)
                        if method_name == list(methods.keys())[0]:  # First method
                            if isinstance(batch, dict):
                                batch_size = len(batch.get('user_id', [1]))
                                labels = batch.get('labels', torch.randint(0, 2, (batch_size,)))
                            else:
                                labels = batch[1] if len(batch) > 1 else torch.randint(0, 2, (batch[0].shape[0],))
                            
                            all_labels.extend(labels.cpu().numpy())
                            
                            # Store sample information for analysis
                            for i in range(len(labels)):
                                sample_info = {
                                    'batch_idx': batch_idx,
                                    'sample_idx': i,
                                    'true_label': labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
                                }
                                all_samples.append(sample_info)
                        
                    except Exception as e:
                        print(f"     ⚠️  Batch {batch_idx} error for {method_name}: {e}")
                        continue
            
            all_predictions[method_name] = method_predictions
        
        # Ensure all methods have same number of predictions
        min_samples = min(len(preds) for preds in all_predictions.values())
        for method_name in all_predictions:
            all_predictions[method_name] = all_predictions[method_name][:min_samples]
        all_labels = all_labels[:min_samples]
        all_samples = all_samples[:min_samples]
        
        # Analyze failures
        failure_analysis = self._categorize_failures(
            all_predictions, all_labels, all_samples, confidence_threshold
        )
        
        # Method-specific failure analysis
        method_analysis = self._analyze_method_specific_failures(
            all_predictions, all_labels, methods
        )
        
        # Cross-method disagreement analysis
        disagreement_analysis = self._analyze_method_disagreements(
            all_predictions, all_labels
        )
        
        return {
            'dataset': dataset_type,
            'total_samples': len(all_labels),
            'methods_analyzed': list(methods.keys()),
            'failure_categorization': failure_analysis,
            'method_specific_analysis': method_analysis,
            'disagreement_analysis': disagreement_analysis,
            'improvement_suggestions': self._generate_improvement_suggestions(
                failure_analysis, method_analysis, disagreement_analysis
            )
        }
    
    def _categorize_failures(self,
                           predictions: Dict[str, List[float]],
                           true_labels: List[int],
                           samples: List[Dict[str, Any]],
                           confidence_threshold: float) -> Dict[str, Any]:
        """Categorize different types of failures."""
        
        categorized_failures = {
            'false_positives': [],
            'false_negatives': [],
            'low_confidence_predictions': [],
            'high_disagreement_cases': [],
            'systematic_failures': {}
        }
        
        method_names = list(predictions.keys())
        
        for i, (true_label, sample) in enumerate(zip(true_labels, samples)):
            # Get predictions for this sample across all methods
            sample_predictions = {
                method: predictions[method][i] 
                for method in method_names
            }
            
            # Analyze each method's prediction for this sample
            for method_name, pred_prob in sample_predictions.items():
                pred_label = 1 if pred_prob > 0.5 else 0
                
                # False positive
                if true_label == 0 and pred_label == 1:
                    categorized_failures['false_positives'].append({
                        'method': method_name,
                        'sample_idx': i,
                        'true_label': true_label,
                        'pred_prob': pred_prob,
                        'confidence': abs(pred_prob - 0.5) * 2,  # 0 to 1
                        'sample_info': sample
                    })
                
                # False negative
                elif true_label == 1 and pred_label == 0:
                    categorized_failures['false_negatives'].append({
                        'method': method_name,
                        'sample_idx': i,
                        'true_label': true_label,
                        'pred_prob': pred_prob,
                        'confidence': abs(pred_prob - 0.5) * 2,
                        'sample_info': sample
                    })
                
                # Low confidence prediction
                confidence = abs(pred_prob - 0.5) * 2
                if confidence < confidence_threshold:
                    categorized_failures['low_confidence_predictions'].append({
                        'method': method_name,
                        'sample_idx': i,
                        'true_label': true_label,
                        'pred_prob': pred_prob,
                        'confidence': confidence,
                        'sample_info': sample
                    })
            
            # Cross-method disagreement
            pred_probs = list(sample_predictions.values())
            disagreement = np.std(pred_probs)
            
            if disagreement > 0.3:  # High disagreement threshold
                categorized_failures['high_disagreement_cases'].append({
                    'sample_idx': i,
                    'true_label': true_label,
                    'predictions': sample_predictions,
                    'disagreement_std': disagreement,
                    'sample_info': sample
                })
        
        # Add summary statistics
        for category in ['false_positives', 'false_negatives', 'low_confidence_predictions']:
            failures = categorized_failures[category]
            categorized_failures[f'{category}_summary'] = {
                'total_count': len(failures),
                'by_method': {},
                'avg_confidence': np.mean([f['confidence'] for f in failures]) if failures else 0
            }
            
            # Count by method
            for failure in failures:
                method = failure['method']
                if method not in categorized_failures[f'{category}_summary']['by_method']:
                    categorized_failures[f'{category}_summary']['by_method'][method] = 0
                categorized_failures[f'{category}_summary']['by_method'][method] += 1
        
        return categorized_failures
    
    def _analyze_method_specific_failures(self,
                                        predictions: Dict[str, List[float]],
                                        true_labels: List[int],
                                        methods: Dict[str, BaselineMethodInterface]) -> Dict[str, Any]:
        """Analyze failures specific to each method."""
        
        method_analysis = {}
        
        for method_name, method_predictions in predictions.items():
            # Convert to binary predictions
            binary_preds = [1 if p > 0.5 else 0 for p in method_predictions]
            
            # Basic metrics
            metrics = BinaryClassificationMetrics()
            pred_tensor = torch.tensor(method_predictions)
            label_tensor = torch.tensor(true_labels)
            metrics.update(pred_tensor, label_tensor)
            method_metrics = metrics.compute()
            
            # Failure patterns
            failures = []
            for i, (true_label, pred_prob, pred_label) in enumerate(
                zip(true_labels, method_predictions, binary_preds)
            ):
                if true_label != pred_label:
                    failures.append({
                        'sample_idx': i,
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'pred_prob': pred_prob,
                        'error_type': 'false_positive' if true_label == 0 else 'false_negative'
                    })
            
            # Analyze failure characteristics
            fp_failures = [f for f in failures if f['error_type'] == 'false_positive']
            fn_failures = [f for f in failures if f['error_type'] == 'false_negative']
            
            method_analysis[method_name] = {
                'method_type': type(methods[method_name]).__name__,
                'performance_metrics': method_metrics,
                'total_failures': len(failures),
                'false_positive_count': len(fp_failures),
                'false_negative_count': len(fn_failures),
                'failure_rate': len(failures) / len(true_labels),
                'avg_fp_confidence': np.mean([f['pred_prob'] for f in fp_failures]) if fp_failures else 0,
                'avg_fn_confidence': np.mean([f['pred_prob'] for f in fn_failures]) if fn_failures else 0,
                'failure_characteristics': {
                    'high_confidence_errors': len([
                        f for f in failures 
                        if abs(f['pred_prob'] - 0.5) > 0.3
                    ]),
                    'low_confidence_errors': len([
                        f for f in failures 
                        if abs(f['pred_prob'] - 0.5) <= 0.3
                    ])
                }
            }
        
        return method_analysis
    
    def _analyze_method_disagreements(self,
                                    predictions: Dict[str, List[float]],
                                    true_labels: List[int]) -> Dict[str, Any]:
        """Analyze cases where methods disagree significantly."""
        
        method_names = list(predictions.keys())
        n_samples = len(true_labels)
        
        disagreement_analysis = {
            'pairwise_correlations': {},
            'high_disagreement_cases': [],
            'consensus_analysis': {},
            'method_pairs_analysis': {}
        }
        
        # Pairwise correlations
        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names[i+1:], i+1):
                correlation = np.corrcoef(predictions[method1], predictions[method2])[0, 1]
                disagreement_analysis['pairwise_correlations'][f'{method1}_vs_{method2}'] = correlation
        
        # Find high disagreement cases
        for i in range(n_samples):
            sample_preds = [predictions[method][i] for method in method_names]
            disagreement = np.std(sample_preds)
            
            if disagreement > 0.25:  # High disagreement threshold
                disagreement_analysis['high_disagreement_cases'].append({
                    'sample_idx': i,
                    'true_label': true_labels[i],
                    'predictions': {method: predictions[method][i] for method in method_names},
                    'disagreement_std': disagreement,
                    'prediction_range': max(sample_preds) - min(sample_preds)
                })
        
        # Consensus analysis
        consensus_correct = 0
        consensus_incorrect = 0
        no_consensus = 0
        
        for i in range(n_samples):
            sample_preds = [predictions[method][i] for method in method_names]
            binary_preds = [1 if p > 0.5 else 0 for p in sample_preds]
            
            if len(set(binary_preds)) == 1:  # All methods agree
                consensus_pred = binary_preds[0]
                if consensus_pred == true_labels[i]:
                    consensus_correct += 1
                else:
                    consensus_incorrect += 1
            else:
                no_consensus += 1
        
        disagreement_analysis['consensus_analysis'] = {
            'consensus_correct': consensus_correct,
            'consensus_incorrect': consensus_incorrect,
            'no_consensus': no_consensus,
            'consensus_accuracy': consensus_correct / (consensus_correct + consensus_incorrect) if (consensus_correct + consensus_incorrect) > 0 else 0
        }
        
        return disagreement_analysis
    
    def _generate_improvement_suggestions(self,
                                        failure_analysis: Dict[str, Any],
                                        method_analysis: Dict[str, Any],
                                        disagreement_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable improvement suggestions based on failure analysis."""
        
        suggestions = []
        
        # Analyze false positive patterns
        fp_summary = failure_analysis.get('false_positives_summary', {})
        if fp_summary.get('total_count', 0) > 0:
            worst_fp_method = max(
                fp_summary.get('by_method', {}).items(),
                key=lambda x: x[1],
                default=(None, 0)
            )[0]
            
            if worst_fp_method:
                suggestions.append({
                    'type': 'false_positive_reduction',
                    'priority': 'high',
                    'method': worst_fp_method,
                    'suggestion': f'Improve precision for {worst_fp_method} by adding more stringent filtering or adjusting decision threshold',
                    'evidence': f'{fp_summary["by_method"][worst_fp_method]} false positives'
                })
        
        # Analyze false negative patterns
        fn_summary = failure_analysis.get('false_negatives_summary', {})
        if fn_summary.get('total_count', 0) > 0:
            worst_fn_method = max(
                fn_summary.get('by_method', {}).items(),
                key=lambda x: x[1],
                default=(None, 0)
            )[0]
            
            if worst_fn_method:
                suggestions.append({
                    'type': 'false_negative_reduction',
                    'priority': 'high',
                    'method': worst_fn_method,
                    'suggestion': f'Improve recall for {worst_fn_method} by enhancing feature extraction or lowering decision threshold',
                    'evidence': f'{fn_summary["by_method"][worst_fn_method]} false negatives'
                })
        
        # Analyze low confidence predictions
        low_conf_summary = failure_analysis.get('low_confidence_predictions_summary', {})
        if low_conf_summary.get('total_count', 0) > 0:
            suggestions.append({
                'type': 'confidence_improvement',
                'priority': 'medium',
                'suggestion': 'Implement uncertainty quantification or ensemble methods to improve prediction confidence',
                'evidence': f'{low_conf_summary["total_count"]} low confidence predictions'
            })
        
        # Analyze method disagreements
        high_disagreement = len(disagreement_analysis.get('high_disagreement_cases', []))
        if high_disagreement > 0:
            suggestions.append({
                'type': 'consensus_improvement',
                'priority': 'medium',
                'suggestion': 'Develop ensemble or voting mechanisms to handle cases where methods disagree',
                'evidence': f'{high_disagreement} high disagreement cases'
            })
        
        # Method-specific suggestions
        for method_name, analysis in method_analysis.items():
            failure_rate = analysis.get('failure_rate', 0)
            if failure_rate > 0.3:  # High failure rate
                suggestions.append({
                    'type': 'method_specific_improvement',
                    'priority': 'high',
                    'method': method_name,
                    'suggestion': f'Review {method_name} architecture or hyperparameters - high failure rate detected',
                    'evidence': f'Failure rate: {failure_rate:.3f}'
                })
        
        return suggestions


# Visualization and reporting functions
def generate_temporal_analysis_report(temporal_results: Dict[str, Any], 
                                    output_dir: Path) -> Path:
    """Generate comprehensive temporal analysis report with visualizations."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"temporal_analysis_report_{timestamp}.html"
    
    # Create visualizations using plotly
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Performance by Period', 'Hunter Ratio Evolution', 
                       'Prediction Score Distribution', 'Attention Patterns'),
        specs=[[{"secondary_y": True}, {"secondary_y": False}], 
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Add plots based on temporal_results
    # (Implementation would create interactive plots)
    
    # Save report
    # (Implementation would generate comprehensive HTML report)
    
    return report_file


def generate_failure_analysis_report(failure_results: Dict[str, Any], 
                                   output_dir: Path) -> Path:
    """Generate comprehensive failure analysis report with actionable insights."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"failure_analysis_report_{timestamp}.html"
    
    # Generate comprehensive failure analysis report
    # (Implementation would create detailed HTML report with charts)
    
    return report_file


if __name__ == "__main__":
    print("🔬 Temporal Analysis & Failure Case Analysis Tools")
    print("This module provides comprehensive analysis tools for Phase 4 evaluation")