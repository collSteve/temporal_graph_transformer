"""
Comprehensive Benchmarking Suite for Temporal Graph Transformer

Complete evaluation framework that brings together all baseline methods,
cross-validation strategies, and performance analysis for comprehensive
comparison and benchmarking of airdrop hunter detection methods.
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
from datetime import datetime
import warnings

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import all baseline methods
from baselines import (
    BaselineMethodInterface, TrustaLabFramework, SubgraphFeaturePropagation,
    EnhancedGNNBaseline, LightGBMBaseline, RandomForestBaseline
)

# Import evaluation framework
from evaluation.cross_validation import CrossValidationFramework, CrossValidationResult

# Import training infrastructure
from scripts.train_enhanced import MultiDatasetTrainer, DatasetFactory


class BenchmarkingExperiment:
    """Single benchmarking experiment configuration."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def add_result(self, method_name: str, result: Dict[str, Any]):
        """Add result for a method."""
        self.results[method_name] = result
    
    def get_duration(self) -> float:
        """Get experiment duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class ComprehensiveBenchmarkingSuite:
    """
    Comprehensive benchmarking suite for airdrop hunter detection.
    
    Provides complete evaluation framework with multiple validation strategies,
    statistical analysis, and performance comparison across all baseline methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.cv_framework = CrossValidationFramework(config.get('cross_validation', {}))
        self.trainer = MultiDatasetTrainer(config)
        
        # Experiment tracking
        self.experiments = []
        self.global_results = {}
        
        print(f"🚀 Comprehensive Benchmarking Suite Initialized")
        print(f"Device: {self.device}")
        print(f"Cross-validation strategies: {list(self.cv_framework.strategies.keys())}")
    
    def create_baseline_methods(self) -> Dict[str, BaselineMethodInterface]:
        """Create all baseline methods for benchmarking."""
        methods = {}
        
        # Our main TGT model
        from scripts.train_enhanced import TemporalGraphTransformerBaseline
        tgt_config = self.config.get('model', {})
        methods['TemporalGraphTransformer'] = TemporalGraphTransformerBaseline(tgt_config)
        
        # TrustaLabs Framework
        trustalab_config = self.config.get('trustalab', {})
        methods['TrustaLabFramework'] = TrustaLabFramework(trustalab_config)
        
        # Subgraph Feature Propagation
        subgraph_config = self.config.get('subgraph_propagation', {})
        methods['SubgraphFeaturePropagation'] = SubgraphFeaturePropagation(subgraph_config)
        
        # Enhanced GNN methods
        gnn_base_config = self.config.get('enhanced_gnns', {})
        
        gat_config = gnn_base_config.copy()
        gat_config['model_type'] = 'gat'
        methods['GAT'] = EnhancedGNNBaseline(gat_config)
        
        sage_config = gnn_base_config.copy()
        sage_config['model_type'] = 'graphsage'
        methods['GraphSAGE'] = EnhancedGNNBaseline(sage_config)
        
        sybilgat_config = gnn_base_config.copy()
        sybilgat_config['model_type'] = 'sybilgat'
        methods['SybilGAT'] = EnhancedGNNBaseline(sybilgat_config)
        
        gcn_config = gnn_base_config.copy()
        gcn_config['model_type'] = 'gcn'
        methods['BasicGCN'] = EnhancedGNNBaseline(gcn_config)
        
        # Traditional ML methods
        traditional_ml_config = self.config.get('traditional_ml', {})
        methods['LightGBM'] = LightGBMBaseline(traditional_ml_config.copy())
        methods['RandomForest'] = RandomForestBaseline(traditional_ml_config.copy())
        
        print(f"✅ Created {len(methods)} baseline methods for benchmarking")
        return methods
    
    def run_comprehensive_benchmark(self, output_dir: str = None) -> Dict[str, Any]:
        """
        Run comprehensive benchmark across all methods and validation strategies.
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Complete benchmarking results
        """
        if output_dir is None:
            output_dir = f"./experiments/comprehensive_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE BENCHMARKING SUITE")
        print(f"{'='*80}")
        print(f"Output directory: {output_dir}")
        
        # Create datasets
        print(f"\n📊 Creating datasets...")
        data_loaders = self.trainer.create_data_loaders()
        
        if not data_loaders['train']:
            print("❌ No training data available")
            return {}
        
        # Create all baseline methods
        methods = self.create_baseline_methods()
        
        # Run experiments
        all_results = {}
        
        # Experiment 1: Standard Training and Evaluation
        print(f"\n{'='*60}")
        print("EXPERIMENT 1: Standard Training & Evaluation")
        print(f"{'='*60}")
        
        exp1 = BenchmarkingExperiment("standard_training", self.config)
        exp1.start_time = datetime.now()
        
        standard_results = self._run_standard_evaluation(methods, data_loaders)
        exp1.results = standard_results
        exp1.end_time = datetime.now()
        
        all_results['standard_training'] = standard_results
        self.experiments.append(exp1)
        
        # Experiment 2: Cross-Validation (if enabled)
        if self.config.get('cross_validation', {}).get('enabled', True):
            print(f"\n{'='*60}")
            print("EXPERIMENT 2: Cross-Validation Analysis")
            print(f"{'='*60}")
            
            exp2 = BenchmarkingExperiment("cross_validation", self.config)
            exp2.start_time = datetime.now()
            
            cv_results = self._run_cross_validation_analysis(methods, data_loaders)
            exp2.results = cv_results
            exp2.end_time = datetime.now()
            
            all_results['cross_validation'] = cv_results
            self.experiments.append(exp2)
        
        # Experiment 3: Performance Analysis
        print(f"\n{'='*60}")
        print("EXPERIMENT 3: Performance Analysis")
        print(f"{'='*60}")
        
        exp3 = BenchmarkingExperiment("performance_analysis", self.config)
        exp3.start_time = datetime.now()
        
        performance_results = self._run_performance_analysis(all_results)
        exp3.results = performance_results
        exp3.end_time = datetime.now()
        
        all_results['performance_analysis'] = performance_results
        self.experiments.append(exp3)
        
        # Save comprehensive results
        self._save_comprehensive_results(all_results, output_dir)
        
        # Print final summary
        self._print_final_summary(all_results)
        
        return all_results
    
    def _run_standard_evaluation(self, methods: Dict[str, BaselineMethodInterface], 
                                data_loaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        """Run standard training and evaluation."""
        results = {}
        
        for name, method in methods.items():
            print(f"\n🔧 Evaluating {name}...")
            
            try:
                start_time = time.time()
                
                # Train method
                train_result = method.train(
                    data_loaders['train'], 
                    data_loaders.get('val', data_loaders['train']), 
                    self.device
                )
                
                # Evaluate method
                test_result = method.evaluate(
                    data_loaders.get('test', data_loaders.get('val', data_loaders['train'])), 
                    self.device
                )
                
                training_time = time.time() - start_time
                
                # Combine results
                method_result = {
                    'training_result': train_result,
                    'test_result': test_result,
                    'training_time_seconds': training_time,
                    'test_f1': test_result.get('f1', 0.0),
                    'test_accuracy': test_result.get('accuracy', 0.0),
                    'test_precision': test_result.get('precision', 0.0),
                    'test_recall': test_result.get('recall', 0.0)
                }
                
                results[name] = method_result
                print(f"  ✅ {name} - F1: {test_result.get('f1', 0.0):.4f}, Time: {training_time:.1f}s")
                
            except Exception as e:
                print(f"  ❌ {name} failed: {e}")
                results[name] = {
                    'error': str(e),
                    'test_f1': 0.0,
                    'test_accuracy': 0.0,
                    'test_precision': 0.0,
                    'test_recall': 0.0,
                    'training_time_seconds': 0.0
                }
        
        return results
    
    def _run_cross_validation_analysis(self, methods: Dict[str, BaselineMethodInterface],
                                     data_loaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        """Run cross-validation analysis."""
        # Create combined dataset for CV
        datasets = []
        if data_loaders['train']:
            datasets.append(data_loaders['train'].dataset)
        if data_loaders.get('val'):
            datasets.append(data_loaders['val'].dataset)
        if data_loaders.get('test'):
            datasets.append(data_loaders['test'].dataset)
        
        if not datasets:
            return {}
        
        # Use first dataset for CV (in practice, combine all)
        cv_dataset = datasets[0]
        
        cv_results = {}
        
        # Stratified cross-validation
        print("Running stratified cross-validation...")
        stratified_results = {}
        
        for name, method in methods.items():
            try:
                result = self.cv_framework.validate_method(method, cv_dataset, 'stratified')
                stratified_results[name] = {
                    'mean_f1': result.get_mean_f1(),
                    'f1_confidence_interval': result.get_f1_confidence_interval(),
                    'summary_stats': result.summary_stats,
                    'fold_results': result.fold_results
                }
                print(f"  ✅ {name} - CV F1: {result.get_mean_f1():.4f}")
            except Exception as e:
                print(f"  ❌ {name} CV failed: {e}")
                stratified_results[name] = {'mean_f1': 0.0, 'error': str(e)}
        
        cv_results['stratified'] = stratified_results
        
        # Print CV comparison
        self._print_cv_summary(stratified_results)
        
        return cv_results
    
    def _run_performance_analysis(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive performance analysis."""
        analysis = {}
        
        # Extract standard results
        standard_results = all_results.get('standard_training', {})
        cv_results = all_results.get('cross_validation', {}).get('stratified', {})
        
        if not standard_results:
            return analysis
        
        # Performance ranking
        method_scores = []
        for method_name, result in standard_results.items():
            if 'error' not in result:
                f1_score = result.get('test_f1', 0.0)
                accuracy = result.get('test_accuracy', 0.0)
                training_time = result.get('training_time_seconds', 0.0)
                
                # Compute composite score (F1 weighted by efficiency)
                efficiency_score = 1.0 / (1.0 + training_time / 60.0)  # Penalty for long training
                composite_score = f1_score * 0.8 + accuracy * 0.1 + efficiency_score * 0.1
                
                method_scores.append({
                    'method': method_name,
                    'f1_score': f1_score,
                    'accuracy': accuracy,
                    'training_time': training_time,
                    'efficiency_score': efficiency_score,
                    'composite_score': composite_score
                })
        
        # Sort by composite score
        method_scores.sort(key=lambda x: x['composite_score'], reverse=True)
        analysis['performance_ranking'] = method_scores
        
        # Category analysis
        categories = {
            'neural_methods': ['TemporalGraphTransformer', 'GAT', 'GraphSAGE', 'SybilGAT', 'BasicGCN', 'SubgraphFeaturePropagation'],
            'traditional_ml': ['LightGBM', 'RandomForest'],
            'specialized_methods': ['TrustaLabFramework', 'SubgraphFeaturePropagation', 'SybilGAT']
        }
        
        category_analysis = {}
        for category, methods in categories.items():
            category_methods = [m for m in method_scores if m['method'] in methods]
            if category_methods:
                avg_f1 = np.mean([m['f1_score'] for m in category_methods])
                best_method = max(category_methods, key=lambda x: x['f1_score'])
                
                category_analysis[category] = {
                    'average_f1': avg_f1,
                    'best_method': best_method['method'],
                    'best_f1': best_method['f1_score'],
                    'method_count': len(category_methods)
                }
        
        analysis['category_analysis'] = category_analysis
        
        # Statistical insights
        if method_scores:
            f1_scores = [m['f1_score'] for m in method_scores]
            analysis['statistical_summary'] = {
                'mean_f1': float(np.mean(f1_scores)),
                'std_f1': float(np.std(f1_scores)),
                'min_f1': float(np.min(f1_scores)),
                'max_f1': float(np.max(f1_scores)),
                'f1_range': float(np.max(f1_scores) - np.min(f1_scores))
            }
        
        return analysis
    
    def _print_cv_summary(self, cv_results: Dict[str, Any]):
        """Print cross-validation summary."""
        print(f"\n📊 Cross-Validation Summary:")
        print(f"{'Method':<25} {'Mean F1':<12} {'95% CI':<20}")
        print(f"{'-'*60}")
        
        # Sort by mean F1
        sorted_methods = sorted(cv_results.items(), 
                              key=lambda x: x[1].get('mean_f1', 0), reverse=True)
        
        for method_name, result in sorted_methods:
            if 'error' not in result:
                mean_f1 = result.get('mean_f1', 0.0)
                ci = result.get('f1_confidence_interval', (0.0, 0.0))
                print(f"{method_name:<25} {mean_f1:<12.4f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    
    def _save_comprehensive_results(self, results: Dict[str, Any], output_dir: str):
        """Save comprehensive results to files."""
        # Save main results
        results_file = os.path.join(output_dir, 'comprehensive_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save experiment metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'device': str(self.device),
            'experiments': [
                {
                    'name': exp.name,
                    'duration_seconds': exp.get_duration(),
                    'start_time': exp.start_time.isoformat() if exp.start_time else None,
                    'end_time': exp.end_time.isoformat() if exp.end_time else None
                }
                for exp in self.experiments
            ]
        }
        
        metadata_file = os.path.join(output_dir, 'experiment_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Create summary CSV for easy analysis
        self._create_summary_csv(results, output_dir)
        
        print(f"✅ Results saved to: {output_dir}")
    
    def _create_summary_csv(self, results: Dict[str, Any], output_dir: str):
        """Create summary CSV for easy analysis."""
        standard_results = results.get('standard_training', {})
        
        if not standard_results:
            return
        
        # Create DataFrame
        rows = []
        for method_name, result in standard_results.items():
            if 'error' not in result:
                row = {
                    'Method': method_name,
                    'F1_Score': result.get('test_f1', 0.0),
                    'Accuracy': result.get('test_accuracy', 0.0),
                    'Precision': result.get('test_precision', 0.0),
                    'Recall': result.get('test_recall', 0.0),
                    'Training_Time_Seconds': result.get('training_time_seconds', 0.0)
                }
                rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values('F1_Score', ascending=False)
            
            csv_file = os.path.join(output_dir, 'results_summary.csv')
            df.to_csv(csv_file, index=False)
            print(f"📊 Summary CSV saved: {csv_file}")
    
    def _print_final_summary(self, results: Dict[str, Any]):
        """Print final benchmarking summary."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE BENCHMARKING SUMMARY")
        print(f"{'='*80}")
        
        # Standard results summary
        standard_results = results.get('standard_training', {})
        if standard_results:
            print(f"\n🏆 PERFORMANCE RANKING:")
            
            # Sort by F1 score
            sorted_methods = sorted(standard_results.items(), 
                                  key=lambda x: x[1].get('test_f1', 0) if 'error' not in x[1] else 0, 
                                  reverse=True)
            
            print(f"{'Rank':<5} {'Method':<25} {'F1':<8} {'Accuracy':<10} {'Time(s)':<10}")
            print(f"{'-'*70}")
            
            for i, (method_name, result) in enumerate(sorted_methods, 1):
                if 'error' not in result:
                    f1 = result.get('test_f1', 0.0)
                    acc = result.get('test_accuracy', 0.0)
                    time_s = result.get('training_time_seconds', 0.0)
                    print(f"{i:<5} {method_name:<25} {f1:<8.4f} {acc:<10.4f} {time_s:<10.1f}")
            
            # Best method
            if sorted_methods and 'error' not in sorted_methods[0][1]:
                best_method, best_result = sorted_methods[0]
                print(f"\n🥇 BEST PERFORMING METHOD: {best_method}")
                print(f"   F1 Score: {best_result.get('test_f1', 0.0):.4f}")
                print(f"   Accuracy: {best_result.get('test_accuracy', 0.0):.4f}")
                print(f"   Training Time: {best_result.get('training_time_seconds', 0.0):.1f}s")
        
        # Performance analysis
        perf_analysis = results.get('performance_analysis', {})
        if perf_analysis:
            category_analysis = perf_analysis.get('category_analysis', {})
            if category_analysis:
                print(f"\n📊 CATEGORY ANALYSIS:")
                for category, stats in category_analysis.items():
                    print(f"  {category.replace('_', ' ').title()}: Best = {stats['best_method']} (F1: {stats['best_f1']:.4f})")
        
        print(f"\n✅ Comprehensive benchmarking completed!")
        print(f"Total experiments run: {len(self.experiments)}")
        total_duration = sum(exp.get_duration() for exp in self.experiments)
        print(f"Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")


def run_full_benchmark(config_path: str = None, output_dir: str = None) -> Dict[str, Any]:
    """
    Convenience function to run full benchmark from config file.
    
    Args:
        config_path: Path to configuration file
        output_dir: Output directory for results
        
    Returns:
        Complete benchmarking results
    """
    # Load configuration
    if config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'datasets': [
                {
                    'type': 'arbitrum',
                    'data_path': './data',
                    'kwargs': {'max_sequence_length': 20}
                }
            ],
            'model': {'d_model': 128, 'epochs': 10},
            'trustalab': {'n_estimators': 50},
            'subgraph_propagation': {'epochs': 20},
            'enhanced_gnns': {'epochs': 20},
            'traditional_ml': {'num_boost_round': 50},
            'cross_validation': {'enabled': True, 'stratified_folds': 3}
        }
    
    # Create and run benchmark
    suite = ComprehensiveBenchmarkingSuite(config)
    return suite.run_comprehensive_benchmark(output_dir)