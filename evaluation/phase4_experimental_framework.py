"""
Phase 4: Experimental Validation Framework

Comprehensive evaluation infrastructure for systematic validation of all
baseline methods across all blockchain ecosystems with statistical rigor,
cross-chain generalization testing, and failure case analysis.

This framework addresses the core research questions:
1. Does TGT significantly outperform existing baselines?
2. Do methods generalize across blockchain ecosystems?
3. What temporal patterns does TGT capture that others miss?
4. Where do methods systematically fail and why?
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import os
import yaml
from pathlib import Path
import time
from datetime import datetime
import warnings
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import all baseline methods
from baselines import (
    BaselineMethodInterface, TrustaLabFramework, SubgraphFeaturePropagation,
    EnhancedGNNBaseline, LightGBMBaseline, RandomForestBaseline,
    TemporalGraphTransformerBaseline
)

# Import evaluation framework
from evaluation.cross_validation import CrossValidationFramework, CrossValidationResult
from evaluation.benchmarking_suite import ComprehensiveBenchmarkingSuite

# Import training infrastructure
from scripts.train_enhanced import MultiDatasetTrainer, DatasetFactory


class ExperimentalConfig:
    """Configuration for Phase 4 experimental validation."""
    
    def __init__(self, config_path: str = None):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default experimental configuration."""
        return {
            'evaluation': {
                'random_seeds': [42, 123, 456, 789, 999],  # 5 runs for statistical significance
                'cv_folds': 5,
                'test_size': 0.2,
                'parallel_jobs': 4
            },
            'datasets': {
                'blockchain_types': ['arbitrum', 'jupiter', 'optimism', 'blur', 'solana'],
                'pure_crypto': ['arbitrum', 'jupiter', 'optimism'],
                'nft_markets': ['blur', 'solana']
            },
            'methods': {
                'all_baselines': [
                    'TemporalGraphTransformer',
                    'TrustaLabFramework', 
                    'SubgraphFeaturePropagation',
                    'GAT', 'GraphSAGE', 'SybilGAT', 'BasicGCN',
                    'LightGBM', 'RandomForest'
                ],
                'primary_comparison': [
                    'TemporalGraphTransformer',
                    'TrustaLabFramework',
                    'SubgraphFeaturePropagation'
                ]
            },
            'experiments': {
                'comprehensive_evaluation': True,
                'cross_chain_generalization': True,
                'temporal_analysis': True,
                'failure_case_analysis': True,
                'ablation_studies': True,
                'interpretability_analysis': True
            },
            'output': {
                'base_dir': './phase4_results',
                'save_models': True,
                'save_predictions': True,
                'generate_visualizations': True
            }
        }
    
    def get_dataset_combinations(self) -> List[Tuple[str, str]]:
        """Get all cross-chain training/testing combinations."""
        chains = self.config['datasets']['blockchain_types']
        return list(itertools.permutations(chains, 2))
    
    def get_method_dataset_combinations(self) -> List[Tuple[str, str]]:
        """Get all method × dataset combinations for evaluation."""
        methods = self.config['methods']['all_baselines']
        datasets = self.config['datasets']['blockchain_types']
        return list(itertools.product(methods, datasets))


class ComprehensiveEvaluationRunner:
    """
    Systematic evaluation of all methods across all datasets.
    
    Handles large-scale experiments with proper statistical testing,
    result aggregation, and performance comparison.
    """
    
    def __init__(self, config: ExperimentalConfig):
        self.config = config
        self.results = {}
        self.experiment_start_time = None
        
    def run_systematic_evaluation(self) -> Dict[str, Any]:
        """
        Run systematic evaluation of all methods on all datasets.
        
        Returns comprehensive results with statistical analysis.
        """
        print("🚀 Starting Phase 4 Comprehensive Evaluation")
        print("=" * 60)
        
        self.experiment_start_time = time.time()
        
        # Get all method × dataset combinations
        combinations = self.config.get_method_dataset_combinations()
        print(f"Total experiments: {len(combinations)} method×dataset combinations")
        print(f"Random seeds: {len(self.config.config['evaluation']['random_seeds'])}")
        print(f"Total runs: {len(combinations) * len(self.config.config['evaluation']['random_seeds'])}")
        
        # Run experiments
        all_results = {}
        
        for method_name, dataset_type in combinations:
            print(f"\n📊 Evaluating {method_name} on {dataset_type}")
            
            method_results = self._evaluate_method_on_dataset(
                method_name, dataset_type
            )
            
            key = f"{method_name}_{dataset_type}"
            all_results[key] = method_results
            
            # Print quick summary
            if 'aggregated' in method_results:
                f1_mean = method_results['aggregated']['f1']['mean']
                f1_std = method_results['aggregated']['f1']['std']
                print(f"   F1: {f1_mean:.4f} ± {f1_std:.4f}")
        
        # Aggregate and analyze results
        print("\n📈 Analyzing Results...")
        analysis_results = self._analyze_comprehensive_results(all_results)
        
        # Save results
        output_dir = Path(self.config.config['output']['base_dir'])
        output_dir.mkdir(exist_ok=True)
        
        self._save_comprehensive_results(all_results, analysis_results, output_dir)
        
        total_time = time.time() - self.experiment_start_time
        print(f"\n✅ Comprehensive evaluation completed in {total_time/3600:.2f} hours")
        
        return {
            'raw_results': all_results,
            'analysis': analysis_results,
            'experiment_info': {
                'total_time_hours': total_time / 3600,
                'total_experiments': len(combinations),
                'random_seeds': len(self.config.config['evaluation']['random_seeds']),
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _evaluate_method_on_dataset(self, method_name: str, dataset_type: str) -> Dict[str, Any]:
        """Evaluate single method on single dataset with multiple random seeds."""
        
        # Results for all random seeds
        seed_results = []
        
        for seed in self.config.config['evaluation']['random_seeds']:
            print(f"   Running seed {seed}...")
            
            try:
                # Create method-specific configuration
                method_config = self._create_method_config(method_name, dataset_type, seed)
                
                # Run single evaluation
                result = self._run_single_evaluation(method_name, method_config)
                result['seed'] = seed
                seed_results.append(result)
                
            except Exception as e:
                print(f"   ❌ Seed {seed} failed: {e}")
                seed_results.append({
                    'seed': seed,
                    'error': str(e),
                    'failed': True
                })
        
        # Aggregate results across seeds
        aggregated = self._aggregate_seed_results(seed_results)
        
        return {
            'method': method_name,
            'dataset': dataset_type,
            'seed_results': seed_results,
            'aggregated': aggregated,
            'successful_runs': len([r for r in seed_results if not r.get('failed', False)])
        }
    
    def _create_method_config(self, method_name: str, dataset_type: str, seed: int) -> Dict[str, Any]:
        """Create configuration for specific method and dataset."""
        
        base_config = {
            'datasets': [{
                'type': dataset_type,
                'data_path': f'./data/{dataset_type}',
                'kwargs': {
                    'max_sequence_length': 50,
                    'min_transactions': 10
                }
            }],
            'evaluation': {
                'random_seed': seed,
                'cv_folds': self.config.config['evaluation']['cv_folds'],
                'test_size': self.config.config['evaluation']['test_size']
            },
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'batch_size': 32
        }
        
        # Method-specific configurations
        if method_name == 'TemporalGraphTransformer':
            base_config['model'] = {
                'd_model': 128,
                'temporal_layers': 3,
                'graph_layers': 3,
                'epochs': 50,
                'lr': 1e-3,
                'patience': 10
            }
        elif method_name == 'TrustaLabFramework':
            base_config['trustalab'] = {
                'n_estimators': 100,
                'random_state': seed
            }
        elif method_name == 'SubgraphFeaturePropagation':
            base_config['subgraph_propagation'] = {
                'input_dim': 128,
                'hidden_dim': 256,
                'epochs': 50,
                'lr': 1e-3,
                'patience': 10
            }
        elif method_name in ['GAT', 'GraphSAGE', 'SybilGAT', 'BasicGCN']:
            base_config['enhanced_gnns'] = {
                'model_type': method_name.lower(),
                'input_dim': 128,
                'hidden_dim': 256,
                'epochs': 50,
                'lr': 1e-3,
                'patience': 10
            }
        elif method_name in ['LightGBM', 'RandomForest']:
            base_config['traditional_ml'] = {
                'num_boost_round': 1000 if method_name == 'LightGBM' else None,
                'n_estimators': 100 if method_name == 'RandomForest' else None,
                'random_state': seed
            }
        
        return base_config
    
    def _run_single_evaluation(self, method_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run single method evaluation with given configuration."""
        
        # Create benchmarking suite
        suite = ComprehensiveBenchmarkingSuite(config)
        
        # Create data loaders
        data_loaders = suite.trainer.create_data_loaders()
        
        if not data_loaders.get('train') or not data_loaders.get('test'):
            raise ValueError("Failed to create required data loaders")
        
        # Create specific method
        if method_name == 'TemporalGraphTransformer':
            from baselines.temporal_graph_transformer_baseline import TemporalGraphTransformerBaseline
            method = TemporalGraphTransformerBaseline(config.get('model', {}))
        elif method_name == 'TrustaLabFramework':
            method = TrustaLabFramework(config.get('trustalab', {}))
        elif method_name == 'SubgraphFeaturePropagation':
            method = SubgraphFeaturePropagation(config.get('subgraph_propagation', {}))
        elif method_name in ['GAT', 'GraphSAGE', 'SybilGAT', 'BasicGCN']:
            gnn_config = config.get('enhanced_gnns', {})
            gnn_config['model_type'] = method_name.lower()
            method = EnhancedGNNBaseline(gnn_config)
        elif method_name == 'LightGBM':
            method = LightGBMBaseline(config.get('traditional_ml', {}))
        elif method_name == 'RandomForest':
            method = RandomForestBaseline(config.get('traditional_ml', {}))
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        # Train method
        device = torch.device(config.get('device', 'cpu'))
        train_results = method.train(data_loaders['train'], data_loaders['val'], device)
        
        # Evaluate method
        test_results = method.evaluate(data_loaders['test'], device)
        
        return {
            'train_results': train_results,
            'test_results': test_results,
            'config': config
        }
    
    def _aggregate_seed_results(self, seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across multiple random seeds."""
        
        successful_results = [r for r in seed_results if not r.get('failed', False)]
        
        if not successful_results:
            return {'error': 'All runs failed'}
        
        # Extract test metrics
        metrics = {}
        for result in successful_results:
            test_results = result.get('test_results', {})
            for metric, value in test_results.items():
                if isinstance(value, (int, float)):
                    if metric not in metrics:
                        metrics[metric] = []
                    metrics[metric].append(value)
        
        # Compute statistics
        aggregated = {}
        for metric, values in metrics.items():
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values,
                'n_runs': len(values)
            }
            
            # Confidence interval
            if len(values) > 1:
                ci = stats.t.interval(0.95, len(values)-1, 
                                    loc=np.mean(values), 
                                    scale=stats.sem(values))
                aggregated[metric]['ci_95'] = ci
        
        return aggregated
    
    def _analyze_comprehensive_results(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze comprehensive results for insights and statistical significance."""
        
        analysis = {
            'method_rankings': {},
            'dataset_difficulty': {},
            'cross_method_comparison': {},
            'statistical_significance': {},
            'best_combinations': []
        }
        
        # Method rankings by average F1 across all datasets
        method_f1_scores = {}
        for key, result in all_results.items():
            if 'aggregated' in result and 'f1' in result['aggregated']:
                method_name = result['method']
                f1_mean = result['aggregated']['f1']['mean']
                
                if method_name not in method_f1_scores:
                    method_f1_scores[method_name] = []
                method_f1_scores[method_name].append(f1_mean)
        
        # Average F1 per method
        for method, scores in method_f1_scores.items():
            analysis['method_rankings'][method] = {
                'avg_f1': np.mean(scores),
                'std_f1': np.std(scores),
                'dataset_count': len(scores)
            }
        
        # Sort by average F1
        analysis['method_rankings'] = dict(
            sorted(analysis['method_rankings'].items(), 
                  key=lambda x: x[1]['avg_f1'], reverse=True)
        )
        
        # Dataset difficulty analysis
        dataset_f1_scores = {}
        for key, result in all_results.items():
            if 'aggregated' in result and 'f1' in result['aggregated']:
                dataset_type = result['dataset']
                f1_mean = result['aggregated']['f1']['mean']
                
                if dataset_type not in dataset_f1_scores:
                    dataset_f1_scores[dataset_type] = []
                dataset_f1_scores[dataset_type].append(f1_mean)
        
        for dataset, scores in dataset_f1_scores.items():
            analysis['dataset_difficulty'][dataset] = {
                'avg_f1': np.mean(scores),
                'std_f1': np.std(scores),
                'method_count': len(scores)
            }
        
        # Find best method-dataset combinations
        best_combinations = []
        for key, result in all_results.items():
            if 'aggregated' in result and 'f1' in result['aggregated']:
                best_combinations.append({
                    'method': result['method'],
                    'dataset': result['dataset'],
                    'f1_mean': result['aggregated']['f1']['mean'],
                    'f1_std': result['aggregated']['f1']['std'],
                    'successful_runs': result['successful_runs']
                })
        
        # Sort by F1 score
        analysis['best_combinations'] = sorted(
            best_combinations, key=lambda x: x['f1_mean'], reverse=True
        )[:20]  # Top 20
        
        return analysis
    
    def _save_comprehensive_results(self, 
                                  raw_results: Dict[str, Any], 
                                  analysis: Dict[str, Any], 
                                  output_dir: Path):
        """Save comprehensive results and analysis."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        raw_file = output_dir / f"comprehensive_results_{timestamp}.json"
        with open(raw_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(raw_results)
            json.dump(serializable_results, f, indent=2)
        
        # Save analysis
        analysis_file = output_dir / f"comprehensive_analysis_{timestamp}.json"
        with open(analysis_file, 'w') as f:
            serializable_analysis = self._make_json_serializable(analysis)
            json.dump(serializable_analysis, f, indent=2)
        
        # Save summary report
        self._generate_summary_report(analysis, output_dir / f"summary_report_{timestamp}.md")
        
        print(f"📁 Results saved to {output_dir}")
        print(f"   Raw results: {raw_file.name}")
        print(f"   Analysis: {analysis_file.name}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return obj
    
    def _generate_summary_report(self, analysis: Dict[str, Any], output_file: Path):
        """Generate human-readable summary report."""
        
        report = []
        report.append("# Phase 4 Comprehensive Evaluation Summary")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Method rankings
        report.append("## 🏆 Method Rankings (by Average F1)")
        report.append("")
        for i, (method, stats) in enumerate(analysis['method_rankings'].items(), 1):
            avg_f1 = stats['avg_f1']
            std_f1 = stats['std_f1']
            count = stats['dataset_count']
            report.append(f"{i}. **{method}**: {avg_f1:.4f} ± {std_f1:.4f} (across {count} datasets)")
        report.append("")
        
        # Dataset difficulty
        report.append("## 📊 Dataset Difficulty Analysis")
        report.append("")
        difficulty_sorted = sorted(analysis['dataset_difficulty'].items(), 
                                 key=lambda x: x[1]['avg_f1'])
        
        for dataset, stats in difficulty_sorted:
            avg_f1 = stats['avg_f1']
            std_f1 = stats['std_f1']
            count = stats['method_count']
            difficulty = "Easy" if avg_f1 > 0.8 else "Medium" if avg_f1 > 0.7 else "Hard"
            report.append(f"- **{dataset}**: {avg_f1:.4f} ± {std_f1:.4f} ({difficulty}) - {count} methods")
        report.append("")
        
        # Best combinations
        report.append("## 🎯 Top Method-Dataset Combinations")
        report.append("")
        for i, combo in enumerate(analysis['best_combinations'][:10], 1):
            method = combo['method']
            dataset = combo['dataset']
            f1_mean = combo['f1_mean']
            f1_std = combo['f1_std']
            runs = combo['successful_runs']
            report.append(f"{i}. {method} on {dataset}: {f1_mean:.4f} ± {f1_std:.4f} ({runs} runs)")
        
        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))


class CrossChainGeneralizationAnalyzer:
    """
    Cross-chain generalization analysis.
    
    Tests whether methods trained on one blockchain generalize to others,
    identifying universal vs blockchain-specific patterns.
    """
    
    def __init__(self, config: ExperimentalConfig):
        self.config = config
        
    def run_cross_chain_analysis(self) -> Dict[str, Any]:
        """Run comprehensive cross-chain generalization analysis."""
        
        print("🔗 Starting Cross-Chain Generalization Analysis")
        print("=" * 50)
        
        # Get train/test combinations
        combinations = self.config.get_dataset_combinations()
        print(f"Testing {len(combinations)} cross-chain combinations")
        
        results = {}
        
        # Test primary methods on cross-chain combinations
        primary_methods = self.config.config['methods']['primary_comparison']
        
        for train_chain, test_chain in combinations:
            print(f"\n📊 Train on {train_chain} → Test on {test_chain}")
            
            for method_name in primary_methods:
                key = f"{method_name}_{train_chain}_to_{test_chain}"
                print(f"   Testing {method_name}...")
                
                try:
                    result = self._test_cross_chain_generalization(
                        method_name, train_chain, test_chain
                    )
                    results[key] = result
                    
                    # Quick summary
                    if 'test_f1' in result:
                        print(f"      F1: {result['test_f1']:.4f}")
                    
                except Exception as e:
                    print(f"      ❌ Failed: {e}")
                    results[key] = {'error': str(e)}
        
        # Analyze generalization patterns
        analysis = self._analyze_generalization_results(results)
        
        return {
            'raw_results': results,
            'analysis': analysis
        }
    
    def _test_cross_chain_generalization(self, 
                                       method_name: str, 
                                       train_chain: str, 
                                       test_chain: str) -> Dict[str, Any]:
        """Test single method's generalization from train_chain to test_chain."""
        
        # Create configurations
        train_config = self._create_cross_chain_config(method_name, train_chain, 'train')
        test_config = self._create_cross_chain_config(method_name, test_chain, 'test')
        
        # Create and train method on train_chain
        method = self._create_method_instance(method_name, train_config)
        
        # Create data loaders
        train_suite = ComprehensiveBenchmarkingSuite(train_config)
        train_loaders = train_suite.trainer.create_data_loaders()
        
        test_suite = ComprehensiveBenchmarkingSuite(test_config)
        test_loaders = test_suite.trainer.create_data_loaders()
        
        if not train_loaders.get('train') or not test_loaders.get('test'):
            raise ValueError("Failed to create required data loaders")
        
        # Train on source chain
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_results = method.train(
            train_loaders['train'], 
            train_loaders['val'], 
            device
        )
        
        # Test on target chain
        test_results = method.evaluate(test_loaders['test'], device)
        
        return {
            'method': method_name,
            'train_chain': train_chain,
            'test_chain': test_chain,
            'train_results': train_results,
            'test_results': test_results,
            **test_results  # Flatten test results for easy access
        }
    
    def _create_cross_chain_config(self, method_name: str, chain: str, split_type: str) -> Dict[str, Any]:
        """Create configuration for cross-chain testing."""
        
        config = {
            'datasets': [{
                'type': chain,
                'data_path': f'./data/{chain}',
                'kwargs': {
                    'max_sequence_length': 50,
                    'min_transactions': 10
                }
            }],
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'batch_size': 32
        }
        
        # Add method-specific config (reuse from ComprehensiveEvaluationRunner)
        if method_name == 'TemporalGraphTransformer':
            config['model'] = {
                'd_model': 128,
                'temporal_layers': 3,
                'graph_layers': 3,
                'epochs': 30,  # Shorter for cross-chain testing
                'lr': 1e-3,
                'patience': 5
            }
        # ... (other method configs similar to ComprehensiveEvaluationRunner)
        
        return config
    
    def _create_method_instance(self, method_name: str, config: Dict[str, Any]):
        """Create method instance for cross-chain testing."""
        
        if method_name == 'TemporalGraphTransformer':
            return TemporalGraphTransformerBaseline(config.get('model', {}))
        elif method_name == 'TrustaLabFramework':
            return TrustaLabFramework(config.get('trustalab', {}))
        elif method_name == 'SubgraphFeaturePropagation':
            return SubgraphFeaturePropagation(config.get('subgraph_propagation', {}))
        else:
            raise ValueError(f"Method {method_name} not supported for cross-chain analysis")
    
    def _analyze_generalization_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cross-chain generalization patterns."""
        
        analysis = {
            'method_generalization': {},
            'chain_compatibility': {},
            'generalization_drop': {},
            'best_generalizers': []
        }
        
        # Analyze by method
        for key, result in results.items():
            if 'error' in result:
                continue
                
            method = result['method']
            train_chain = result['train_chain']
            test_chain = result['test_chain']
            test_f1 = result.get('f1', 0.0)
            
            # Method generalization tracking
            if method not in analysis['method_generalization']:
                analysis['method_generalization'][method] = []
            
            analysis['method_generalization'][method].append({
                'train_chain': train_chain,
                'test_chain': test_chain,
                'f1': test_f1
            })
            
            # Best generalizers
            analysis['best_generalizers'].append({
                'method': method,
                'train_chain': train_chain,
                'test_chain': test_chain,
                'f1': test_f1
            })
        
        # Sort best generalizers by F1
        analysis['best_generalizers'] = sorted(
            analysis['best_generalizers'], 
            key=lambda x: x['f1'], 
            reverse=True
        )[:15]  # Top 15
        
        # Compute average generalization by method
        for method, results_list in analysis['method_generalization'].items():
            f1_scores = [r['f1'] for r in results_list]
            analysis['method_generalization'][method] = {
                'results': results_list,
                'avg_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'min_f1': np.min(f1_scores),
                'max_f1': np.max(f1_scores)
            }
        
        return analysis


# Main execution functions
def run_phase4_comprehensive_evaluation(config_path: str = None) -> Dict[str, Any]:
    """Run Phase 4 comprehensive evaluation."""
    
    config = ExperimentalConfig(config_path)
    runner = ComprehensiveEvaluationRunner(config)
    
    return runner.run_systematic_evaluation()


def run_phase4_cross_chain_analysis(config_path: str = None) -> Dict[str, Any]:
    """Run Phase 4 cross-chain generalization analysis."""
    
    config = ExperimentalConfig(config_path)
    analyzer = CrossChainGeneralizationAnalyzer(config)
    
    return analyzer.run_cross_chain_analysis()


if __name__ == "__main__":
    # Example usage
    print("🔬 Phase 4: Experimental Validation Framework")
    print("Select evaluation type:")
    print("1. Comprehensive Evaluation (all methods × all datasets)")
    print("2. Cross-Chain Generalization Analysis")
    print("3. Both (full Phase 4)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        results = run_phase4_comprehensive_evaluation()
        print("✅ Comprehensive evaluation completed")
    elif choice == "2":
        results = run_phase4_cross_chain_analysis()
        print("✅ Cross-chain analysis completed")
    elif choice == "3":
        print("Running full Phase 4 evaluation...")
        comp_results = run_phase4_comprehensive_evaluation()
        cross_results = run_phase4_cross_chain_analysis()
        print("✅ Full Phase 4 evaluation completed")
    else:
        print("Invalid choice")