#!/usr/bin/env python3
"""
Phase 3 Evaluation Framework Test Suite

Comprehensive testing of evaluation components:
- Cross-validation framework
- Benchmarking suite
- Statistical analysis
- Result management
"""

import sys
import os
import tempfile
import shutil
import torch
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_cross_validation_framework():
    """Test Cross-Validation Framework implementation."""
    print("Testing Cross-Validation Framework...")
    
    try:
        from evaluation.cross_validation import CrossValidationFramework, CrossValidationResult
        
        # Test initialization
        config = {
            'stratified_folds': 3,
            'temporal_folds': 3,
            'random_state': 42
        }
        
        cv_framework = CrossValidationFramework(config)
        assert 'stratified' in cv_framework.strategies
        assert 'temporal' in cv_framework.strategies
        assert 'chain' in cv_framework.strategies
        print("   ✅ CV Framework initialization successful")
        
        # Test with a simple mock method
        from baselines.base_interface import BaselineMethodInterface
        
        class MockMethod(BaselineMethodInterface):
            def __init__(self):
                super().__init__({'test': True})
                self.name = 'MockMethod'
            
            def train(self, train_loader, val_loader, device):
                return {'best_val_f1': 0.75}
            
            def evaluate(self, test_loader, device):
                return {'f1': 0.8, 'accuracy': 0.85, 'precision': 0.82, 'recall': 0.78}
            
            def predict(self, batch, device):
                batch_size = len(batch.get('user_id', [1]))
                return torch.rand(batch_size, 2)
        
        mock_method = MockMethod()
        
        # Create dummy dataset
        dummy_dataset = TensorDataset(torch.randn(20, 5), torch.randint(0, 2, (20,)))
        
        # Test cross-validation
        result = cv_framework.validate_method(mock_method, dummy_dataset, 'stratified')
        
        assert isinstance(result, CrossValidationResult)
        assert result.method_name == 'MockMethod'
        assert len(result.fold_results) > 0
        assert 'test_f1' in result.summary_stats
        print("   ✅ Cross-validation execution successful")
        
        # Test confidence intervals
        f1_ci = result.get_f1_confidence_interval()
        assert len(f1_ci) == 2
        assert f1_ci[0] <= f1_ci[1]
        print("   ✅ Confidence intervals working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ CV Framework test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_benchmarking_suite():
    """Test Comprehensive Benchmarking Suite."""
    print("Testing Comprehensive Benchmarking Suite...")
    
    try:
        from evaluation.benchmarking_suite import ComprehensiveBenchmarkingSuite, BenchmarkingExperiment
        
        # Test with minimal config
        config = {
            'datasets': [
                {
                    'type': 'arbitrum',
                    'data_path': '/tmp/test',
                    'kwargs': {'max_sequence_length': 10}
                }
            ],
            'model': {'d_model': 64, 'epochs': 2},
            'trustalab': {'n_estimators': 5},
            'enhanced_gnns': {'epochs': 2},
            'traditional_ml': {'num_boost_round': 5},
            'cross_validation': {'enabled': False}  # Disable for quick testing
        }
        
        suite = ComprehensiveBenchmarkingSuite(config)
        assert hasattr(suite, 'cv_framework')
        assert hasattr(suite, 'trainer')
        print("   ✅ Benchmarking suite initialization successful")
        
        # Test baseline creation
        methods = suite.create_baseline_methods()
        assert len(methods) > 0
        
        expected_methods = [
            'TemporalGraphTransformer', 'TrustaLabFramework', 
            'GAT', 'LightGBM', 'RandomForest'
        ]
        
        for method_name in expected_methods:
            if method_name in methods:
                assert hasattr(methods[method_name], 'train')
                assert hasattr(methods[method_name], 'evaluate')
        
        print(f"   ✅ Created {len(methods)} baseline methods")
        
        # Test experiment tracking
        exp = BenchmarkingExperiment("test_exp", config)
        exp.add_result("method1", {"f1": 0.8})
        assert "method1" in exp.results
        print("   ✅ Experiment tracking working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Benchmarking suite test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cross_validation_strategies():
    """Test different cross-validation strategies."""
    print("Testing Cross-Validation Strategies...")
    
    try:
        from evaluation.cross_validation import (
            StratifiedCrossValidation, TemporalCrossValidation, ChainCrossValidation
        )
        
        # Create dummy data
        n_samples = 20
        labels = [0, 1] * (n_samples // 2)
        timestamps = list(range(n_samples))
        chain_labels = ['chain_a', 'chain_b'] * (n_samples // 2)
        
        dummy_dataset = TensorDataset(torch.randn(n_samples, 5))
        
        # Test Stratified CV
        stratified_cv = StratifiedCrossValidation(n_folds=3, random_state=42)
        stratified_splits = stratified_cv.split(dummy_dataset, labels)
        
        assert len(stratified_splits) == 3
        for train_idx, test_idx in stratified_splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            assert len(set(train_idx) & set(test_idx)) == 0  # No overlap
        
        print("   ✅ Stratified CV working correctly")
        
        # Test Temporal CV
        temporal_cv = TemporalCrossValidation(n_folds=3, random_state=42)
        temporal_splits = temporal_cv.split(dummy_dataset, timestamps)
        
        assert len(temporal_splits) == 3
        print("   ✅ Temporal CV working correctly")
        
        # Test Chain CV
        chain_info = {i: chain for i, chain in enumerate(chain_labels)}
        chain_cv = ChainCrossValidation(chain_info, random_state=42)
        chain_splits = chain_cv.split(dummy_dataset, chain_labels)
        
        assert len(chain_splits) > 0  # Should have at least one split
        print("   ✅ Chain CV working correctly")
        
        return True
        
    except Exception as e:
        print(f"   ❌ CV strategies test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_training_infrastructure():
    """Test enhanced training infrastructure."""
    print("Testing Enhanced Training Infrastructure...")
    
    try:
        from scripts.train_enhanced import MultiDatasetTrainer, DatasetFactory
        
        # Test DatasetFactory
        factory = DatasetFactory()
        assert hasattr(factory, 'DATASET_CLASSES')
        assert 'arbitrum' in factory.DATASET_CLASSES
        assert 'jupiter' in factory.DATASET_CLASSES
        print("   ✅ DatasetFactory working correctly")
        
        # Test MultiDatasetTrainer with minimal config
        config = {
            'datasets': [
                {
                    'type': 'arbitrum',
                    'data_path': '/tmp/test',
                    'kwargs': {'max_sequence_length': 5}
                }
            ],
            'model': {'d_model': 32, 'epochs': 1},
            'batch_size': 4,
            'device': 'cpu'
        }
        
        trainer = MultiDatasetTrainer(config)
        assert hasattr(trainer, 'device')
        assert hasattr(trainer, 'baseline_methods')
        print("   ✅ MultiDatasetTrainer initialization successful")
        
        # Test baseline registration
        from baselines.base_interface import BaselineMethodInterface
        
        class TestBaseline(BaselineMethodInterface):
            def train(self, train_loader, val_loader, device):
                return {'best_val_f1': 0.7}
            def evaluate(self, test_loader, device):
                return {'f1': 0.7}
            def predict(self, batch, device):
                return torch.rand(1, 2)
        
        test_baseline = TestBaseline({'test': True})
        trainer.register_baseline('TestMethod', test_baseline)
        
        assert 'TestMethod' in trainer.baseline_methods
        print("   ✅ Baseline registration working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Training infrastructure test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_result_management():
    """Test result saving and management."""
    print("Testing Result Management...")
    
    try:
        from evaluation.cross_validation import CrossValidationFramework
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test result saving
            config = {'stratified_folds': 2}
            cv_framework = CrossValidationFramework(config)
            
            # Create mock results
            from evaluation.cross_validation import CrossValidationResult
            
            mock_results = {
                'method1': CrossValidationResult(
                    method_name='method1',
                    fold_results=[
                        {'test_f1': 0.8, 'test_accuracy': 0.85},
                        {'test_f1': 0.82, 'test_accuracy': 0.87}
                    ],
                    summary_stats={
                        'test_f1': {'mean': 0.81, 'std': 0.01},
                        'test_accuracy': {'mean': 0.86, 'std': 0.01}
                    },
                    confidence_intervals={
                        'test_f1': (0.79, 0.83),
                        'test_accuracy': (0.84, 0.88)
                    },
                    statistical_tests={}
                )
            }
            
            # Test saving
            results_file = os.path.join(temp_dir, 'test_results.json')
            cv_framework.save_results(mock_results, results_file)
            
            assert os.path.exists(results_file)
            
            # Test loading
            with open(results_file, 'r') as f:
                loaded_results = json.load(f)
            
            assert 'method1' in loaded_results
            assert 'mean_f1' in loaded_results['method1']
            print("   ✅ Result saving and loading working")
            
            # Test summary printing (should not crash)
            cv_framework.print_comparison_summary(mock_results)
            print("   ✅ Summary printing working")
            
            return True
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f"   ❌ Result management test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_statistical_analysis():
    """Test statistical analysis capabilities."""
    print("Testing Statistical Analysis...")
    
    try:
        from utils.metrics import BinaryClassificationMetrics
        
        # Test statistical significance testing
        metrics1 = BinaryClassificationMetrics()
        metrics2 = BinaryClassificationMetrics()
        
        # Add different performance data
        predictions1 = torch.tensor([0.8, 0.7, 0.9, 0.6, 0.8])
        labels1 = torch.tensor([1, 1, 1, 0, 1])
        metrics1.update(predictions1, labels1)
        
        predictions2 = torch.tensor([0.6, 0.5, 0.7, 0.4, 0.6])
        labels2 = torch.tensor([1, 1, 1, 0, 1])
        metrics2.update(predictions2, labels2)
        
        # Test statistical test
        test_result = metrics1.statistical_test(metrics2, 'f1')
        
        expected_keys = ['p_value', 'statistic', 'significant', 'mean_difference']
        for key in expected_keys:
            assert key in test_result
        
        print("   ✅ Statistical significance testing working")
        
        # Test confidence intervals
        ci = metrics1.compute_confidence_intervals()
        assert 'f1' in ci
        assert len(ci['f1']) == 2
        print("   ✅ Confidence interval computation working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Statistical analysis test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 3 evaluation framework tests."""
    print("🧪 Phase 3 Evaluation Framework Test Suite")
    print("=" * 55)
    
    tests = [
        test_cross_validation_framework,
        test_cross_validation_strategies,
        test_benchmarking_suite,
        test_enhanced_training_infrastructure,
        test_result_management,
        test_statistical_analysis
    ]
    
    results = []
    for test in tests:
        print(f"\n{len(results)+1}. {test.__doc__.strip()}")
        result = test()
        results.append(result)
    
    print("\n" + "=" * 55)
    print("📊 Phase 3 Evaluation Framework Test Summary")
    print("=" * 55)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if all(results):
        print("\n🎉 ALL PHASE 3 EVALUATION TESTS PASSED!")
        print("✅ Cross-validation framework working correctly")
        print("✅ Benchmarking suite functioning properly")
        print("✅ Statistical analysis capabilities verified")
        print("✅ Result management system operational")
        return 0
    else:
        print("\n❌ Some Phase 3 evaluation tests failed!")
        failed_tests = [i for i, r in enumerate(results) if not r]
        print(f"Failed tests: {failed_tests}")
        return 1


if __name__ == "__main__":
    exit(main())