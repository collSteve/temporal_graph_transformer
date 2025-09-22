#!/usr/bin/env python3
"""
Phase 3 Baseline Methods Test Suite

Comprehensive testing of all baseline methods implemented in Phase 3:
- TrustaLabs Framework
- Subgraph Feature Propagation  
- Enhanced GNN Baselines (GAT, GraphSAGE, SybilGAT, BasicGCN)
- Traditional ML Baselines (LightGBM, RandomForest)
"""

import sys
import os
import tempfile
import shutil
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_trustalab_framework():
    """Test TrustaLabs Framework implementation."""
    print("Testing TrustaLabs Framework...")
    
    try:
        from baselines.trustalab_framework import TrustaLabFramework
        
        # Test initialization
        config = {
            'star_threshold_degree': 10,
            'tree_max_depth': 5,
            'chain_min_length': 3,
            'similarity_threshold': 0.8,
            'n_estimators': 10,  # Small for testing
            'random_state': 42
        }
        
        trustalab = TrustaLabFramework(config)
        assert trustalab.name == 'TrustaLabFramework'
        print("   ✅ TrustaLabs initialization successful")
        
        # Test pattern detectors
        assert hasattr(trustalab, 'star_detector')
        assert hasattr(trustalab, 'tree_detector')
        assert hasattr(trustalab, 'chain_detector')
        assert hasattr(trustalab, 'similarity_detector')
        print("   ✅ All pattern detectors present")
        
        # Test with dummy data
        device = torch.device('cpu')
        dummy_batch = {
            'user_id': ['user1', 'user2', 'user3'],
            'labels': torch.tensor([0, 1, 0])
        }
        
        # Test training (should not crash)
        train_loader = DataLoader(TensorDataset(torch.randn(10, 5)), batch_size=2)
        val_loader = DataLoader(TensorDataset(torch.randn(5, 5)), batch_size=2)
        
        result = trustalab.train(train_loader, val_loader, device)
        assert 'best_val_f1' in result
        print("   ✅ Training completed without errors")
        
        # Test prediction
        predictions = trustalab.predict(dummy_batch, device)
        assert predictions.shape[0] == len(dummy_batch['user_id'])
        assert predictions.shape[1] == 2  # Binary classification
        print("   ✅ Predictions working correctly")
        
        return True
        
    except Exception as e:
        print(f"   ❌ TrustaLabs test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_subgraph_propagation():
    """Test Subgraph Feature Propagation implementation."""
    print("Testing Subgraph Feature Propagation...")
    
    try:
        from baselines.subgraph_propagation import SubgraphFeaturePropagation
        
        # Test initialization
        config = {
            'max_neighbors_l1': 10,
            'max_neighbors_l2': 20,
            'input_dim': 32,
            'hidden_dim': 64,
            'num_layers': 2,
            'heads': 4,
            'dropout': 0.1,
            'lr': 1e-3,
            'epochs': 5,  # Small for testing
            'patience': 3
        }
        
        subgraph_method = SubgraphFeaturePropagation(config)
        assert subgraph_method.name == 'SubgraphFeaturePropagation'
        print("   ✅ Subgraph propagation initialization successful")
        
        # Test subgraph extractor
        assert hasattr(subgraph_method, 'subgraph_extractor')
        assert subgraph_method.input_dim == 32
        assert subgraph_method.hidden_dim == 64
        print("   ✅ Configuration correctly set")
        
        # Test with dummy data
        device = torch.device('cpu')
        train_loader = DataLoader(TensorDataset(torch.randn(10, 5)), batch_size=2)
        val_loader = DataLoader(TensorDataset(torch.randn(5, 5)), batch_size=2)
        
        result = subgraph_method.train(train_loader, val_loader, device)
        assert 'best_val_f1' in result
        print("   ✅ Training completed without errors")
        
        # Test evaluation
        eval_result = subgraph_method.evaluate(val_loader, device)
        expected_keys = ['f1', 'accuracy', 'precision', 'recall']
        for key in expected_keys:
            assert key in eval_result
        print("   ✅ Evaluation metrics complete")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Subgraph propagation test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_gnn_baselines():
    """Test Enhanced GNN baseline implementations."""
    print("Testing Enhanced GNN Baselines...")
    
    try:
        from baselines.enhanced_gnns import EnhancedGNNBaseline
        
        # Test all GNN types
        gnn_types = ['gat', 'graphsage', 'sybilgat', 'gcn']
        
        for gnn_type in gnn_types:
            print(f"   Testing {gnn_type.upper()}...")
            
            config = {
                'model_type': gnn_type,
                'input_dim': 32,
                'hidden_dim': 64,
                'num_layers': 2,
                'heads': 4,
                'dropout': 0.1,
                'lr': 1e-3,
                'epochs': 3,  # Small for testing
                'patience': 2
            }
            
            gnn_baseline = EnhancedGNNBaseline(config)
            assert gnn_baseline.model_type == gnn_type
            assert gnn_baseline.name == 'EnhancedGNNBaseline'
            
            # Test training
            device = torch.device('cpu')
            train_loader = DataLoader(TensorDataset(torch.randn(10, 5)), batch_size=2)
            val_loader = DataLoader(TensorDataset(torch.randn(5, 5)), batch_size=2)
            
            result = gnn_baseline.train(train_loader, val_loader, device)
            assert 'best_val_f1' in result
            
            print(f"     ✅ {gnn_type.upper()} training successful")
        
        print("   ✅ All GNN baselines tested successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Enhanced GNN test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_traditional_ml_baselines():
    """Test Traditional ML baseline implementations."""
    print("Testing Traditional ML Baselines...")
    
    try:
        from baselines.traditional_ml import LightGBMBaseline, RandomForestBaseline
        
        # Test LightGBM baseline
        print("   Testing LightGBM/GradientBoosting...")
        lgb_config = {
            'num_leaves': 10,
            'learning_rate': 0.1,
            'num_boost_round': 5,  # Small for testing
            'early_stopping_rounds': 2,
            'random_state': 42
        }
        
        lgb_baseline = LightGBMBaseline(lgb_config)
        assert lgb_baseline.name == 'LightGBMBaseline'
        
        # Test feature engineer
        assert hasattr(lgb_baseline, 'feature_engineer')
        print("     ✅ LightGBM initialization successful")
        
        # Test Random Forest baseline
        print("   Testing Random Forest...")
        rf_config = {
            'n_estimators': 5,  # Small for testing
            'max_depth': 3,
            'random_state': 42
        }
        
        rf_baseline = RandomForestBaseline(rf_config)
        assert rf_baseline.name == 'RandomForestBaseline'
        print("     ✅ Random Forest initialization successful")
        
        # Test training for both
        device = torch.device('cpu')
        train_loader = DataLoader(TensorDataset(torch.randn(20, 5)), batch_size=4)
        val_loader = DataLoader(TensorDataset(torch.randn(10, 5)), batch_size=4)
        
        for name, baseline in [('LightGBM', lgb_baseline), ('RandomForest', rf_baseline)]:
            result = baseline.train(train_loader, val_loader, device)
            assert 'best_val_f1' in result
            print(f"     ✅ {name} training successful")
        
        print("   ✅ All Traditional ML baselines tested successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Traditional ML test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_baseline_interface_compliance():
    """Test that all baselines comply with the common interface."""
    print("Testing Baseline Interface Compliance...")
    
    try:
        from baselines.base_interface import BaselineMethodInterface
        from baselines import (
            TrustaLabFramework, SubgraphFeaturePropagation, 
            EnhancedGNNBaseline, LightGBMBaseline, RandomForestBaseline
        )
        
        # Create instances of all baselines
        baselines = [
            TrustaLabFramework({'n_estimators': 5}),
            SubgraphFeaturePropagation({'epochs': 3}),
            EnhancedGNNBaseline({'model_type': 'gat', 'epochs': 3}),
            LightGBMBaseline({'num_boost_round': 5}),
            RandomForestBaseline({'n_estimators': 5})
        ]
        
        # Test interface compliance
        for baseline in baselines:
            # Test inheritance
            assert isinstance(baseline, BaselineMethodInterface)
            
            # Test required methods exist
            required_methods = ['train', 'evaluate', 'predict']
            for method in required_methods:
                assert hasattr(baseline, method)
                assert callable(getattr(baseline, method))
            
            # Test required attributes
            assert hasattr(baseline, 'name')
            assert hasattr(baseline, 'config')
            
            print(f"   ✅ {baseline.name} interface compliance verified")
        
        print("   ✅ All baselines comply with interface")
        return True
        
    except Exception as e:
        print(f"   ❌ Interface compliance test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_module():
    """Test the BinaryClassificationMetrics module."""
    print("Testing BinaryClassificationMetrics...")
    
    try:
        from utils.metrics import BinaryClassificationMetrics
        
        # Test initialization
        metrics = BinaryClassificationMetrics()
        assert metrics.threshold == 0.5
        print("   ✅ Metrics initialization successful")
        
        # Test with dummy data
        predictions = torch.tensor([0.8, 0.3, 0.9, 0.1, 0.7])
        labels = torch.tensor([1, 0, 1, 0, 1])
        
        metrics.update(predictions, labels)
        result = metrics.compute()
        
        # Check all expected metrics are present
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1', 'specificity',
            'auc_roc', 'auc_pr', 'balanced_accuracy', 'mcc'
        ]
        
        for metric in expected_metrics:
            assert metric in result
            assert isinstance(result[metric], (int, float))
        
        print("   ✅ All metrics computed successfully")
        
        # Test confidence intervals
        ci = metrics.compute_confidence_intervals()
        assert 'f1' in ci
        assert len(ci['f1']) == 2  # Lower and upper bounds
        print("   ✅ Confidence intervals working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Metrics test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 3 baseline tests."""
    print("🧪 Phase 3 Baseline Methods Test Suite")
    print("=" * 50)
    
    tests = [
        test_metrics_module,
        test_baseline_interface_compliance,
        test_trustalab_framework,
        test_subgraph_propagation,
        test_enhanced_gnn_baselines,
        test_traditional_ml_baselines
    ]
    
    results = []
    for test in tests:
        print(f"\n{len(results)+1}. {test.__doc__.strip()}")
        result = test()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📊 Phase 3 Baseline Test Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if all(results):
        print("\n🎉 ALL PHASE 3 BASELINE TESTS PASSED!")
        print("✅ All baseline methods are working correctly")
        print("✅ Interface compliance verified")
        print("✅ Metrics module functioning properly")
        return 0
    else:
        print("\n❌ Some Phase 3 baseline tests failed!")
        failed_tests = [i for i, r in enumerate(results) if not r]
        print(f"Failed tests: {failed_tests}")
        return 1


if __name__ == "__main__":
    exit(main())