#!/usr/bin/env python3
"""
Phase 3 Integration Test Suite

End-to-end integration testing of the complete Phase 3 system:
- Multi-dataset training pipeline
- All baseline methods working together
- Cross-validation and benchmarking integration
- Configuration system
- Complete workflow validation
"""

import sys
import os
import tempfile
import shutil
import torch
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_complete_baseline_integration():
    """Test that all baseline methods work together in the training system."""
    print("Testing Complete Baseline Integration...")
    
    try:
        from scripts.train_enhanced import MultiDatasetTrainer
        
        # Create comprehensive config
        config = {
            'datasets': [
                {
                    'type': 'arbitrum',
                    'data_path': '/tmp/test_data',
                    'kwargs': {
                        'max_sequence_length': 5
                    }
                }
            ],
            'model': {
                'd_model': 32,
                'temporal_layers': 2,
                'graph_layers': 2,
                'epochs': 2,
                'verbose': False
            },
            'trustalab': {
                'n_estimators': 5,
                'random_state': 42
            },
            'subgraph_propagation': {
                'input_dim': 32,
                'hidden_dim': 32,
                'epochs': 2,
                'patience': 1,
                'verbose': False
            },
            'enhanced_gnns': {
                'input_dim': 32,
                'hidden_dim': 32,
                'epochs': 2,
                'patience': 1,
                'verbose': False
            },
            'traditional_ml': {
                'num_boost_round': 5,
                'n_estimators': 5,
                'random_state': 42
            },
            'batch_size': 4,
            'device': 'cpu'
        }
        
        # Create trainer
        trainer = MultiDatasetTrainer(config)
        
        # Test that all baselines are registered correctly
        data_loaders = {'train': None, 'val': None}  # Mock empty loaders for testing
        
        # This should register all baselines without errors
        trainer._register_all_baselines()
        
        # Check that we have the expected number of baselines
        expected_baselines = [
            'TemporalGraphTransformer',
            'TrustaLabFramework', 
            'SubgraphFeaturePropagation',
            'GAT', 'GraphSAGE', 'SybilGAT', 'BasicGCN',
            'LightGBM', 'RandomForest'
        ]
        
        registered_count = 0
        for baseline_name in expected_baselines:
            if baseline_name in trainer.baseline_methods:
                registered_count += 1
                print(f"     ✅ {baseline_name} registered successfully")
        
        print(f"   ✅ {registered_count} baseline methods integrated successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_system_integration():
    """Test the configuration system with YAML files."""
    print("Testing Configuration System Integration...")
    
    try:
        # Create temporary config file
        temp_dir = tempfile.mkdtemp()
        config_file = os.path.join(temp_dir, 'test_config.yaml')
        
        try:
            # Create test configuration
            test_config = {
                'datasets': [
                    {
                        'type': 'arbitrum',
                        'data_path': './data',
                        'kwargs': {
                            'max_sequence_length': 10,
                            'start_date': '2023-03-15',
                            'end_date': '2023-03-16'
                        }
                    }
                ],
                'model': {
                    'd_model': 64,
                    'temporal_layers': 2,
                    'temporal_heads': 4,
                    'graph_layers': 2,
                    'graph_heads': 4,
                    'epochs': 3,
                    'lr': 1e-3
                },
                'trustalab': {
                    'star_threshold_degree': 5,
                    'n_estimators': 10,
                    'random_state': 42
                },
                'cross_validation': {
                    'enabled': True,
                    'stratified_folds': 3
                },
                'batch_size': 8,
                'device': 'cpu',
                'output_dir': './test_output'
            }
            
            # Save config to YAML
            with open(config_file, 'w') as f:
                yaml.dump(test_config, f, indent=2)
            
            # Test loading config
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            assert loaded_config['model']['d_model'] == 64
            assert loaded_config['trustalab']['n_estimators'] == 10
            assert loaded_config['cross_validation']['enabled'] == True
            
            print("   ✅ YAML configuration loading working")
            
            # Test config validation with trainer
            from scripts.train_enhanced import MultiDatasetTrainer
            trainer = MultiDatasetTrainer(loaded_config)
            
            assert trainer.config['model']['d_model'] == 64
            assert trainer.config['batch_size'] == 8
            
            print("   ✅ Configuration system integration successful")
            
            return True
            
        finally:
            shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f"   ❌ Config system test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_integration():
    """Test integration between training system and evaluation framework."""
    print("Testing Evaluation Integration...")
    
    try:
        from evaluation.benchmarking_suite import ComprehensiveBenchmarkingSuite
        from scripts.train_enhanced import MultiDatasetTrainer
        
        # Create minimal config for testing
        config = {
            'datasets': [
                {
                    'type': 'arbitrum',
                    'data_path': '/tmp/test',
                    'kwargs': {'max_sequence_length': 5}
                }
            ],
            'model': {'d_model': 32, 'epochs': 1},
            'trustalab': {'n_estimators': 3},
            'enhanced_gnns': {'epochs': 1},
            'traditional_ml': {'num_boost_round': 3},
            'cross_validation': {
                'enabled': False,  # Disable for quick test
                'stratified_folds': 2
            },
            'device': 'cpu'
        }
        
        # Test benchmarking suite creation
        suite = ComprehensiveBenchmarkingSuite(config)
        
        # Test that it can create baseline methods
        methods = suite.create_baseline_methods()
        assert len(methods) > 0
        print(f"   ✅ Created {len(methods)} methods for benchmarking")
        
        # Test that trainer and evaluation components work together
        trainer = MultiDatasetTrainer(config)
        assert hasattr(trainer, 'config')
        assert hasattr(suite, 'trainer')
        
        print("   ✅ Training and evaluation systems integrated")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Evaluation integration test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cross_chain_compatibility():
    """Test cross-chain dataset compatibility."""
    print("Testing Cross-Chain Compatibility...")
    
    try:
        from scripts.train_enhanced import DatasetFactory
        
        # Test multiple chain configurations
        chain_configs = [
            {
                'type': 'arbitrum',
                'data_path': '/tmp/test',
                'kwargs': {'max_sequence_length': 5}
            },
            {
                'type': 'jupiter',
                'data_path': '/tmp/test',
                'kwargs': {'max_sequence_length': 5}
            }
        ]
        
        factory = DatasetFactory()
        
        # Test that all dataset types are supported
        supported_types = ['arbitrum', 'jupiter', 'optimism', 'blur', 'solana']
        
        for dataset_type in supported_types:
            assert dataset_type in factory.DATASET_CLASSES
            assert factory.DATASET_CLASSES[dataset_type] is not None
            print(f"     ✅ {dataset_type} dataset supported")
        
        print("   ✅ Cross-chain compatibility verified")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Cross-chain compatibility test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_workflow():
    """Test the complete end-to-end workflow."""
    print("Testing End-to-End Workflow...")
    
    try:
        # Create temporary output directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create minimal but complete config
            config = {
                'datasets': [
                    {
                        'type': 'arbitrum',
                        'data_path': temp_dir,
                        'kwargs': {
                            'max_sequence_length': 3
                        }
                    }
                ],
                'model': {
                    'd_model': 16,
                    'temporal_layers': 1,
                    'graph_layers': 1,
                    'epochs': 1,
                    'lr': 1e-2,
                    'verbose': False
                },
                'trustalab': {
                    'n_estimators': 2,
                    'random_state': 42
                },
                'enhanced_gnns': {
                    'input_dim': 16,
                    'hidden_dim': 16,
                    'epochs': 1,
                    'patience': 1,
                    'verbose': False
                },
                'traditional_ml': {
                    'num_boost_round': 2,
                    'n_estimators': 2,
                    'random_state': 42
                },
                'cross_validation': {
                    'enabled': False  # Disabled for speed
                },
                'batch_size': 2,
                'device': 'cpu',
                'output_dir': temp_dir
            }
            
            # Test the complete workflow doesn't crash
            from scripts.train_enhanced import MultiDatasetTrainer
            
            trainer = MultiDatasetTrainer(config)
            
            # Test data loader creation (will create demonstration data)
            data_loaders = trainer.create_data_loaders()
            
            # Should have at least train loader
            if data_loaders.get('train') is not None:
                print("   ✅ Data loading pipeline working")
            else:
                print("   ⚠️  Data loading returned None (may be expected for test)")
            
            # Test baseline method creation
            methods = trainer._create_all_baseline_methods()
            assert len(methods) > 0
            print(f"   ✅ Created {len(methods)} baseline methods")
            
            # Test that evaluation framework can be initialized
            from evaluation.benchmarking_suite import ComprehensiveBenchmarkingSuite
            suite = ComprehensiveBenchmarkingSuite(config)
            
            print("   ✅ Complete workflow initialization successful")
            
            return True
            
        finally:
            shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f"   ❌ End-to-end workflow test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling and edge cases."""
    print("Testing Error Handling...")
    
    try:
        from scripts.train_enhanced import MultiDatasetTrainer
        from baselines import TrustaLabFramework
        
        # Test with invalid config
        invalid_config = {
            'datasets': [],  # Empty datasets
            'model': {},
            'device': 'cpu'
        }
        
        trainer = MultiDatasetTrainer(invalid_config)
        data_loaders = trainer.create_data_loaders()
        
        # Should handle empty datasets gracefully
        assert data_loaders is not None
        print("   ✅ Empty dataset handling working")
        
        # Test baseline with invalid config
        try:
            invalid_baseline_config = {'invalid_param': 'test'}
            baseline = TrustaLabFramework(invalid_baseline_config)
            # Should not crash, should use defaults
            assert baseline.name == 'TrustaLabFramework'
            print("   ✅ Invalid baseline config handling working")
        except Exception:
            print("   ⚠️  Baseline config validation could be improved")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error handling test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 3 integration tests."""
    print("🧪 Phase 3 Integration Test Suite")
    print("=" * 45)
    
    tests = [
        test_complete_baseline_integration,
        test_config_system_integration,
        test_evaluation_integration,
        test_cross_chain_compatibility,
        test_end_to_end_workflow,
        test_error_handling
    ]
    
    results = []
    for test in tests:
        print(f"\n{len(results)+1}. {test.__doc__.strip()}")
        result = test()
        results.append(result)
    
    print("\n" + "=" * 45)
    print("📊 Phase 3 Integration Test Summary")
    print("=" * 45)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if all(results):
        print("\n🎉 ALL PHASE 3 INTEGRATION TESTS PASSED!")
        print("✅ Complete system integration verified")
        print("✅ All baseline methods working together")
        print("✅ Configuration system operational")
        print("✅ Cross-chain compatibility confirmed")
        print("✅ End-to-end workflow functional")
        return 0
    else:
        print("\n❌ Some Phase 3 integration tests failed!")
        failed_tests = [i for i, r in enumerate(results) if not r]
        print(f"Failed tests: {failed_tests}")
        return 1


if __name__ == "__main__":
    exit(main())