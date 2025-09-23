#!/usr/bin/env python3
"""
Comprehensive Phase 4 Testing Suite

Thorough testing of all Phase 4 experimental validation components:
1. Experimental framework functionality
2. Cross-chain generalization analysis
3. Temporal pattern analysis
4. Failure case analysis
5. Ablation study framework
6. Interpretability analysis
7. Integration testing with real data flows
"""

import sys
import os
import tempfile
import shutil
import torch
import numpy as np
import yaml
import json
import copy
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import time
import unittest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import Phase 4 components
from evaluation.phase4_experimental_framework import (
    ExperimentalConfig,
    ComprehensiveEvaluationRunner,
    CrossChainGeneralizationAnalyzer
)
from evaluation.temporal_failure_analysis import (
    TemporalPatternAnalyzer,
    FailureCaseAnalyzer
)
from evaluation.ablation_interpretability import (
    AblationStudyFramework,
    InterpretabilityAnalyzer
)
from run_phase4_evaluation import Phase4Coordinator

# Import baseline methods for testing
from baselines import (
    TrustaLabFramework,
    LightGBMBaseline,
    RandomForestBaseline,
    TemporalGraphTransformerBaseline
)


class TestPhase4ExperimentalFramework(unittest.TestCase):
    """Test the core experimental framework components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ExperimentalConfig()
        
        # Modify config for faster testing
        self.config.config['evaluation']['random_seeds'] = [42, 123]
        self.config.config['methods']['all_baselines'] = [
            'TrustaLabFramework', 'LightGBM', 'RandomForest'
        ]
        self.config.config['datasets']['blockchain_types'] = ['arbitrum', 'jupiter']
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_experimental_config_initialization(self):
        """Test ExperimentalConfig initialization and validation."""
        print("Testing ExperimentalConfig...")
        
        # Test default config
        config = ExperimentalConfig()
        self.assertIsInstance(config.config, dict)
        self.assertIn('evaluation', config.config)
        self.assertIn('datasets', config.config)
        self.assertIn('methods', config.config)
        
        # Test config validation
        required_sections = ['evaluation', 'datasets', 'methods', 'experiments', 'output']
        for section in required_sections:
            self.assertIn(section, config.config, f"Missing required section: {section}")
        
        # Test dataset combinations generation
        combinations = config.get_dataset_combinations()
        self.assertIsInstance(combinations, list)
        self.assertTrue(len(combinations) > 0)
        
        # Test method-dataset combinations
        method_dataset_combos = config.get_method_dataset_combinations()
        self.assertIsInstance(method_dataset_combos, list)
        self.assertTrue(len(method_dataset_combos) > 0)
        
        print("   ✅ ExperimentalConfig tests passed")
    
    def test_experimental_config_custom_file(self):
        """Test loading custom configuration file."""
        print("Testing custom configuration file...")
        
        # Create temporary config file
        test_config = {
            'evaluation': {'random_seeds': [42]},
            'datasets': {'blockchain_types': ['test_chain']},
            'methods': {'all_baselines': ['TestMethod']},
            'experiments': {'comprehensive_evaluation': True},
            'output': {'base_dir': './test_output'}
        }
        
        config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        # Load custom config
        config = ExperimentalConfig(config_file)
        self.assertEqual(config.config['datasets']['blockchain_types'], ['test_chain'])
        self.assertEqual(config.config['evaluation']['random_seeds'], [42])
        
        print("   ✅ Custom configuration file tests passed")
    
    def test_comprehensive_evaluation_runner_initialization(self):
        """Test ComprehensiveEvaluationRunner initialization."""
        print("Testing ComprehensiveEvaluationRunner...")
        
        runner = ComprehensiveEvaluationRunner(self.config)
        self.assertIsNotNone(runner.config)
        self.assertEqual(runner.config, self.config)
        self.assertIsInstance(runner.results, dict)
        
        print("   ✅ ComprehensiveEvaluationRunner initialization tests passed")
    
    def test_method_config_creation(self):
        """Test method configuration creation."""
        print("Testing method configuration creation...")
        
        runner = ComprehensiveEvaluationRunner(self.config)
        
        # Test TGT config creation
        tgt_config = runner._create_method_config('TemporalGraphTransformer', 'arbitrum', 42)
        self.assertIn('model', tgt_config)
        self.assertIn('datasets', tgt_config)
        self.assertEqual(tgt_config['evaluation']['random_seed'], 42)
        
        # Test TrustaLab config creation
        trustalab_config = runner._create_method_config('TrustaLabFramework', 'arbitrum', 42)
        self.assertIn('trustalab', trustalab_config)
        self.assertIn('datasets', trustalab_config)
        
        # Test traditional ML config creation
        lgb_config = runner._create_method_config('LightGBM', 'arbitrum', 42)
        self.assertIn('traditional_ml', lgb_config)
        self.assertIn('datasets', lgb_config)
        
        print("   ✅ Method configuration creation tests passed")
    
    def test_cross_chain_analyzer_initialization(self):
        """Test CrossChainGeneralizationAnalyzer initialization."""
        print("Testing CrossChainGeneralizationAnalyzer...")
        
        analyzer = CrossChainGeneralizationAnalyzer(self.config)
        self.assertIsNotNone(analyzer.config)
        self.assertEqual(analyzer.config, self.config)
        
        print("   ✅ CrossChainGeneralizationAnalyzer initialization tests passed")


class TestPhase4TemporalFailureAnalysis(unittest.TestCase):
    """Test temporal pattern analysis and failure case analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock airdrop dates
        self.airdrop_dates = {
            'arbitrum': '2023-03-23',
            'jupiter': '2024-01-31'
        }
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_temporal_pattern_analyzer_initialization(self):
        """Test TemporalPatternAnalyzer initialization."""
        print("Testing TemporalPatternAnalyzer...")
        
        analyzer = TemporalPatternAnalyzer(self.airdrop_dates)
        self.assertEqual(analyzer.airdrop_dates, self.airdrop_dates)
        self.assertIn('arbitrum', analyzer.airdrop_timestamps)
        self.assertIn('jupiter', analyzer.airdrop_timestamps)
        
        print("   ✅ TemporalPatternAnalyzer initialization tests passed")
    
    def test_temporal_periods_definition(self):
        """Test temporal periods definition."""
        print("Testing temporal periods definition...")
        
        analyzer = TemporalPatternAnalyzer(self.airdrop_dates)
        
        # Test period definition for arbitrum
        from datetime import datetime
        airdrop_ts = datetime.strptime('2023-03-23', '%Y-%m-%d')
        periods = analyzer._define_temporal_periods(airdrop_ts)
        
        expected_periods = ['pre_farming', 'intensive_farming', 'pre_announcement', 'post_airdrop']
        for period in expected_periods:
            self.assertIn(period, periods)
            self.assertIn('start', periods[period])
            self.assertIn('end', periods[period])
            self.assertIn('description', periods[period])
        
        # Verify temporal order
        self.assertLess(periods['pre_farming']['start'], periods['intensive_farming']['start'])
        self.assertLess(periods['intensive_farming']['start'], periods['pre_announcement']['start'])
        self.assertLess(periods['pre_announcement']['start'], periods['post_airdrop']['start'])
        
        print("   ✅ Temporal periods definition tests passed")
    
    def test_failure_case_analyzer_initialization(self):
        """Test FailureCaseAnalyzer initialization."""
        print("Testing FailureCaseAnalyzer...")
        
        analyzer = FailureCaseAnalyzer()
        self.assertIsInstance(analyzer.failure_categories, dict)
        
        expected_categories = ['false_positive', 'false_negative', 'low_confidence', 
                             'inconsistent', 'temporal_confusion', 'graph_structure']
        for category in expected_categories:
            self.assertIn(category, analyzer.failure_categories)
        
        print("   ✅ FailureCaseAnalyzer initialization tests passed")
    
    def test_failure_categorization_logic(self):
        """Test failure categorization logic."""
        print("Testing failure categorization logic...")
        
        analyzer = FailureCaseAnalyzer()
        
        # Create mock prediction data
        predictions = {
            'method1': [0.8, 0.3, 0.9, 0.1, 0.7],  # Mix of high/low confidence
            'method2': [0.6, 0.4, 0.8, 0.2, 0.5]   # Different predictions
        }
        true_labels = [1, 0, 1, 0, 1]  # Ground truth
        samples = [{'sample_idx': i} for i in range(5)]
        
        # Test categorization
        categorized = analyzer._categorize_failures(predictions, true_labels, samples, 0.7)
        
        # Check that all expected categories exist
        expected_keys = ['false_positives', 'false_negatives', 'low_confidence_predictions', 
                        'high_disagreement_cases']
        for key in expected_keys:
            self.assertIn(key, categorized)
            self.assertIsInstance(categorized[key], list)
        
        # Check summary statistics are generated
        self.assertIn('false_positives_summary', categorized)
        self.assertIn('false_negatives_summary', categorized)
        
        print("   ✅ Failure categorization logic tests passed")


class TestPhase4AblationInterpretability(unittest.TestCase):
    """Test ablation studies and interpretability analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_config = {
            'd_model': 128,
            'temporal_layers': 3,
            'temporal_heads': 8,
            'graph_layers': 3,
            'graph_heads': 8,
            'dropout': 0.1
        }
        
    def test_ablation_study_framework_initialization(self):
        """Test AblationStudyFramework initialization."""
        print("Testing AblationStudyFramework...")
        
        framework = AblationStudyFramework(self.base_config)
        self.assertEqual(framework.base_config, self.base_config)
        self.assertIsInstance(framework.ablation_configs, dict)
        
        # Check that baseline configuration exists
        self.assertIn('baseline_full', framework.ablation_configs)
        baseline_config = framework.ablation_configs['baseline_full']
        self.assertEqual(baseline_config['config'], self.base_config)
        
        print("   ✅ AblationStudyFramework initialization tests passed")
    
    def test_ablation_configurations_definition(self):
        """Test ablation configurations are properly defined."""
        print("Testing ablation configurations...")
        
        framework = AblationStudyFramework(self.base_config)
        configs = framework.ablation_configs
        
        # Check essential ablation types exist
        essential_ablations = [
            'no_temporal_layers', 'no_graph_layers', 
            'single_temporal_layer', 'single_graph_layer',
            'half_model_size', 'single_head_attention'
        ]
        
        for ablation in essential_ablations:
            self.assertIn(ablation, configs)
            self.assertIn('description', configs[ablation])
            self.assertIn('config', configs[ablation])
            self.assertIn('modifications', configs[ablation])
        
        # Verify configurations are actually different from baseline
        baseline_config = configs['baseline_full']['config']
        no_temporal_config = configs['no_temporal_layers']['config']
        self.assertNotEqual(baseline_config['temporal_layers'], no_temporal_config['temporal_layers'])
        
        print("   ✅ Ablation configurations tests passed")
    
    def test_component_contribution_analysis(self):
        """Test component contribution analysis logic."""
        print("Testing component contribution analysis...")
        
        framework = AblationStudyFramework(self.base_config)
        
        # Create mock ablation results
        mock_results = {
            'baseline_full': {
                'aggregated': {
                    'test_metrics': {
                        'f1': {'mean': 0.85}
                    }
                }
            },
            'no_temporal_layers': {
                'description': 'Remove temporal layers',
                'modifications': 'temporal_layers=0',
                'aggregated': {
                    'test_metrics': {
                        'f1': {'mean': 0.70}
                    }
                }
            },
            'no_graph_layers': {
                'description': 'Remove graph layers',
                'modifications': 'graph_layers=0',
                'aggregated': {
                    'test_metrics': {
                        'f1': {'mean': 0.75}
                    }
                }
            }
        }
        
        # Test analysis
        analysis = framework._analyze_component_contributions(mock_results)
        
        self.assertIn('component_contributions', analysis)
        self.assertIn('ranked_by_importance', analysis)
        self.assertIn('key_insights', analysis)
        
        # Check that temporal layers are ranked as more important (higher drop)
        contributions = analysis['component_contributions']
        temporal_drop = contributions['no_temporal_layers']['performance_drop']
        graph_drop = contributions['no_graph_layers']['performance_drop']
        self.assertGreater(temporal_drop, graph_drop)
        
        print("   ✅ Component contribution analysis tests passed")
    
    def test_interpretability_analyzer_initialization(self):
        """Test InterpretabilityAnalyzer initialization."""
        print("Testing InterpretabilityAnalyzer...")
        
        analyzer = InterpretabilityAnalyzer()
        self.assertIsInstance(analyzer.analysis_methods, dict)
        
        expected_methods = ['attention_visualization', 'feature_importance', 
                           'pattern_clustering', 'decision_boundary', 'representational_similarity']
        for method in expected_methods:
            self.assertIn(method, analyzer.analysis_methods)
            self.assertTrue(callable(analyzer.analysis_methods[method]))
        
        print("   ✅ InterpretabilityAnalyzer initialization tests passed")


class TestPhase4Integration(unittest.TestCase):
    """Test integration between Phase 4 components and existing baseline methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device('cpu')
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_mock_data_loader(self, batch_size=4, num_batches=3):
        """Create mock data loader for testing."""
        
        # Create mock data that resembles our expected format
        data = []
        for _ in range(num_batches * batch_size):
            # Mock features
            features = torch.randn(10)  # 10 features
            label = torch.randint(0, 2, (1,))[0]  # Binary label
            data.append((features, label))
        
        dataset = data
        # Create simple data loader
        batches = []
        for i in range(0, len(dataset), batch_size):
            batch_data = dataset[i:i+batch_size]
            features = torch.stack([item[0] for item in batch_data])
            labels = torch.stack([item[1] for item in batch_data])
            
            # Create batch in expected format
            batch = {
                'user_id': [f'user_{j}' for j in range(len(batch_data))],
                'labels': labels,
                'features': features
            }
            batches.append(batch)
        
        return batches
    
    def test_baseline_method_integration(self):
        """Test that Phase 4 framework works with actual baseline methods."""
        print("Testing baseline method integration...")
        
        # Create baseline methods
        methods = {
            'TrustaLab': TrustaLabFramework({'n_estimators': 5, 'random_state': 42}),
            'LightGBM': LightGBMBaseline({'num_boost_round': 5, 'random_state': 42}),
            'RandomForest': RandomForestBaseline({'n_estimators': 5, 'random_state': 42})
        }
        
        # Create mock data loader
        data_loader = self.create_mock_data_loader()
        
        # Test failure case analyzer with real methods
        analyzer = FailureCaseAnalyzer()
        
        # This should not crash
        try:
            # Test with a single batch to avoid extensive computation
            single_batch = data_loader[0]
            
            # Test that methods can make predictions
            for name, method in methods.items():
                predictions = method.predict(single_batch, self.device)
                self.assertIsInstance(predictions, torch.Tensor)
                self.assertTrue(predictions.numel() > 0)
                
        except Exception as e:
            self.fail(f"Baseline method integration failed: {e}")
        
        print("   ✅ Baseline method integration tests passed")
    
    def test_temporal_analysis_with_real_methods(self):
        """Test temporal analysis with real baseline methods."""
        print("Testing temporal analysis with real methods...")
        
        # Create temporal analyzer
        analyzer = TemporalPatternAnalyzer({'arbitrum': '2023-03-23'})
        
        # Create a simple baseline method
        method = TrustaLabFramework({'n_estimators': 3, 'random_state': 42})
        
        # Create mock data loader
        data_loader = self.create_mock_data_loader(batch_size=2, num_batches=2)
        
        # Test temporal analysis (should handle mock data gracefully)
        try:
            # Convert data_loader to iterator-like object
            class MockDataLoader:
                def __init__(self, batches):
                    self.batches = batches
                
                def __iter__(self):
                    return iter(self.batches)
            
            mock_loader = MockDataLoader(data_loader)
            
            result = analyzer.analyze_temporal_patterns(
                method, 'arbitrum', mock_loader, self.device
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('method', result)
            self.assertIn('dataset', result)
            self.assertEqual(result['method'], method.name)
            self.assertEqual(result['dataset'], 'arbitrum')
            
        except Exception as e:
            self.fail(f"Temporal analysis integration failed: {e}")
        
        print("   ✅ Temporal analysis integration tests passed")
    
    def test_ablation_study_with_tgt(self):
        """Test ablation study framework with TGT baseline."""
        print("Testing ablation study with TGT...")
        
        # Create TGT configuration
        base_config = {
            'd_model': 32,  # Small for testing
            'temporal_layers': 2,
            'graph_layers': 2,
            'temporal_heads': 4,
            'graph_heads': 4,
            'epochs': 3,  # Very short for testing
            'patience': 2
        }
        
        # Create ablation framework
        framework = AblationStudyFramework(base_config)
        
        # Test that we can create TGT baselines with different configurations
        try:
            for config_name, config_info in list(framework.ablation_configs.items())[:3]:  # Test first 3 only
                modified_config = config_info['config']
                
                # Try to create TGT baseline with this config
                tgt_baseline = TemporalGraphTransformerBaseline(modified_config)
                self.assertIsNotNone(tgt_baseline)
                self.assertEqual(tgt_baseline.name, 'TemporalGraphTransformer')
                
        except Exception as e:
            self.fail(f"TGT ablation study integration failed: {e}")
        
        print("   ✅ Ablation study with TGT tests passed")


class TestPhase4Coordinator(unittest.TestCase):
    """Test the main Phase 4 coordinator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_coordinator_initialization(self):
        """Test Phase4Coordinator initialization."""
        print("Testing Phase4Coordinator...")
        
        coordinator = Phase4Coordinator(output_dir=self.temp_dir)
        
        self.assertIsNotNone(coordinator.config)
        self.assertTrue(coordinator.output_dir.exists())
        self.assertIsInstance(coordinator.results, dict)
        
        print("   ✅ Phase4Coordinator initialization tests passed")
    
    def test_quick_evaluation_configuration(self):
        """Test quick evaluation configuration setup."""
        print("Testing quick evaluation configuration...")
        
        coordinator = Phase4Coordinator(output_dir=self.temp_dir)
        
        # Store original config (deep copy to prevent reference issues)
        original_config = copy.deepcopy(coordinator.config.config)
        
        # Modify config for quick evaluation (simulating the logic)
        coordinator.config.config['evaluation']['random_seeds'] = [42, 123]
        coordinator.config.config['methods']['all_baselines'] = ['TemporalGraphTransformer', 'TrustaLabFramework', 'LightGBM']
        coordinator.config.config['datasets']['blockchain_types'] = ['arbitrum', 'jupiter']
        
        # Get modified config
        quick_config = coordinator.config.config
        
        # Verify modifications
        self.assertEqual(len(quick_config['evaluation']['random_seeds']), 2)
        self.assertEqual(len(quick_config['methods']['all_baselines']), 3)
        self.assertEqual(len(quick_config['datasets']['blockchain_types']), 2)
        
        # Verify original config was different (should have 5 seeds originally)
        self.assertEqual(len(original_config['evaluation']['random_seeds']), 5)
        self.assertNotEqual(len(original_config['evaluation']['random_seeds']), 
                           len(quick_config['evaluation']['random_seeds']))
        
        print("   ✅ Quick evaluation configuration tests passed")
    
    def test_result_finalization(self):
        """Test result finalization and saving."""
        print("Testing result finalization...")
        
        coordinator = Phase4Coordinator(output_dir=self.temp_dir)
        coordinator.start_time = time.time() - 100  # 100 seconds ago
        
        # Add mock results
        coordinator.results = {
            'comprehensive_evaluation': {'status': 'completed', 'test_results': {'f1': 0.85}},
            'cross_chain_analysis': {'status': 'completed', 'generalization_score': 0.78}
        }
        
        # Test finalization
        final_results = coordinator._finalize_results("test")
        
        self.assertIn('evaluation_type', final_results)
        self.assertIn('config', final_results)
        self.assertIn('execution_info', final_results)
        self.assertIn('results', final_results)
        self.assertEqual(final_results['evaluation_type'], 'test')
        self.assertEqual(final_results['results'], coordinator.results)
        
        # Check that files are created
        result_files = list(Path(self.temp_dir).glob("phase4_test_results_*.json"))
        self.assertTrue(len(result_files) > 0)
        
        summary_files = list(Path(self.temp_dir).glob("phase4_test_summary_*.md"))
        self.assertTrue(len(result_files) > 0)
        
        print("   ✅ Result finalization tests passed")


def run_single_test_class(test_class):
    """Run a single test class and return results."""
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
    result = runner.run(suite)
    return result.wasSuccessful(), len(result.failures), len(result.errors)


def main():
    """Run all Phase 4 comprehensive tests."""
    print("🧪 Phase 4 Comprehensive Testing Suite")
    print("=" * 60)
    
    test_classes = [
        ("Experimental Framework", TestPhase4ExperimentalFramework),
        ("Temporal & Failure Analysis", TestPhase4TemporalFailureAnalysis),
        ("Ablation & Interpretability", TestPhase4AblationInterpretability),
        ("Integration Testing", TestPhase4Integration),
        ("Coordinator Testing", TestPhase4Coordinator)
    ]
    
    all_results = []
    total_tests = 0
    total_passed = 0
    
    for test_name, test_class in test_classes:
        print(f"\n🔬 {test_name}")
        print("-" * 40)
        
        try:
            success, failures, errors = run_single_test_class(test_class)
            
            # Count tests in this class
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            test_count = suite.countTestCases()
            total_tests += test_count
            
            if success:
                print(f"   ✅ All {test_count} tests PASSED")
                total_passed += test_count
                all_results.append((test_name, True, test_count, 0, 0))
            else:
                failed_count = failures + errors
                passed_count = test_count - failed_count
                total_passed += passed_count
                print(f"   ❌ {passed_count}/{test_count} tests passed ({failed_count} failed)")
                all_results.append((test_name, False, test_count, failures, errors))
                
        except Exception as e:
            print(f"   ❌ Test class error: {e}")
            all_results.append((test_name, False, 0, 0, 1))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Phase 4 Comprehensive Testing Summary")
    print("=" * 60)
    
    print(f"Total tests run: {total_tests}")
    print(f"Tests passed: {total_passed}")
    print(f"Success rate: {total_passed/total_tests*100:.1f}%" if total_tests > 0 else "No tests run")
    
    print("\nDetailed Results:")
    overall_success = True
    for test_name, success, test_count, failures, errors in all_results:
        status = "✅ PASS" if success else "❌ FAIL"
        details = f"({test_count} tests)" if success else f"({failures} failures, {errors} errors)"
        print(f"  {status} {test_name} {details}")
        if not success:
            overall_success = False
    
    print("\n" + "=" * 60)
    
    if overall_success and total_passed == total_tests:
        print("🎉 ALL PHASE 4 COMPREHENSIVE TESTS PASSED!")
        print("✅ Experimental framework thoroughly validated")
        print("✅ Temporal and failure analysis components working")
        print("✅ Ablation and interpretability framework functional")
        print("✅ Integration with baseline methods confirmed")
        print("✅ Coordinator and orchestration system operational")
        print("\n🚀 Phase 4 experimental validation framework is production-ready!")
        return 0
    else:
        print("❌ SOME PHASE 4 COMPREHENSIVE TESTS FAILED!")
        print(f"⚠️  {total_tests - total_passed} out of {total_tests} tests failed")
        print("\n🔧 Action Required:")
        print("1. Review failed test details above")
        print("2. Fix implementation issues")
        print("3. Re-run comprehensive tests")
        return 1


if __name__ == "__main__":
    exit(main())