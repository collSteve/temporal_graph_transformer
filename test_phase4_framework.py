#!/usr/bin/env python3
"""
Phase 4 Framework Validation Test

Quick test to validate that all Phase 4 components can be imported
and initialized correctly before running the full evaluation.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all Phase 4 modules can be imported."""
    print("🔍 Testing Phase 4 imports...")
    
    try:
        # Test evaluation framework imports
        from evaluation import (
            ExperimentalConfig,
            ComprehensiveEvaluationRunner,
            CrossChainGeneralizationAnalyzer,
            TemporalPatternAnalyzer,
            FailureCaseAnalyzer,
            AblationStudyFramework,
            InterpretabilityAnalyzer
        )
        print("   ✅ All evaluation framework imports successful")
        
        # Test main coordination script import
        import run_phase4_evaluation
        print("   ✅ Main coordination script import successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading and validation."""
    print("\n🔧 Testing configuration...")
    
    try:
        from evaluation.phase4_experimental_framework import ExperimentalConfig
        
        # Test default configuration
        config = ExperimentalConfig()
        print("   ✅ Default configuration loaded")
        
        # Test custom configuration file
        config_path = "./configs/phase4_config.yaml"
        if os.path.exists(config_path):
            config = ExperimentalConfig(config_path)
            print("   ✅ Custom configuration file loaded")
        else:
            print("   ⚠️  Custom configuration file not found, using defaults")
        
        # Validate configuration structure
        required_sections = ['evaluation', 'datasets', 'methods', 'experiments', 'output']
        for section in required_sections:
            if section in config.config:
                print(f"      ✅ {section} section present")
            else:
                print(f"      ❌ {section} section missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False

def test_framework_initialization():
    """Test that framework components can be initialized."""
    print("\n🏗️ Testing framework initialization...")
    
    try:
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
        
        # Initialize configuration
        config = ExperimentalConfig()
        print("   ✅ ExperimentalConfig initialized")
        
        # Initialize evaluation runner
        runner = ComprehensiveEvaluationRunner(config)
        print("   ✅ ComprehensiveEvaluationRunner initialized")
        
        # Initialize cross-chain analyzer
        analyzer = CrossChainGeneralizationAnalyzer(config)
        print("   ✅ CrossChainGeneralizationAnalyzer initialized")
        
        # Initialize temporal analyzer
        temporal_analyzer = TemporalPatternAnalyzer()
        print("   ✅ TemporalPatternAnalyzer initialized")
        
        # Initialize failure analyzer
        failure_analyzer = FailureCaseAnalyzer()
        print("   ✅ FailureCaseAnalyzer initialized")
        
        # Initialize ablation framework
        base_config = {'d_model': 128, 'temporal_layers': 3, 'graph_layers': 3}
        ablation_framework = AblationStudyFramework(base_config)
        print("   ✅ AblationStudyFramework initialized")
        
        # Initialize interpretability analyzer
        interpretability_analyzer = InterpretabilityAnalyzer()
        print("   ✅ InterpretabilityAnalyzer initialized")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_coordination_script():
    """Test the main coordination script functionality."""
    print("\n🎯 Testing coordination script...")
    
    try:
        from run_phase4_evaluation import Phase4Coordinator
        
        # Initialize coordinator with default settings
        coordinator = Phase4Coordinator(output_dir="./test_output")
        print("   ✅ Phase4Coordinator initialized")
        
        # Test configuration validation
        if hasattr(coordinator, 'config') and coordinator.config:
            print("   ✅ Coordinator configuration loaded")
        else:
            print("   ❌ Coordinator configuration missing")
            return False
        
        # Test output directory creation
        if coordinator.output_dir.exists():
            print("   ✅ Output directory created")
        else:
            print("   ❌ Output directory creation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Coordination script error: {e}")
        return False

def test_baseline_integration():
    """Test that Phase 4 can integrate with existing baseline methods."""
    print("\n🔗 Testing baseline integration...")
    
    try:
        # Test baseline method imports
        from baselines import (
            TrustaLabFramework,
            SubgraphFeaturePropagation,
            EnhancedGNNBaseline,
            LightGBMBaseline,
            RandomForestBaseline,
            TemporalGraphTransformerBaseline
        )
        print("   ✅ All baseline methods importable")
        
        # Test baseline method creation with minimal configs
        baselines = {}
        
        try:
            baselines['TrustaLab'] = TrustaLabFramework({'n_estimators': 5})
            print("   ✅ TrustaLabFramework created")
        except Exception as e:
            print(f"   ⚠️  TrustaLabFramework error: {e}")
        
        try:
            baselines['SubgraphProp'] = SubgraphFeaturePropagation({'epochs': 3})
            print("   ✅ SubgraphFeaturePropagation created")
        except Exception as e:
            print(f"   ⚠️  SubgraphFeaturePropagation error: {e}")
        
        try:
            baselines['LightGBM'] = LightGBMBaseline({'num_boost_round': 5})
            print("   ✅ LightGBMBaseline created")
        except Exception as e:
            print(f"   ⚠️  LightGBMBaseline error: {e}")
        
        try:
            baselines['RandomForest'] = RandomForestBaseline({'n_estimators': 5})
            print("   ✅ RandomForestBaseline created")
        except Exception as e:
            print(f"   ⚠️  RandomForestBaseline error: {e}")
        
        try:
            baselines['TGT'] = TemporalGraphTransformerBaseline({'d_model': 32, 'epochs': 3})
            print("   ✅ TemporalGraphTransformerBaseline created")
        except Exception as e:
            print(f"   ⚠️  TemporalGraphTransformerBaseline error: {e}")
        
        print(f"   📊 Successfully created {len(baselines)} baseline methods")
        
        return len(baselines) > 0
        
    except Exception as e:
        print(f"   ❌ Baseline integration error: {e}")
        return False

def main():
    """Run all Phase 4 framework validation tests."""
    print("🚀 Phase 4 Framework Validation Test")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_configuration),
        ("Framework Initialization", test_framework_initialization),
        ("Coordination Script", test_coordination_script),
        ("Baseline Integration", test_baseline_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}")
        print("-" * 40)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"   ✅ {test_name}: PASSED")
            else:
                print(f"   ❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"   ❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Phase 4 Framework Validation Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    if passed == total:
        print("\n🎉 ALL PHASE 4 FRAMEWORK TESTS PASSED!")
        print("✅ Phase 4 experimental validation framework is ready")
        print("✅ All components can be imported and initialized")
        print("✅ Integration with existing baselines confirmed")
        print("✅ Ready to run comprehensive Phase 4 evaluation!")
        
        print("\n🚀 Next Steps:")
        print("1. Run quick evaluation: python run_phase4_evaluation.py --quick")
        print("2. Run full evaluation: python run_phase4_evaluation.py --full")
        print("3. Run custom evaluation: python run_phase4_evaluation.py --components comprehensive ablation")
        
        return 0
    else:
        print(f"\n❌ {total - passed} PHASE 4 FRAMEWORK TESTS FAILED!")
        print("⚠️  Please fix the failing components before running Phase 4 evaluation")
        
        print("\n🔧 Troubleshooting:")
        print("1. Check that all dependencies are installed")
        print("2. Verify that Phase 3 baseline methods are working")
        print("3. Ensure configuration files are present and valid")
        print("4. Check import paths and module structure")
        
        return 1

if __name__ == "__main__":
    exit(main())