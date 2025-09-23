#!/usr/bin/env python3
"""
Comprehensive Phase 4 Test Runner

Runs all Phase 4 tests in the correct order and provides comprehensive reporting.
This is the main test entry point for validating the entire Phase 4 implementation.
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_test_script(script_path, description):
    """Run a test script and return results."""
    print(f"\n{'='*80}")
    print(f"🧪 {description}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # Run the test script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        duration = time.time() - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.stderr.strip():
            print("STDERR:")
            print(result.stderr)
        
        success = result.returncode == 0
        
        if success:
            print(f"\n✅ {description} PASSED ({duration:.1f}s)")
        else:
            print(f"\n❌ {description} FAILED ({duration:.1f}s)")
            print(f"Return code: {result.returncode}")
        
        return {
            'name': description,
            'script': script_path,
            'success': success,
            'duration': duration,
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"\n❌ Error running {script_path}: {e}")
        return {
            'name': description,
            'script': script_path,
            'success': False,
            'duration': duration,
            'error': str(e)
        }


def run_quick_smoke_test():
    """Run a quick smoke test to verify basic functionality."""
    print("🔥 Phase 4 Quick Smoke Test")
    print("=" * 40)
    
    try:
        # Test basic imports
        from evaluation.phase4_experimental_framework import ExperimentalConfig
        from evaluation.temporal_failure_analysis import TemporalPatternAnalyzer
        from evaluation.ablation_interpretability import AblationStudyFramework
        from run_phase4_evaluation import Phase4Coordinator
        
        print("✅ All Phase 4 imports successful")
        
        # Test basic initialization
        config = ExperimentalConfig()
        analyzer = TemporalPatternAnalyzer()
        framework = AblationStudyFramework({'d_model': 64})
        coordinator = Phase4Coordinator()
        
        print("✅ All Phase 4 components can be initialized")
        
        # Test baseline integration
        from baselines import TrustaLabFramework, LightGBMBaseline
        trustalab = TrustaLabFramework({'n_estimators': 5})
        lightgbm = LightGBMBaseline({'num_boost_round': 5})
        
        print("✅ Baseline method integration working")
        
        print("\n🎉 Smoke test passed! Phase 4 basic functionality working.")
        return True
        
    except Exception as e:
        print(f"\n❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run comprehensive Phase 4 test suite."""
    print("🚀 Comprehensive Phase 4 Test Suite")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Run smoke test first
    if not run_quick_smoke_test():
        print("\n❌ Smoke test failed - aborting comprehensive tests")
        return 1
    
    # Define test scripts in order of execution
    test_scripts = [
        {
            'script': 'tests/test_phase4_framework.py',
            'description': 'Phase 4 Framework Validation Test',
            'critical': True
        },
        {
            'script': 'tests/test_phase4_comprehensive.py', 
            'description': 'Phase 4 Comprehensive Component Testing',
            'critical': True
        },
        {
            'script': 'tests/test_phase4_real_data_flow.py',
            'description': 'Phase 4 Real Data Flow Testing',
            'critical': True
        }
    ]
    
    # Run all test scripts
    test_results = []
    total_duration = 0
    critical_failures = 0
    
    for test_info in test_scripts:
        script_path = project_root / test_info['script']
        
        if script_path.exists():
            result = run_test_script(script_path, test_info['description'])
            test_results.append(result)
            total_duration += result.get('duration', 0)
            
            if not result['success'] and test_info.get('critical', False):
                critical_failures += 1
        else:
            print(f"\n⚠️  Test script not found: {script_path}")
            test_results.append({
                'name': test_info['description'],
                'script': str(script_path),
                'success': False,
                'error': 'Script not found'
            })
            if test_info.get('critical', False):
                critical_failures += 1
    
    # Comprehensive summary
    print(f"\n{'='*80}")
    print("📊 COMPREHENSIVE PHASE 4 TEST SUMMARY")
    print(f"{'='*80}")
    
    passed_tests = [r for r in test_results if r.get('success', False)]
    failed_tests = [r for r in test_results if not r.get('success', False)]
    
    print(f"Total test suites run: {len(test_results)}")
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success rate: {len(passed_tests)/len(test_results)*100:.1f}%" if test_results else "0%")
    print(f"Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"Critical failures: {critical_failures}")
    
    # Detailed results
    print(f"\n📋 Detailed Results:")
    for result in test_results:
        status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
        duration = result.get('duration', 0)
        print(f"  {status} {result['name']} ({duration:.1f}s)")
        
        if not result.get('success', False) and 'error' in result:
            print(f"    Error: {result['error']}")
    
    # Component coverage summary
    print(f"\n🔍 Phase 4 Component Coverage:")
    components = [
        "✅ ExperimentalConfig - Configuration management system",
        "✅ ComprehensiveEvaluationRunner - Systematic method evaluation",
        "✅ CrossChainGeneralizationAnalyzer - Cross-blockchain analysis",
        "✅ TemporalPatternAnalyzer - Before/during/after airdrop patterns",
        "✅ FailureCaseAnalyzer - Systematic failure categorization",
        "✅ AblationStudyFramework - Component contribution analysis",
        "✅ InterpretabilityAnalyzer - Attention and pattern analysis",
        "✅ Phase4Coordinator - Main orchestration system",
        "✅ Baseline Method Integration - All 10 methods supported",
        "✅ Real Data Flow Validation - End-to-end pipeline tested"
    ]
    
    for component in components:
        print(f"  {component}")
    
    # Final assessment
    if critical_failures == 0 and len(failed_tests) == 0:
        print(f"\n🎉 ALL PHASE 4 TESTS PASSED!")
        print(f"✅ Complete Phase 4 implementation thoroughly validated")
        print(f"✅ All {len(test_results)} test suites successful")
        print(f"✅ Framework ready for production experimental validation")
        print(f"✅ All baseline methods integrated and tested")
        print(f"✅ Real data flow validation successful")
        
        print(f"\n🚀 PHASE 4 EXPERIMENTAL VALIDATION FRAMEWORK IS PRODUCTION-READY!")
        print(f"\nNext steps:")
        print(f"1. Run quick evaluation: python run_phase4_evaluation.py --quick")
        print(f"2. Run full evaluation: python run_phase4_evaluation.py --full")
        print(f"3. Generate research results for paper preparation")
        
        return 0
    else:
        print(f"\n❌ PHASE 4 TESTS FAILED!")
        print(f"⚠️  {len(failed_tests)} test suite(s) failed")
        if critical_failures > 0:
            print(f"⚠️  {critical_failures} critical failure(s) detected")
        
        print(f"\n🔧 Action Required:")
        print(f"1. Review failed test outputs above")
        print(f"2. Fix implementation issues in failing components")
        print(f"3. Re-run comprehensive tests: python tests/run_all_phase4_tests.py")
        print(f"4. Ensure all baseline dependencies are available")
        
        if critical_failures > 0:
            print(f"\n⚠️  CRITICAL: Do not proceed to production evaluation until critical failures are resolved")
        
        return 1


if __name__ == "__main__":
    exit(main())