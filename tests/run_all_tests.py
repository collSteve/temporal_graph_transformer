#!/usr/bin/env python3
"""
Master test runner for the Temporal Graph Transformer project.

This script runs all unit tests, integration tests, and performance tests
in a systematic manner with detailed reporting.
"""

import sys
import os
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def print_banner(title: str):
    """Print formatted banner."""
    print(f"\n{'='*80}")
    print(f"🚀 {title}")
    print(f"{'='*80}")

def print_section(section: str):
    """Print section header."""
    print(f"\n{'─'*60}")
    print(f"📁 {section}")
    print(f"{'─'*60}")

def run_test_file(test_file: Path, env_python: str) -> Tuple[bool, float, str]:
    """
    Run a single test file and return results.
    
    Returns:
        (success, duration, output)
    """
    start_time = time.time()
    
    try:
        # Run test with conda environment python
        result = subprocess.run(
            [env_python, str(test_file)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per test
        )
        
        duration = time.time() - start_time
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        return success, duration, output
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return False, duration, "Test timed out after 5 minutes"
    except Exception as e:
        duration = time.time() - start_time
        return False, duration, f"Error running test: {str(e)}"

def main():
    """Main test runner."""
    print_banner("TEMPORAL GRAPH TRANSFORMER - COMPREHENSIVE TEST SUITE")
    
    # Environment setup
    conda_env_python = "/opt/homebrew/anaconda3/envs/temporal_graph_transformer/bin/python"
    
    # Verify conda environment
    if not Path(conda_env_python).exists():
        print("❌ Conda environment not found!")
        print(f"Expected: {conda_env_python}")
        print("Please create the conda environment first.")
        sys.exit(1)
    
    print(f"🐍 Using Python: {conda_env_python}")
    
    # Test file organization
    test_files = {
        "Unit Tests": [
            "tests/unit/test_time_encoding.py",
            "tests/unit/test_temporal_encoder.py", 
            "tests/unit/test_graph_encoder.py",
            "tests/unit/test_loss_functions.py"
        ],
        "Integration Tests": [
            "tests/integration/test_end_to_end.py"
        ]
    }
    
    # Results tracking
    all_results = {}
    total_tests = 0
    total_passed = 0
    total_duration = 0
    
    # Run all test categories
    for category, files in test_files.items():
        print_section(category)
        category_results = {}
        
        for test_file in files:
            test_path = PROJECT_ROOT / test_file
            
            if not test_path.exists():
                print(f"⚠️  Test file not found: {test_file}")
                category_results[test_file] = (False, 0, "File not found")
                continue
            
            print(f"🧪 Running: {test_file}")
            
            success, duration, output = run_test_file(test_path, conda_env_python)
            category_results[test_file] = (success, duration, output)
            
            total_tests += 1
            total_duration += duration
            
            if success:
                total_passed += 1
                print(f"   ✅ PASSED ({duration:.2f}s)")
            else:
                print(f"   ❌ FAILED ({duration:.2f}s)")
                # Print first few lines of error for quick debugging
                error_lines = output.split('\n')[:10]
                for line in error_lines:
                    if line.strip():
                        print(f"      {line}")
                if len(output.split('\n')) > 10:
                    print(f"      ... (see full output below)")
        
        all_results[category] = category_results
    
    # Final summary
    print_banner("TEST RESULTS SUMMARY")
    
    print(f"📊 Overall Results:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {total_passed}")
    print(f"   Failed: {total_tests - total_passed}")
    print(f"   Success Rate: {total_passed/total_tests*100:.1f}%")
    print(f"   Total Duration: {total_duration:.2f}s")
    
    # Detailed results by category
    for category, results in all_results.items():
        category_passed = sum(1 for success, _, _ in results.values() if success)
        category_total = len(results)
        
        print(f"\n📁 {category}:")
        print(f"   Passed: {category_passed}/{category_total}")
        
        for test_file, (success, duration, output) in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {test_file} ({duration:.2f}s)")
    
    # Failed test details
    failed_tests = []
    for category, results in all_results.items():
        for test_file, (success, duration, output) in results.items():
            if not success:
                failed_tests.append((test_file, output))
    
    if failed_tests:
        print_banner("FAILED TEST DETAILS")
        for test_file, output in failed_tests:
            print(f"\n❌ {test_file}:")
            print("-" * 40)
            print(output)
            print("-" * 40)
    
    # Environment information
    print_banner("ENVIRONMENT INFORMATION")
    try:
        # Get Python version
        python_result = subprocess.run([conda_env_python, "--version"], capture_output=True, text=True)
        print(f"Python: {python_result.stdout.strip()}")
        
        # Get key package versions
        packages = ["torch", "torch_geometric", "numpy", "pandas"]
        for package in packages:
            try:
                version_result = subprocess.run(
                    [conda_env_python, "-c", f"import {package}; print(f'{package}: {{getattr({package}, '__version__', 'unknown')}}')"],
                    capture_output=True, text=True
                )
                if version_result.returncode == 0:
                    print(f"{version_result.stdout.strip()}")
            except:
                print(f"{package}: version check failed")
    except:
        print("Could not retrieve environment information")
    
    # Test file management summary
    print_banner("TEST FILE MANAGEMENT")
    print("📂 Test Organization:")
    print("   tests/")
    print("   ├── unit/              # Individual component tests")
    print("   │   ├── test_time_encoding.py")
    print("   │   ├── test_temporal_encoder.py")
    print("   │   ├── test_graph_encoder.py")
    print("   │   └── test_loss_functions.py")
    print("   ├── integration/       # End-to-end tests")
    print("   │   └── test_end_to_end.py")
    print("   ├── utils/             # Test utilities")
    print("   │   └── test_config.py")
    print("   └── run_all_tests.py   # This master runner")
    
    print("\n🔄 Running Individual Tests:")
    print("   # Run all tests:")
    print("   python tests/run_all_tests.py")
    print()
    print("   # Run specific test category:")
    print("   /opt/homebrew/anaconda3/envs/temporal_graph_transformer/bin/python tests/unit/test_time_encoding.py")
    print("   /opt/homebrew/anaconda3/envs/temporal_graph_transformer/bin/python tests/integration/test_end_to_end.py")
    
    # Exit with appropriate code
    if total_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total_tests - total_passed} TESTS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()