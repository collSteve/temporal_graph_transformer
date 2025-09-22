#!/usr/bin/env python3
"""
Phase 2 Basic Import Tests

Tests that all new Phase 2 components can be imported successfully.
This validates that our multi-asset dataset interface is properly structured.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_base_class_imports():
    """Test that base classes can be imported."""
    print("Testing base class imports...")
    
    try:
        from data import BaseTemporalGraphDataset, PureCryptoDataset
        print("   ✅ Base classes imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Base class import error: {e}")
        return False

def test_dataset_imports():
    """Test that dataset implementations can be imported."""
    print("Testing dataset implementation imports...")
    
    try:
        from data import ArbitrumDeFiDataset, JupiterSolanaDataset
        print("   ✅ DeFi dataset classes imported successfully")
        
        from data import SolanaNFTDataset, EthereumNFTDataset, L2NetworkDataset
        print("   ✅ NFT dataset classes imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Dataset import error: {e}")
        return False

def test_schema_imports():
    """Test that transaction schema can be imported."""
    print("Testing transaction schema imports...")
    
    try:
        from data.transaction_schema import (
            BaseTransaction, 
            ArbitrumDeFiTransaction, 
            SolanaDeFiTransaction,
            OptimismDeFiTransaction,
            TransactionSchemaValidator,
            create_transaction_from_dict,
            transactions_to_dataframe
        )
        print("   ✅ Transaction schema classes imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Schema import error: {e}")
        return False

def test_utility_imports():
    """Test that utility components can be imported."""
    print("Testing utility component imports...")
    
    try:
        from data import (
            TemporalGraphPreprocessor,
            UnifiedDataLoader,
            create_data_loaders
        )
        print("   ✅ Utility components imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Utility import error: {e}")
        return False

def main():
    """Run all basic import tests."""
    print("🔧 Phase 2 Basic Import Tests")
    print("=" * 40)
    
    tests = [
        test_base_class_imports,
        test_dataset_imports,
        test_schema_imports,
        test_utility_imports
    ]
    
    results = []
    for test in tests:
        print(f"\n{len(results)+1}. {test.__doc__.strip()}")
        result = test()
        results.append(result)
    
    print("\n" + "=" * 40)
    print("📊 Import Test Summary")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if all(results):
        print("\n🎉 All import tests passed!")
        return 0
    else:
        print("\n❌ Some import tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())