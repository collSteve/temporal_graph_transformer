#!/usr/bin/env python3
"""
Cross-chain compatibility validation for Phase 2 implementation.

Tests the unified dataset interface across Arbitrum DeFi and Jupiter Solana
to ensure our Temporal Graph Transformer can handle different blockchain
ecosystems consistently.
"""

import sys
import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data import (
    ArbitrumDeFiDataset,
    JupiterSolanaDataset,
    BaseTemporalGraphDataset,
    PureCryptoDataset,
    UnifiedDataLoader,
    create_data_loaders,
    TransactionSchemaValidator
)


class CrossChainCompatibilityTester:
    """
    Comprehensive tester for cross-chain dataset compatibility.
    
    Validates that our unified interface works consistently across
    different blockchain ecosystems and transaction types.
    """
    
    def __init__(self, test_data_dir: str = './test_data'):
        self.test_data_dir = test_data_dir
        os.makedirs(test_data_dir, exist_ok=True)
        
        self.arbitrum_dataset = None
        self.jupiter_dataset = None
        self.test_results = {}
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run comprehensive cross-chain compatibility tests."""
        print("🚀 Starting Cross-Chain Compatibility Tests for Phase 2")
        print("=" * 60)
        
        # Test 1: Dataset instantiation
        test1_result = self.test_dataset_instantiation()
        self.test_results['dataset_instantiation'] = test1_result
        
        # Test 2: Unified interface compatibility
        test2_result = self.test_unified_interface()
        self.test_results['unified_interface'] = test2_result
        
        # Test 3: Transaction schema validation
        test3_result = self.test_transaction_schema()
        self.test_results['transaction_schema'] = test3_result
        
        # Test 4: Feature extraction consistency
        test4_result = self.test_feature_extraction()
        self.test_results['feature_extraction'] = test4_result
        
        # Test 5: Graph construction compatibility
        test5_result = self.test_graph_construction()
        self.test_results['graph_construction'] = test5_result
        
        # Test 6: Data loader integration
        test6_result = self.test_data_loader_integration()
        self.test_results['data_loader_integration'] = test6_result
        
        # Test 7: Cross-chain analysis capability
        test7_result = self.test_cross_chain_analysis()
        self.test_results['cross_chain_analysis'] = test7_result
        
        # Print summary
        self.print_test_summary()
        
        return self.test_results
    
    def test_dataset_instantiation(self) -> bool:
        """Test that both datasets can be instantiated successfully."""
        print("\n🔧 Test 1: Dataset Instantiation")
        
        try:
            # Test Arbitrum dataset
            print("   Creating ArbitrumDeFiDataset...")
            self.arbitrum_dataset = ArbitrumDeFiDataset(
                data_path=os.path.join(self.test_data_dir, 'arbitrum'),
                split='test',
                start_date='2023-03-01',
                end_date='2023-03-22',
                max_sequence_length=50,  # Smaller for testing
                include_known_hunters=True
            )
            
            # Test Jupiter dataset
            print("   Creating JupiterSolanaDataset...")
            self.jupiter_dataset = JupiterSolanaDataset(
                data_path=os.path.join(self.test_data_dir, 'jupiter'),
                split='test',
                start_date='2024-01-01',
                end_date='2024-01-30',
                max_sequence_length=50,  # Smaller for testing
                anti_farming_analysis=True
            )
            
            print("   ✅ Dataset instantiation successful")
            return True
            
        except Exception as e:
            print(f"   ❌ Dataset instantiation failed: {e}")
            return False
    
    def test_unified_interface(self) -> bool:
        """Test that both datasets implement the unified interface correctly."""
        print("\n🔧 Test 2: Unified Interface Compatibility")
        
        if not self.arbitrum_dataset or not self.jupiter_dataset:
            print("   ❌ Datasets not available for testing")
            return False
        
        try:
            # Test required methods exist and work
            datasets = [
                ('Arbitrum', self.arbitrum_dataset),
                ('Jupiter', self.jupiter_dataset)
            ]
            
            for name, dataset in datasets:
                print(f"   Testing {name} dataset interface...")
                
                # Test basic properties
                assert hasattr(dataset, 'transactions'), f"{name}: Missing transactions"
                assert hasattr(dataset, 'users'), f"{name}: Missing users"
                assert hasattr(dataset, 'labels'), f"{name}: Missing labels"
                
                # Test required methods
                assert hasattr(dataset, '__len__'), f"{name}: Missing __len__"
                assert hasattr(dataset, '__getitem__'), f"{name}: Missing __getitem__"
                assert hasattr(dataset, 'extract_transaction_features'), f"{name}: Missing extract_transaction_features"
                assert hasattr(dataset, 'build_user_graph'), f"{name}: Missing build_user_graph"
                
                # Test inheritance
                assert isinstance(dataset, BaseTemporalGraphDataset), f"{name}: Not BaseTemporalGraphDataset"
                assert isinstance(dataset, PureCryptoDataset), f"{name}: Not PureCryptoDataset"
                
                # Test dataset length
                dataset_len = len(dataset)
                assert dataset_len > 0, f"{name}: Empty dataset"
                print(f"     {name} dataset size: {dataset_len}")
            
            print("   ✅ Unified interface compatibility confirmed")
            return True
            
        except Exception as e:
            print(f"   ❌ Unified interface test failed: {e}")
            return False
    
    def test_transaction_schema(self) -> bool:
        """Test transaction schema validation and compatibility."""
        print("\n🔧 Test 3: Transaction Schema Validation")
        
        try:
            validator = TransactionSchemaValidator()
            
            # Test Arbitrum transactions
            if self.arbitrum_dataset and hasattr(self.arbitrum_dataset, 'transactions'):
                arb_txs = self.arbitrum_dataset.transactions
                if len(arb_txs) > 0:
                    is_valid, errors = validator.validate_dataframe(arb_txs)
                    if not is_valid:
                        print(f"     Arbitrum validation errors: {errors}")
                    else:
                        print("     ✅ Arbitrum transactions valid")
            
            # Test Jupiter transactions  
            if self.jupiter_dataset and hasattr(self.jupiter_dataset, 'transactions'):
                jup_txs = self.jupiter_dataset.transactions
                if len(jup_txs) > 0:
                    is_valid, errors = validator.validate_dataframe(jup_txs)
                    if not is_valid:
                        print(f"     Jupiter validation errors: {errors}")
                    else:
                        print("     ✅ Jupiter transactions valid")
            
            # Test cross-chain schema compatibility
            print("     Testing cross-chain schema compatibility...")
            
            # Both should have common base fields
            required_fields = ['user_id', 'timestamp', 'chain_id', 'transaction_type', 'value_usd', 'gas_fee']
            
            for name, dataset in [('Arbitrum', self.arbitrum_dataset), ('Jupiter', self.jupiter_dataset)]:
                if dataset and hasattr(dataset, 'transactions') and len(dataset.transactions) > 0:
                    txs = dataset.transactions
                    missing_fields = set(required_fields) - set(txs.columns)
                    if missing_fields:
                        print(f"     ❌ {name} missing required fields: {missing_fields}")
                        return False
                    else:
                        print(f"     ✅ {name} has all required fields")
            
            print("   ✅ Transaction schema validation successful")
            return True
            
        except Exception as e:
            print(f"   ❌ Transaction schema test failed: {e}")
            return False
    
    def test_feature_extraction(self) -> bool:
        """Test that feature extraction works consistently across chains."""
        print("\n🔧 Test 4: Feature Extraction Consistency")
        
        try:
            # Test feature extraction for both datasets
            for name, dataset in [('Arbitrum', self.arbitrum_dataset), ('Jupiter', self.jupiter_dataset)]:
                if not dataset or len(dataset) == 0:
                    continue
                    
                print(f"   Testing {name} feature extraction...")
                
                # Get a sample user
                sample_idx = 0
                sample_item = dataset[sample_idx]
                
                # Check required components in sample
                required_keys = [
                    'transaction_features', 'timestamps', 'node_features',
                    'edge_index', 'edge_features', 'attention_mask', 'label'
                ]
                
                for key in required_keys:
                    assert key in sample_item, f"{name}: Missing {key} in sample"
                    assert isinstance(sample_item[key], torch.Tensor), f"{name}: {key} not tensor"
                
                # Check feature dimensions
                tx_features = sample_item['transaction_features']
                assert isinstance(tx_features, dict), f"{name}: transaction_features not dict"
                
                # Check timestamps
                timestamps = sample_item['timestamps']
                assert timestamps.shape[0] == dataset.max_sequence_length, f"{name}: Wrong timestamp length"
                
                # Check node features
                node_features = sample_item['node_features']
                assert len(node_features.shape) == 1, f"{name}: Wrong node feature shape"
                
                print(f"     ✅ {name} feature extraction successful")
                print(f"        Transaction features: {len(tx_features)} types")
                print(f"        Node feature size: {node_features.shape[0]}")
                print(f"        Edge count: {sample_item['edge_index'].shape[1]}")
            
            print("   ✅ Feature extraction consistency confirmed")
            return True
            
        except Exception as e:
            print(f"   ❌ Feature extraction test failed: {e}")
            return False
    
    def test_graph_construction(self) -> bool:
        """Test graph construction across different blockchain types."""
        print("\n🔧 Test 5: Graph Construction Compatibility")
        
        try:
            for name, dataset in [('Arbitrum', self.arbitrum_dataset), ('Jupiter', self.jupiter_dataset)]:
                if not dataset or len(dataset) == 0:
                    continue
                    
                print(f"   Testing {name} graph construction...")
                
                # Test graph building
                edge_index, edge_features_dict = dataset.build_user_graph()
                
                # Validate graph structure
                assert isinstance(edge_index, torch.Tensor), f"{name}: edge_index not tensor"
                assert edge_index.shape[0] == 2, f"{name}: Wrong edge_index shape"
                assert isinstance(edge_features_dict, dict), f"{name}: edge_features not dict"
                
                # Check edge features
                if 'edge_features' in edge_features_dict:
                    edge_features = edge_features_dict['edge_features']
                    assert isinstance(edge_features, torch.Tensor), f"{name}: edge_features not tensor"
                    assert edge_features.shape[0] == edge_index.shape[1], f"{name}: Mismatched edge counts"
                
                print(f"     ✅ {name} graph construction successful")
                print(f"        Nodes: {len(dataset.users) if hasattr(dataset, 'users') else 0}")
                print(f"        Edges: {edge_index.shape[1]}")
                print(f"        Edge feature size: {edge_features.shape[1] if 'edge_features' in edge_features_dict else 0}")
            
            print("   ✅ Graph construction compatibility confirmed")
            return True
            
        except Exception as e:
            print(f"   ❌ Graph construction test failed: {e}")
            return False
    
    def test_data_loader_integration(self) -> bool:
        """Test that datasets work with unified data loader."""
        print("\n🔧 Test 6: Data Loader Integration")
        
        try:
            # Test individual data loaders
            for name, dataset in [('Arbitrum', self.arbitrum_dataset), ('Jupiter', self.jupiter_dataset)]:
                if not dataset or len(dataset) == 0:
                    continue
                    
                print(f"   Testing {name} data loader...")
                
                # Create data loader
                data_loader = UnifiedDataLoader(
                    dataset=dataset,
                    batch_size=4,
                    shuffle=False,
                    model_type='temporal_graph_transformer'
                )
                
                # Test batch loading
                batch_count = 0
                for batch in data_loader:
                    # Validate batch structure
                    assert isinstance(batch, dict), f"{name}: Batch not dict"
                    
                    required_batch_keys = [
                        'transaction_features', 'timestamps', 'node_features',
                        'edge_index', 'edge_features', 'labels'
                    ]
                    
                    for key in required_batch_keys:
                        assert key in batch, f"{name}: Missing {key} in batch"
                    
                    # Check batch dimensions
                    batch_size = batch['labels'].shape[0]
                    assert batch_size <= 4, f"{name}: Batch size too large"
                    
                    batch_count += 1
                    if batch_count >= 2:  # Test first 2 batches
                        break
                
                print(f"     ✅ {name} data loader successful ({batch_count} batches tested)")
            
            print("   ✅ Data loader integration confirmed")
            return True
            
        except Exception as e:
            print(f"   ❌ Data loader integration test failed: {e}")
            return False
    
    def test_cross_chain_analysis(self) -> bool:
        """Test cross-chain analysis capabilities."""
        print("\n🔧 Test 7: Cross-Chain Analysis Capability")
        
        try:
            # Test dataset statistics comparison
            print("   Comparing dataset statistics...")
            
            stats_comparison = {}
            
            for name, dataset in [('Arbitrum', self.arbitrum_dataset), ('Jupiter', self.jupiter_dataset)]:
                if not dataset:
                    continue
                    
                stats = dataset.get_dataset_stats()
                stats_comparison[name] = stats
                
                print(f"     {name} Statistics:")
                print(f"       Users: {stats.get('num_users', 0)}")
                print(f"       Transactions: {stats.get('num_transactions', 0)}")
                print(f"       Hunter ratio: {stats.get('hunter_ratio', 0):.3f}")
                print(f"       Avg txs/user: {stats.get('avg_transactions_per_user', 0):.1f}")
            
            # Test protocol-specific features
            if hasattr(self.arbitrum_dataset, 'get_protocol_statistics'):
                arb_protocol_stats = self.arbitrum_dataset.get_protocol_statistics()
                print(f"     Arbitrum Protocol Distribution: {arb_protocol_stats.get('protocol_distribution', {})}")
            
            if hasattr(self.jupiter_dataset, 'get_protocol_statistics'):
                jup_protocol_stats = self.jupiter_dataset.get_protocol_statistics()
                print(f"     Jupiter Protocol Distribution: {jup_protocol_stats.get('protocol_distribution', {})}")
            
            # Test cross-chain feature compatibility
            print("   Testing cross-chain feature compatibility...")
            
            # Sample features from both datasets
            arb_sample = self.arbitrum_dataset[0] if self.arbitrum_dataset and len(self.arbitrum_dataset) > 0 else None
            jup_sample = self.jupiter_dataset[0] if self.jupiter_dataset and len(self.jupiter_dataset) > 0 else None
            
            if arb_sample and jup_sample:
                # Both should have the same interface
                arb_features = arb_sample['transaction_features']
                jup_features = jup_sample['transaction_features']
                
                # Check that base features exist in both
                base_features = ['timestamps', 'values_usd', 'gas_fees', 'transaction_types']
                
                common_features = set(arb_features.keys()) & set(jup_features.keys())
                missing_base = set(base_features) - common_features
                
                if missing_base:
                    print(f"     ⚠️  Some base features missing in cross-chain compatibility: {missing_base}")
                else:
                    print(f"     ✅ All base features present in both datasets")
                
                print(f"     Common features: {len(common_features)}")
                print(f"     Arbitrum-specific features: {len(set(arb_features.keys()) - common_features)}")
                print(f"     Jupiter-specific features: {len(set(jup_features.keys()) - common_features)}")
            
            print("   ✅ Cross-chain analysis capability confirmed")
            return True
            
        except Exception as e:
            print(f"   ❌ Cross-chain analysis test failed: {e}")
            return False
    
    def print_test_summary(self) -> None:
        """Print comprehensive test summary."""
        print("\n" + "=" * 60)
        print("📊 CROSS-CHAIN COMPATIBILITY TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name:<25}: {status}")
        
        if all(self.test_results.values()):
            print("\n🎉 ALL TESTS PASSED! Cross-chain compatibility confirmed.")
            print("   Ready for Phase 3: Baseline Implementation & Training")
        else:
            print("\n⚠️  Some tests failed. Review implementation before proceeding.")
        
        print("=" * 60)


def main():
    """Run cross-chain compatibility tests."""
    # Create test directory
    test_data_dir = '/tmp/temporal_graph_transformer_test'
    
    # Run tests
    tester = CrossChainCompatibilityTester(test_data_dir)
    results = tester.run_all_tests()
    
    # Clean up test data if all tests passed
    if all(results.values()):
        import shutil
        if os.path.exists(test_data_dir):
            shutil.rmtree(test_data_dir)
        print(f"\n🧹 Cleaned up test directory: {test_data_dir}")
    
    return results


if __name__ == '__main__':
    main()