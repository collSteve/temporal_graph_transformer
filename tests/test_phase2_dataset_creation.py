#!/usr/bin/env python3
"""
Phase 2 Dataset Creation Tests

Tests that ArbitrumDeFiDataset and JupiterSolanaDataset can be created
and generate demonstration data properly.
"""

import sys
import os
import tempfile
import shutil
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_arbitrum_dataset_creation():
    """Test ArbitrumDeFiDataset creation and data generation."""
    print("Testing ArbitrumDeFiDataset creation...")
    
    # Create temporary directory for test data
    test_dir = tempfile.mkdtemp(prefix='arbitrum_test_')
    
    try:
        from data import ArbitrumDeFiDataset
        
        # Create dataset (will generate demonstration data)
        dataset = ArbitrumDeFiDataset(
            data_path=test_dir,
            split='test',
            start_date='2023-03-15',
            end_date='2023-03-22',
            max_sequence_length=20,  # Smaller for testing
            include_known_hunters=True
        )
        
        print(f"   ✅ ArbitrumDeFiDataset created successfully")
        
        # Test dataset properties
        assert hasattr(dataset, 'transactions')
        assert hasattr(dataset, 'users')
        assert hasattr(dataset, 'labels')
        assert hasattr(dataset, 'blockchain')
        assert dataset.blockchain == 'arbitrum'
        
        print(f"   ✅ Dataset has required attributes")
        
        # Test that data was generated
        assert len(dataset) > 0, "Dataset should not be empty"
        print(f"   ✅ Dataset has {len(dataset)} users")
        
        # Test known hunter addresses
        if hasattr(dataset, 'known_hunter_addresses'):
            hunter_count = len(dataset.known_hunter_addresses)
            print(f"   ✅ {hunter_count} known hunter addresses loaded")
        
        # Test protocols
        expected_protocols = ['uniswap_v3', 'gmx', 'camelot', 'sushiswap', 'aave']
        assert set(dataset.protocols) == set(expected_protocols)
        print(f"   ✅ Arbitrum protocols configured: {dataset.protocols}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ArbitrumDeFiDataset creation error: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

def test_jupiter_dataset_creation():
    """Test JupiterSolanaDataset creation and data generation."""
    print("Testing JupiterSolanaDataset creation...")
    
    # Create temporary directory for test data
    test_dir = tempfile.mkdtemp(prefix='jupiter_test_')
    
    try:
        from data import JupiterSolanaDataset
        
        # Create dataset (will generate demonstration data)
        dataset = JupiterSolanaDataset(
            data_path=test_dir,
            split='test',
            start_date='2024-01-15',
            end_date='2024-01-30',
            max_sequence_length=20,  # Smaller for testing
            anti_farming_analysis=True,
            include_stablecoin_farming=True
        )
        
        print(f"   ✅ JupiterSolanaDataset created successfully")
        
        # Test dataset properties
        assert hasattr(dataset, 'transactions')
        assert hasattr(dataset, 'users')
        assert hasattr(dataset, 'labels')
        assert hasattr(dataset, 'blockchain')
        assert dataset.blockchain == 'solana'
        
        print(f"   ✅ Dataset has required attributes")
        
        # Test that data was generated
        assert len(dataset) > 0, "Dataset should not be empty"
        print(f"   ✅ Dataset has {len(dataset)} users")
        
        # Test Jupiter anti-farming criteria
        if hasattr(dataset, 'jupiter_anti_farming_criteria'):
            criteria = dataset.jupiter_anti_farming_criteria
            assert 'min_interactions' in criteria
            assert 'stablecoin_penalty' in criteria
            print(f"   ✅ Jupiter anti-farming criteria configured")
        
        # Test protocols
        expected_protocols = ['jupiter', 'raydium', 'orca', 'drift', 'kamino', 'marinade']
        assert set(dataset.protocols) == set(expected_protocols)
        print(f"   ✅ Solana protocols configured: {dataset.protocols}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ JupiterSolanaDataset creation error: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

def test_dataset_interface_consistency():
    """Test that both datasets implement the same interface."""
    print("Testing dataset interface consistency...")
    
    # Create temporary directories
    arb_dir = tempfile.mkdtemp(prefix='arb_interface_test_')
    jup_dir = tempfile.mkdtemp(prefix='jup_interface_test_')
    
    try:
        from data import ArbitrumDeFiDataset, JupiterSolanaDataset, BaseTemporalGraphDataset, PureCryptoDataset
        
        # Create both datasets
        arb_dataset = ArbitrumDeFiDataset(
            data_path=arb_dir,
            split='test',
            max_sequence_length=10
        )
        
        jup_dataset = JupiterSolanaDataset(
            data_path=jup_dir,
            split='test',
            max_sequence_length=10
        )
        
        datasets = [
            ('Arbitrum', arb_dataset),
            ('Jupiter', jup_dataset)
        ]
        
        # Test inheritance
        for name, dataset in datasets:
            assert isinstance(dataset, BaseTemporalGraphDataset), f"{name} not BaseTemporalGraphDataset"
            assert isinstance(dataset, PureCryptoDataset), f"{name} not PureCryptoDataset"
            print(f"   ✅ {name} inheritance correct")
        
        # Test required methods exist
        required_methods = [
            '__len__', '__getitem__', 'extract_transaction_features',
            'extract_defi_features', 'build_user_graph', 'load_raw_data',
            'verify_data_integrity'
        ]
        
        for name, dataset in datasets:
            for method in required_methods:
                assert hasattr(dataset, method), f"{name} missing {method}"
            print(f"   ✅ {name} has all required methods")
        
        # Test common attributes
        common_attributes = ['blockchain', 'protocols', 'max_sequence_length', 'airdrop_events']
        
        for name, dataset in datasets:
            for attr in common_attributes:
                assert hasattr(dataset, attr), f"{name} missing {attr}"
            print(f"   ✅ {name} has all common attributes")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Interface consistency test error: {e}")
        return False
    finally:
        # Clean up
        for test_dir in [arb_dir, jup_dir]:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

def test_dataset_sample_generation():
    """Test that datasets can generate valid samples."""
    print("Testing dataset sample generation...")
    
    # Create temporary directories
    arb_dir = tempfile.mkdtemp(prefix='arb_sample_test_')
    jup_dir = tempfile.mkdtemp(prefix='jup_sample_test_')
    
    try:
        from data import ArbitrumDeFiDataset, JupiterSolanaDataset
        import torch
        
        # Create datasets
        arb_dataset = ArbitrumDeFiDataset(
            data_path=arb_dir,
            split='test',
            max_sequence_length=15
        )
        
        jup_dataset = JupiterSolanaDataset(
            data_path=jup_dir,
            split='test',
            max_sequence_length=15
        )
        
        datasets = [
            ('Arbitrum', arb_dataset),
            ('Jupiter', jup_dataset)
        ]
        
        # Test sample generation
        for name, dataset in datasets:
            if len(dataset) == 0:
                print(f"   ⚠️  {name} dataset is empty, skipping sample test")
                continue
                
            sample = dataset[0]
            print(f"   ✅ {name} generated sample successfully")
            
            # Test sample structure
            required_keys = [
                'transaction_features', 'timestamps', 'node_features',
                'edge_index', 'edge_features', 'attention_mask', 'label', 'user_id'
            ]
            
            for key in required_keys:
                assert key in sample, f"{name} sample missing {key}"
                if key != 'user_id':  # user_id is string
                    assert isinstance(sample[key], (torch.Tensor, dict)), f"{name} {key} not tensor/dict"
            
            print(f"   ✅ {name} sample structure correct")
            
            # Test transaction features
            tx_features = sample['transaction_features']
            assert isinstance(tx_features, dict), f"{name} transaction_features not dict"
            
            # Both should have common base features
            expected_base_features = ['timestamps', 'values_usd', 'gas_fees', 'transaction_types']
            
            for feature in expected_base_features:
                if feature in tx_features:
                    assert isinstance(tx_features[feature], torch.Tensor), f"{name} {feature} not tensor"
                    assert tx_features[feature].shape[0] == dataset.max_sequence_length, f"{name} {feature} wrong length"
            
            print(f"   ✅ {name} transaction features valid")
            
            # Test timestamps
            timestamps = sample['timestamps']
            assert timestamps.shape[0] == dataset.max_sequence_length, f"{name} wrong timestamp length"
            
            # Test attention mask
            attention_mask = sample['attention_mask']
            assert attention_mask.dtype == torch.bool, f"{name} attention_mask not bool"
            assert attention_mask.shape[0] == dataset.max_sequence_length, f"{name} attention_mask wrong length"
            
            print(f"   ✅ {name} sample validation passed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Sample generation test error: {e}")
        return False
    finally:
        # Clean up
        for test_dir in [arb_dir, jup_dir]:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

def test_dataset_statistics():
    """Test that datasets provide meaningful statistics."""
    print("Testing dataset statistics...")
    
    # Create temporary directories
    arb_dir = tempfile.mkdtemp(prefix='arb_stats_test_')
    jup_dir = tempfile.mkdtemp(prefix='jup_stats_test_')
    
    try:
        from data import ArbitrumDeFiDataset, JupiterSolanaDataset
        
        # Create datasets
        arb_dataset = ArbitrumDeFiDataset(
            data_path=arb_dir,
            split='test',
            max_sequence_length=10
        )
        
        jup_dataset = JupiterSolanaDataset(
            data_path=jup_dir,
            split='test',
            max_sequence_length=10
        )
        
        datasets = [
            ('Arbitrum', arb_dataset),
            ('Jupiter', jup_dataset)
        ]
        
        # Test statistics
        for name, dataset in datasets:
            if len(dataset) == 0:
                print(f"   ⚠️  {name} dataset is empty, skipping stats test")
                continue
            
            # Test get_dataset_stats
            stats = dataset.get_dataset_stats()
            assert isinstance(stats, dict), f"{name} stats not dict"
            
            expected_stats = ['num_users', 'num_transactions', 'hunter_ratio']
            for stat in expected_stats:
                assert stat in stats, f"{name} missing {stat} in stats"
            
            print(f"   ✅ {name} basic statistics available")
            print(f"       Users: {stats['num_users']}")
            print(f"       Transactions: {stats['num_transactions']}")
            print(f"       Hunter ratio: {stats['hunter_ratio']:.3f}")
            
            # Test protocol statistics (if available)
            if hasattr(dataset, 'get_protocol_statistics'):
                protocol_stats = dataset.get_protocol_statistics()
                if protocol_stats:
                    print(f"   ✅ {name} protocol statistics available")
                    if 'protocol_distribution' in protocol_stats:
                        protocol_dist = protocol_stats['protocol_distribution']
                        print(f"       Top protocol: {max(protocol_dist, key=protocol_dist.get)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Dataset statistics test error: {e}")
        return False
    finally:
        # Clean up
        for test_dir in [arb_dir, jup_dir]:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)

def main():
    """Run all dataset creation tests."""
    print("🔧 Phase 2 Dataset Creation Tests")
    print("=" * 40)
    
    tests = [
        test_arbitrum_dataset_creation,
        test_jupiter_dataset_creation,
        test_dataset_interface_consistency,
        test_dataset_sample_generation,
        test_dataset_statistics
    ]
    
    results = []
    for test in tests:
        print(f"\n{len(results)+1}. {test.__doc__.strip()}")
        result = test()
        results.append(result)
    
    print("\n" + "=" * 40)
    print("📊 Dataset Creation Test Summary")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if all(results):
        print("\n🎉 All dataset creation tests passed!")
        print("✅ Ready for data loader and cross-chain compatibility testing!")
        return 0
    else:
        print("\n❌ Some dataset creation tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())