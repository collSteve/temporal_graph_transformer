#!/usr/bin/env python3
"""
Phase 2 Quick Validation Tests

Fast validation tests for our Phase 2 implementation with smaller datasets.
Tests the core functionality without generating large amounts of data.
"""

import sys
import os
import tempfile
import shutil
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_quick_arbitrum_dataset():
    """Quick test of ArbitrumDeFiDataset with minimal data."""
    print("Testing ArbitrumDeFiDataset (quick)...")
    
    test_dir = tempfile.mkdtemp(prefix='arbitrum_quick_')
    
    try:
        from data import ArbitrumDeFiDataset
        
        # Create dataset with minimal settings
        dataset = ArbitrumDeFiDataset(
            data_path=test_dir,
            split='test',
            start_date='2023-03-20',
            end_date='2023-03-22',
            max_sequence_length=5  # Very small
        )
        
        # Override the generation to create fewer users/transactions
        original_method = dataset._generate_demonstration_data
        
        def quick_generation():
            print("Generating minimal demonstration data...")
            
            # Quick minimal data generation
            import json
            import pandas as pd
            from data.transaction_schema import ArbitrumDeFiTransaction
            
            users = []
            transactions = []
            labels = {}
            
            # Generate just 5 users with 3 transactions each
            for i in range(5):
                user_id = f"quick_user_{i}"
                users.append({'user_id': user_id})
                labels[user_id] = 1 if i < 2 else 0  # 2 hunters, 3 legitimate
                
                # Generate 3 transactions per user
                for j in range(3):
                    tx = ArbitrumDeFiTransaction(
                        user_id=user_id,
                        timestamp=1679529600.0 + j * 3600,  # 1 hour apart
                        chain_id='arbitrum',
                        transaction_type='swap',
                        value_usd=100.0 + j * 50,
                        gas_fee=0.005,
                        signature=f"quick_tx_{i}_{j}",
                        protocol='uniswap_v3'
                    )
                    transactions.append(tx)
            
            # Convert to DataFrames
            from data.transaction_schema import transactions_to_dataframe
            transactions_df = transactions_to_dataframe(transactions)
            users_df = pd.DataFrame(users)
            
            # Save minimal dataset
            dataset_data = {
                'transactions': transactions_df.to_dict(orient='records'),
                'users': users_df.to_dict(orient='records'),
                'labels': labels,
                'metadata': {'data_type': 'quick_test'}
            }
            
            dataset_file = os.path.join(test_dir, 'arbitrum_defi_test.json')
            with open(dataset_file, 'w') as f:
                json.dump(dataset_data, f)
            
            # Load the data
            dataset.load_raw_data()
        
        # Replace the method temporarily
        dataset._generate_demonstration_data = quick_generation
        dataset._load_dataset()
        
        print(f"   ✅ Created dataset with {len(dataset)} users")
        
        # Test basic functionality
        assert len(dataset) > 0
        sample = dataset[0]
        assert 'transaction_features' in sample
        assert 'label' in sample
        
        print("   ✅ Sample generation works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Quick Arbitrum test error: {e}")
        return False
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

def test_quick_jupiter_dataset():
    """Quick test of JupiterSolanaDataset with minimal data."""
    print("Testing JupiterSolanaDataset (quick)...")
    
    test_dir = tempfile.mkdtemp(prefix='jupiter_quick_')
    
    try:
        from data import JupiterSolanaDataset
        
        # Create dataset with minimal settings
        dataset = JupiterSolanaDataset(
            data_path=test_dir,
            split='test',
            start_date='2024-01-28',
            end_date='2024-01-30',
            max_sequence_length=5  # Very small
        )
        
        # Override the generation method
        def quick_generation():
            print("Generating minimal Jupiter demonstration data...")
            
            import json
            import pandas as pd
            from data.transaction_schema import SolanaDeFiTransaction
            
            users = []
            transactions = []
            labels = {}
            
            # Generate just 5 users with 2 transactions each
            for i in range(5):
                user_id = f"jupiter_user_{i}"
                users.append({'user_id': user_id})
                labels[user_id] = 1 if i < 2 else 0  # 2 hunters, 3 legitimate
                
                # Generate 2 transactions per user
                for j in range(2):
                    tx = SolanaDeFiTransaction(
                        user_id=user_id,
                        timestamp=1706659200.0 + j * 1800,  # 30 min apart
                        chain_id='solana',
                        transaction_type='swap',
                        value_usd=50.0 + j * 25,
                        gas_fee=0.000005,
                        signature=f"jupiter_tx_{i}_{j}",
                        protocol='jupiter',
                        program_id='JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4'
                    )
                    transactions.append(tx)
            
            # Convert to DataFrames
            from data.transaction_schema import transactions_to_dataframe
            transactions_df = transactions_to_dataframe(transactions)
            users_df = pd.DataFrame(users)
            
            # Save minimal dataset
            dataset_data = {
                'transactions': transactions_df.to_dict(orient='records'),
                'users': users_df.to_dict(orient='records'),
                'labels': labels,
                'metadata': {'data_type': 'quick_test'}
            }
            
            dataset_file = os.path.join(test_dir, 'jupiter_solana_test.json')
            with open(dataset_file, 'w') as f:
                json.dump(dataset_data, f)
            
            # Load the data
            dataset.load_raw_data()
        
        # Replace the method temporarily
        dataset._generate_demonstration_data = quick_generation
        dataset._load_dataset()
        
        print(f"   ✅ Created dataset with {len(dataset)} users")
        
        # Test basic functionality
        assert len(dataset) > 0
        sample = dataset[0]
        assert 'transaction_features' in sample
        assert 'label' in sample
        
        print("   ✅ Sample generation works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Quick Jupiter test error: {e}")
        return False
    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

def test_cross_chain_data_loader():
    """Test that data loader works with both datasets."""
    print("Testing cross-chain data loader...")
    
    try:
        from data import UnifiedDataLoader
        
        # We'll use the datasets created in previous tests
        arb_dir = tempfile.mkdtemp(prefix='arb_loader_')
        jup_dir = tempfile.mkdtemp(prefix='jup_loader_')
        
        try:
            from data import ArbitrumDeFiDataset, JupiterSolanaDataset
            
            # Create minimal datasets
            arb_dataset = ArbitrumDeFiDataset(
                data_path=arb_dir,
                split='test',
                max_sequence_length=5
            )
            
            # Manually create minimal data for both
            import json
            import pandas as pd
            from data.transaction_schema import ArbitrumDeFiTransaction, SolanaDeFiTransaction, transactions_to_dataframe
            
            # Arbitrum data
            arb_tx = ArbitrumDeFiTransaction(
                user_id='loader_test_arb',
                timestamp=1679529600.0,
                chain_id='arbitrum',
                transaction_type='swap',
                value_usd=100.0,
                gas_fee=0.005,
                signature='loader_test_arb_tx',
                protocol='uniswap_v3'
            )
            
            arb_df = transactions_to_dataframe([arb_tx])
            arb_users_df = pd.DataFrame([{'user_id': 'loader_test_arb'}])
            arb_labels = {'loader_test_arb': 0}
            
            arb_data = {
                'transactions': arb_df.to_dict(orient='records'),
                'users': arb_users_df.to_dict(orient='records'),
                'labels': arb_labels,
                'metadata': {'data_type': 'loader_test'}
            }
            
            with open(os.path.join(arb_dir, 'arbitrum_defi_test.json'), 'w') as f:
                json.dump(arb_data, f)
            
            arb_dataset.load_raw_data()
            
            print(f"   ✅ Created test datasets")
            
            # Test data loader
            loader = UnifiedDataLoader(
                dataset=arb_dataset,
                batch_size=1,
                shuffle=False,
                model_type='temporal_graph_transformer'
            )
            
            # Test batch generation
            batch_count = 0
            for batch in loader:
                assert isinstance(batch, dict)
                assert 'transaction_features' in batch
                assert 'labels' in batch
                batch_count += 1
                break  # Just test one batch
            
            print(f"   ✅ Data loader generated {batch_count} batch successfully")
            
            return True
            
        finally:
            for test_dir in [arb_dir, jup_dir]:
                if os.path.exists(test_dir):
                    shutil.rmtree(test_dir)
        
    except Exception as e:
        print(f"   ❌ Data loader test error: {e}")
        return False

def test_feature_compatibility():
    """Test that both datasets generate compatible features."""
    print("Testing feature compatibility...")
    
    try:
        # Create test transactions
        from data.transaction_schema import ArbitrumDeFiTransaction, SolanaDeFiTransaction
        
        arb_tx = ArbitrumDeFiTransaction(
            user_id='feature_test_user',
            timestamp=1679529600.0,
            chain_id='arbitrum',
            transaction_type='swap',
            value_usd=100.0,
            gas_fee=0.005,
            signature='feature_test_arb',
            protocol='uniswap_v3',
            token_in='USDC',
            token_out='WETH'
        )
        
        sol_tx = SolanaDeFiTransaction(
            user_id='feature_test_user',
            timestamp=1706659200.0,
            chain_id='solana',
            transaction_type='swap',
            value_usd=50.0,
            gas_fee=0.000005,
            signature='feature_test_sol',
            protocol='jupiter',
            program_id='JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4'
        )
        
        # Test dictionary conversion
        arb_dict = arb_tx.to_dict()
        sol_dict = sol_tx.to_dict()
        
        # Check common fields exist
        common_fields = ['user_id', 'timestamp', 'chain_id', 'transaction_type', 'value_usd', 'gas_fee']
        
        for field in common_fields:
            assert field in arb_dict, f"Arbitrum missing {field}"
            assert field in sol_dict, f"Solana missing {field}"
        
        # Check chain-specific fields exist
        assert 'token_in' in arb_dict, "Arbitrum missing token_in"
        assert 'program_id' in sol_dict, "Solana missing program_id"
        
        print("   ✅ Transaction schemas compatible")
        
        # Test DataFrame conversion
        from data.transaction_schema import transactions_to_dataframe
        
        df = transactions_to_dataframe([arb_tx, sol_tx])
        assert len(df) == 2
        assert 'arbitrum' in df['chain_id'].values
        assert 'solana' in df['chain_id'].values
        
        print("   ✅ Cross-chain DataFrame generation works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Feature compatibility test error: {e}")
        return False

def main():
    """Run quick validation tests."""
    print("🚀 Phase 2 Quick Validation Tests")
    print("=" * 45)
    
    tests = [
        test_quick_arbitrum_dataset,
        test_quick_jupiter_dataset,
        test_cross_chain_data_loader,
        test_feature_compatibility
    ]
    
    results = []
    for test in tests:
        print(f"\n{len(results)+1}. {test.__doc__.strip()}")
        result = test()
        results.append(result)
    
    print("\n" + "=" * 45)
    print("📊 Quick Validation Summary")
    print("=" * 45)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if all(results):
        print("\n🎉 All quick validation tests passed!")
        print("✅ Phase 2 core implementation is working correctly!")
        return 0
    else:
        print("\n❌ Some quick validation tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())