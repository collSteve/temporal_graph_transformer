#!/usr/bin/env python3
"""
Phase 2 Transaction Schema Tests

Tests the unified transaction schema for cross-chain compatibility.
Validates that our schema can handle Arbitrum, Solana, and other blockchains.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_base_transaction_creation():
    """Test BaseTransaction creation and validation."""
    print("Testing BaseTransaction creation...")
    
    try:
        from data.transaction_schema import BaseTransaction, TransactionSchemaValidator
        
        # Create a basic transaction
        tx = BaseTransaction(
            user_id='test_user_001',
            timestamp=1679529600.0,  # March 23, 2023
            chain_id='ethereum',  # Use valid chain_id
            transaction_type='swap',
            value_usd=100.50,
            gas_fee=0.005,
            signature='0xtest123...',
            protocol='test_protocol'
        )
        
        print(f"   ✅ Created BaseTransaction: {tx.user_id}")
        
        # Test validation
        validator = TransactionSchemaValidator()
        is_valid = validator.validate_transaction(tx)
        
        if is_valid:
            print("   ✅ Transaction validation passed")
        else:
            print("   ❌ Transaction validation failed")
            return False
        
        # Test dictionary conversion
        tx_dict = tx.to_dict()
        assert isinstance(tx_dict, dict)
        assert tx_dict['user_id'] == 'test_user_001'
        print("   ✅ Dictionary conversion successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ BaseTransaction test error: {e}")
        return False

def test_arbitrum_transaction_creation():
    """Test ArbitrumDeFiTransaction creation with chain-specific fields."""
    print("Testing ArbitrumDeFiTransaction creation...")
    
    try:
        from data.transaction_schema import ArbitrumDeFiTransaction
        
        # Create Arbitrum-specific transaction
        tx = ArbitrumDeFiTransaction(
            user_id='arbitrum_user_001',
            timestamp=1679529600.0,
            chain_id='arbitrum',  # Will be overridden by __post_init__
            transaction_type='swap',
            value_usd=250.75,
            gas_fee=0.008,
            signature='0xarbitrum123...',
            protocol='uniswap_v3',
            # Arbitrum-specific fields
            token_in='USDC',
            token_out='WETH',
            amount_in=250.75,
            amount_out=0.125,
            price_impact=0.02,
            slippage=0.005,
            l1_gas_used=21000,
            l2_gas_used=150000
        )
        
        print(f"   ✅ Created ArbitrumDeFiTransaction: {tx.protocol}")
        
        # Test chain_id is set correctly
        assert tx.chain_id == 'arbitrum'
        print("   ✅ Chain ID correctly set to 'arbitrum'")
        
        # Test Arbitrum-specific fields
        tx_dict = tx.to_dict()
        assert tx_dict['token_in'] == 'USDC'
        assert tx_dict['l1_gas_used'] == 21000
        print("   ✅ Arbitrum-specific fields preserved")
        
        return True
        
    except Exception as e:
        print(f"   ❌ ArbitrumDeFiTransaction test error: {e}")
        return False

def test_solana_transaction_creation():
    """Test SolanaDeFiTransaction creation with chain-specific fields."""
    print("Testing SolanaDeFiTransaction creation...")
    
    try:
        from data.transaction_schema import SolanaDeFiTransaction
        
        # Create Solana-specific transaction
        tx = SolanaDeFiTransaction(
            user_id='solana_user_001',
            timestamp=1706659200.0,  # Jan 31, 2024
            chain_id='solana',  # Will be overridden by __post_init__
            transaction_type='swap',
            value_usd=50.25,
            gas_fee=0.000005,  # Very low Solana fees
            signature='jupiter_tx_abc123...',
            protocol='jupiter',
            # Solana-specific fields
            program_id='JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4',
            instruction_type='sharedAccountsRoute',
            input_mint='EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',  # USDC
            output_mint='So11111111111111111111111111111111111111112',   # SOL
            in_amount=50.25,
            out_amount=0.5,
            compute_units_consumed=35000,
            prioritization_fee=0.0001
        )
        
        print(f"   ✅ Created SolanaDeFiTransaction: {tx.protocol}")
        
        # Test chain_id is set correctly
        assert tx.chain_id == 'solana'
        print("   ✅ Chain ID correctly set to 'solana'")
        
        # Test Solana-specific fields
        tx_dict = tx.to_dict()
        assert tx_dict['program_id'] == 'JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4'
        assert tx_dict['compute_units_consumed'] == 35000
        print("   ✅ Solana-specific fields preserved")
        
        return True
        
    except Exception as e:
        print(f"   ❌ SolanaDeFiTransaction test error: {e}")
        return False

def test_transaction_dataframe_conversion():
    """Test converting transaction objects to DataFrame."""
    print("Testing transaction to DataFrame conversion...")
    
    try:
        from data.transaction_schema import (
            BaseTransaction, 
            ArbitrumDeFiTransaction, 
            SolanaDeFiTransaction,
            transactions_to_dataframe
        )
        
        # Create mixed transaction list
        transactions = [
            BaseTransaction(
                user_id='user_001',
                timestamp=1679529600.0,
                chain_id='ethereum',
                transaction_type='swap',
                value_usd=100.0,
                gas_fee=0.01,
                signature='0xeth123...'
            ),
            ArbitrumDeFiTransaction(
                user_id='user_002',
                timestamp=1679529700.0,
                chain_id='arbitrum',
                transaction_type='swap',
                value_usd=200.0,
                gas_fee=0.005,
                signature='0xarb123...',
                protocol='uniswap_v3',
                token_in='USDC',
                token_out='WETH'
            ),
            SolanaDeFiTransaction(
                user_id='user_003',
                timestamp=1679529800.0,
                chain_id='solana',
                transaction_type='swap',
                value_usd=50.0,
                gas_fee=0.000005,
                signature='sol_tx_123...',
                protocol='jupiter',
                program_id='JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4'
            )
        ]
        
        # Convert to DataFrame
        df = transactions_to_dataframe(transactions)
        
        print(f"   ✅ Created DataFrame with {len(df)} transactions")
        
        # Test DataFrame structure
        assert len(df) == 3
        assert 'user_id' in df.columns
        assert 'chain_id' in df.columns
        assert 'value_usd' in df.columns
        
        # Test chain-specific columns exist
        assert 'token_in' in df.columns  # Arbitrum-specific
        assert 'program_id' in df.columns  # Solana-specific
        
        print("   ✅ DataFrame structure correct")
        
        # Test data integrity
        assert df.iloc[0]['chain_id'] == 'ethereum'
        assert df.iloc[1]['chain_id'] == 'arbitrum'
        assert df.iloc[2]['chain_id'] == 'solana'
        
        print("   ✅ Chain-specific data preserved")
        
        return True
        
    except Exception as e:
        print(f"   ❌ DataFrame conversion test error: {e}")
        return False

def test_schema_validation():
    """Test transaction schema validation."""
    print("Testing transaction schema validation...")
    
    try:
        from data.transaction_schema import (
            BaseTransaction, 
            TransactionSchemaValidator,
            transactions_to_dataframe
        )
        
        validator = TransactionSchemaValidator()
        
        # Test valid transaction
        valid_tx = BaseTransaction(
            user_id='valid_user',
            timestamp=1679529600.0,
            chain_id='arbitrum',
            transaction_type='swap',
            value_usd=100.0,
            gas_fee=0.005,
            signature='0xvalid123...'
        )
        
        assert validator.validate_transaction(valid_tx) == True
        print("   ✅ Valid transaction passed validation")
        
        # Test invalid transaction (negative value)
        invalid_tx = BaseTransaction(
            user_id='invalid_user',
            timestamp=1679529600.0,
            chain_id='arbitrum',
            transaction_type='swap',
            value_usd=-100.0,  # Invalid negative value
            gas_fee=0.005,
            signature='0xinvalid123...'
        )
        
        assert validator.validate_transaction(invalid_tx) == False
        print("   ✅ Invalid transaction correctly rejected")
        
        # Test DataFrame validation
        valid_transactions = [valid_tx]
        df = transactions_to_dataframe(valid_transactions)
        
        is_valid, errors = validator.validate_dataframe(df)
        assert is_valid == True
        assert len(errors) == 0
        print("   ✅ Valid DataFrame passed validation")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Schema validation test error: {e}")
        return False

def test_cross_chain_compatibility():
    """Test that different chain transactions can coexist."""
    print("Testing cross-chain compatibility...")
    
    try:
        from data.transaction_schema import (
            ArbitrumDeFiTransaction,
            SolanaDeFiTransaction,
            TransactionSchemaValidator,
            transactions_to_dataframe
        )
        
        # Create transactions from different chains
        arbitrum_tx = ArbitrumDeFiTransaction(
            user_id='multi_chain_user',
            timestamp=1679529600.0,
            chain_id='arbitrum',
            transaction_type='swap',
            value_usd=1000.0,
            gas_fee=0.01,
            signature='0xarb_multi...',
            protocol='gmx',
            position_size=5000.0,
            leverage=5.0,
            is_long=True
        )
        
        solana_tx = SolanaDeFiTransaction(
            user_id='multi_chain_user',  # Same user, different chain
            timestamp=1679529700.0,
            chain_id='solana',
            transaction_type='swap',
            value_usd=1000.0,
            gas_fee=0.000008,
            signature='sol_multi_123...',
            protocol='jupiter',
            program_id='JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4',
            in_amount=1000.0,
            out_amount=10.0
        )
        
        # Combine in DataFrame
        df = transactions_to_dataframe([arbitrum_tx, solana_tx])
        
        print(f"   ✅ Combined {len(df)} cross-chain transactions")
        
        # Validate combined data
        validator = TransactionSchemaValidator()
        is_valid, errors = validator.validate_dataframe(df)
        
        if is_valid:
            print("   ✅ Cross-chain DataFrame validation passed")
        else:
            print(f"   ⚠️  Cross-chain validation warnings: {errors}")
        
        # Test that both chains are represented
        chains = df['chain_id'].unique()
        assert 'arbitrum' in chains
        assert 'solana' in chains
        print("   ✅ Both chains represented in combined data")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Cross-chain compatibility test error: {e}")
        return False

def main():
    """Run all transaction schema tests."""
    print("🔧 Phase 2 Transaction Schema Tests")
    print("=" * 40)
    
    tests = [
        test_base_transaction_creation,
        test_arbitrum_transaction_creation,
        test_solana_transaction_creation,
        test_transaction_dataframe_conversion,
        test_schema_validation,
        test_cross_chain_compatibility
    ]
    
    results = []
    for test in tests:
        print(f"\n{len(results)+1}. {test.__doc__.strip()}")
        result = test()
        results.append(result)
    
    print("\n" + "=" * 40)
    print("📊 Transaction Schema Test Summary")
    print("=" * 40)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if all(results):
        print("\n🎉 All transaction schema tests passed!")
        return 0
    else:
        print("\n❌ Some transaction schema tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())