#!/usr/bin/env python3
"""
Unit tests for temporal encoder components.

Tests:
1. TransactionFeatureEmbedding - transaction feature processing
2. ChangePointAttention - behavioral change detection
3. TransactionSequenceTransformer - complete temporal modeling
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add test utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.test_config import print_test_header, print_test_result, print_test_section, get_temporal_config

# Import components to test
from models.temporal_encoder import (
    TransactionFeatureEmbedding,
    ChangePointAttention, 
    TransactionSequenceTransformer,
    create_transaction_sequence_config
)

def test_transaction_feature_embedding():
    """Test TransactionFeatureEmbedding component."""
    print_test_section("Transaction Feature Embedding Tests")
    
    config = get_temporal_config()
    config.update({
        'num_tx_types': 5,
        'num_collections': 10,
        'use_nft_features': True,
        'nft_visual_dim': 64,
        'nft_text_dim': 64
    })
    
    # Test 1: Basic embedding functionality
    try:
        embedding = TransactionFeatureEmbedding(config)
        batch_size, seq_len = 2, 8
        
        # Create mock transaction features
        transaction_features = {
            'value': torch.randn(batch_size, seq_len),
            'gas_fee': torch.randn(batch_size, seq_len),
            'volume': torch.randn(batch_size, seq_len),
            'tx_type': torch.randint(0, 5, (batch_size, seq_len)),
            'nft_collection': torch.randint(0, 10, (batch_size, seq_len)),
            'nft_visual': torch.randn(batch_size, seq_len, 64),
            'nft_text': torch.randn(batch_size, seq_len, 64)
        }
        
        output = embedding(transaction_features)
        expected_shape = (batch_size, seq_len, config['d_model'])
        
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        
        print_test_result("Basic embedding", True, f"Shape: {output.shape}")
        
    except Exception as e:
        print_test_result("Basic embedding", False, str(e))
        return False
    
    # Test 2: Minimal features (only required ones)
    try:
        minimal_features = {
            'value': torch.randn(batch_size, seq_len),
            'tx_type': torch.randint(0, 5, (batch_size, seq_len))
        }
        
        output_minimal = embedding(minimal_features)
        assert output_minimal.shape == expected_shape, "Minimal features failed"
        
        print_test_result("Minimal features", True, "Handles sparse feature sets")
        
    except Exception as e:
        print_test_result("Minimal features", False, str(e))
        return False
    
    # Test 3: Empty features fallback
    try:
        empty_features = {}
        output_empty = embedding(empty_features)
        
        # Should return zeros but with correct shape
        assert output_empty.shape == expected_shape, "Empty features shape incorrect"
        
        print_test_result("Empty features fallback", True, "Graceful degradation")
        
    except Exception as e:
        print_test_result("Empty features fallback", False, str(e))
        return False
    
    return True

def test_change_point_attention():
    """Test ChangePointAttention component."""
    print_test_section("Change Point Attention Tests")
    
    config = get_temporal_config()
    d_model = config['d_model']
    num_heads = config['num_heads']
    
    # Test 1: Basic attention functionality
    try:
        attention = ChangePointAttention(d_model, num_heads)
        batch_size, seq_len = 2, 10
        
        x = torch.randn(batch_size, seq_len, d_model)
        timestamps = torch.randn(batch_size, seq_len) * 1000000 + 1600000000
        
        attended_output, change_scores = attention(x, timestamps)
        
        assert attended_output.shape == (batch_size, seq_len, d_model), f"Attended output shape: {attended_output.shape}"
        assert change_scores.shape == (batch_size, seq_len), f"Change scores shape: {change_scores.shape}"
        assert not torch.isnan(attended_output).any(), "Attended output contains NaN"
        assert not torch.isnan(change_scores).any(), "Change scores contain NaN"
        
        print_test_result("Basic attention", True, f"Output: {attended_output.shape}, Scores: {change_scores.shape}")
        
    except Exception as e:
        print_test_result("Basic attention", False, str(e))
        return False
    
    # Test 2: With airdrop events
    try:
        airdrop_events = torch.randn(3) * 1000000 + 1600000000
        attended_output_ad, change_scores_ad = attention(x, timestamps, airdrop_events)
        
        assert attended_output_ad.shape == attended_output.shape, "Airdrop events changed output shape"
        assert change_scores_ad.shape == change_scores.shape, "Airdrop events changed scores shape"
        
        # Scores should be different when airdrop events are provided
        assert not torch.allclose(change_scores, change_scores_ad, atol=1e-4), "Airdrop events had no effect"
        
        print_test_result("With airdrop events", True, "Airdrop events affect change detection")
        
    except Exception as e:
        print_test_result("With airdrop events", False, str(e))
        return False
    
    return True

def test_transaction_sequence_transformer():
    """Test complete TransactionSequenceTransformer."""
    print_test_section("Transaction Sequence Transformer Tests")
    
    # Test 1: Full model functionality
    try:
        config = create_transaction_sequence_config(
            d_model=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1
        )
        
        model = TransactionSequenceTransformer(config)
        batch_size, seq_len = 2, 12
        
        # Create comprehensive transaction features
        transaction_features = {
            'prices': torch.randn(batch_size, seq_len),
            'gas_fees': torch.randn(batch_size, seq_len),
            'timestamps': torch.randn(batch_size, seq_len) * 1000000 + 1600000000,
            'transaction_types': torch.randint(0, 5, (batch_size, seq_len)).float(),
            'sequence_length': torch.tensor([seq_len] * batch_size)
        }
        
        timestamps = torch.randn(batch_size, seq_len) * 1000000 + 1600000000
        airdrop_events = torch.randn(2) * 1000000 + 1600000000
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        model.eval()
        with torch.no_grad():
            outputs = model(transaction_features, timestamps, airdrop_events, attention_mask)
        
        # Check outputs
        assert 'sequence_embedding' in outputs, "Missing sequence_embedding"
        assert 'change_scores' in outputs, "Missing change_scores"
        assert 'temporal_sequence' in outputs, "Missing temporal_sequence"
        
        seq_emb = outputs['sequence_embedding']
        change_scores = outputs['change_scores']
        
        assert seq_emb.shape == (batch_size, 64), f"Sequence embedding shape: {seq_emb.shape}"
        assert change_scores.shape == (batch_size,), f"Change scores shape: {change_scores.shape}"
        
        print_test_result("Full model forward", True, f"Seq emb: {seq_emb.shape}, Scores: {change_scores.shape}")
        
    except Exception as e:
        print_test_result("Full model forward", False, str(e))
        return False
    
    # Test 2: Different pooling strategies
    try:
        pooling_strategies = ['mean', 'max', 'attention', 'last']
        results = {}
        
        for pooling in pooling_strategies:
            pool_config = config.copy()
            pool_config['pooling'] = pooling
            pool_model = TransactionSequenceTransformer(pool_config)
            
            pool_model.eval()
            with torch.no_grad():
                pool_outputs = pool_model(transaction_features, timestamps, airdrop_events, attention_mask)
            
            results[pooling] = pool_outputs['sequence_embedding'].shape
            assert pool_outputs['sequence_embedding'].shape == (batch_size, 64), f"Pooling {pooling} failed"
        
        print_test_result("Different pooling strategies", True, f"All strategies work: {list(results.keys())}")
        
    except Exception as e:
        print_test_result("Different pooling strategies", False, str(e))
        return False
    
    # Test 3: Gradient flow
    try:
        model.train()
        outputs = model(transaction_features, timestamps, airdrop_events, attention_mask)
        
        # Simple loss for gradient test
        loss = outputs['sequence_embedding'].sum() + outputs['change_scores'].sum()
        loss.backward()
        
        # Check gradients exist
        has_gradients = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for param in model.parameters()
        )
        
        assert has_gradients, "No gradients found"
        print_test_result("Gradient flow", True, "Gradients computed successfully")
        
    except Exception as e:
        print_test_result("Gradient flow", False, str(e))
        return False
    
    return True

def run_all_tests():
    """Run all temporal encoder tests."""
    print_test_header("Temporal Encoder Components")
    
    results = {
        'transaction_feature_embedding': test_transaction_feature_embedding(),
        'change_point_attention': test_change_point_attention(),
        'transaction_sequence_transformer': test_transaction_sequence_transformer()
    }
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n📊 TEMPORAL ENCODER TEST SUMMARY:")
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   ❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All temporal encoder tests PASSED!")
        return True
    else:
        print("⚠️  Some temporal encoder tests FAILED!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)