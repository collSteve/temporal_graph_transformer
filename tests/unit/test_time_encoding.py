#!/usr/bin/env python3
"""
Unit tests for time encoding components.

Tests:
1. FunctionalTimeEncoding - dimension consistency and functionality
2. BehaviorChangeTimeEncoding - airdrop-aware encoding
3. Edge cases and error handling
"""

import torch
import sys
from pathlib import Path

# Add test utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.test_config import print_test_header, print_test_result, print_test_section, get_temporal_config

# Import components to test
from utils.time_encoding import (
    FunctionalTimeEncoding,
    BehaviorChangeTimeEncoding,
    create_time_mask,
    normalize_timestamps
)

def test_functional_time_encoding():
    """Test FunctionalTimeEncoding component."""
    print_test_section("Functional Time Encoding Tests")
    
    config = get_temporal_config()
    d_model = config['d_model']
    
    # Test 1: Basic dimension consistency
    try:
        encoder = FunctionalTimeEncoding(d_model)
        batch_size, seq_len = 3, 10
        timestamps = torch.randn(batch_size, seq_len) * 1000000 + 1600000000
        
        output = encoder(timestamps)
        expected_shape = (batch_size, seq_len, d_model)
        
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains infinite values"
        
        print_test_result("Basic dimension consistency", True, f"Shape: {output.shape}")
        
    except Exception as e:
        print_test_result("Basic dimension consistency", False, str(e))
        return False
    
    # Test 2: Single sequence input
    try:
        timestamps_1d = torch.randn(seq_len) * 1000000 + 1600000000
        output_1d = encoder(timestamps_1d)
        expected_shape_1d = (1, seq_len, d_model)
        
        assert output_1d.shape == expected_shape_1d, f"1D input failed: {output_1d.shape}"
        print_test_result("Single sequence handling", True, f"1D→2D: {output_1d.shape}")
        
    except Exception as e:
        print_test_result("Single sequence handling", False, str(e))
        return False
    
    # Test 3: Different time ranges
    try:
        # Test with different time scales
        recent_times = torch.randn(2, 5) * 3600 + 1700000000  # Recent times
        old_times = torch.randn(2, 5) * 3600 + 1500000000     # Older times
        
        recent_out = encoder(recent_times)
        old_out = encoder(old_times)
        
        assert recent_out.shape == old_out.shape, "Different time ranges produce different shapes"
        assert not torch.equal(recent_out, old_out), "Different times should produce different encodings"
        
        print_test_result("Different time ranges", True, "Produces distinct encodings")
        
    except Exception as e:
        print_test_result("Different time ranges", False, str(e))
        return False
    
    return True

def test_behavior_change_time_encoding():
    """Test BehaviorChangeTimeEncoding component."""
    print_test_section("Behavior Change Time Encoding Tests")
    
    config = get_temporal_config()
    d_model = config['d_model']
    
    # Test 1: Basic functionality with airdrop events
    try:
        encoder = BehaviorChangeTimeEncoding(d_model)
        batch_size, seq_len = 2, 8
        timestamps = torch.randn(batch_size, seq_len) * 1000000 + 1600000000
        airdrop_events = torch.randn(3) * 1000000 + 1600000000
        
        output = encoder(timestamps, airdrop_events)
        expected_shape = (batch_size, seq_len, d_model)
        
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        
        print_test_result("With airdrop events", True, f"Shape: {output.shape}")
        
    except Exception as e:
        print_test_result("With airdrop events", False, str(e))
        return False
    
    # Test 2: Without airdrop events
    try:
        output_no_events = encoder(timestamps, None)
        assert output_no_events.shape == expected_shape, "No events case failed"
        
        output_empty_events = encoder(timestamps, torch.tensor([]))
        assert output_empty_events.shape == expected_shape, "Empty events case failed"
        
        print_test_result("Without airdrop events", True, "Handles None and empty events")
        
    except Exception as e:
        print_test_result("Without airdrop events", False, str(e))
        return False
    
    # Test 3: Airdrop proximity effect
    try:
        # Create timestamps around an airdrop event
        airdrop_time = 1600000000.0
        near_airdrop = torch.tensor([[airdrop_time - 3600, airdrop_time, airdrop_time + 3600]])  # ±1 hour
        far_airdrop = torch.tensor([[airdrop_time - 86400*7, airdrop_time - 86400*6, airdrop_time - 86400*5]])  # 1 week before
        
        near_encoding = encoder(near_airdrop, torch.tensor([airdrop_time]))
        far_encoding = encoder(far_airdrop, torch.tensor([airdrop_time]))
        
        # Encodings should be different for near vs far from airdrop
        assert not torch.allclose(near_encoding, far_encoding, atol=1e-3), "Near and far airdrop encodings too similar"
        
        print_test_result("Airdrop proximity effect", True, "Near vs far encodings differ")
        
    except Exception as e:
        print_test_result("Airdrop proximity effect", False, str(e))
        return False
    
    return True

def test_utility_functions():
    """Test utility functions."""
    print_test_section("Utility Functions Tests")
    
    # Test 1: create_time_mask
    try:
        batch_size, seq_len = 2, 5
        timestamps = torch.randn(batch_size, seq_len) * 3600 + 1600000000
        
        mask = create_time_mask(timestamps, window_size=3600.0)  # 1 hour window
        expected_shape = (batch_size, seq_len, seq_len)
        
        assert mask.shape == expected_shape, f"Time mask shape: {mask.shape}"
        assert mask.dtype == torch.bool, "Time mask should be boolean"
        
        print_test_result("create_time_mask", True, f"Shape: {mask.shape}")
        
    except Exception as e:
        print_test_result("create_time_mask", False, str(e))
        return False
    
    # Test 2: normalize_timestamps
    try:
        timestamps = torch.tensor([1600000000, 1600003600, 1600007200])  # 1 hour intervals
        normalized, stats = normalize_timestamps(timestamps)
        
        assert 0.0 <= normalized.min() <= normalized.max() <= 1.0, "Normalization out of [0,1] range"
        assert 'min_time' in stats and 'max_time' in stats, "Missing normalization stats"
        
        print_test_result("normalize_timestamps", True, f"Range: [{normalized.min():.3f}, {normalized.max():.3f}]")
        
    except Exception as e:
        print_test_result("normalize_timestamps", False, str(e))
        return False
    
    return True

def run_all_tests():
    """Run all time encoding tests."""
    print_test_header("Time Encoding Components")
    
    results = {
        'functional_encoding': test_functional_time_encoding(),
        'behavior_change_encoding': test_behavior_change_time_encoding(),
        'utility_functions': test_utility_functions()
    }
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n📊 TIME ENCODING TEST SUMMARY:")
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   ❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All time encoding tests PASSED!")
        return True
    else:
        print("⚠️  Some time encoding tests FAILED!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)