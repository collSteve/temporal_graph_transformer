#!/usr/bin/env python3
"""
Fixed unit tests for time encoding components with proper imports.
"""

import torch
import sys
import os
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now we can import directly without relative imports
print("Testing time encoding with fixed imports...")

try:
    # Test basic imports first
    import utils.time_encoding as time_encoding
    print("✅ Successfully imported utils.time_encoding")
    
    # Test individual components
    encoder = time_encoding.FunctionalTimeEncoding(64)
    print("✅ Created FunctionalTimeEncoding")
    
    # Test basic functionality
    batch_size, seq_len = 2, 10
    timestamps = torch.randn(batch_size, seq_len) * 1000000 + 1600000000
    
    output = encoder(timestamps)
    expected_shape = (batch_size, seq_len, 64)
    
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"
    
    print(f"✅ FunctionalTimeEncoding test passed - Shape: {output.shape}")
    
    # Test BehaviorChangeTimeEncoding
    behavior_encoder = time_encoding.BehaviorChangeTimeEncoding(64)
    airdrop_events = torch.randn(3) * 1000000 + 1600000000
    
    behavior_output = behavior_encoder(timestamps, airdrop_events)
    assert behavior_output.shape == expected_shape, f"Behavior encoding shape: {behavior_output.shape}"
    
    print(f"✅ BehaviorChangeTimeEncoding test passed - Shape: {behavior_output.shape}")
    
    print("\n🎉 All time encoding tests PASSED with fixed imports!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)