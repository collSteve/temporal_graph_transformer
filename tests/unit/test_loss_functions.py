#!/usr/bin/env python3
"""
Unit tests for loss function components.

Tests:
1. Individual loss components (InfoNCE, Focal, etc.)
2. TemporalGraphLoss - combined loss function
3. Edge cases and numerical stability
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add test utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.test_config import print_test_header, print_test_result, print_test_section, get_temporal_config

# Import components to test
from utils.loss_functions import (
    InfoNCE,
    FocalLoss,
    TemporalConsistencyLoss,
    BehavioralChangeLoss,
    ConfidenceRegularizationLoss,
    TemporalGraphLoss
)

def test_infonce_loss():
    """Test InfoNCE contrastive loss."""
    print_test_section("InfoNCE Loss Tests")
    
    # Test 1: Basic functionality
    try:
        loss_fn = InfoNCE(temperature=0.1)
        batch_size = 4
        d_model = 64
        
        features = torch.randn(batch_size, d_model)
        labels = torch.tensor([0, 0, 1, 1])  # Two classes
        
        loss = loss_fn(features, labels)
        
        assert isinstance(loss, torch.Tensor), "Loss should be tensor"
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
        
        print_test_result("Basic InfoNCE", True, f"Loss: {loss.item():.4f}")
        
    except Exception as e:
        print_test_result("Basic InfoNCE", False, str(e))
        return False
    
    # Test 2: Single class (should handle gracefully)
    try:
        single_labels = torch.tensor([0, 0, 0, 0])
        loss_single = loss_fn(features, single_labels)
        
        assert not torch.isnan(loss_single), "Single class should not produce NaN"
        print_test_result("Single class handling", True, f"Loss: {loss_single.item():.4f}")
        
    except Exception as e:
        print_test_result("Single class handling", False, str(e))
        return False
    
    return True

def test_focal_loss():
    """Test Focal Loss for class imbalance."""
    print_test_section("Focal Loss Tests")
    
    # Test 1: Basic functionality
    try:
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        batch_size = 6
        num_classes = 2
        
        logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))
        
        loss = loss_fn(logits, labels)
        
        assert isinstance(loss, torch.Tensor), "Loss should be tensor"
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
        
        print_test_result("Basic Focal Loss", True, f"Loss: {loss.item():.4f}")
        
    except Exception as e:
        print_test_result("Basic Focal Loss", False, str(e))
        return False
    
    # Test 2: Compare with standard cross-entropy
    try:
        ce_loss = F.cross_entropy(logits, labels)
        focal_loss = loss_fn(logits, labels)
        
        # Focal loss should generally be different from CE
        assert not torch.allclose(ce_loss, focal_loss, atol=1e-4), "Focal loss too similar to CE"
        
        print_test_result("Focal vs CrossEntropy", True, f"CE: {ce_loss.item():.4f}, Focal: {focal_loss.item():.4f}")
        
    except Exception as e:
        print_test_result("Focal vs CrossEntropy", False, str(e))
        return False
    
    return True

def test_temporal_consistency_loss():
    """Test Temporal Consistency Loss."""
    print_test_section("Temporal Consistency Loss Tests")
    
    # Test 1: Basic functionality
    try:
        loss_fn = TemporalConsistencyLoss()
        batch_size = 3
        seq_len = 8
        d_model = 32
        
        embeddings = torch.randn(batch_size, seq_len, d_model)
        timestamps = torch.randn(batch_size, seq_len) * 3600 + 1600000000  # Reasonable timestamps
        
        loss = loss_fn(embeddings, timestamps)
        
        assert isinstance(loss, torch.Tensor), "Loss should be tensor"
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert torch.isfinite(loss), "Loss should be finite"
        
        print_test_result("Basic Temporal Consistency", True, f"Loss: {loss.item():.4f}")
        
    except Exception as e:
        print_test_result("Basic Temporal Consistency", False, str(e))
        return False
    
    # Test 2: Extreme time differences (numerical stability)
    try:
        # Very large time differences
        extreme_timestamps = torch.tensor([[0.0, 1e9, 2e9], [1e6, 1e7, 1e8]])
        extreme_embeddings = torch.randn(2, 3, d_model)
        
        extreme_loss = loss_fn(extreme_embeddings, extreme_timestamps)
        assert torch.isfinite(extreme_loss), "Extreme timestamps should not cause infinite loss"
        
        print_test_result("Extreme timestamps", True, f"Loss: {extreme_loss.item():.4f}")
        
    except Exception as e:
        print_test_result("Extreme timestamps", False, str(e))
        return False
    
    return True

def test_behavioral_change_loss():
    """Test Behavioral Change Loss."""
    print_test_section("Behavioral Change Loss Tests")
    
    # Test 1: Basic functionality  
    try:
        loss_fn = BehavioralChangeLoss(margin=1.0)
        batch_size = 6
        
        change_scores = torch.randn(batch_size)
        labels = torch.tensor([0, 0, 0, 1, 1, 1])  # Balanced
        
        loss = loss_fn(change_scores, labels)
        
        assert isinstance(loss, torch.Tensor), "Loss should be tensor"
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss >= 0, "Loss should be non-negative"
        assert not torch.isnan(loss), "Loss should not be NaN"
        
        print_test_result("Basic Behavioral Change", True, f"Loss: {loss.item():.4f}")
        
    except Exception as e:
        print_test_result("Basic Behavioral Change", False, str(e))
        return False
    
    # Test 2: Imbalanced classes
    try:
        imbalanced_labels = torch.tensor([0, 0, 0, 0, 0, 1])  # Very imbalanced
        imbalanced_scores = torch.randn(6)
        
        imbalanced_loss = loss_fn(imbalanced_scores, imbalanced_labels)
        assert not torch.isnan(imbalanced_loss), "Imbalanced classes should not cause NaN"
        
        print_test_result("Imbalanced classes", True, f"Loss: {imbalanced_loss.item():.4f}")
        
    except Exception as e:
        print_test_result("Imbalanced classes", False, str(e))
        return False
    
    # Test 3: Single class fallback
    try:
        single_class_labels = torch.tensor([0, 0, 0, 0])
        single_class_scores = torch.randn(4)
        
        single_loss = loss_fn(single_class_scores, single_class_labels)
        assert not torch.isnan(single_loss), "Single class should not cause NaN"
        
        print_test_result("Single class fallback", True, f"Loss: {single_loss.item():.4f}")
        
    except Exception as e:
        print_test_result("Single class fallback", False, str(e))
        return False
    
    return True

def test_temporal_graph_loss():
    """Test combined TemporalGraphLoss."""
    print_test_section("Combined Temporal Graph Loss Tests")
    
    # Test 1: Full loss computation
    try:
        loss_fn = TemporalGraphLoss()
        batch_size = 4
        
        # Create mock model outputs
        outputs = {
            'logits': torch.randn(batch_size, 2),
            'probabilities': torch.softmax(torch.randn(batch_size, 2), dim=-1),
            'confidence': torch.sigmoid(torch.randn(batch_size)),
            'behavioral_scores': torch.randn(batch_size),
            'fused_representation': torch.randn(batch_size, 64)
        }
        
        # Create mock batch data
        batch = {
            'timestamps': torch.randn(batch_size, 10) * 3600 + 1600000000,
            'airdrop_events': [torch.randn(2) * 3600 + 1600000000 for _ in range(batch_size)]
        }
        
        labels = torch.randint(0, 2, (batch_size,))
        
        loss_dict = loss_fn(outputs, batch, labels)
        
        # Check all loss components exist
        expected_components = ['classification', 'contrastive', 'temporal_consistency', 
                             'behavioral_change', 'confidence_regularization', 'total']
        
        for component in expected_components:
            assert component in loss_dict, f"Missing loss component: {component}"
            assert isinstance(loss_dict[component], torch.Tensor), f"{component} should be tensor"
            assert not torch.isnan(loss_dict[component]), f"{component} should not be NaN"
        
        total_loss = loss_dict['total']
        assert total_loss >= 0, "Total loss should be non-negative"
        
        print_test_result("Full loss computation", True, f"Total: {total_loss.item():.4f}")
        
    except Exception as e:
        print_test_result("Full loss computation", False, str(e))
        return False
    
    # Test 2: Loss component weights
    try:
        # Test with different weights
        custom_weights = {
            'classification': 2.0,
            'contrastive': 0.5,
            'temporal_consistency': 0.1,
            'behavioral_change': 1.5,
            'confidence_regularization': 0.01
        }
        
        custom_loss_fn = TemporalGraphLoss(**custom_weights)
        custom_loss_dict = custom_loss_fn(outputs, batch, labels)
        
        # Should produce different total loss
        default_total = loss_dict['total']
        custom_total = custom_loss_dict['total']
        
        assert not torch.allclose(default_total, custom_total, atol=1e-4), "Custom weights had no effect"
        
        print_test_result("Custom loss weights", True, f"Default: {default_total.item():.4f}, Custom: {custom_total.item():.4f}")
        
    except Exception as e:
        print_test_result("Custom loss weights", False, str(e))
        return False
    
    # Test 3: Gradient flow
    try:
        # Test that loss can be used for training
        total_loss = loss_dict['total']
        total_loss.backward()
        
        # Check that outputs have gradients
        for key, value in outputs.items():
            if value.requires_grad:
                assert value.grad is not None, f"No gradient for {key}"
        
        print_test_result("Gradient flow", True, "All outputs have gradients")
        
    except Exception as e:
        print_test_result("Gradient flow", False, str(e))
        return False
    
    return True

def run_all_tests():
    """Run all loss function tests."""
    print_test_header("Loss Function Components")
    
    results = {
        'infonce_loss': test_infonce_loss(),
        'focal_loss': test_focal_loss(),
        'temporal_consistency_loss': test_temporal_consistency_loss(),
        'behavioral_change_loss': test_behavioral_change_loss(),
        'temporal_graph_loss': test_temporal_graph_loss()
    }
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n📊 LOSS FUNCTIONS TEST SUMMARY:")
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   ❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All loss function tests PASSED!")
        return True
    else:
        print("⚠️  Some loss function tests FAILED!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)