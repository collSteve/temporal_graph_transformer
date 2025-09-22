#!/usr/bin/env python3
"""
Debug script to identify NaN sources in loss functions.
"""

import torch
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.loss_functions import TemporalGraphLoss

def debug_loss_components():
    """Debug each loss component individually."""
    print("🔍 DEBUGGING LOSS FUNCTION NaN ISSUES")
    print("=" * 50)
    
    # Create loss function
    loss_fn = TemporalGraphLoss()
    
    # Create synthetic model outputs (similar to comprehensive test)
    batch_size = 2
    d_model = 64
    seq_len = 10
    
    outputs = {
        'logits': torch.randn(batch_size, 2) * 0.1,  # Small values to prevent explosion
        'probabilities': torch.softmax(torch.randn(batch_size, 2) * 0.1, dim=-1),
        'confidence': torch.sigmoid(torch.randn(batch_size) * 0.1),
        'behavioral_scores': torch.sigmoid(torch.randn(batch_size, 4) * 0.1),
        'fused_representation': torch.randn(batch_size, d_model) * 0.1,
        'intermediate': {
            'temporal_sequences': torch.randn(batch_size, seq_len, d_model) * 0.1
        }
    }
    
    # Create synthetic batch data
    batch_data = {
        'timestamps': torch.randn(batch_size, seq_len) * 1000000 + 1600000000,  # Reasonable timestamps
        'airdrop_events': torch.randn(2) * 1000000 + 1600000000
    }
    
    # Create labels
    labels = torch.randint(0, 2, (batch_size,))
    
    print(f"Batch size: {batch_size}")
    print(f"Labels: {labels}")
    print(f"Outputs shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test each loss component individually
    print("\n🧪 Testing Individual Loss Components:")
    
    # 1. Classification Loss
    try:
        cls_loss = loss_fn.classification_loss(outputs['logits'], labels)
        print(f"✅ Classification Loss: {cls_loss.item():.6f}")
        if torch.isnan(cls_loss):
            print("  ❌ Classification loss is NaN!")
    except Exception as e:
        print(f"❌ Classification Loss Failed: {e}")
    
    # 2. Contrastive Loss
    try:
        contrastive_loss = loss_fn.contrastive_loss(outputs['fused_representation'], labels)
        print(f"✅ Contrastive Loss: {contrastive_loss.item():.6f}")
        if torch.isnan(contrastive_loss):
            print("  ❌ Contrastive loss is NaN!")
    except Exception as e:
        print(f"❌ Contrastive Loss Failed: {e}")
    
    # 3. Temporal Consistency Loss
    try:
        temporal_loss = loss_fn.temporal_consistency_loss(
            outputs['intermediate']['temporal_sequences'],
            batch_data['timestamps'],
            batch_data.get('airdrop_events', None),
            labels
        )
        print(f"✅ Temporal Consistency Loss: {temporal_loss.item():.6f}")
        if torch.isnan(temporal_loss):
            print("  ❌ Temporal consistency loss is NaN!")
    except Exception as e:
        print(f"❌ Temporal Consistency Loss Failed: {e}")
    
    # 4. Behavioral Change Loss
    try:
        change_loss = loss_fn.behavioral_change_loss(
            outputs['behavioral_scores'][:, 0],  # First behavioral score
            labels,
            outputs.get('confidence', None)
        )
        print(f"✅ Behavioral Change Loss: {change_loss.item():.6f}")
        if torch.isnan(change_loss):
            print("  ❌ Behavioral change loss is NaN!")
    except Exception as e:
        print(f"❌ Behavioral Change Loss Failed: {e}")
    
    # 5. Confidence Regularization Loss
    try:
        conf_reg_loss = loss_fn.confidence_regularization_loss(
            outputs['confidence'],
            outputs['probabilities'],
            labels
        )
        print(f"✅ Confidence Regularization Loss: {conf_reg_loss.item():.6f}")
        if torch.isnan(conf_reg_loss):
            print("  ❌ Confidence regularization loss is NaN!")
    except Exception as e:
        print(f"❌ Confidence Regularization Loss Failed: {e}")
    
    # 6. Combined Loss
    print("\n🔗 Testing Combined Loss:")
    try:
        loss_dict = loss_fn(outputs, batch_data, labels)
        print("Loss Components:")
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.6f} {'[NaN]' if torch.isnan(value) else ''}")
        
        total_loss = loss_dict['total']
        if torch.isnan(total_loss):
            print("❌ TOTAL LOSS IS NaN!")
            
            # Find which component is causing NaN
            print("\n🔍 Identifying NaN source:")
            for key, value in loss_dict.items():
                if key != 'total' and isinstance(value, torch.Tensor) and torch.isnan(value):
                    print(f"  🚨 {key} is NaN!")
        else:
            print(f"✅ Total Loss: {total_loss.item():.6f}")
            
    except Exception as e:
        print(f"❌ Combined Loss Failed: {e}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    debug_loss_components()