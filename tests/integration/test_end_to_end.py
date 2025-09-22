#!/usr/bin/env python3
"""
End-to-end integration tests for the complete Temporal Graph Transformer system.

Tests:
1. Complete TGT model pipeline
2. Data loading and preprocessing
3. Training loop simulation
4. Model inference
"""

import torch
import tempfile
import sys
from pathlib import Path

# Add test utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.test_config import print_test_header, print_test_result, print_test_section, get_temporal_config, get_graph_config

# Import components to test
from models.temporal_graph_transformer import TemporalGraphTransformer
from models.temporal_encoder import create_transaction_sequence_config
from models.graph_encoder import create_graph_structure_config
from models.artemis_baseline import ARTEMISBaseline
from data.solana_dataset import SolanaNFTDataset
from utils.loss_functions import TemporalGraphLoss

def test_complete_tgt_model():
    """Test complete Temporal Graph Transformer model."""
    print_test_section("Complete TGT Model Tests")
    
    # Test 1: Model creation and forward pass
    try:
        config = {
            'd_model': 64,
            'temporal_config': create_transaction_sequence_config(d_model=64),
            'graph_config': create_graph_structure_config(d_model=64),
            'output_dim': 64,
            'num_classes': 2,
            'fusion_type': 'cross_attention'
        }
        
        model = TemporalGraphTransformer(config)
        
        # Create realistic test data
        batch_size = 2
        seq_len = 12
        num_nodes = 50
        num_edges = 120
        
        sample = {
            'timestamps': torch.randn(batch_size, seq_len) * 3600 + 1600000000,
            'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.bool),
            'node_features': torch.randn(num_nodes, 64),
            'edge_index': torch.randint(0, num_nodes, (2, num_edges)),
            'edge_features': {'edge_features': torch.randn(num_edges, 64)},
            'airdrop_events': torch.randn(3) * 3600 + 1600000000,
            'user_indices': torch.tensor([0, 1]),
            'transaction_features': {
                'prices': torch.randn(batch_size, seq_len),
                'gas_fees': torch.randn(batch_size, seq_len),
                'sequence_length': torch.tensor([seq_len, seq_len])
            }
        }
        
        model.eval()
        with torch.no_grad():
            outputs = model(sample)
        
        # Verify outputs
        expected_outputs = ['logits', 'probabilities', 'confidence', 'behavioral_scores', 'fused_representation']
        for key in expected_outputs:
            assert key in outputs, f"Missing output: {key}"
        
        assert outputs['logits'].shape == (batch_size, 2), f"Logits shape: {outputs['logits'].shape}"
        assert outputs['probabilities'].shape == (batch_size, 2), f"Probabilities shape: {outputs['probabilities'].shape}"
        assert outputs['confidence'].shape == (batch_size,), f"Confidence shape: {outputs['confidence'].shape}"
        
        print_test_result("Complete model forward", True, f"Logits: {outputs['logits'].shape}")
        
    except Exception as e:
        print_test_result("Complete model forward", False, str(e))
        return False
    
    # Test 2: Different fusion strategies
    try:
        fusion_types = ['cross_attention', 'concatenation', 'addition']
        results = {}
        
        for fusion_type in fusion_types:
            fusion_config = config.copy()
            fusion_config['fusion_type'] = fusion_type
            fusion_model = TemporalGraphTransformer(fusion_config)
            
            fusion_model.eval()
            with torch.no_grad():
                fusion_outputs = fusion_model(sample)
            
            results[fusion_type] = fusion_outputs['logits'].shape
            assert fusion_outputs['logits'].shape == (batch_size, 2), f"Fusion {fusion_type} failed"
        
        print_test_result("Different fusion strategies", True, f"All work: {list(results.keys())}")
        
    except Exception as e:
        print_test_result("Different fusion strategies", False, str(e))
        return False
    
    return True

def test_data_pipeline_integration():
    """Test data loading and preprocessing pipeline."""
    print_test_section("Data Pipeline Integration Tests")
    
    # Test 1: Dataset creation and loading
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = SolanaNFTDataset(
                data_path=temp_dir,
                split='train',
                marketplace='magic_eden',
                max_sequence_length=15,
                airdrop_window_days=7
            )
            
            assert len(dataset) > 0, "Dataset should not be empty"
            
            # Test sample extraction
            sample = dataset[0]
            required_keys = ['transaction_features', 'timestamps', 'node_features', 
                           'edge_index', 'edge_features', 'label', 'user_id']
            
            for key in required_keys:
                assert key in sample, f"Missing sample key: {key}"
            
            print_test_result("Dataset creation", True, f"Size: {len(dataset)}")
            
    except Exception as e:
        print_test_result("Dataset creation", False, str(e))
        return False
    
    # Test 2: Data integrity
    try:
        # Check multiple samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            
            # Check dimensions consistency
            timestamps = sample['timestamps']
            tx_features = sample['transaction_features']
            
            assert timestamps.dim() == 1, "Timestamps should be 1D"
            assert timestamps.shape[0] <= 15, "Sequence too long"
            
            if 'sequence_length' in tx_features:
                seq_len = tx_features['sequence_length'].item()
                assert seq_len <= timestamps.shape[0], "Sequence length inconsistent"
        
        print_test_result("Data integrity", True, "All samples well-formed")
        
    except Exception as e:
        print_test_result("Data integrity", False, str(e))
        return False
    
    return True

def test_training_simulation():
    """Test training loop simulation."""
    print_test_section("Training Simulation Tests")
    
    # Test 1: Forward and backward pass
    try:
        # Small model for fast testing
        config = {
            'd_model': 32,
            'temporal_config': create_transaction_sequence_config(d_model=32, num_layers=1),
            'graph_config': create_graph_structure_config(d_model=32, num_layers=2),
            'output_dim': 32,
            'num_classes': 2,
            'fusion_type': 'cross_attention'
        }
        
        model = TemporalGraphTransformer(config)
        loss_fn = TemporalGraphLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create training batch
        batch_size = 3
        seq_len = 8
        num_nodes = 15
        
        sample = {
            'timestamps': torch.randn(batch_size, seq_len) * 3600 + 1600000000,
            'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.bool),
            'node_features': torch.randn(num_nodes, 32),
            'edge_index': torch.randint(0, num_nodes, (2, 30)),
            'edge_features': {'edge_features': torch.randn(30, 32)},
            'airdrop_events': torch.randn(2) * 3600 + 1600000000,
            'user_indices': torch.tensor([0, 1, 2]),
            'transaction_features': {
                'prices': torch.randn(batch_size, seq_len),
                'gas_fees': torch.randn(batch_size, seq_len),
                'sequence_length': torch.tensor([seq_len] * batch_size)
            }
        }
        
        labels = torch.randint(0, 2, (batch_size,))
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(sample)
        
        # Loss computation
        batch_data = {
            'timestamps': sample['timestamps'],
            'airdrop_events': [sample['airdrop_events']] * batch_size
        }
        
        loss_dict = loss_fn(outputs, batch_data, labels)
        total_loss = loss_dict['total']
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Check gradients
        has_gradients = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for param in model.parameters()
        )
        
        assert has_gradients, "No gradients found"
        assert not torch.isnan(total_loss), "Loss is NaN"
        assert torch.isfinite(total_loss), "Loss is infinite"
        
        print_test_result("Training step", True, f"Loss: {total_loss.item():.4f}")
        
    except Exception as e:
        print_test_result("Training step", False, str(e))
        return False
    
    # Test 2: Multiple training steps
    try:
        initial_loss = total_loss.item()
        losses = [initial_loss]
        
        # Run a few more steps
        for step in range(3):
            optimizer.zero_grad()
            outputs = model(sample)
            loss_dict = loss_fn(outputs, batch_data, labels)
            total_loss = loss_dict['total']
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())
        
        # Loss should be finite in all steps
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses), "Some losses are infinite"
        
        print_test_result("Multiple training steps", True, f"Losses: {[f'{l:.3f}' for l in losses]}")
        
    except Exception as e:
        print_test_result("Multiple training steps", False, str(e))
        return False
    
    return True

def test_model_comparison():
    """Test comparison with ARTEMIS baseline."""
    print_test_section("Model Comparison Tests")
    
    # Test 1: ARTEMIS baseline functionality
    try:
        artemis_config = {
            'input_dim': 64,
            'hidden_channels': 32,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.1,
            'manual_feature_dim': 20,
            'num_classes': 2
        }
        
        artemis_model = ARTEMISBaseline(artemis_config)
        
        # Create ARTEMIS-compatible data
        batch_size = 2
        num_nodes = 20
        
        artemis_batch = {
            'x': torch.randn(num_nodes, 64),
            'edge_index': torch.randint(0, num_nodes, (2, 40)),
            'edge_attr': torch.randn(40, 16),
            'manual_features': torch.randn(batch_size, 20),
            'batch': torch.repeat_interleave(torch.arange(batch_size), num_nodes // batch_size)
        }
        
        artemis_model.eval()
        with torch.no_grad():
            artemis_outputs = artemis_model(artemis_batch)
        
        assert 'logits' in artemis_outputs, "ARTEMIS missing logits"
        assert artemis_outputs['logits'].shape == (batch_size, 2), f"ARTEMIS logits shape: {artemis_outputs['logits'].shape}"
        
        print_test_result("ARTEMIS baseline", True, f"Logits: {artemis_outputs['logits'].shape}")
        
    except Exception as e:
        print_test_result("ARTEMIS baseline", False, str(e))
        return False
    
    # Test 2: Output comparison
    try:
        # Both models should produce valid probability distributions
        tgt_probs = torch.softmax(outputs['logits'], dim=-1)
        artemis_probs = torch.softmax(artemis_outputs['logits'], dim=-1)
        
        # Check probability constraints
        assert torch.allclose(tgt_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5), "TGT probabilities don't sum to 1"
        assert torch.allclose(artemis_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5), "ARTEMIS probabilities don't sum to 1"
        
        print_test_result("Probability distributions", True, "Both models produce valid distributions")
        
    except Exception as e:
        print_test_result("Probability distributions", False, str(e))
        return False
    
    return True

def run_all_tests():
    """Run all integration tests."""
    print_test_header("End-to-End Integration Tests")
    
    results = {
        'complete_tgt_model': test_complete_tgt_model(),
        'data_pipeline_integration': test_data_pipeline_integration(),
        'training_simulation': test_training_simulation(),
        'model_comparison': test_model_comparison()
    }
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n📊 INTEGRATION TEST SUMMARY:")
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   ❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All integration tests PASSED!")
        return True
    else:
        print("⚠️  Some integration tests FAILED!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)