"""Integration tests for the complete pipeline."""

import pytest
import torch
import tempfile
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from temporal_graph_transformer.models.temporal_graph_transformer import TemporalGraphTransformer
from temporal_graph_transformer.models.temporal_encoder import create_transaction_sequence_config
from temporal_graph_transformer.models.graph_encoder import create_graph_structure_config
from temporal_graph_transformer.data.solana_dataset import SolanaNFTDataset
from temporal_graph_transformer.utils.loss_functions import TemporalGraphLoss


class TestEndToEndIntegration:
    """Test the complete end-to-end pipeline."""
    
    def test_complete_pipeline(self):
        """Test the complete pipeline from data loading to model output."""
        # Create test configuration
        config = {
            'd_model': 64,
            'temporal_config': create_transaction_sequence_config(d_model=64),
            'graph_config': create_graph_structure_config(d_model=64),
            'output_dim': 64,
            'num_classes': 2,
            'fusion_type': 'cross_attention'
        }
        
        # Create model
        model = TemporalGraphTransformer(config)
        
        # Create minimal dataset
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = SolanaNFTDataset(
                data_path=temp_dir,
                split='train',
                marketplace='magic_eden',
                max_sequence_length=10,
                airdrop_window_days=7
            )
            
            if len(dataset) == 0:
                pytest.skip("Empty dataset, skipping test")
            
            # Get sample and fix node features
            sample = dataset[0]
            num_nodes = sample['edge_index'].max().item() + 1
            all_node_features = torch.randn(num_nodes, 64)
            
            # Prepare batched sample
            batched_sample = {
                'timestamps': sample['timestamps'].unsqueeze(0),
                'attention_mask': sample['attention_mask'].unsqueeze(0),
                'node_features': all_node_features,
                'edge_index': sample['edge_index'],
                'edge_features': sample['edge_features'],
                'airdrop_events': sample['airdrop_events'],
                'user_indices': sample['user_indices'],
                'transaction_features': {
                    k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.dim() > 0 else v
                    for k, v in sample['transaction_features'].items()
                }
            }
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                outputs = model(batched_sample)
            
            # Verify outputs
            assert 'logits' in outputs
            assert 'probabilities' in outputs
            assert 'confidence' in outputs
            assert outputs['logits'].shape == (1, 2)
            assert outputs['probabilities'].shape == (1, 2)
            assert outputs['confidence'].shape == (1,)
            
            # Test loss computation
            loss_fn = TemporalGraphLoss()
            batch = {
                'timestamps': batched_sample['timestamps'],
                'airdrop_events': [sample['airdrop_events']] if len(sample['airdrop_events']) > 0 else [torch.tensor([])]
            }
            labels = sample['label'].unsqueeze(0)
            
            loss_dict = loss_fn(outputs, batch, labels)
            assert 'total' in loss_dict
            assert isinstance(loss_dict['total'], torch.Tensor)
    
    def test_model_backward_pass(self):
        """Test that the model supports gradient computation."""
        config = {
            'd_model': 32,  # Smaller for faster testing
            'temporal_config': create_transaction_sequence_config(d_model=32),
            'graph_config': create_graph_structure_config(d_model=32),
            'output_dim': 32,
            'num_classes': 2,
            'fusion_type': 'cross_attention'
        }
        
        model = TemporalGraphTransformer(config)
        model.train()
        
        # Create simple test data
        batch_size = 1
        seq_len = 5
        num_nodes = 10
        
        sample = {
            'timestamps': torch.randn(batch_size, seq_len) * 1000000 + 1000000000,
            'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.bool),
            'node_features': torch.randn(num_nodes, 32),
            'edge_index': torch.randint(0, num_nodes, (2, 20)),
            'edge_features': {'edge_features': torch.randn(20, 32)},
            'airdrop_events': torch.randn(2) * 1000000 + 1000000000,
            'user_indices': torch.tensor([0]),
            'transaction_features': {
                'prices': torch.randn(batch_size, seq_len),
                'gas_fees': torch.randn(batch_size, seq_len),
                'sequence_length': torch.tensor([seq_len])
            }
        }
        
        # Forward pass
        outputs = model(sample)
        
        # Simple loss
        loss = outputs['logits'].sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        has_gradients = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for param in model.parameters()
        )
        assert has_gradients, "Model should have gradients after backward pass"


if __name__ == "__main__":
    pytest.main([__file__])