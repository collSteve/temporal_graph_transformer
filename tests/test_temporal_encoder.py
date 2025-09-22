"""Tests for temporal encoder components."""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from temporal_graph_transformer.models.temporal_encoder import (
    TransactionSequenceTransformer,
    create_transaction_sequence_config
)
from temporal_graph_transformer.utils.time_encoding import (
    FunctionalTimeEncoding,
    BehaviorChangeTimeEncoding
)


class TestFunctionalTimeEncoding:
    """Test cases for FunctionalTimeEncoding."""
    
    def test_functional_time_encoding_dimensions(self):
        """Test that FunctionalTimeEncoding returns correct dimensions."""
        d_model = 64
        batch_size = 2
        seq_len = 10
        
        encoder = FunctionalTimeEncoding(d_model)
        timestamps = torch.randn(batch_size, seq_len) * 1000000 + 1000000000
        
        output = encoder(timestamps)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
    
    def test_behavior_change_time_encoding_dimensions(self):
        """Test that BehaviorChangeTimeEncoding returns correct dimensions."""
        d_model = 64
        batch_size = 2
        seq_len = 10
        
        encoder = BehaviorChangeTimeEncoding(d_model)
        timestamps = torch.randn(batch_size, seq_len) * 1000000 + 1000000000
        airdrop_events = torch.randn(3) * 1000000 + 1000000000
        
        # Test with airdrop events
        output_with_events = encoder(timestamps, airdrop_events)
        assert output_with_events.shape == (batch_size, seq_len, d_model)
        
        # Test without airdrop events
        output_without_events = encoder(timestamps, None)
        assert output_without_events.shape == (batch_size, seq_len, d_model)


class TestTransactionSequenceTransformer:
    """Test cases for TransactionSequenceTransformer."""
    
    def test_transaction_sequence_transformer_forward(self):
        """Test forward pass of TransactionSequenceTransformer."""
        config = create_transaction_sequence_config(d_model=64)
        model = TransactionSequenceTransformer(config)
        
        batch_size = 2
        seq_len = 10
        
        # Create mock transaction features
        transaction_features = {
            'prices': torch.randn(batch_size, seq_len),
            'gas_fees': torch.randn(batch_size, seq_len),
            'timestamps': torch.randn(batch_size, seq_len) * 1000000 + 1000000000,
            'transaction_types': torch.randint(0, 5, (batch_size, seq_len)).float(),
            'sequence_length': torch.tensor([seq_len] * batch_size)
        }
        
        timestamps = torch.randn(batch_size, seq_len) * 1000000 + 1000000000
        airdrop_events = torch.randn(2) * 1000000 + 1000000000
        
        model.eval()
        with torch.no_grad():
            outputs = model(transaction_features, timestamps, airdrop_events)
        
        assert 'sequence_embedding' in outputs
        assert 'change_scores' in outputs
        assert outputs['sequence_embedding'].shape == (batch_size, 64)
        assert outputs['change_scores'].shape == (batch_size,)


if __name__ == "__main__":
    pytest.main([__file__])