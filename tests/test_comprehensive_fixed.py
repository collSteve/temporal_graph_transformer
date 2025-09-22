#!/usr/bin/env python3
"""
Comprehensive test with fixed imports to verify all components work.
"""

import torch
import tempfile
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_all_components():
    """Test all major components with fixed imports."""
    print("🧪 COMPREHENSIVE TEST WITH FIXED IMPORTS")
    print("=" * 50)
    
    # Test 1: Time Encoding
    print("\n📊 Testing Time Encoding...")
    try:
        from utils.time_encoding import FunctionalTimeEncoding, BehaviorChangeTimeEncoding
        
        encoder = FunctionalTimeEncoding(64)
        behavior_encoder = BehaviorChangeTimeEncoding(64)
        
        timestamps = torch.randn(2, 10) * 1000000 + 1600000000
        airdrop_events = torch.randn(3) * 1000000 + 1600000000
        
        output1 = encoder(timestamps)
        output2 = behavior_encoder(timestamps, airdrop_events)
        
        assert output1.shape == (2, 10, 64), f"FunctionalTimeEncoding shape: {output1.shape}"
        assert output2.shape == (2, 10, 64), f"BehaviorChangeTimeEncoding shape: {output2.shape}"
        
        print("  ✅ Time encoding components working")
        
    except Exception as e:
        print(f"  ❌ Time encoding failed: {e}")
        return False
    
    # Test 2: Temporal Encoder
    print("\n🕐 Testing Temporal Encoder...")
    try:
        from models.temporal_encoder import TransactionSequenceTransformer, create_transaction_sequence_config
        
        config = create_transaction_sequence_config(d_model=64, num_layers=2)
        model = TransactionSequenceTransformer(config)
        
        transaction_features = {
            'prices': torch.randn(2, 10),
            'gas_fees': torch.randn(2, 10),
            'sequence_length': torch.tensor([10, 10])
        }
        
        timestamps = torch.randn(2, 10) * 1000000 + 1600000000
        airdrop_events = torch.randn(2) * 1000000 + 1600000000
        
        outputs = model(transaction_features, timestamps, airdrop_events)
        
        assert 'sequence_embedding' in outputs, "Missing sequence_embedding"
        assert outputs['sequence_embedding'].shape == (2, 64), f"Sequence embedding shape: {outputs['sequence_embedding'].shape}"
        
        print("  ✅ Temporal encoder working")
        
    except Exception as e:
        print(f"  ❌ Temporal encoder failed: {e}")
        return False
    
    # Test 3: Graph Encoder
    print("\n🕸️  Testing Graph Encoder...")
    try:
        from models.graph_encoder import GraphStructureTransformer, create_graph_structure_config
        
        config = create_graph_structure_config(d_model=64, num_layers=2)
        model = GraphStructureTransformer(config)
        
        num_nodes = 20
        node_features = torch.randn(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, 40))
        edge_features = {'edge_features': torch.randn(40, 64)}
        
        outputs = model(node_features, edge_index, edge_features)
        
        assert 'node_embeddings' in outputs, "Missing node_embeddings"
        assert 'graph_embedding' in outputs, "Missing graph_embedding"
        assert outputs['node_embeddings'].shape == (num_nodes, 64), f"Node embeddings shape: {outputs['node_embeddings'].shape}"
        
        print("  ✅ Graph encoder working")
        
    except Exception as e:
        print(f"  ❌ Graph encoder failed: {e}")
        return False
    
    # Test 4: Complete TGT Model
    print("\n🤖 Testing Complete TGT Model...")
    try:
        from models.temporal_graph_transformer import TemporalGraphTransformer
        
        config = {
            'd_model': 64,
            'temporal_config': create_transaction_sequence_config(d_model=64),
            'graph_config': create_graph_structure_config(d_model=64),
            'output_dim': 64,
            'num_classes': 2,
            'fusion_type': 'cross_attention'
        }
        
        model = TemporalGraphTransformer(config)
        
        sample = {
            'timestamps': torch.randn(2, 10) * 1000000 + 1600000000,
            'attention_mask': torch.ones(2, 10, dtype=torch.bool),
            'node_features': torch.randn(20, 64),
            'edge_index': torch.randint(0, 20, (2, 40)),
            'edge_features': {'edge_features': torch.randn(40, 64)},
            'airdrop_events': torch.randn(2) * 1000000 + 1600000000,
            'user_indices': torch.tensor([0, 1]),
            'transaction_features': {
                'prices': torch.randn(2, 10),
                'gas_fees': torch.randn(2, 10),
                'sequence_length': torch.tensor([10, 10])
            }
        }
        
        outputs = model(sample)
        
        expected_outputs = ['logits', 'probabilities', 'confidence', 'behavioral_scores']
        for key in expected_outputs:
            assert key in outputs, f"Missing output: {key}"
        
        assert outputs['logits'].shape == (2, 2), f"Logits shape: {outputs['logits'].shape}"
        
        print("  ✅ Complete TGT model working")
        
    except Exception as e:
        print(f"  ❌ Complete TGT model failed: {e}")
        return False
    
    # Test 5: Loss Functions
    print("\n📉 Testing Loss Functions...")
    try:
        from utils.loss_functions import TemporalGraphLoss
        
        loss_fn = TemporalGraphLoss()
        
        batch_data = {
            'timestamps': torch.randn(2, 10) * 1000000 + 1600000000,
            'airdrop_events': torch.randn(2) * 1000000 + 1600000000  # Fixed: single tensor, not list
        }
        
        labels = torch.randint(0, 2, (2,))
        
        loss_dict = loss_fn(outputs, batch_data, labels)
        
        assert 'total' in loss_dict, "Missing total loss"
        
        # Debug: Print all loss components
        print("    Loss components:")
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                is_nan = torch.isnan(value)
                print(f"      {key}: {value.item():.6f} {'[NaN]' if is_nan else ''}")
                
        assert not torch.isnan(loss_dict['total']), f"Total loss is NaN. Loss dict: {loss_dict}"
        
        print("  ✅ Loss functions working")
        
    except Exception as e:
        print(f"  ❌ Loss functions failed: {e}")
        return False
    
    # Test 6: Data Pipeline
    print("\n📊 Testing Data Pipeline...")
    try:
        from data.solana_dataset import SolanaNFTDataset
        
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = SolanaNFTDataset(
                data_path=temp_dir,
                split='train',
                marketplace='magic_eden',
                max_sequence_length=10,
                airdrop_window_days=7
            )
            
            assert len(dataset) > 0, "Dataset should not be empty"
            
            sample = dataset[0]
            required_keys = ['transaction_features', 'timestamps', 'node_features', 'label']
            for key in required_keys:
                assert key in sample, f"Missing sample key: {key}"
        
        print("  ✅ Data pipeline working")
        
    except Exception as e:
        print(f"  ❌ Data pipeline failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 ALL COMPONENTS WORKING WITH CONDA ENVIRONMENT!")
    print("✅ Time Encoding")
    print("✅ Temporal Encoder") 
    print("✅ Graph Encoder")
    print("✅ Complete TGT Model")
    print("✅ Loss Functions")
    print("✅ Data Pipeline")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = test_all_components()
    sys.exit(0 if success else 1)