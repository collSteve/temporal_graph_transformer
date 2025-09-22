"""
Simplified demo of the data preprocessing pipeline.
"""

import torch
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from temporal_graph_transformer.data.solana_dataset import SolanaNFTDataset


def simple_demo():
    """Simple demonstration of the preprocessing pipeline."""
    print("=== Simple Temporal Graph Transformer Demo ===\n")
    
    # 1. Create a single dataset
    print("1. Creating Solana NFT dataset...")
    data_path = "./data/simple_demo"
    os.makedirs(data_path, exist_ok=True)
    
    dataset = SolanaNFTDataset(
        data_path=data_path,
        split='train',
        marketplace='magic_eden',
        max_sequence_length=20,  # Smaller for demo
        airdrop_window_days=7
    )
    
    print(f"✓ Dataset created with {len(dataset)} users")
    
    # 2. Get dataset statistics
    stats = dataset.get_dataset_stats()
    print(f"\nDataset Statistics:")
    print(f"  - Users: {stats['num_users']}")
    print(f"  - Transactions: {stats['num_transactions']}")
    print(f"  - Hunter ratio: {stats['hunter_ratio']:.1%}")
    print(f"  - Avg transactions per user: {stats['avg_transactions_per_user']:.1f}")
    print(f"  - Time span: {stats['time_span_days']:.1f} days")
    
    # 3. Test single sample
    print(f"\n2. Testing sample extraction...")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"User ID: {sample['user_id']}")
    print(f"Label: {'Hunter' if sample['label'].item() == 1 else 'Legitimate'}")
    print(f"Sequence length: {sample['transaction_features']['sequence_length'].item()}")
    print(f"Timestamps shape: {sample['timestamps'].shape}")
    print(f"Node features shape: {sample['node_features'].shape}")
    print(f"Airdrop events: {len(sample['airdrop_events'])} events")
    
    # 4. Test feature extraction
    print(f"\n3. Testing feature extraction...")
    tx_features = sample['transaction_features']
    print(f"Transaction feature keys: {list(tx_features.keys())}")
    
    # Show some actual values
    seq_len = tx_features['sequence_length'].item()
    if seq_len > 0:
        print(f"First few prices: {tx_features['prices'][:min(5, seq_len)].tolist()}")
        print(f"First few timestamps: {sample['timestamps'][:min(5, seq_len)].tolist()}")
    
    # 5. Test graph structure
    print(f"\n4. Testing graph structure...")
    print(f"Edge index shape: {sample['edge_index'].shape}")
    print(f"Number of edges: {sample['edge_index'].shape[1]}")
    print(f"Edge features shape: {sample['edge_features']['edge_features'].shape}")
    
    # 6. Test class balance
    print(f"\n5. Class balance information:")
    class_weights = dataset.get_class_weights()
    print(f"Class weights: {class_weights.tolist()}")
    
    # 7. Verify data integrity
    print(f"\n6. Data integrity check:")
    is_valid = dataset.verify_data_integrity()
    print(f"Dataset integrity: {'✓ PASSED' if is_valid else '✗ FAILED'}")
    
    print(f"\n=== Demo completed successfully! ===")
    print(f"\nKey achievements:")
    print(f"✓ Unified interface for blockchain datasets")
    print(f"✓ Automatic synthetic data generation for testing")
    print(f"✓ Transaction sequence feature extraction")
    print(f"✓ Graph structure construction")
    print(f"✓ Multi-modal edge features")
    print(f"✓ Airdrop event processing")
    print(f"✓ Class balancing support")
    print(f"✓ Data integrity validation")
    
    return dataset


if __name__ == "__main__":
    dataset = simple_demo()
    
    print(f"\n" + "="*50)
    print(f"NEXT STEPS:")
    print(f"="*50)
    print(f"1. ✓ Data preprocessing pipeline complete")
    print(f"2. → Set up training pipeline with loss functions")  
    print(f"3. → Run model comparison experiments")
    print(f"4. → Collect real data from Solana/Magic Eden")
    print(f"5. → Expand to other blockchain ecosystems")
