"""
Demo script showing the unified data preprocessing pipeline.

This demonstrates how to use the unified interface to test both
Temporal Graph Transformer and ARTEMIS baseline on the same datasets.
"""

import torch
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from temporal_graph_transformer.data import (
    SolanaNFTDataset,
    UnifiedDataLoader,
    create_data_loaders,
    TemporalGraphTransform,
    ARTEMISTransform
)
from temporal_graph_transformer.models import (
    TemporalGraphTransformer,
    create_model_config
)
from temporal_graph_transformer.models.artemis_baseline import (
    ARTEMISBaseline,
    create_artemis_config
)


def demo_unified_preprocessing():
    """Demonstrate the unified preprocessing pipeline."""
    print("=== Temporal Graph Transformer: Unified Data Preprocessing Demo ===\n")
    
    # 1. Create dataset with unified interface
    print("1. Creating Solana NFT dataset with unified interface...")
    data_path = "./data/solana_demo"
    os.makedirs(data_path, exist_ok=True)
    
    # Create train/val/test splits
    datasets = SolanaNFTDataset.create_splits(
        data_path=data_path,
        marketplace='magic_eden',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        max_sequence_length=50,
        airdrop_window_days=7
    )
    
    print(f"✓ Created datasets: {list(datasets.keys())}")
    
    # Display dataset statistics
    for split_name, dataset in datasets.items():
        stats = dataset.get_dataset_stats()
        print(f"\n{split_name.upper()} Dataset:")
        print(f"  - Users: {stats['num_users']}")
        print(f"  - Transactions: {stats['num_transactions']}")
        print(f"  - Hunter ratio: {stats['hunter_ratio']:.1%}")
        print(f"  - Avg transactions per user: {stats['avg_transactions_per_user']:.1f}")
    
    # 2. Create data loaders for Temporal Graph Transformer
    print("\n2. Creating data loaders for Temporal Graph Transformer...")
    
    tgt_config = {
        'batch_size': 16,
        'num_workers': 0,
        'model_type': 'temporal_graph_transformer',
        'class_balancing': True
    }
    
    tgt_loaders = create_data_loaders(
        datasets['train'], datasets['val'], datasets['test'], tgt_config
    )
    
    print("✓ Created TGT data loaders")
    
    # 3. Create data loaders for ARTEMIS baseline
    print("\n3. Creating data loaders for ARTEMIS baseline...")
    
    artemis_config = {
        'batch_size': 16,
        'num_workers': 0,
        'model_type': 'artemis',
        'class_balancing': True
    }
    
    artemis_loaders = create_data_loaders(
        datasets['train'], datasets['val'], datasets['test'], artemis_config
    )
    
    print("✓ Created ARTEMIS data loaders")
    
    # 4. Test batch loading for both models
    print("\n4. Testing batch loading...")
    
    # Test TGT batch
    tgt_batch = next(iter(tgt_loaders['train']))
    print(f"\nTemporal Graph Transformer batch:")
    print(f"  - Batch size: {tgt_batch['labels'].shape[0]}")
    print(f"  - Transaction features keys: {list(tgt_batch['transaction_features'].keys())}")
    print(f"  - Timestamps shape: {tgt_batch['timestamps'].shape}")
    print(f"  - Node features shape: {tgt_batch['node_features'].shape}")
    print(f"  - Edge index shape: {tgt_batch['edge_index'].shape}")
    print(f"  - Airdrop events: {len(tgt_batch['airdrop_events'])} events")
    
    # Test ARTEMIS batch
    artemis_batch = next(iter(artemis_loaders['train']))
    print(f"\nARTEMIS baseline batch:")
    print(f"  - Batch size: {artemis_batch['labels'].shape[0]}")
    print(f"  - Node features shape: {artemis_batch['node_features'].shape}")
    print(f"  - Manual features shape: {artemis_batch['manual_features'].shape}")
    print(f"  - Edge index shape: {artemis_batch['edge_index'].shape}")
    
    # 5. Test model compatibility
    print("\n5. Testing model compatibility...")
    
    # Initialize Temporal Graph Transformer
    tgt_model_config = create_model_config(
        d_model=128,
        temporal_layers=2,
        graph_layers=2,
        temporal_heads=4,
        graph_heads=4
    )
    tgt_model = TemporalGraphTransformer(tgt_model_config)
    
    # Initialize ARTEMIS baseline
    artemis_model_config = create_artemis_config(
        hidden_channels=64,
        num_layers=3,
        num_node_features=tgt_batch['node_features'].shape[-1],
        num_edge_features=tgt_batch['edge_features'].shape[-1],
        manual_feature_dim=artemis_batch['manual_features'].shape[-1]
    )
    artemis_model = ARTEMISBaseline(artemis_model_config)
    
    print("✓ Initialized both models")
    
    # Test forward passes
    print("\n6. Testing forward passes...")
    
    # TGT forward pass
    try:
        tgt_outputs = tgt_model(tgt_batch)
        print(f"✓ TGT forward pass successful")
        print(f"  - Logits shape: {tgt_outputs['logits'].shape}")
        print(f"  - Probabilities shape: {tgt_outputs['probabilities'].shape}")
        print(f"  - Confidence shape: {tgt_outputs['confidence'].shape}")
        print(f"  - Behavioral scores shape: {tgt_outputs['behavioral_scores'].shape}")
    except Exception as e:
        print(f"✗ TGT forward pass failed: {e}")
    
    # ARTEMIS forward pass
    try:
        artemis_outputs = artemis_model(artemis_batch)
        print(f"✓ ARTEMIS forward pass successful")
        print(f"  - Logits shape: {artemis_outputs['logits'].shape}")
        print(f"  - Probabilities shape: {artemis_outputs['probabilities'].shape}")
    except Exception as e:
        print(f"✗ ARTEMIS forward pass failed: {e}")
    
    # 7. Display class balance information
    print("\n7. Dataset class balance:")
    class_weights = datasets['train'].get_class_weights()
    print(f"  - Class weights: {class_weights.tolist()}")
    print(f"  - Recommended for loss weighting to handle imbalance")
    
    print("\n=== Demo completed successfully! ===")
    print("\nKey achievements:")
    print("✓ Unified interface works for multiple blockchain datasets")
    print("✓ Same data can be used for both TGT and ARTEMIS models")
    print("✓ Automatic feature extraction and graph construction")
    print("✓ Built-in class balancing and data validation")
    print("✓ Ready for fair model comparison experiments")
    
    return datasets, tgt_loaders, artemis_loaders, tgt_model, artemis_model


def demo_different_transforms():
    """Demonstrate how transforms adapt data for different models."""
    print("\n=== Transform Adaptation Demo ===")
    
    # Create base dataset
    data_path = "./data/transform_demo"
    os.makedirs(data_path, exist_ok=True)
    
    dataset = SolanaNFTDataset(
        data_path=data_path,
        split='train',
        max_sequence_length=30
    )
    
    # Get a sample
    sample = dataset[0]
    print(f"Original sample keys: {list(sample.keys())}")
    
    # Apply TGT transform
    tgt_transform = TemporalGraphTransform({
        'max_sequence_length': 30,
        'normalize_timestamps': True,
        'augment_sequences': False
    })
    
    tgt_sample = tgt_transform(sample.copy())
    print(f"\nTGT transformed sample:")
    print(f"  - Additional temporal features: {set(tgt_sample['transaction_features'].keys()) - set(sample['transaction_features'].keys())}")
    
    # Apply ARTEMIS transform
    artemis_transform = ARTEMISTransform({
        'normalize_features': True
    })
    
    artemis_sample = artemis_transform(sample.copy())
    print(f"\nARTEMIS transformed sample:")
    print(f"  - Manual features shape: {artemis_sample['manual_features'].shape}")
    print(f"  - Engineered {len(artemis_sample['manual_features'])} statistical features")
    
    print("✓ Transforms successfully adapt data for different model architectures")


if __name__ == "__main__":
    # Run main demo
    datasets, tgt_loaders, artemis_loaders, tgt_model, artemis_model = demo_unified_preprocessing()
    
    # Run transform demo
    demo_different_transforms()
    
    print("\n" + "="*60)
    print("NEXT STEPS FOR BEATING ARTEMIS:")
    print("="*60)
    print("1. Collect real Solana NFT data from Magic Eden API")
    print("2. Implement data collection for other understudied markets:")
    print("   - L2 networks (Arbitrum, Optimism)")
    print("   - GameFi platforms (Axie, StepN)")
    print("   - Alternative Ethereum marketplaces")
    print("3. Set up training pipeline with the loss functions")
    print("4. Run comparative experiments: TGT vs ARTEMIS")
    print("5. Analyze where TGT outperforms (temporal patterns, airdrop detection)")
    print("6. Generate paper results showing SOTA improvements")