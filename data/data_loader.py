"""
Unified data loader for temporal graph datasets.

Provides consistent batching and data loading interface for both
Temporal Graph Transformer and ARTEMIS models.
"""

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Optional, Any, Union, Callable
import numpy as np
from collections import defaultdict
import random


class UnifiedDataLoader:
    """
    Unified data loader that provides consistent interface for all models.
    
    Handles batching, padding, and data transformation for both 
    Temporal Graph Transformer and ARTEMIS baseline.
    """
    
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 model_type: str = 'temporal_graph_transformer',
                 class_balancing: bool = False,
                 max_graph_size: int = 1000):
        """
        Args:
            dataset: Dataset implementing the unified interface
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            model_type: 'temporal_graph_transformer' or 'artemis'
            class_balancing: Whether to balance classes in batches
            max_graph_size: Maximum number of nodes in graph batches
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.model_type = model_type
        self.class_balancing = class_balancing
        self.max_graph_size = max_graph_size
        
        # Set up sampler for class balancing if needed
        sampler = None
        if class_balancing:
            sampler = BalancedBatchSampler(dataset, batch_size)
            shuffle = False  # Sampler handles shuffling
        
        # Create the underlying DataLoader
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=self._get_collate_fn(),
            pin_memory=torch.cuda.is_available()
        )
    
    def _get_collate_fn(self) -> Callable:
        """Get appropriate collate function based on model type."""
        if self.model_type == 'temporal_graph_transformer':
            return self._collate_temporal_graph
        elif self.model_type == 'artemis':
            return self._collate_artemis
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _collate_temporal_graph(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for Temporal Graph Transformer.
        
        Handles graph batching and temporal sequence padding.
        """
        batch_size = len(batch)
        
        # Separate components
        transaction_features = []
        timestamps = []
        node_features = []
        labels = []
        user_ids = []
        attention_masks = []
        airdrop_events_list = []
        
        for item in batch:
            transaction_features.append(item['transaction_features'])
            timestamps.append(item['timestamps'])
            node_features.append(item['node_features'])
            labels.append(item['label'])
            user_ids.append(item['user_id'])
            attention_masks.append(item['attention_mask'])
            airdrop_events_list.append(item['airdrop_events'])
        
        # Stack temporal components
        batched_timestamps = torch.stack(timestamps)
        batched_attention_masks = torch.stack(attention_masks)
        batched_labels = torch.stack(labels)
        
        # Combine transaction features
        batched_transaction_features = {}
        feature_keys = transaction_features[0].keys()
        for key in feature_keys:
            batched_transaction_features[key] = torch.stack([
                item[key] for item in transaction_features
            ])
        
        # Handle graph components - use first item's graph structure
        # In practice, you might want to build a dynamic graph per batch
        reference_item = batch[0]
        edge_index = reference_item['edge_index']
        edge_features = reference_item['edge_features']
        
        # Stack node features and create user indices
        batched_node_features = torch.stack(node_features)
        user_indices = torch.arange(batch_size, dtype=torch.long)
        
        # Combine airdrop events (union of all relevant events)
        all_airdrop_events = []
        for events in airdrop_events_list:
            if len(events) > 0:
                all_airdrop_events.extend(events.tolist())
        
        unique_airdrop_events = torch.tensor(
            sorted(set(all_airdrop_events)), dtype=torch.float32
        ) if all_airdrop_events else torch.tensor([], dtype=torch.float32)
        
        return {
            'transaction_features': batched_transaction_features,
            'timestamps': batched_timestamps,
            'node_features': batched_node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'airdrop_events': unique_airdrop_events,
            'user_indices': user_indices,
            'attention_mask': batched_attention_masks,
            'labels': batched_labels,
            'user_ids': user_ids
        }
    
    def _collate_artemis(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for ARTEMIS baseline.
        
        Focuses on static graph features and manual feature engineering.
        """
        batch_size = len(batch)
        
        # Extract components needed for ARTEMIS
        node_features = []
        labels = []
        user_ids = []
        
        # Extract manual features from transaction data
        manual_features = []
        
        for item in batch:
            # Use node features as base
            node_features.append(item['node_features'])
            labels.append(item['label'])
            user_ids.append(item['user_id'])
            
            # Extract ARTEMIS-style manual features from transaction data
            artemis_features = self._extract_artemis_features(item)
            manual_features.append(artemis_features)
        
        # Use graph structure from first item
        reference_item = batch[0]
        edge_index = reference_item['edge_index']
        edge_features = reference_item['edge_features']
        
        # Stack features
        batched_node_features = torch.stack(node_features)
        batched_manual_features = torch.stack(manual_features)
        batched_labels = torch.stack(labels)
        user_indices = torch.arange(batch_size, dtype=torch.long)
        
        return {
            'node_features': batched_node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'manual_features': batched_manual_features,
            'user_indices': user_indices,
            'labels': batched_labels,
            'user_ids': user_ids
        }
    
    def _extract_artemis_features(self, item: Dict[str, Any]) -> torch.Tensor:
        """Extract ARTEMIS-style manual features from item data."""
        # Get transaction features
        tx_features = item['transaction_features']
        timestamps = item['timestamps']
        attention_mask = item['attention_mask']
        
        # Only use non-padded transactions
        valid_length = attention_mask.sum().item()
        if valid_length == 0:
            return torch.zeros(32)  # Default feature size
        
        # Extract relevant sequences
        prices = tx_features['prices'][:valid_length]
        gas_fees = tx_features['gas_fees'][:valid_length]
        time_deltas = tx_features['time_deltas'][:valid_length]
        transaction_frequency = tx_features['transaction_frequency'][:valid_length]
        
        # Compute ARTEMIS-style statistics
        features = []
        
        # Price statistics
        features.extend([
            prices.mean().item(),
            prices.std().item() if len(prices) > 1 else 0.0,
            prices.median().item(),
            prices.max().item(),
            prices.min().item()
        ])
        
        # Gas statistics
        features.extend([
            gas_fees.mean().item(),
            gas_fees.std().item() if len(gas_fees) > 1 else 0.0
        ])
        
        # Temporal patterns
        features.extend([
            time_deltas.mean().item(),
            time_deltas.std().item() if len(time_deltas) > 1 else 0.0,
            transaction_frequency.mean().item()
        ])
        
        # Activity patterns
        features.extend([
            float(valid_length),  # Transaction count
            timestamps[:valid_length].max().item() - timestamps[:valid_length].min().item()  # Time span
        ])
        
        # Pad to standard size
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        padded_features = torch.zeros(32)
        padded_features[:min(len(feature_tensor), 32)] = feature_tensor[:32]
        
        return padded_features
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Return number of batches."""
        return len(self.dataloader)
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for loss function."""
        return self.dataset.get_class_weights()
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return self.dataset.get_dataset_stats()


class BalancedBatchSampler(Sampler):
    """
    Sampler that creates balanced batches with equal representation of classes.
    """
    
    def __init__(self, dataset: Dataset, batch_size: int, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx in range(len(dataset)):
            item = dataset[idx]
            label = item['label'].item()
            self.class_indices[label].append(idx)
        
        self.classes = list(self.class_indices.keys())
        self.num_classes = len(self.classes)
        
        # Calculate samples per class per batch
        self.samples_per_class = max(1, batch_size // self.num_classes)
        self.total_batch_size = self.samples_per_class * self.num_classes
        
        # Calculate number of batches
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        self.num_batches = min_class_size // self.samples_per_class
        
        if not self.drop_last and min_class_size % self.samples_per_class > 0:
            self.num_batches += 1
    
    def __iter__(self):
        # Shuffle indices within each class
        shuffled_indices = {}
        for class_label, indices in self.class_indices.items():
            shuffled_indices[class_label] = indices.copy()
            random.shuffle(shuffled_indices[class_label])
        
        # Generate balanced batches
        for batch_idx in range(self.num_batches):
            batch = []
            
            for class_label in self.classes:
                start_idx = batch_idx * self.samples_per_class
                end_idx = min(start_idx + self.samples_per_class, 
                            len(shuffled_indices[class_label]))
                
                # Add samples from this class
                class_samples = shuffled_indices[class_label][start_idx:end_idx]
                batch.extend(class_samples)
            
            # Shuffle batch order
            random.shuffle(batch)
            
            # Only yield if batch has minimum size or not drop_last
            if len(batch) >= self.total_batch_size or not self.drop_last:
                yield batch
    
    def __len__(self):
        return self.num_batches


class GraphBatchSampler(Sampler):
    """
    Sampler that creates batches suitable for graph processing.
    
    Tries to balance computational load by considering graph sizes.
    """
    
    def __init__(self, 
                 dataset: Dataset, 
                 batch_size: int, 
                 max_nodes_per_batch: int = 10000,
                 drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_nodes_per_batch = max_nodes_per_batch
        self.drop_last = drop_last
        
        # Estimate node counts (simplified - assumes each user is one node)
        self.indices = list(range(len(dataset)))
        
    def __iter__(self):
        if hasattr(self.dataset, 'shuffle') and self.dataset.shuffle:
            random.shuffle(self.indices)
        
        batch = []
        current_nodes = 0
        
        for idx in self.indices:
            # Simplified: assume each sample adds one node
            nodes_to_add = 1
            
            # Check if adding this sample would exceed limits
            if (len(batch) >= self.batch_size or 
                current_nodes + nodes_to_add > self.max_nodes_per_batch):
                
                if batch:  # Yield current batch if not empty
                    yield batch
                    batch = []
                    current_nodes = 0
            
            batch.append(idx)
            current_nodes += nodes_to_add
        
        # Yield final batch if not empty
        if batch and (not self.drop_last or len(batch) >= self.batch_size):
            yield batch
    
    def __len__(self):
        # Approximate calculation
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


def create_data_loaders(train_dataset: Dataset,
                       val_dataset: Dataset,
                       test_dataset: Dataset,
                       config: Dict[str, Any]) -> Dict[str, UnifiedDataLoader]:
    """
    Create train, validation, and test data loaders with consistent configuration.
    
    Args:
        train_dataset, val_dataset, test_dataset: Dataset instances
        config: Configuration dictionary containing:
            - batch_size: Batch size for training
            - val_batch_size: Batch size for validation (optional)
            - test_batch_size: Batch size for testing (optional)
            - num_workers: Number of data loading workers
            - model_type: Model type ('temporal_graph_transformer' or 'artemis')
            - class_balancing: Whether to use balanced sampling for training
            
    Returns:
        Dictionary containing train, val, and test data loaders
    """
    batch_size = config.get('batch_size', 32)
    val_batch_size = config.get('val_batch_size', batch_size)
    test_batch_size = config.get('test_batch_size', batch_size)
    num_workers = config.get('num_workers', 0)
    model_type = config.get('model_type', 'temporal_graph_transformer')
    class_balancing = config.get('class_balancing', False)
    
    # Training data loader with shuffling and optional class balancing
    train_loader = UnifiedDataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        model_type=model_type,
        class_balancing=class_balancing
    )
    
    # Validation data loader
    val_loader = UnifiedDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        model_type=model_type,
        class_balancing=False
    )
    
    # Test data loader
    test_loader = UnifiedDataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        model_type=model_type,
        class_balancing=False
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }