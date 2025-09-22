"""
Base dataset class defining the unified interface for temporal graph datasets.

This ensures consistent data format across different blockchain ecosystems
for fair comparison between models.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
import numpy as np


class BaseTemporalGraphDataset(Dataset, ABC):
    """
    Abstract base class for temporal graph datasets.
    
    Defines the unified interface that all blockchain datasets must implement,
    ensuring consistent data format for both Temporal Graph Transformer and ARTEMIS.
    """
    
    def __init__(self, 
                 data_path: str,
                 split: str = 'train',
                 max_sequence_length: int = 100,
                 airdrop_window_days: int = 7,
                 include_market_features: bool = True,
                 cache_processed: bool = True):
        """
        Args:
            data_path: Path to dataset directory
            split: 'train', 'val', or 'test'
            max_sequence_length: Maximum transaction sequence length
            airdrop_window_days: Window around airdrop events for analysis
            include_market_features: Whether to include market manipulation features
            cache_processed: Whether to cache preprocessed data
        """
        self.data_path = data_path
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.airdrop_window_days = airdrop_window_days
        self.include_market_features = include_market_features
        self.cache_processed = cache_processed
        
        # Will be populated by subclasses
        self.users = None
        self.transactions = None
        self.nft_metadata = None
        self.graph_edges = None
        self.airdrop_events = None
        self.labels = None
        
    @abstractmethod
    def load_raw_data(self) -> None:
        """Load raw blockchain data from source files."""
        pass
    
    @abstractmethod
    def extract_transaction_features(self, user_id: str) -> Dict[str, torch.Tensor]:
        """Extract transaction features for a user."""
        pass
    
    @abstractmethod
    def extract_nft_features(self, nft_id: str) -> Dict[str, torch.Tensor]:
        """Extract NFT multimodal features (visual + textual)."""
        pass
    
    @abstractmethod
    def build_user_graph(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Build user interaction graph with multi-modal edge features."""
        pass
    
    def __len__(self) -> int:
        """Return number of users in dataset."""
        return len(self.users) if self.users is not None else 0
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single user sample with all required components.
        
        Returns:
            Dictionary containing all data needed for both models:
            - transaction_features: Dict of transaction attributes
            - timestamps: Transaction timestamps
            - node_features: Initial node features for graph
            - edge_index: Graph edge indices
            - edge_features: Multi-modal edge features
            - airdrop_events: Airdrop event timestamps
            - user_indices: Mapping from batch to graph nodes
            - attention_mask: Padding mask for transactions
            - label: Ground truth (0=legitimate, 1=hunter)
            - user_id: Original user identifier
        """
        if self.users is None:
            self.load_raw_data()
            
        user_id = self.users.iloc[idx]['user_id']
        
        # Extract transaction sequence features
        transaction_features = self.extract_transaction_features(user_id)
        
        # Get user's transaction timestamps
        user_txs = self.transactions[self.transactions['user_id'] == user_id].copy()
        user_txs = user_txs.sort_values('timestamp')
        
        # Limit sequence length and create padding mask
        if len(user_txs) > self.max_sequence_length:
            user_txs = user_txs.tail(self.max_sequence_length)
            
        seq_len = len(user_txs)
        attention_mask = torch.ones(self.max_sequence_length, dtype=torch.bool)
        if seq_len < self.max_sequence_length:
            attention_mask[seq_len:] = False
            
        # Convert timestamps to tensor
        timestamps = torch.tensor(user_txs['timestamp'].values, dtype=torch.float32)
        if len(timestamps) < self.max_sequence_length:
            # Pad with last timestamp
            last_timestamp = timestamps[-1] if len(timestamps) > 0 else 0.0
            padding = torch.full((self.max_sequence_length - len(timestamps),), last_timestamp)
            timestamps = torch.cat([timestamps, padding])
        
        # Get graph components (shared across all users in batch)
        edge_index, edge_features_dict = self.build_user_graph()
        edge_features = edge_features_dict
        
        # Get node features (one per user, will be expanded in batch)
        node_features = self.get_user_node_features(user_id)
        
        # Get airdrop events relevant to this user
        airdrop_events = self.get_relevant_airdrop_events(user_id, timestamps)
        
        # Get label
        label = self.labels.get(user_id, 0)  # Default to legitimate if not found
        
        return {
            'transaction_features': transaction_features,
            'timestamps': timestamps,
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'airdrop_events': airdrop_events,
            'user_indices': torch.tensor([idx], dtype=torch.long),  # Will be updated in collate
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long),
            'user_id': user_id
        }
    
    def get_user_node_features(self, user_id: str) -> torch.Tensor:
        """
        Get initial node features for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            node_features: Initial features for this user node
        """
        # Extract basic user statistics
        user_txs = self.transactions[self.transactions['user_id'] == user_id]
        
        if len(user_txs) == 0:
            return torch.zeros(64)  # Default feature size
        
        # Compute basic statistics
        total_volume = user_txs['price'].sum()
        avg_price = user_txs['price'].mean()
        tx_count = len(user_txs)
        unique_nfts = user_txs['nft_id'].nunique()
        time_span = user_txs['timestamp'].max() - user_txs['timestamp'].min()
        
        # Activity patterns
        daily_activity = user_txs.groupby(user_txs['timestamp'] // (24*3600))['price'].count().std()
        
        features = torch.tensor([
            np.log1p(total_volume),
            np.log1p(avg_price),
            np.log1p(tx_count),
            np.log1p(unique_nfts),
            np.log1p(time_span),
            daily_activity if not np.isnan(daily_activity) else 0.0
        ], dtype=torch.float32)
        
        # Pad to standard size
        padded_features = torch.zeros(64)
        padded_features[:min(len(features), 64)] = features[:64]
        
        return padded_features
    
    def get_relevant_airdrop_events(self, user_id: str, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Get airdrop events relevant to this user's time period.
        
        Args:
            user_id: User identifier  
            timestamps: User's transaction timestamps
            
        Returns:
            airdrop_events: Relevant airdrop event timestamps
        """
        if self.airdrop_events is None or len(timestamps) == 0:
            return torch.tensor([], dtype=torch.float32)
            
        # Get time range for this user
        start_time = timestamps[timestamps > 0].min().item()
        end_time = timestamps[timestamps > 0].max().item()
        
        # Expand window
        window_seconds = self.airdrop_window_days * 24 * 3600
        start_time -= window_seconds
        end_time += window_seconds
        
        # Convert to numpy array for filtering
        airdrop_array = np.array(self.airdrop_events)
        
        # Filter relevant airdrop events
        relevant_events = airdrop_array[
            (airdrop_array >= start_time) & 
            (airdrop_array <= end_time)
        ]
        
        return torch.tensor(relevant_events, dtype=torch.float32)
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        if self.users is None:
            self.load_raw_data()
            
        stats = {
            'num_users': len(self.users),
            'num_transactions': len(self.transactions),
            'num_hunters': sum(self.labels.values()),
            'hunter_ratio': sum(self.labels.values()) / len(self.labels),
            'avg_transactions_per_user': len(self.transactions) / len(self.users),
            'time_span_days': (self.transactions['timestamp'].max() - 
                             self.transactions['timestamp'].min()) / (24 * 3600),
            'unique_nfts': self.transactions['nft_id'].nunique() if 'nft_id' in self.transactions.columns else 0
        }
        
        return stats
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for handling imbalanced data."""
        if self.labels is None:
            return torch.tensor([1.0, 1.0])
            
        num_legitimate = sum(1 for v in self.labels.values() if v == 0)
        num_hunters = sum(1 for v in self.labels.values() if v == 1)
        total = num_legitimate + num_hunters
        
        if num_hunters == 0:
            return torch.tensor([1.0, 1.0])
            
        # Inverse frequency weighting
        weight_legitimate = total / (2 * num_legitimate)
        weight_hunter = total / (2 * num_hunters)
        
        return torch.tensor([weight_legitimate, weight_hunter])
    
    @abstractmethod
    def download_data(self) -> None:
        """Download raw data if not already present."""
        pass
    
    @abstractmethod
    def verify_data_integrity(self) -> bool:
        """Verify that downloaded data is complete and valid."""
        pass