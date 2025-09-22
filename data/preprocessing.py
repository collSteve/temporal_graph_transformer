"""
Core preprocessing pipeline for temporal graph datasets.

This module handles the transformation of raw blockchain data into formats
suitable for both Temporal Graph Transformer and ARTEMIS models.
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import networkx as nx
from collections import defaultdict
import pickle
import os
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings


class TemporalGraphPreprocessor:
    """
    Main preprocessing pipeline for temporal graph datasets.
    
    Handles feature extraction, graph construction, and data normalization
    with support for multiple blockchain ecosystems.
    """
    
    def __init__(self, 
                 config: Dict[str, Any]):
        """
        Args:
            config: Preprocessing configuration containing:
                - max_sequence_length: Maximum transaction sequence length
                - min_transactions: Minimum transactions per user
                - airdrop_window_days: Window around airdrop events
                - normalize_features: Whether to normalize transaction features
                - include_nft_features: Whether to include NFT multimodal features
                - graph_construction_method: 'transaction_based' or 'similarity_based'
                - feature_cache_dir: Directory for caching extracted features
        """
        self.config = config
        self.max_sequence_length = config.get('max_sequence_length', 100)
        self.min_transactions = config.get('min_transactions', 5)
        self.airdrop_window_days = config.get('airdrop_window_days', 7)
        self.normalize_features = config.get('normalize_features', True)
        self.include_nft_features = config.get('include_nft_features', True)
        self.graph_construction_method = config.get('graph_construction_method', 'transaction_based')
        self.feature_cache_dir = config.get('feature_cache_dir', './cache')
        
        # Feature scalers
        self.transaction_scaler = RobustScaler()
        self.user_scaler = StandardScaler()
        self.fitted_scalers = False
        
        # Feature extractors
        self.nft_feature_extractor = None
        if self.include_nft_features:
            self.nft_feature_extractor = NFTFeatureExtractor()
        
        os.makedirs(self.feature_cache_dir, exist_ok=True)
    
    def process_dataset(self, 
                       transactions_df: pd.DataFrame,
                       users_df: pd.DataFrame,
                       nft_metadata_df: Optional[pd.DataFrame] = None,
                       airdrop_events: Optional[List[float]] = None,
                       labels: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Process complete dataset into temporal graph format.
        
        Args:
            transactions_df: Transaction data with columns:
                ['user_id', 'nft_id', 'timestamp', 'price', 'gas_fee', 'transaction_type']
            users_df: User data with basic information
            nft_metadata_df: Optional NFT metadata for multimodal features
            airdrop_events: Optional list of airdrop announcement timestamps
            labels: Optional user labels (0=legitimate, 1=hunter)
            
        Returns:
            Dictionary containing processed components
        """
        print("Processing temporal graph dataset...")
        
        # 1. Clean and filter data
        transactions_df = self._clean_transaction_data(transactions_df)
        users_df = self._filter_valid_users(users_df, transactions_df)
        
        # 2. Extract temporal features
        print("Extracting temporal features...")
        temporal_features = self._extract_temporal_features(transactions_df)
        
        # 3. Extract user features
        print("Extracting user features...")
        user_features = self._extract_user_features(users_df, transactions_df)
        
        # 4. Extract NFT features if available
        nft_features = None
        if nft_metadata_df is not None and self.include_nft_features:
            print("Extracting NFT multimodal features...")
            nft_features = self._extract_nft_features(nft_metadata_df)
        
        # 5. Build interaction graph
        print("Building user interaction graph...")
        graph_data = self._build_interaction_graph(transactions_df, nft_features)
        
        # 6. Process airdrop events
        airdrop_data = self._process_airdrop_events(airdrop_events, transactions_df)
        
        # 7. Normalize features
        if self.normalize_features:
            print("Normalizing features...")
            temporal_features = self._normalize_temporal_features(temporal_features)
            user_features = self._normalize_user_features(user_features)
        
        return {
            'temporal_features': temporal_features,
            'user_features': user_features,
            'nft_features': nft_features,
            'graph_data': graph_data,
            'airdrop_data': airdrop_data,
            'labels': labels or {},
            'metadata': {
                'num_users': len(users_df),
                'num_transactions': len(transactions_df),
                'time_span': transactions_df['timestamp'].max() - transactions_df['timestamp'].min(),
                'processing_config': self.config
            }
        }
    
    def _clean_transaction_data(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate transaction data."""
        print(f"Input transactions: {len(transactions_df)}")
        
        # Remove invalid transactions
        transactions_df = transactions_df.dropna(subset=['user_id', 'timestamp', 'price'])
        transactions_df = transactions_df[transactions_df['price'] > 0]
        transactions_df = transactions_df[transactions_df['timestamp'] > 0]
        
        # Sort by timestamp
        transactions_df = transactions_df.sort_values(['user_id', 'timestamp'])
        
        # Remove duplicate transactions
        transactions_df = transactions_df.drop_duplicates(
            subset=['user_id', 'nft_id', 'timestamp'], keep='first'
        )
        
        print(f"Cleaned transactions: {len(transactions_df)}")
        return transactions_df
    
    def _filter_valid_users(self, users_df: pd.DataFrame, 
                          transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Filter users based on minimum transaction requirements."""
        # Count transactions per user
        tx_counts = transactions_df['user_id'].value_counts()
        valid_users = tx_counts[tx_counts >= self.min_transactions].index
        
        # Filter users dataframe
        users_df = users_df[users_df['user_id'].isin(valid_users)]
        
        print(f"Valid users (>= {self.min_transactions} transactions): {len(users_df)}")
        return users_df
    
    def _extract_temporal_features(self, transactions_df: pd.DataFrame) -> Dict[str, Dict[str, torch.Tensor]]:
        """Extract temporal sequence features for each user."""
        temporal_features = {}
        
        for user_id in transactions_df['user_id'].unique():
            user_txs = transactions_df[transactions_df['user_id'] == user_id].copy()
            user_txs = user_txs.sort_values('timestamp')
            
            # Limit sequence length
            if len(user_txs) > self.max_sequence_length:
                user_txs = user_txs.tail(self.max_sequence_length)
            
            # Extract features
            features = self._extract_transaction_sequence_features(user_txs)
            temporal_features[user_id] = features
        
        return temporal_features
    
    def _extract_transaction_sequence_features(self, user_txs: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Extract features from a user's transaction sequence."""
        seq_len = len(user_txs)
        
        # Basic transaction features
        prices = user_txs['price'].values
        gas_fees = user_txs.get('gas_fee', pd.Series([0] * seq_len)).values
        timestamps = user_txs['timestamp'].values
        
        # Transaction type encoding (if available)
        tx_types = user_txs.get('transaction_type', pd.Series(['buy'] * seq_len))
        type_mapping = {'buy': 0, 'sell': 1, 'mint': 2, 'transfer': 3}
        tx_type_ids = [type_mapping.get(t, 0) for t in tx_types]
        
        # Time-based features
        time_deltas = np.diff(timestamps, prepend=timestamps[0])
        time_of_day = (timestamps % (24 * 3600)) / (24 * 3600)  # Normalized hour of day
        day_of_week = ((timestamps // (24 * 3600)) % 7) / 7  # Normalized day of week
        
        # Price-based features
        price_ratios = np.ones(seq_len)
        if seq_len > 1:
            price_ratios[1:] = prices[1:] / (prices[:-1] + 1e-8)
        
        # Gas efficiency (price per gas)
        gas_efficiency = prices / (gas_fees + 1e-8)
        
        # Volume features
        cumulative_volume = np.cumsum(prices)
        rolling_avg_price = pd.Series(prices).rolling(window=5, min_periods=1).mean().values
        
        # Behavioral features
        transaction_frequency = np.ones(seq_len)
        if seq_len > 1:
            # Transactions per day in rolling window
            for i in range(1, seq_len):
                window_start = max(0, i - 6)  # 7-day window
                window_txs = timestamps[window_start:i+1]
                time_span = timestamps[i] - timestamps[window_start] + 1
                transaction_frequency[i] = len(window_txs) / (time_span / (24 * 3600))
        
        # Pad sequences to max length
        def pad_sequence(arr):
            padded = np.zeros(self.max_sequence_length)
            padded[:len(arr)] = arr
            return torch.tensor(padded, dtype=torch.float32)
        
        return {
            'prices': pad_sequence(np.log1p(prices)),
            'gas_fees': pad_sequence(np.log1p(gas_fees)),
            'timestamps': pad_sequence(timestamps),
            'transaction_types': pad_sequence(tx_type_ids),
            'time_deltas': pad_sequence(np.log1p(time_deltas)),
            'time_of_day': pad_sequence(time_of_day),
            'day_of_week': pad_sequence(day_of_week),
            'price_ratios': pad_sequence(np.log(price_ratios + 1e-8)),
            'gas_efficiency': pad_sequence(np.log1p(gas_efficiency)),
            'cumulative_volume': pad_sequence(np.log1p(cumulative_volume)),
            'rolling_avg_price': pad_sequence(np.log1p(rolling_avg_price)),
            'transaction_frequency': pad_sequence(transaction_frequency),
            'sequence_length': torch.tensor(seq_len, dtype=torch.long)
        }
    
    def _extract_user_features(self, users_df: pd.DataFrame, 
                             transactions_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Extract user-level features for graph nodes."""
        user_features = {}
        
        for user_id in users_df['user_id'].unique():
            user_txs = transactions_df[transactions_df['user_id'] == user_id]
            features = self._compute_user_statistics(user_txs)
            user_features[user_id] = features
        
        return user_features
    
    def _compute_user_statistics(self, user_txs: pd.DataFrame) -> torch.Tensor:
        """Compute statistical features for a user."""
        if len(user_txs) == 0:
            return torch.zeros(32)
        
        # Basic statistics
        total_volume = user_txs['price'].sum()
        avg_price = user_txs['price'].mean()
        median_price = user_txs['price'].median()
        price_std = user_txs['price'].std()
        tx_count = len(user_txs)
        
        # Time-based features
        time_span = user_txs['timestamp'].max() - user_txs['timestamp'].min()
        avg_time_between_txs = time_span / max(tx_count - 1, 1)
        
        # Diversity features
        unique_nfts = user_txs['nft_id'].nunique() if 'nft_id' in user_txs.columns else 1
        nft_diversity = unique_nfts / tx_count
        
        # Transaction pattern features
        buy_sell_ratio = len(user_txs[user_txs.get('transaction_type', 'buy') == 'buy']) / max(tx_count, 1)
        
        # Price behavior
        price_volatility = price_std / (avg_price + 1e-8)
        
        # Activity patterns
        timestamps = user_txs['timestamp'].values
        hour_activity = np.bincount((timestamps % (24 * 3600) // 3600).astype(int), minlength=24)
        peak_hour_ratio = hour_activity.max() / max(hour_activity.sum(), 1)
        
        # Benford's Law features (for fraud detection)
        first_digits = [int(str(int(p))[0]) for p in user_txs['price'] if p >= 1]
        if first_digits:
            digit_dist = np.bincount(first_digits, minlength=10)[1:10]  # Digits 1-9
            benford_expected = np.log10(1 + 1/np.arange(1, 10))
            benford_actual = digit_dist / digit_dist.sum()
            benford_score = np.sum((benford_actual - benford_expected) ** 2)
        else:
            benford_score = 0
        
        features = torch.tensor([
            np.log1p(total_volume),
            np.log1p(avg_price),
            np.log1p(median_price),
            np.log1p(price_std + 1e-8),
            np.log1p(tx_count),
            np.log1p(time_span),
            np.log1p(avg_time_between_txs),
            unique_nfts,
            nft_diversity,
            buy_sell_ratio,
            price_volatility,
            peak_hour_ratio,
            benford_score,
            # Add more features as needed
        ], dtype=torch.float32)
        
        # Pad to standard size
        padded_features = torch.zeros(32)
        padded_features[:min(len(features), 32)] = features[:32]
        
        return padded_features
    
    def _extract_nft_features(self, nft_metadata_df: pd.DataFrame) -> Dict[str, Dict[str, torch.Tensor]]:
        """Extract NFT multimodal features using the feature extractor."""
        if self.nft_feature_extractor is None:
            return {}
        
        nft_features = {}
        
        for _, nft in nft_metadata_df.iterrows():
            nft_id = nft['nft_id']
            
            # Extract visual and textual features
            visual_features = self.nft_feature_extractor.extract_visual_features(nft)
            text_features = self.nft_feature_extractor.extract_text_features(nft)
            
            nft_features[nft_id] = {
                'visual': visual_features,
                'text': text_features,
                'metadata': self.nft_feature_extractor.extract_metadata_features(nft)
            }
        
        return nft_features
    
    def _build_interaction_graph(self, transactions_df: pd.DataFrame, 
                               nft_features: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """Build user interaction graph with multi-modal edge features."""
        if self.graph_construction_method == 'transaction_based':
            return self._build_transaction_based_graph(transactions_df, nft_features)
        elif self.graph_construction_method == 'similarity_based':
            return self._build_similarity_based_graph(transactions_df, nft_features)
        else:
            raise ValueError(f"Unknown graph construction method: {self.graph_construction_method}")
    
    def _build_transaction_based_graph(self, transactions_df: pd.DataFrame,
                                     nft_features: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """Build graph based on direct transaction relationships."""
        # Find user pairs that interacted with the same NFTs
        user_pairs = defaultdict(list)
        
        for nft_id in transactions_df['nft_id'].unique():
            nft_users = transactions_df[transactions_df['nft_id'] == nft_id]['user_id'].unique()
            
            # Create edges between users who traded the same NFT
            for i, user1 in enumerate(nft_users):
                for user2 in nft_users[i+1:]:
                    user_pairs[(user1, user2)].append(nft_id)
        
        # Create edge index and features
        edges = []
        edge_features = []
        
        user_to_idx = {user: idx for idx, user in enumerate(transactions_df['user_id'].unique())}
        
        for (user1, user2), shared_nfts in user_pairs.items():
            if user1 in user_to_idx and user2 in user_to_idx:
                # Add bidirectional edges
                edges.extend([(user_to_idx[user1], user_to_idx[user2]),
                            (user_to_idx[user2], user_to_idx[user1])])
                
                # Compute edge features
                edge_feature = self._compute_edge_features(user1, user2, shared_nfts, 
                                                         transactions_df, nft_features)
                edge_features.extend([edge_feature, edge_feature])  # Same for both directions
        
        if not edges:
            # Create minimal graph if no edges found
            num_users = len(user_to_idx)
            edges = [(i, i) for i in range(num_users)]  # Self-loops
            edge_features = [torch.zeros(64) for _ in range(num_users)]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_features = torch.stack(edge_features)
        
        return {
            'edge_index': edge_index,
            'edge_features': edge_features,
            'user_to_idx': user_to_idx
        }
    
    def _compute_edge_features(self, user1: str, user2: str, shared_nfts: List[str],
                             transactions_df: pd.DataFrame, 
                             nft_features: Optional[Dict] = None) -> torch.Tensor:
        """Compute features for an edge between two users."""
        # Transaction-based features
        user1_txs = transactions_df[transactions_df['user_id'] == user1]
        user2_txs = transactions_df[transactions_df['user_id'] == user2]
        
        # Shared NFT features
        num_shared_nfts = len(shared_nfts)
        
        # Price interaction features
        shared_nft_txs1 = user1_txs[user1_txs['nft_id'].isin(shared_nfts)]
        shared_nft_txs2 = user2_txs[user2_txs['nft_id'].isin(shared_nfts)]
        
        avg_price_diff = 0
        if len(shared_nft_txs1) > 0 and len(shared_nft_txs2) > 0:
            avg_price_diff = abs(shared_nft_txs1['price'].mean() - shared_nft_txs2['price'].mean())
        
        # Temporal overlap
        time_overlap = 0
        if len(shared_nft_txs1) > 0 and len(shared_nft_txs2) > 0:
            min_time1, max_time1 = shared_nft_txs1['timestamp'].min(), shared_nft_txs1['timestamp'].max()
            min_time2, max_time2 = shared_nft_txs2['timestamp'].min(), shared_nft_txs2['timestamp'].max()
            
            overlap_start = max(min_time1, min_time2)
            overlap_end = min(max_time1, max_time2)
            time_overlap = max(0, overlap_end - overlap_start)
        
        # NFT multimodal features (average over shared NFTs)
        visual_similarity = 0
        text_similarity = 0
        
        if nft_features and shared_nfts:
            visual_features = []
            text_features = []
            
            for nft_id in shared_nfts:
                if nft_id in nft_features:
                    visual_features.append(nft_features[nft_id]['visual'])
                    text_features.append(nft_features[nft_id]['text'])
            
            if visual_features:
                visual_similarity = torch.stack(visual_features).mean(dim=0).norm().item()
            if text_features:
                text_similarity = torch.stack(text_features).mean(dim=0).norm().item()
        
        # Combine features
        features = torch.tensor([
            num_shared_nfts,
            np.log1p(avg_price_diff),
            np.log1p(time_overlap),
            visual_similarity,
            text_similarity
        ], dtype=torch.float32)
        
        # Pad to standard edge feature size
        padded_features = torch.zeros(64)
        padded_features[:min(len(features), 64)] = features[:64]
        
        return padded_features
    
    def _build_similarity_based_graph(self, transactions_df: pd.DataFrame,
                                    nft_features: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """Build graph based on user behavior similarity."""
        # This would implement k-NN or threshold-based similarity graph
        # For now, fall back to transaction-based method
        warnings.warn("Similarity-based graph construction not fully implemented, using transaction-based")
        return self._build_transaction_based_graph(transactions_df, nft_features)
    
    def _process_airdrop_events(self, airdrop_events: Optional[List[float]], 
                              transactions_df: pd.DataFrame) -> Dict[str, Any]:
        """Process airdrop event data for temporal analysis."""
        if airdrop_events is None:
            return {'events': torch.tensor([], dtype=torch.float32)}
        
        # Convert to tensor
        events_tensor = torch.tensor(airdrop_events, dtype=torch.float32)
        
        # Filter events within transaction time range
        min_time = transactions_df['timestamp'].min()
        max_time = transactions_df['timestamp'].max()
        
        relevant_events = events_tensor[
            (events_tensor >= min_time) & (events_tensor <= max_time)
        ]
        
        return {
            'events': relevant_events,
            'all_events': events_tensor,
            'num_relevant': len(relevant_events)
        }
    
    def _normalize_temporal_features(self, temporal_features: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Normalize temporal features across users."""
        if not self.fitted_scalers:
            # Fit scalers on training data
            all_features = []
            for user_features in temporal_features.values():
                # Stack numerical features
                numerical_features = torch.stack([
                    user_features['prices'],
                    user_features['gas_fees'],
                    user_features['time_deltas'],
                    user_features['gas_efficiency'],
                    user_features['cumulative_volume'],
                    user_features['rolling_avg_price'],
                    user_features['transaction_frequency']
                ], dim=1)  # (seq_len, num_features)
                
                # Only use non-padded values
                seq_len = user_features['sequence_length'].item()
                if seq_len > 0:
                    all_features.append(numerical_features[:seq_len])
            
            if all_features:
                combined_features = torch.cat(all_features, dim=0).numpy()
                self.transaction_scaler.fit(combined_features)
                self.fitted_scalers = True
        
        # Apply normalization
        normalized_features = {}
        for user_id, user_features in temporal_features.items():
            normalized_features[user_id] = user_features.copy()
            
            # Normalize numerical features
            numerical_features = torch.stack([
                user_features['prices'],
                user_features['gas_fees'],
                user_features['time_deltas'],
                user_features['gas_efficiency'],
                user_features['cumulative_volume'],
                user_features['rolling_avg_price'],
                user_features['transaction_frequency']
            ], dim=1)
            
            seq_len = user_features['sequence_length'].item()
            if seq_len > 0:
                # Normalize non-padded values
                normalized_numerical = numerical_features[:seq_len].numpy()
                normalized_numerical = self.transaction_scaler.transform(normalized_numerical)
                
                # Put back into tensors
                normalized_tensor = torch.tensor(normalized_numerical, dtype=torch.float32)
                normalized_features[user_id]['prices'] = torch.zeros_like(user_features['prices'])
                normalized_features[user_id]['gas_fees'] = torch.zeros_like(user_features['gas_fees'])
                normalized_features[user_id]['time_deltas'] = torch.zeros_like(user_features['time_deltas'])
                normalized_features[user_id]['gas_efficiency'] = torch.zeros_like(user_features['gas_efficiency'])
                normalized_features[user_id]['cumulative_volume'] = torch.zeros_like(user_features['cumulative_volume'])
                normalized_features[user_id]['rolling_avg_price'] = torch.zeros_like(user_features['rolling_avg_price'])
                normalized_features[user_id]['transaction_frequency'] = torch.zeros_like(user_features['transaction_frequency'])
                
                # Fill in normalized values
                normalized_features[user_id]['prices'][:seq_len] = normalized_tensor[:, 0]
                normalized_features[user_id]['gas_fees'][:seq_len] = normalized_tensor[:, 1]
                normalized_features[user_id]['time_deltas'][:seq_len] = normalized_tensor[:, 2]
                normalized_features[user_id]['gas_efficiency'][:seq_len] = normalized_tensor[:, 3]
                normalized_features[user_id]['cumulative_volume'][:seq_len] = normalized_tensor[:, 4]
                normalized_features[user_id]['rolling_avg_price'][:seq_len] = normalized_tensor[:, 5]
                normalized_features[user_id]['transaction_frequency'][:seq_len] = normalized_tensor[:, 6]
        
        return normalized_features
    
    def _normalize_user_features(self, user_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize user-level features."""
        if not user_features:
            return user_features
        
        # Stack all user features
        feature_matrix = torch.stack(list(user_features.values())).numpy()
        
        # Fit and transform
        if not hasattr(self, 'user_scaler_fitted'):
            self.user_scaler.fit(feature_matrix)
            self.user_scaler_fitted = True
        
        normalized_matrix = self.user_scaler.transform(feature_matrix)
        normalized_tensor = torch.tensor(normalized_matrix, dtype=torch.float32)
        
        # Convert back to dictionary
        normalized_features = {}
        for i, user_id in enumerate(user_features.keys()):
            normalized_features[user_id] = normalized_tensor[i]
        
        return normalized_features


class NFTFeatureExtractor:
    """Extract multimodal features from NFT metadata."""
    
    def __init__(self):
        # Placeholder for actual feature extractors
        # In practice, these would be pre-trained ViT and BERT models
        self.visual_model = None  # Would load ViT
        self.text_model = None    # Would load BERT
    
    def extract_visual_features(self, nft_metadata: pd.Series) -> torch.Tensor:
        """Extract visual features from NFT image."""
        # Placeholder implementation
        # In practice, this would process the image through ViT
        if 'image_url' in nft_metadata and nft_metadata['image_url']:
            # Would download and process image
            return torch.randn(768)  # ViT feature size
        else:
            return torch.zeros(768)
    
    def extract_text_features(self, nft_metadata: pd.Series) -> torch.Tensor:
        """Extract textual features from NFT description/name."""
        # Placeholder implementation
        # In practice, this would process text through BERT
        text_content = str(nft_metadata.get('name', '')) + ' ' + str(nft_metadata.get('description', ''))
        if text_content.strip():
            # Would tokenize and process through BERT
            return torch.randn(768)  # BERT feature size
        else:
            return torch.zeros(768)
    
    def extract_metadata_features(self, nft_metadata: pd.Series) -> torch.Tensor:
        """Extract features from NFT metadata attributes."""
        # Basic metadata features
        features = []
        
        # Collection size
        collection_size = nft_metadata.get('collection_size', 1)
        features.append(np.log1p(collection_size))
        
        # Rarity score
        rarity_score = nft_metadata.get('rarity_score', 0)
        features.append(rarity_score)
        
        # Number of attributes
        num_attributes = len(nft_metadata.get('attributes', []))
        features.append(num_attributes)
        
        # Creator reputation (placeholder)
        creator_reputation = nft_metadata.get('creator_reputation', 0.5)
        features.append(creator_reputation)
        
        return torch.tensor(features, dtype=torch.float32)