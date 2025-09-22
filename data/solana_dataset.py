"""
Solana NFT dataset implementation for airdrop hunter detection.

This implements the unified interface for Solana NFT marketplace data,
targeting Magic Eden and other Solana-based NFT platforms where ARTEMIS
has not been tested extensively.
"""

import torch
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import requests
import time
from collections import defaultdict
import warnings

from .base_dataset import BaseTemporalGraphDataset
from .preprocessing import TemporalGraphPreprocessor


class SolanaNFTDataset(BaseTemporalGraphDataset):
    """
    Solana NFT marketplace dataset focusing on Magic Eden and other platforms.
    
    This dataset targets an ecosystem where ARTEMIS has not been extensively
    tested, providing an opportunity to demonstrate superior performance.
    """
    
    def __init__(self, 
                 data_path: str,
                 split: str = 'train',
                 marketplace: str = 'magic_eden',
                 collection_filter: Optional[List[str]] = None,
                 time_range: Optional[Tuple[str, str]] = None,
                 min_transaction_value: float = 0.1,  # SOL
                 **kwargs):
        """
        Args:
            data_path: Path to Solana dataset directory
            split: 'train', 'val', or 'test'
            marketplace: Target marketplace ('magic_eden', 'solanart', 'alpha_art')
            collection_filter: Optional list of collection addresses to include
            time_range: Optional (start_date, end_date) in YYYY-MM-DD format
            min_transaction_value: Minimum transaction value in SOL
        """
        super().__init__(data_path, split, **kwargs)
        
        self.marketplace = marketplace
        self.collection_filter = collection_filter
        self.time_range = time_range
        self.min_transaction_value = min_transaction_value
        
        # Solana-specific configuration
        self.sol_to_lamports = 1e9
        self.preprocessor = TemporalGraphPreprocessor({
            'max_sequence_length': self.max_sequence_length,
            'min_transactions': 5,
            'airdrop_window_days': self.airdrop_window_days,
            'normalize_features': True,
            'include_nft_features': True,
            'graph_construction_method': 'transaction_based'
        })
        
        # Known Solana airdrop events (timestamps)
        self.known_airdrop_events = [
            1640995200,  # Dec 31, 2021 - Various Solana ecosystem airdrops
            1646092800,  # Mar 1, 2022 - Solana ecosystem expansion
            1651363200,  # May 1, 2022 - Magic Eden token discussions
            1659312000,  # Aug 1, 2022 - Solana ecosystem airdrops
            1667260800,  # Nov 1, 2022 - Various protocol airdrops
        ]
        
        # Load data if available
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """Load the dataset, downloading if necessary."""
        dataset_file = os.path.join(self.data_path, f'solana_{self.marketplace}_{self.split}.json')
        
        if os.path.exists(dataset_file):
            print(f"Loading existing Solana dataset from {dataset_file}")
            self.load_raw_data()
        else:
            print(f"Dataset not found at {dataset_file}")
            print("To collect Solana data, you would need to:")
            print("1. Set up access to Solana RPC endpoints")
            print("2. Query transaction data from Magic Eden and other marketplaces")
            print("3. Extract NFT metadata and user interaction patterns")
            print("4. Label known airdrop hunters vs legitimate users")
            print("\nFor now, generating synthetic data for demonstration...")
            self._generate_synthetic_data()
    
    def download_data(self) -> None:
        """Download Solana NFT data from various sources."""
        print("Downloading Solana NFT data...")
        
        # This would implement actual data collection from:
        # 1. Solana RPC nodes for transaction data
        # 2. Magic Eden API for marketplace data
        # 3. Solana NFT metadata standards
        # 4. On-chain program logs for trading activities
        
        # For now, create placeholder structure
        os.makedirs(self.data_path, exist_ok=True)
        
        print("Note: Actual Solana data collection requires:")
        print("- Solana RPC endpoint access")
        print("- Magic Eden API integration")
        print("- NFT metadata parsing")
        print("- Trading pattern analysis")
        
        # Generate synthetic data for demonstration
        self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> None:
        """Generate synthetic Solana NFT data for testing."""
        print("Generating synthetic Solana NFT data...")
        
        # Create synthetic user population
        num_users = 1000 if self.split == 'train' else 300
        user_ids = [f"solana_user_{i:04d}" for i in range(num_users)]
        
        # Create synthetic NFT collections
        collections = [
            {"id": "degenerate_apes", "name": "Degenerate Ape Academy", "size": 10000},
            {"id": "solana_monkeys", "name": "Solana Monkey Business", "size": 5000},
            {"id": "thugbirdz", "name": "Thugbirdz", "size": 3333},
            {"id": "catalina_whales", "name": "Catalina Whale Mixer", "size": 3333},
            {"id": "aurory", "name": "Aurory", "size": 10000}
        ]
        
        # Generate synthetic transactions
        transactions = []
        users_data = []
        nft_metadata = []
        labels = {}
        
        # Simulate airdrop hunter behavior around known events
        hunter_ratio = 0.15  # 15% hunters
        num_hunters = int(num_users * hunter_ratio)
        
        for i, user_id in enumerate(user_ids):
            is_hunter = i < num_hunters
            labels[user_id] = 1 if is_hunter else 0
            
            # Generate user profile
            if is_hunter:
                # Hunters: more active around airdrops, price manipulation patterns
                base_activity = np.random.uniform(20, 100)
                price_manipulation = True
            else:
                # Legitimate users: steady activity, organic trading
                base_activity = np.random.uniform(5, 30)
                price_manipulation = False
            
            users_data.append({
                'user_id': user_id,
                'first_seen': datetime(2021, 1, 1) + timedelta(days=np.random.randint(0, 365)),
                'total_transactions': int(base_activity),
                'avg_transaction_value': np.random.uniform(0.5, 10.0),
                'is_hunter': is_hunter
            })
            
            # Generate transactions for this user
            user_txs = self._generate_user_transactions(
                user_id, base_activity, is_hunter, collections, price_manipulation
            )
            transactions.extend(user_txs)
            
            # Generate NFT metadata for user's NFTs
            user_nfts = [tx['nft_id'] for tx in user_txs]
            for nft_id in set(user_nfts):
                if not any(nft['nft_id'] == nft_id for nft in nft_metadata):
                    nft_data = self._generate_nft_metadata(nft_id, collections)
                    nft_metadata.append(nft_data)
        
        # Convert to DataFrames
        self.transactions = pd.DataFrame(transactions)
        self.users = pd.DataFrame(users_data)
        self.nft_metadata = pd.DataFrame(nft_metadata)
        self.labels = labels
        self.airdrop_events = self.known_airdrop_events
        
        print(f"Generated {len(transactions)} transactions for {len(user_ids)} users")
        print(f"Hunter ratio: {hunter_ratio:.1%} ({num_hunters} hunters)")
    
    def _generate_user_transactions(self, user_id: str, base_activity: float, 
                                  is_hunter: bool, collections: List[Dict], 
                                  price_manipulation: bool) -> List[Dict]:
        """Generate transaction sequence for a user."""
        transactions = []
        
        # Time range for transactions
        start_time = datetime(2021, 6, 1)
        end_time = datetime(2023, 12, 31)
        total_days = (end_time - start_time).days
        
        # Generate transaction timestamps
        num_txs = max(5, int(np.random.poisson(base_activity)))
        
        if is_hunter:
            # Hunters cluster activity around airdrop events
            tx_times = self._generate_hunter_timestamps(start_time, total_days, num_txs)
        else:
            # Legitimate users have more uniform activity
            tx_times = self._generate_uniform_timestamps(start_time, total_days, num_txs)
        
        for i, tx_time in enumerate(tx_times):
            # Select collection (hunters prefer certain collections)
            if is_hunter and np.random.random() < 0.7:
                # Hunters focus on high-value collections
                collection = np.random.choice([c for c in collections if c['size'] >= 5000])
            else:
                collection = np.random.choice(collections)
            
            # Generate NFT ID
            nft_id = f"{collection['id']}_{np.random.randint(1, collection['size'])}"
            
            # Generate transaction details
            base_price = np.random.uniform(1.0, 20.0)  # SOL
            
            if price_manipulation and np.random.random() < 0.3:
                # Price manipulation patterns
                if i > 0 and np.random.random() < 0.5:
                    # Wash trading - similar to previous price
                    prev_price = transactions[-1]['price']
                    price = prev_price * np.random.uniform(0.95, 1.05)
                else:
                    # Artificial price inflation
                    price = base_price * np.random.uniform(2.0, 5.0)
            else:
                # Normal price variation
                price = base_price * np.random.uniform(0.7, 1.3)
            
            # Transaction type
            tx_types = ['buy', 'sell', 'bid', 'list']
            if is_hunter:
                # Hunters do more buying around airdrops
                tx_type = np.random.choice(tx_types, p=[0.5, 0.3, 0.1, 0.1])
            else:
                tx_type = np.random.choice(tx_types, p=[0.3, 0.3, 0.2, 0.2])
            
            # Gas fees (in SOL)
            gas_fee = np.random.uniform(0.000005, 0.00001)  # Solana has low fees
            
            transaction = {
                'user_id': user_id,
                'nft_id': nft_id,
                'collection_id': collection['id'],
                'timestamp': tx_time.timestamp(),
                'price': price,
                'gas_fee': gas_fee,
                'transaction_type': tx_type,
                'marketplace': self.marketplace,
                'signature': f"sig_{user_id}_{i}_{int(tx_time.timestamp())}"
            }
            
            transactions.append(transaction)
        
        return sorted(transactions, key=lambda x: x['timestamp'])
    
    def _generate_hunter_timestamps(self, start_time: datetime, total_days: int, num_txs: int) -> List[datetime]:
        """Generate timestamps clustered around airdrop events for hunters."""
        timestamps = []
        
        # Distribute transactions around airdrop events
        airdrop_dates = [datetime.fromtimestamp(ts) for ts in self.known_airdrop_events]
        airdrop_dates = [d for d in airdrop_dates if start_time <= d <= start_time + timedelta(days=total_days)]
        
        if not airdrop_dates:
            # No airdrops in range, fall back to uniform
            return self._generate_uniform_timestamps(start_time, total_days, num_txs)
        
        # 70% of transactions around airdrops, 30% background
        airdrop_txs = int(num_txs * 0.7)
        background_txs = num_txs - airdrop_txs
        
        # Airdrop-clustered transactions
        for _ in range(airdrop_txs):
            # Pick random airdrop event
            airdrop_date = np.random.choice(airdrop_dates)
            
            # Cluster around the event (±14 days, with stronger clustering ±7 days)
            if np.random.random() < 0.7:
                # Close to airdrop (±7 days)
                offset_days = np.random.uniform(-7, 7)
            else:
                # Further from airdrop (±14 days)
                offset_days = np.random.uniform(-14, 14)
            
            tx_time = airdrop_date + timedelta(days=offset_days)
            
            # Add some hour-level randomness
            tx_time += timedelta(hours=np.random.uniform(0, 24))
            
            timestamps.append(tx_time)
        
        # Background transactions
        for _ in range(background_txs):
            random_day = np.random.uniform(0, total_days)
            tx_time = start_time + timedelta(days=random_day)
            timestamps.append(tx_time)
        
        return sorted(timestamps)
    
    def _generate_uniform_timestamps(self, start_time: datetime, total_days: int, num_txs: int) -> List[datetime]:
        """Generate uniformly distributed timestamps for legitimate users."""
        timestamps = []
        
        for _ in range(num_txs):
            random_day = np.random.uniform(0, total_days)
            tx_time = start_time + timedelta(days=random_day)
            # Add hour-level randomness
            tx_time += timedelta(hours=np.random.uniform(0, 24))
            timestamps.append(tx_time)
        
        return sorted(timestamps)
    
    def _generate_nft_metadata(self, nft_id: str, collections: List[Dict]) -> Dict:
        """Generate synthetic NFT metadata."""
        collection_id = nft_id.split('_')[0] + '_' + nft_id.split('_')[1]
        collection = next((c for c in collections if c['id'] == collection_id), collections[0])
        
        # Generate synthetic attributes
        attributes = []
        num_attributes = np.random.randint(3, 8)
        
        attribute_types = ['background', 'body', 'clothing', 'eyes', 'hat', 'mouth', 'accessories']
        for attr_type in np.random.choice(attribute_types, num_attributes, replace=False):
            attributes.append({
                'trait_type': attr_type,
                'value': f'{attr_type}_{np.random.randint(1, 20)}',
                'rarity': np.random.uniform(0.01, 0.5)
            })
        
        # Compute rarity score
        rarity_score = sum(1/attr['rarity'] for attr in attributes)
        rarity_rank = np.random.randint(1, collection['size'])
        
        return {
            'nft_id': nft_id,
            'collection_id': collection['id'],
            'collection_name': collection['name'],
            'collection_size': collection['size'],
            'name': f"{collection['name']} #{nft_id.split('_')[-1]}",
            'description': f"A unique NFT from the {collection['name']} collection",
            'image_url': f"https://example.com/nfts/{nft_id}.png",
            'attributes': attributes,
            'rarity_score': rarity_score,
            'rarity_rank': rarity_rank,
            'creator_reputation': np.random.uniform(0.3, 0.9),
            'mint_date': datetime(2021, 1, 1) + timedelta(days=np.random.randint(0, 365))
        }
    
    def load_raw_data(self) -> None:
        """Load raw data from files or generate synthetic data."""
        if hasattr(self, 'transactions') and self.transactions is not None:
            return  # Already loaded
        
        # Try to load from files
        data_file = os.path.join(self.data_path, f'solana_{self.marketplace}_{self.split}.json')
        
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            self.transactions = pd.DataFrame(data['transactions'])
            self.users = pd.DataFrame(data['users'])
            self.nft_metadata = pd.DataFrame(data['nft_metadata'])
            self.labels = data['labels']
            self.airdrop_events = data.get('airdrop_events', self.known_airdrop_events)
        else:
            # Generate synthetic data
            self._generate_synthetic_data()
    
    def extract_transaction_features(self, user_id: str) -> Dict[str, torch.Tensor]:
        """Extract transaction features for a user."""
        user_txs = self.transactions[self.transactions['user_id'] == user_id].copy()
        user_txs = user_txs.sort_values('timestamp')
        
        if len(user_txs) == 0:
            # Return empty features
            return {key: torch.zeros(self.max_sequence_length) 
                   for key in ['prices', 'gas_fees', 'timestamps', 'transaction_types']}
        
        # Limit sequence length
        if len(user_txs) > self.max_sequence_length:
            user_txs = user_txs.tail(self.max_sequence_length)
        
        seq_len = len(user_txs)
        
        # Extract basic features
        prices = user_txs['price'].values
        gas_fees = user_txs['gas_fee'].values
        timestamps = user_txs['timestamp'].values
        
        # Transaction type encoding
        tx_type_mapping = {'buy': 0, 'sell': 1, 'bid': 2, 'list': 3}
        tx_types = [tx_type_mapping.get(t, 0) for t in user_txs['transaction_type']]
        
        # Derived features
        time_deltas = np.diff(timestamps, prepend=timestamps[0])
        
        # Price ratios
        price_ratios = np.ones(seq_len)
        if seq_len > 1:
            price_ratios[1:] = prices[1:] / (prices[:-1] + 1e-8)
        
        # Cumulative volume
        cumulative_volume = np.cumsum(prices)
        
        # Transaction frequency (transactions per day in rolling window)
        transaction_frequency = np.ones(seq_len)
        for i in range(seq_len):
            window_start = max(0, i - 6)  # 7-day window
            window_timestamps = timestamps[window_start:i+1]
            time_span_days = (timestamps[i] - timestamps[window_start]) / (24 * 3600) + 1
            transaction_frequency[i] = len(window_timestamps) / time_span_days
        
        # Pad sequences
        def pad_sequence(arr):
            padded = np.zeros(self.max_sequence_length)
            padded[:len(arr)] = arr
            return torch.tensor(padded, dtype=torch.float32)
        
        return {
            'prices': pad_sequence(np.log1p(prices)),
            'gas_fees': pad_sequence(np.log1p(gas_fees * self.sol_to_lamports)),  # Convert to lamports for scaling
            'timestamps': pad_sequence(timestamps),
            'transaction_types': pad_sequence(tx_types),
            'time_deltas': pad_sequence(np.log1p(time_deltas)),
            'price_ratios': pad_sequence(np.log(price_ratios + 1e-8)),
            'cumulative_volume': pad_sequence(np.log1p(cumulative_volume)),
            'transaction_frequency': pad_sequence(transaction_frequency),
            'sequence_length': torch.tensor(seq_len, dtype=torch.long)
        }
    
    def extract_nft_features(self, nft_id: str) -> Dict[str, torch.Tensor]:
        """Extract NFT multimodal features."""
        nft_data = self.nft_metadata[self.nft_metadata['nft_id'] == nft_id]
        
        if len(nft_data) == 0:
            return {
                'visual': torch.zeros(768),
                'text': torch.zeros(768),
                'metadata': torch.zeros(32)
            }
        
        nft_row = nft_data.iloc[0]
        
        # Simulate visual features (would use actual ViT in practice)
        visual_features = torch.randn(768)
        
        # Simulate text features (would use actual BERT in practice)
        text_features = torch.randn(768)
        
        # Extract metadata features
        metadata_features = torch.tensor([
            np.log1p(nft_row['collection_size']),
            nft_row['rarity_score'] / 1000,  # Normalize rarity score
            nft_row['rarity_rank'] / nft_row['collection_size'],  # Relative rarity
            nft_row['creator_reputation'],
            len(nft_row.get('attributes', [])),
        ], dtype=torch.float32)
        
        # Pad metadata features
        padded_metadata = torch.zeros(32)
        padded_metadata[:min(len(metadata_features), 32)] = metadata_features[:32]
        
        return {
            'visual': visual_features,
            'text': text_features,
            'metadata': padded_metadata
        }
    
    def build_user_graph(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Build user interaction graph based on NFT trading relationships."""
        # Create user-to-user edges based on shared NFT interactions
        user_interactions = defaultdict(list)
        
        # Group transactions by NFT
        for _, tx in self.transactions.iterrows():
            nft_id = tx['nft_id']
            user_id = tx['user_id']
            user_interactions[nft_id].append(user_id)
        
        # Create edges between users who traded the same NFTs
        edges = []
        edge_features = []
        
        users = sorted(self.transactions['user_id'].unique())
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        
        for nft_id, nft_users in user_interactions.items():
            unique_users = list(set(nft_users))
            
            # Create edges between all pairs of users who traded this NFT
            for i, user1 in enumerate(unique_users):
                for user2 in unique_users[i+1:]:
                    if user1 in user_to_idx and user2 in user_to_idx:
                        idx1, idx2 = user_to_idx[user1], user_to_idx[user2]
                        
                        # Add bidirectional edges
                        edges.extend([(idx1, idx2), (idx2, idx1)])
                        
                        # Compute edge features
                        edge_feature = self._compute_edge_features(user1, user2, [nft_id])
                        edge_features.extend([edge_feature, edge_feature])
        
        if not edges:
            # Create minimal graph with self-loops
            num_users = len(users)
            edges = [(i, i) for i in range(num_users)]
            edge_features = [torch.zeros(64) for _ in range(num_users)]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_features = torch.stack(edge_features)
        
        return edge_index, {'edge_features': edge_features}
    
    def _compute_edge_features(self, user1: str, user2: str, shared_nfts: List[str]) -> torch.Tensor:
        """Compute features for an edge between two users."""
        user1_txs = self.transactions[self.transactions['user_id'] == user1]
        user2_txs = self.transactions[self.transactions['user_id'] == user2]
        
        # Shared NFT statistics
        shared_nft_txs1 = user1_txs[user1_txs['nft_id'].isin(shared_nfts)]
        shared_nft_txs2 = user2_txs[user2_txs['nft_id'].isin(shared_nfts)]
        
        # Price similarity
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
            time_overlap = max(0, overlap_end - overlap_start) / (24 * 3600)  # Days
        
        # Trading pattern similarity
        pattern_similarity = 0
        if len(shared_nft_txs1) > 0 and len(shared_nft_txs2) > 0:
            # Compare transaction types
            types1 = set(shared_nft_txs1['transaction_type'])
            types2 = set(shared_nft_txs2['transaction_type'])
            pattern_similarity = len(types1 & types2) / len(types1 | types2) if types1 | types2 else 0
        
        features = torch.tensor([
            len(shared_nfts),
            np.log1p(avg_price_diff),
            time_overlap,
            pattern_similarity,
            # Additional features can be added here
        ], dtype=torch.float32)
        
        # Pad to standard edge feature size
        padded_features = torch.zeros(64)
        padded_features[:min(len(features), 64)] = features[:64]
        
        return padded_features
    
    def verify_data_integrity(self) -> bool:
        """Verify that the dataset is complete and valid."""
        try:
            # Check that all required components exist
            assert self.transactions is not None and len(self.transactions) > 0
            assert self.users is not None and len(self.users) > 0
            assert self.labels is not None and len(self.labels) > 0
            
            # Check data consistency
            user_ids_in_txs = set(self.transactions['user_id'].unique())
            user_ids_in_labels = set(self.labels.keys())
            assert user_ids_in_txs.issubset(user_ids_in_labels)
            
            # Check timestamp validity
            timestamps = self.transactions['timestamp']
            assert timestamps.min() > 0 and timestamps.max() < time.time() + 365*24*3600
            
            # Check price validity
            prices = self.transactions['price']
            assert prices.min() >= 0 and prices.max() < 1000  # Reasonable SOL prices
            
            print("✓ Solana dataset integrity check passed")
            return True
            
        except AssertionError as e:
            print(f"✗ Solana dataset integrity check failed: {e}")
            return False
        except Exception as e:
            print(f"✗ Solana dataset integrity check error: {e}")
            return False
    
    def save_dataset(self, output_path: Optional[str] = None) -> None:
        """Save the processed dataset to disk."""
        if output_path is None:
            output_path = os.path.join(self.data_path, f'solana_{self.marketplace}_{self.split}.json')
        
        data_to_save = {
            'transactions': self.transactions.to_dict('records'),
            'users': self.users.to_dict('records'),
            'nft_metadata': self.nft_metadata.to_dict('records'),
            'labels': self.labels,
            'airdrop_events': self.airdrop_events,
            'metadata': {
                'marketplace': self.marketplace,
                'split': self.split,
                'collection_filter': self.collection_filter,
                'time_range': self.time_range,
                'min_transaction_value': self.min_transaction_value,
                'created_at': datetime.now().isoformat()
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data_to_save, f, indent=2, default=str)
        
        print(f"Saved Solana dataset to {output_path}")
    
    @classmethod
    def create_splits(cls, data_path: str, train_ratio: float = 0.7, 
                     val_ratio: float = 0.15, test_ratio: float = 0.15,
                     **kwargs) -> Dict[str, 'SolanaNFTDataset']:
        """Create train/val/test splits of the dataset."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        # Create full dataset first
        full_dataset = cls(data_path, split='full', **kwargs)
        full_dataset.load_raw_data()
        
        # Split users
        all_users = list(full_dataset.labels.keys())
        np.random.shuffle(all_users)
        
        n_train = int(len(all_users) * train_ratio)
        n_val = int(len(all_users) * val_ratio)
        
        train_users = set(all_users[:n_train])
        val_users = set(all_users[n_train:n_train + n_val])
        test_users = set(all_users[n_train + n_val:])
        
        # Create split datasets
        splits = {}
        for split_name, user_set in [('train', train_users), ('val', val_users), ('test', test_users)]:
            dataset = cls(data_path, split=split_name, **kwargs)
            
            # Filter data for this split
            dataset.transactions = full_dataset.transactions[
                full_dataset.transactions['user_id'].isin(user_set)
            ].copy()
            dataset.users = full_dataset.users[
                full_dataset.users['user_id'].isin(user_set)
            ].copy()
            dataset.nft_metadata = full_dataset.nft_metadata.copy()  # Shared across splits
            dataset.labels = {k: v for k, v in full_dataset.labels.items() if k in user_set}
            dataset.airdrop_events = full_dataset.airdrop_events
            
            # Save split
            dataset.save_dataset()
            splits[split_name] = dataset
        
        return splits