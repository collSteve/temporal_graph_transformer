"""
Blur NFT marketplace dataset implementation for ARTEMIS comparison.

Implements the unified interface for Blur marketplace data, enabling direct
comparison between our Temporal Graph Transformer and the ARTEMIS baseline
on the same NFT marketplace ecosystem.

Target period: 2022-2024 (Blur's major growth period)
Focus: Wash trading detection, airdrop farming around BLUR token launch
"""

import torch
import pandas as pd
import numpy as np
import json
import os
import requests
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict
import warnings

from .base_dataset import BaseTemporalGraphDataset
from .preprocessing import TemporalGraphPreprocessor


class BlurNFTDataset(BaseTemporalGraphDataset):
    """
    Blur NFT marketplace dataset for direct ARTEMIS comparison.
    
    Focuses on the same type of NFT marketplace data that ARTEMIS was designed for,
    enabling fair comparison between our temporal approach and their static methods.
    """
    
    def __init__(self, 
                 data_path: str,
                 split: str = 'train',
                 marketplace: str = 'blur',
                 collection_filter: Optional[List[str]] = None,
                 time_range: Optional[Tuple[str, str]] = None,
                 min_transaction_value_eth: float = 0.01,
                 include_blur_specific_features: bool = True,
                 focus_on_wash_trading: bool = True,
                 **kwargs):
        """
        Args:
            data_path: Path to dataset directory
            split: 'train', 'val', or 'test'
            marketplace: Target marketplace (should be 'blur')
            collection_filter: Optional list of collection addresses to include
            time_range: Optional (start_date, end_date) in YYYY-MM-DD format
            min_transaction_value_eth: Minimum transaction value in ETH
            include_blur_specific_features: Whether to include Blur-specific features
            focus_on_wash_trading: Whether to focus on wash trading detection
        """
        super().__init__(data_path, split, **kwargs)
        
        self.marketplace = marketplace
        self.collection_filter = collection_filter
        self.time_range = time_range
        self.min_transaction_value_eth = min_transaction_value_eth
        self.include_blur_specific_features = include_blur_specific_features
        self.focus_on_wash_trading = focus_on_wash_trading
        
        # Blur-specific configuration
        self.blur_contract_address = '0x000000000000Ad05Ccc4F10045630fb830B95127'  # Blur marketplace
        self.blur_token_address = '0x5283d291dbcf85356a21ba090e6db59121208b44'    # BLUR token
        
        # Blur ecosystem events
        self.blur_milestones = {
            'launch': 1666137600,        # Oct 19, 2022 - Blur launch
            'airdrop_1': 1676505600,     # Feb 16, 2023 - First BLUR airdrop
            'airdrop_2': 1679270400,     # Mar 20, 2023 - Second BLUR airdrop
            'royalty_wars': 1672531200,  # Jan 1, 2023 - Royalty wars period
        }
        
        # Set airdrop events
        self.airdrop_events = [
            self.blur_milestones['airdrop_1'],
            self.blur_milestones['airdrop_2']
        ]
        
        # High-value NFT collections (common wash trading targets)
        self.target_collections = [
            {
                'id': 'cryptopunks',
                'name': 'CryptoPunks',
                'address': '0xb47e3cd837dDF8e4c57F05d70Ab865de6e193BBB',
                'size': 10000,
                'floor_price_range': (50, 500)  # ETH
            },
            {
                'id': 'bayc',
                'name': 'Bored Ape Yacht Club', 
                'address': '0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D',
                'size': 10000,
                'floor_price_range': (20, 200)
            },
            {
                'id': 'mayc',
                'name': 'Mutant Ape Yacht Club',
                'address': '0x60E4d786628Fea6478F785A6d7e704777c86a7c6',
                'size': 20000,
                'floor_price_range': (5, 50)
            },
            {
                'id': 'azuki',
                'name': 'Azuki',
                'address': '0xED5AF388653567Af2F388E6224dC7C4b3241C544',
                'size': 10000,
                'floor_price_range': (3, 30)
            },
            {
                'id': 'milady',
                'name': 'Milady Maker',
                'address': '0x5Af0D9827E0c53E4799BB226655A1de152A425a5',
                'size': 9999,
                'floor_price_range': (1, 15)
            }
        ]
        
        # Filter collections if specified
        if collection_filter:
            self.target_collections = [
                c for c in self.target_collections 
                if c['id'] in collection_filter or c['address'] in collection_filter
            ]
        
        self.preprocessor = TemporalGraphPreprocessor({
            'max_sequence_length': self.max_sequence_length,
            'min_transactions': 3,  # Lower threshold for NFT trading
            'airdrop_window_days': self.airdrop_window_days,
            'normalize_features': True,
            'include_nft_features': True,
            'graph_construction_method': 'transaction_based'
        })
        
        # Load data
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """Load the dataset, downloading if necessary."""
        dataset_file = os.path.join(self.data_path, f'blur_nft_{self.split}.json')
        
        if os.path.exists(dataset_file):
            print(f"Loading existing Blur dataset from {dataset_file}")
            self.load_raw_data()
        else:
            print(f"Dataset not found at {dataset_file}")
            print("To collect Blur data, you would need to:")
            print("1. Set up access to Ethereum RPC endpoints")
            print("2. Query Blur marketplace contract events")
            print("3. Extract NFT metadata and trading patterns")
            print("4. Label wash trading and legitimate activity")
            print("\\nFor now, generating synthetic data for ARTEMIS comparison...")
            self._generate_synthetic_data()
    
    def download_data(self) -> None:
        """Download Blur NFT data from various sources."""
        print("Downloading Blur NFT data...")
        
        # In practice, this would integrate with:
        # - Ethereum RPC providers (Alchemy, Infura, QuickNode)
        # - NFT metadata services (OpenSea API, Alchemy NFT API)
        # - Blur-specific APIs or subgraphs
        
        print("Real Blur data collection not yet implemented")
        print("Falling back to synthetic data generation...")
        self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> None:
        """Generate synthetic Blur marketplace data for ARTEMIS comparison."""
        print("Generating synthetic Blur NFT marketplace data...")
        
        # Generate realistic NFT marketplace activity
        num_users = 2000  # Larger for NFT marketplace
        base_transactions_per_user = 8
        
        users = []
        transactions = []
        nft_metadata = []
        labels = {}
        
        # Generate NFT metadata first
        all_nfts = []
        for collection in self.target_collections:
            collection_nfts = self._generate_collection_nfts(collection)
            all_nfts.extend(collection_nfts)
            nft_metadata.extend(collection_nfts)
        
        # Generate users and their trading behavior
        for i in range(num_users):
            user_id = f"blur_user_{i:04d}"
            
            # Determine user type and wash trading propensity
            user_type = self._determine_blur_user_type(i)
            is_wash_trader = user_type in ['wash_trader', 'sophisticated_wash_trader']
            
            users.append({
                'user_id': user_id,
                'user_type': user_type
            })
            
            labels[user_id] = 1 if is_wash_trader else 0
            
            # Generate transactions for this user
            user_transactions = self._generate_blur_user_transactions(
                user_id, base_transactions_per_user, user_type, all_nfts
            )
            transactions.extend(user_transactions)
        
        # Convert to DataFrames
        users_df = pd.DataFrame(users)
        transactions_df = pd.DataFrame(transactions)
        nft_metadata_df = pd.DataFrame(nft_metadata)
        
        # Save synthetic data
        dataset = {
            'transactions': transactions_df.to_dict(orient='records'),
            'users': users_df.to_dict(orient='records'),
            'nft_metadata': nft_metadata_df.to_dict(orient='records'),
            'labels': labels,
            'metadata': {
                'data_type': 'synthetic_blur',
                'generation_date': datetime.now().isoformat(),
                'marketplace': self.marketplace,
                'total_transactions': len(transactions_df),
                'total_users': len(users_df),
                'total_nfts': len(nft_metadata_df),
                'wash_trader_ratio': sum(labels.values()) / len(labels),
                'collections': [c['id'] for c in self.target_collections]
            }
        }
        
        dataset_file = os.path.join(self.data_path, f'blur_nft_{self.split}.json')
        os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
        
        with open(dataset_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Generated synthetic Blur dataset with {len(transactions)} transactions")
        
        # Load the generated data
        self.load_raw_data()
    
    def _generate_collection_nfts(self, collection: Dict) -> List[Dict]:
        """Generate NFT metadata for a collection."""
        nfts = []
        collection_size = min(collection['size'], 100)  # Limit for demo
        
        for token_id in range(1, collection_size + 1):
            nft = {
                'nft_id': f"{collection['id']}_{token_id}",
                'collection_id': collection['id'],
                'collection_name': collection['name'],
                'collection_address': collection['address'],
                'token_id': token_id,
                'name': f"{collection['name']} #{token_id}",
                'description': f"NFT #{token_id} from {collection['name']} collection",
                'image_url': f"https://example.com/{collection['id']}/{token_id}.png",
                'attributes': self._generate_nft_attributes(collection['id']),
                'rarity_score': np.random.uniform(0.1, 100.0),
                'collection_size': collection['size']
            }
            nfts.append(nft)
        
        return nfts
    
    def _generate_nft_attributes(self, collection_id: str) -> List[Dict]:
        """Generate realistic NFT attributes."""
        if collection_id == 'cryptopunks':
            attributes = [
                {'trait_type': 'Type', 'value': np.random.choice(['Alien', 'Ape', 'Human', 'Zombie'])},
                {'trait_type': 'Accessory', 'value': np.random.choice(['None', 'Hat', 'Glasses', 'Pipe'])}
            ]
        elif collection_id == 'bayc':
            attributes = [
                {'trait_type': 'Fur', 'value': np.random.choice(['Brown', 'Golden', 'Black', 'White'])},
                {'trait_type': 'Eyes', 'value': np.random.choice(['Bored', 'Angry', 'Sleepy', 'Surprised'])},
                {'trait_type': 'Mouth', 'value': np.random.choice(['Grin', 'Smile', 'Frown', 'Tongue Out'])}
            ]
        else:
            # Generic attributes
            attributes = [
                {'trait_type': 'Background', 'value': np.random.choice(['Blue', 'Red', 'Green', 'Purple'])},
                {'trait_type': 'Rarity', 'value': np.random.choice(['Common', 'Rare', 'Epic', 'Legendary'])}
            ]
        
        return attributes
    
    def _determine_blur_user_type(self, user_index: int) -> str:
        """Determine user behavior type for Blur marketplace."""
        # Distribution based on estimated Blur marketplace patterns
        if user_index < 50:
            return 'sophisticated_wash_trader'  # ~2.5% - Advanced wash trading
        elif user_index < 200:
            return 'wash_trader'               # ~7.5% - Basic wash trading  
        elif user_index < 300:
            return 'bot_trader'                # ~5% - Automated trading
        elif user_index < 500:
            return 'flipper'                   # ~10% - Professional flippers
        elif user_index < 1000:
            return 'collector'                 # ~25% - Serious collectors
        else:
            return 'casual_user'               # ~50% - Casual users
    
    def _generate_blur_user_transactions(self, user_id: str, base_count: int, user_type: str, available_nfts: List[Dict]) -> List[Dict]:
        """Generate Blur marketplace transactions for a user."""
        transactions = []
        
        # Adjust transaction count based on user type
        if user_type == 'sophisticated_wash_trader':
            tx_count = int(np.random.poisson(25))  # High activity
        elif user_type == 'wash_trader':
            tx_count = int(np.random.poisson(15))
        elif user_type == 'bot_trader':
            tx_count = int(np.random.poisson(20))
        elif user_type == 'flipper':
            tx_count = int(np.random.poisson(12))
        elif user_type == 'collector':
            tx_count = int(np.random.poisson(8))
        else:  # casual_user
            tx_count = int(np.random.poisson(3))
        
        # Time range
        start_time = datetime(2022, 10, 19)  # Blur launch
        end_time = datetime(2024, 6, 1)
        
        # Track user's NFT inventory for realistic trading
        user_inventory = set()
        
        for i in range(tx_count):
            # Generate timestamp
            if user_type in ['wash_trader', 'sophisticated_wash_trader']:
                # Wash traders cluster activity around airdrops
                time_weight = self._get_airdrop_weighted_time()
            else:
                time_weight = np.random.random()
            
            timestamp = start_time + timedelta(
                seconds=time_weight * (end_time - start_time).total_seconds()
            )
            
            # Choose NFT based on user type and wash trading patterns
            nft = self._choose_nft_for_transaction(user_type, available_nfts, user_inventory)
            
            # Determine transaction type and execute
            tx_type, price_eth = self._determine_blur_transaction(
                user_type, nft, user_inventory, i
            )
            
            # Update inventory
            if tx_type == 'buy':
                user_inventory.add(nft['nft_id'])
            elif tx_type == 'sell' and nft['nft_id'] in user_inventory:
                user_inventory.remove(nft['nft_id'])
            
            # Generate gas fee (higher for NFT transactions)
            gas_fee = np.random.uniform(0.005, 0.02)  # ETH
            
            transaction = {
                'user_id': user_id,
                'nft_id': nft['nft_id'],
                'collection_id': nft['collection_id'],
                'timestamp': timestamp.timestamp(),
                'price': price_eth,
                'gas_fee': gas_fee,
                'transaction_type': tx_type,
                'marketplace': 'blur',
                'signature': f"blur_tx_{user_id}_{i}_{int(timestamp.timestamp())}"
            }
            
            transactions.append(transaction)
        
        return sorted(transactions, key=lambda x: x['timestamp'])
    
    def _get_airdrop_weighted_time(self) -> float:
        """Get time weight that clusters around airdrop events."""
        # Choose which airdrop to cluster around
        airdrop_choice = np.random.choice(['airdrop_1', 'airdrop_2'], p=[0.6, 0.4])
        airdrop_timestamp = self.blur_milestones[airdrop_choice]
        
        # Create time around airdrop (±30 days)
        airdrop_date = datetime.fromtimestamp(airdrop_timestamp)
        start_time = datetime(2022, 10, 19)
        end_time = datetime(2024, 6, 1)
        
        # Weight towards 30 days before airdrop
        target_date = airdrop_date - timedelta(days=np.random.uniform(0, 60))
        
        # Convert back to weight
        total_seconds = (end_time - start_time).total_seconds()
        target_seconds = (target_date - start_time).total_seconds()
        
        return max(0, min(1, target_seconds / total_seconds))
    
    def _choose_nft_for_transaction(self, user_type: str, available_nfts: List[Dict], user_inventory: set) -> Dict:
        """Choose NFT based on user behavior patterns."""
        if user_type in ['wash_trader', 'sophisticated_wash_trader']:
            # Wash traders prefer high-value collections for maximum impact
            high_value_collections = ['cryptopunks', 'bayc', 'mayc']
            preferred_nfts = [
                nft for nft in available_nfts 
                if nft['collection_id'] in high_value_collections
            ]
            
            if preferred_nfts:
                return np.random.choice(preferred_nfts)
        
        elif user_type == 'bot_trader':
            # Bots focus on liquid collections
            liquid_collections = ['bayc', 'mayc', 'azuki']
            liquid_nfts = [
                nft for nft in available_nfts
                if nft['collection_id'] in liquid_collections
            ]
            
            if liquid_nfts:
                return np.random.choice(liquid_nfts)
        
        elif user_type == 'flipper':
            # Flippers look for undervalued pieces
            return np.random.choice(available_nfts)
        
        elif user_type == 'collector':
            # Collectors focus on specific collections
            favorite_collection = np.random.choice([c['id'] for c in self.target_collections])
            collection_nfts = [
                nft for nft in available_nfts
                if nft['collection_id'] == favorite_collection
            ]
            
            if collection_nfts:
                return np.random.choice(collection_nfts)
        
        # Default: random selection
        return np.random.choice(available_nfts)
    
    def _determine_blur_transaction(self, user_type: str, nft: Dict, user_inventory: set, tx_index: int) -> Tuple[str, float]:
        """Determine transaction type and price for Blur marketplace."""
        collection_id = nft['collection_id']
        
        # Get collection price range
        collection_info = next((c for c in self.target_collections if c['id'] == collection_id), None)
        if collection_info:
            min_price, max_price = collection_info['floor_price_range']
        else:
            min_price, max_price = (0.1, 10.0)  # Default range
        
        # Base price with some randomness
        base_price = np.random.uniform(min_price, max_price)
        
        # Determine transaction type
        if user_type in ['wash_trader', 'sophisticated_wash_trader']:
            # Wash traders create artificial activity
            if nft['nft_id'] in user_inventory:
                tx_type = 'sell'
                # Wash traders often sell at similar prices to create volume
                if user_type == 'sophisticated_wash_trader':
                    price_eth = base_price * np.random.uniform(0.98, 1.02)  # Tight range
                else:
                    price_eth = base_price * np.random.uniform(0.95, 1.05)
            else:
                tx_type = 'buy'
                price_eth = base_price * np.random.uniform(0.9, 1.1)
        
        elif user_type == 'bot_trader':
            # Bots arbitrage and provide liquidity
            tx_type = np.random.choice(['buy', 'sell'])
            price_eth = base_price * np.random.uniform(0.92, 1.08)  # Tighter spreads
        
        elif user_type == 'flipper':
            # Flippers buy low, sell high
            if nft['nft_id'] in user_inventory and tx_index > 2:
                tx_type = 'sell'
                price_eth = base_price * np.random.uniform(1.1, 1.5)  # Higher sell price
            else:
                tx_type = 'buy'
                price_eth = base_price * np.random.uniform(0.8, 0.95)  # Lower buy price
        
        elif user_type == 'collector':
            # Collectors mostly buy and hold
            if np.random.random() < 0.8:
                tx_type = 'buy'
                price_eth = base_price * np.random.uniform(0.95, 1.2)  # Willing to pay premium
            else:
                tx_type = 'sell'
                price_eth = base_price * np.random.uniform(1.0, 1.3)
        
        else:  # casual_user
            # Random trading behavior
            tx_type = np.random.choice(['buy', 'sell'])
            price_eth = base_price * np.random.uniform(0.8, 1.3)
        
        return tx_type, max(0.01, price_eth)  # Minimum price
    
    def load_raw_data(self) -> None:
        """Load raw Blur marketplace data from saved files."""
        dataset_file = os.path.join(self.data_path, f'blur_nft_{self.split}.json')
        
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        
        # Load components
        self.transactions = pd.DataFrame(dataset['transactions'])
        self.users = pd.DataFrame(dataset['users'])
        
        if 'nft_metadata' in dataset:
            self.nft_metadata = pd.DataFrame(dataset['nft_metadata'])
        else:
            self.nft_metadata = pd.DataFrame()
        
        self.labels = dataset['labels']
        
        print(f"Loaded Blur dataset: {len(self.transactions)} transactions, {len(self.users)} users")
        
        if 'metadata' in dataset:
            metadata = dataset['metadata']
            print(f"Collections: {metadata.get('collections', [])}")
            print(f"Wash trader ratio: {metadata.get('wash_trader_ratio', 0):.3f}")
    
    def extract_transaction_features(self, user_id: str) -> Dict[str, torch.Tensor]:
        """Extract transaction features for Blur marketplace (NFT-focused)."""
        user_txs = self.transactions[self.transactions['user_id'] == user_id].copy()
        user_txs = user_txs.sort_values('timestamp')
        
        # Limit sequence length
        if len(user_txs) > self.max_sequence_length:
            user_txs = user_txs.tail(self.max_sequence_length)
        
        seq_len = len(user_txs)
        if seq_len == 0:
            return self._get_empty_nft_features()
        
        # NFT-specific transaction features
        prices = user_txs['price'].values
        gas_fees = user_txs['gas_fee'].values
        timestamps = user_txs['timestamp'].values
        
        # Transaction type encoding for NFT markets
        nft_tx_types = {'buy': 0, 'sell': 1, 'bid': 2, 'list': 3, 'transfer': 4}
        tx_type_ids = [nft_tx_types.get(t, 0) for t in user_txs['transaction_type']]
        
        # Collection encoding
        collections = user_txs['collection_id']
        collection_to_id = {c['id']: i for i, c in enumerate(self.target_collections)}
        collection_ids = [collection_to_id.get(c, 0) for c in collections]
        
        # NFT-specific features
        time_deltas = np.diff(timestamps, prepend=timestamps[0])
        time_of_day = (timestamps % (24 * 3600)) / (24 * 3600)
        day_of_week = ((timestamps // (24 * 3600)) % 7) / 7
        
        # Price patterns (important for wash trading detection)
        price_ratios = np.ones(seq_len)
        if seq_len > 1:
            price_ratios[1:] = prices[1:] / (prices[:-1] + 1e-8)
        
        # Round number detection (wash traders often use round numbers)
        round_number_indicators = []
        for price in prices:
            # Check if price is suspiciously round
            if price == round(price) or price == round(price, 1):
                round_number_indicators.append(1.0)
            else:
                round_number_indicators.append(0.0)
        round_number_indicators = np.array(round_number_indicators)
        
        # Collection diversity
        collection_diversity = []
        for i in range(seq_len):
            window_start = max(0, i - 4)  # 5-transaction window
            window_collections = collections.iloc[window_start:i+1]
            unique_collections = window_collections.nunique()
            collection_diversity.append(unique_collections)
        collection_diversity = np.array(collection_diversity)
        
        # Buy/sell pattern analysis
        buy_sell_pattern = []
        for i, tx_type in enumerate(user_txs['transaction_type']):
            if i == 0:
                buy_sell_pattern.append(0.0)
            else:
                prev_type = user_txs['transaction_type'].iloc[i-1]
                # Suspicious: immediate buy-sell or sell-buy
                if (tx_type == 'sell' and prev_type == 'buy') or (tx_type == 'buy' and prev_type == 'sell'):
                    buy_sell_pattern.append(1.0)
                else:
                    buy_sell_pattern.append(0.0)
        buy_sell_pattern = np.array(buy_sell_pattern)
        
        # Pad sequences
        def pad_sequence(arr):
            padded = np.zeros(self.max_sequence_length)
            padded[:len(arr)] = arr
            return torch.tensor(padded, dtype=torch.float32)
        
        return {
            'prices': pad_sequence(np.log1p(prices)),
            'gas_fees': pad_sequence(np.log1p(gas_fees)),
            'timestamps': pad_sequence(timestamps),
            'transaction_types': pad_sequence(tx_type_ids),
            'collection_ids': pad_sequence(collection_ids),
            'time_deltas': pad_sequence(np.log1p(time_deltas)),
            'time_of_day': pad_sequence(time_of_day),
            'day_of_week': pad_sequence(day_of_week),
            'price_ratios': pad_sequence(np.log(price_ratios + 1e-8)),
            'round_number_indicators': pad_sequence(round_number_indicators),
            'collection_diversity': pad_sequence(collection_diversity),
            'buy_sell_pattern': pad_sequence(buy_sell_pattern),
            'sequence_length': torch.tensor(seq_len, dtype=torch.long)
        }
    
    def extract_nft_features(self, nft_id: str) -> Dict[str, torch.Tensor]:
        """Extract NFT multimodal features (visual + textual)."""
        if self.nft_metadata is None or len(self.nft_metadata) == 0:
            return {
                'visual_features': torch.zeros(768),    # Placeholder ViT features
                'text_features': torch.zeros(768),      # Placeholder BERT features
                'metadata_features': torch.zeros(32)    # Basic metadata
            }
        
        nft_data = self.nft_metadata[self.nft_metadata['nft_id'] == nft_id]
        
        if len(nft_data) == 0:
            return {
                'visual_features': torch.zeros(768),
                'text_features': torch.zeros(768),
                'metadata_features': torch.zeros(32)
            }
        
        nft = nft_data.iloc[0]
        
        # Placeholder multimodal features (in practice, would use ViT + BERT)
        visual_features = torch.randn(768)  # Would extract from image_url
        
        # Text features from name + description
        text_content = f"{nft['name']} {nft['description']}"
        text_features = torch.randn(768)  # Would extract with BERT
        
        # Metadata features
        metadata_features = torch.tensor([
            float(nft.get('token_id', 0)),
            float(nft.get('rarity_score', 0)),
            float(nft.get('collection_size', 0)),
            len(nft.get('attributes', [])),
        ], dtype=torch.float32)
        
        # Pad metadata features
        padded_metadata = torch.zeros(32)
        padded_metadata[:min(len(metadata_features), 32)] = metadata_features[:32]
        
        return {
            'visual_features': visual_features,
            'text_features': text_features,
            'metadata_features': padded_metadata
        }
    
    def build_user_graph(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Build user interaction graph based on NFT marketplace relationships."""
        # Find users who traded the same NFTs (wash trading indicator)
        user_pairs = defaultdict(list)
        
        for nft_id in self.transactions['nft_id'].unique():
            nft_traders = self.transactions[self.transactions['nft_id'] == nft_id]['user_id'].unique()
            
            # Create edges between users who traded the same NFT
            for i, user1 in enumerate(nft_traders):
                for user2 in nft_traders[i+1:]:
                    user_pairs[(user1, user2)].append(nft_id)
        
        # Build edge index and features
        edges = []
        edge_features = []
        
        user_to_idx = {user: idx for idx, user in enumerate(self.users['user_id'])}
        
        for (user1, user2), shared_nfts in user_pairs.items():
            if user1 in user_to_idx and user2 in user_to_idx:
                edges.extend([(user_to_idx[user1], user_to_idx[user2]),
                            (user_to_idx[user2], user_to_idx[user1])])
                
                # Compute edge features (important for wash trading detection)
                edge_feature = self._compute_nft_edge_features(user1, user2, shared_nfts)
                edge_features.extend([edge_feature, edge_feature])
        
        if not edges:
            num_users = len(user_to_idx)
            edges = [(i, i) for i in range(num_users)]
            edge_features = [torch.zeros(64) for _ in range(num_users)]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_features = torch.stack(edge_features)
        
        return edge_index, {'edge_features': edge_features}
    
    def _compute_nft_edge_features(self, user1: str, user2: str, shared_nfts: List[str]) -> torch.Tensor:
        """Compute edge features for NFT marketplace relationships."""
        user1_txs = self.transactions[self.transactions['user_id'] == user1]
        user2_txs = self.transactions[self.transactions['user_id'] == user2]
        
        # Shared NFT features
        num_shared_nfts = len(shared_nfts)
        
        # Wash trading indicators
        wash_trading_score = 0.0
        
        for nft_id in shared_nfts:
            nft_txs1 = user1_txs[user1_txs['nft_id'] == nft_id].sort_values('timestamp')
            nft_txs2 = user2_txs[user2_txs['nft_id'] == nft_id].sort_values('timestamp')
            
            # Check for suspiciously close transactions
            for _, tx1 in nft_txs1.iterrows():
                for _, tx2 in nft_txs2.iterrows():
                    time_diff = abs(tx1['timestamp'] - tx2['timestamp'])
                    price_diff = abs(tx1['price'] - tx2['price']) / max(tx1['price'], tx2['price'])
                    
                    # Suspicious if transactions are very close in time with similar prices
                    if time_diff < 3600 and price_diff < 0.05:  # 1 hour, 5% price difference
                        wash_trading_score += 1.0
        
        # Normalize wash trading score
        wash_trading_score = wash_trading_score / max(num_shared_nfts, 1)
        
        # Price correlation
        shared_nft_prices1 = []
        shared_nft_prices2 = []
        
        for nft_id in shared_nfts:
            nft_txs1 = user1_txs[user1_txs['nft_id'] == nft_id]
            nft_txs2 = user2_txs[user2_txs['nft_id'] == nft_id]
            
            if len(nft_txs1) > 0 and len(nft_txs2) > 0:
                shared_nft_prices1.append(nft_txs1['price'].mean())
                shared_nft_prices2.append(nft_txs2['price'].mean())
        
        price_correlation = 0.0
        if len(shared_nft_prices1) > 1:
            correlation = np.corrcoef(shared_nft_prices1, shared_nft_prices2)[0, 1]
            price_correlation = correlation if not np.isnan(correlation) else 0.0
        
        # Temporal overlap
        temporal_overlap = 0.0
        if len(user1_txs) > 0 and len(user2_txs) > 0:
            min_time1, max_time1 = user1_txs['timestamp'].min(), user1_txs['timestamp'].max()
            min_time2, max_time2 = user2_txs['timestamp'].min(), user2_txs['timestamp'].max()
            
            overlap_start = max(min_time1, min_time2)
            overlap_end = min(max_time1, max_time2)
            temporal_overlap = max(0, overlap_end - overlap_start)
        
        # Combine features
        features = torch.tensor([
            num_shared_nfts,
            wash_trading_score,
            price_correlation,
            np.log1p(temporal_overlap)
        ], dtype=torch.float32)
        
        # Pad to standard size
        padded_features = torch.zeros(64)
        padded_features[:min(len(features), 64)] = features[:64]
        
        return padded_features
    
    def _get_empty_nft_features(self) -> Dict[str, torch.Tensor]:
        """Return empty features when user has no transactions."""
        feature_names = [
            'prices', 'gas_fees', 'timestamps', 'transaction_types', 'collection_ids',
            'time_deltas', 'time_of_day', 'day_of_week', 'price_ratios', 
            'round_number_indicators', 'collection_diversity', 'buy_sell_pattern'
        ]
        
        empty_features = {}
        for name in feature_names:
            empty_features[name] = torch.zeros(self.max_sequence_length, dtype=torch.float32)
        
        empty_features['sequence_length'] = torch.tensor(0, dtype=torch.long)
        return empty_features
    
    def verify_data_integrity(self) -> bool:
        """Verify that Blur NFT data is complete and valid."""
        if self.transactions is None or len(self.transactions) == 0:
            return False
        
        required_columns = [
            'user_id', 'nft_id', 'timestamp', 'price', 'transaction_type'
        ]
        
        # Check required columns exist
        for col in required_columns:
            if col not in self.transactions.columns:
                warnings.warn(f"Missing required column: {col}")
                return False
        
        # Check data validity
        if self.transactions['price'].isna().any():
            warnings.warn("Found NaN values in price")
            return False
        
        if (self.transactions['price'] <= 0).any():
            warnings.warn("Found non-positive values in price")
            return False
        
        return True