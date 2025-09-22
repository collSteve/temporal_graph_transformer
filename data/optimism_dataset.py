"""
Optimism DeFi dataset implementation for longitudinal airdrop analysis.

Analyzes multiple airdrop rounds on Optimism to understand farming evolution
across different waves of token distribution. Supports Optimism's multi-round
airdrop strategy and unique L2 ecosystem characteristics.

Primary protocols: Uniswap V3, Synthetix, Aave, Velodrome
Target period: Multiple airdrop rounds (2022-2024)
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

from .pure_crypto_dataset import PureCryptoDataset
from .transaction_schema import OptimismDeFiTransaction, TransactionSchemaValidator, transactions_to_dataframe
from .preprocessing import TemporalGraphPreprocessor


class OptimismDataset(PureCryptoDataset):
    """
    Optimism DeFi dataset for multi-round airdrop longitudinal analysis.
    
    Focuses on understanding how airdrop farming strategies evolved across
    multiple OP token distribution rounds and the unique Optimism ecosystem.
    """
    
    def __init__(self, 
                 data_path: str,
                 split: str = 'train',
                 protocols: Optional[List[str]] = None,
                 airdrop_rounds: Optional[List[int]] = None,
                 start_date: str = '2022-01-01',
                 end_date: str = '2024-06-01',
                 alchemy_api_key: Optional[str] = None,
                 longitudinal_analysis: bool = True,
                 **kwargs):
        """
        Args:
            data_path: Path to dataset directory
            split: 'train', 'val', or 'test'
            protocols: List of Optimism protocols to include
            airdrop_rounds: List of airdrop rounds to analyze (1, 2, 3, 4)
            start_date: Start date for data collection
            end_date: End date for data collection
            alchemy_api_key: Alchemy RPC API key for Optimism
            longitudinal_analysis: Whether to perform cross-round analysis
        """
        
        # Set Optimism-specific defaults
        if protocols is None:
            protocols = ['uniswap_v3', 'synthetix', 'aave', 'velodrome']
        
        if airdrop_rounds is None:
            airdrop_rounds = [1, 2, 3, 4]  # All rounds
        
        super().__init__(
            data_path=data_path,
            split=split,
            blockchain='optimism',
            protocols=protocols,
            **kwargs
        )
        
        self.airdrop_rounds = airdrop_rounds
        self.start_date = start_date
        self.end_date = end_date
        self.alchemy_api_key = alchemy_api_key
        self.longitudinal_analysis = longitudinal_analysis
        
        # Optimism airdrop events (multiple rounds)
        self.airdrop_round_dates = {
            1: 1654041600,  # June 1, 2022 - First airdrop
            2: 1675209600,  # Feb 1, 2023 - Second airdrop  
            3: 1693526400,  # Sep 1, 2023 - Third airdrop
            4: 1706745600   # Feb 1, 2024 - Fourth airdrop
        }
        
        # Set airdrop events based on selected rounds
        self.airdrop_events = [
            self.airdrop_round_dates[round_num] 
            for round_num in self.airdrop_rounds 
            if round_num in self.airdrop_round_dates
        ]
        
        # Protocol configurations for Optimism
        self.protocol_configs = {
            'uniswap_v3': {
                'contract_address': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
                'subgraph_endpoint': 'optimism-uniswap-v3',
                'transaction_types': ['swap', 'mint', 'burn']
            },
            'synthetix': {
                'contract_address': '0x8700dAec35aF8Ff88c16BdF0418774CB3D7599B4',
                'subgraph_endpoint': 'synthetix-optimism',
                'transaction_types': ['exchange', 'mint', 'burn']
            },
            'aave': {
                'contract_address': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
                'subgraph_endpoint': 'aave-v3-optimism',
                'transaction_types': ['deposit', 'withdraw', 'borrow', 'repay']
            },
            'velodrome': {
                'contract_address': '0x9c12939390052919aF3155f41Bf4160Fd3666A6e',
                'subgraph_endpoint': 'velodrome-optimism',
                'transaction_types': ['swap', 'add_liquidity', 'remove_liquidity']
            }
        }
        
        # Transaction schema validator
        self.schema_validator = TransactionSchemaValidator()
        
        # Load dataset
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """Load the dataset, collecting from APIs if necessary."""
        dataset_file = os.path.join(self.data_path, f'optimism_defi_{self.split}.json')
        
        if os.path.exists(dataset_file) and not self._should_refresh_data():
            print(f"Loading existing Optimism dataset from {dataset_file}")
            self.load_raw_data()
        else:
            print(f"Dataset not found or needs refresh at {dataset_file}")
            if self.alchemy_api_key:
                print("Collecting real data from Optimism APIs...")
                self.download_data()
            else:
                print("No API keys provided. Generating demonstration data...")
                print("To collect real data, provide alchemy_api_key")
                self._generate_demonstration_data()
    
    def _should_refresh_data(self) -> bool:
        """Check if data should be refreshed."""
        dataset_file = os.path.join(self.data_path, f'optimism_defi_{self.split}.json')
        
        if not os.path.exists(dataset_file):
            return True
        
        # Check file age (refresh if older than 7 days)
        file_age = time.time() - os.path.getmtime(dataset_file)
        if file_age > 7 * 24 * 3600:
            return True
        
        return False
    
    def download_data(self) -> None:
        """Download real Optimism DeFi data."""
        print("Starting Optimism DeFi data collection...")
        
        all_transactions = []
        
        # Collect data for each airdrop round
        for round_num in self.airdrop_rounds:
            if round_num not in self.airdrop_round_dates:
                continue
                
            print(f"Collecting data for airdrop round {round_num}...")
            
            # Define time window around airdrop
            airdrop_date = datetime.fromtimestamp(self.airdrop_round_dates[round_num])
            start_window = airdrop_date - timedelta(days=90)  # 3 months before
            end_window = airdrop_date + timedelta(days=30)    # 1 month after
            
            round_transactions = self._collect_round_data(round_num, start_window, end_window)
            all_transactions.extend(round_transactions)
        
        if not all_transactions:
            print("No data collected from APIs, falling back to demonstration data")
            self._generate_demonstration_data()
            return
        
        # Convert to DataFrame and validate
        transactions_df = transactions_to_dataframe(all_transactions)
        
        # Validate schema
        is_valid, errors = self.schema_validator.validate_dataframe(transactions_df)
        if not is_valid:
            print(f"Data validation errors: {errors}")
        
        # Generate users and labels based on longitudinal analysis
        users_df = self._generate_users_from_transactions(transactions_df)
        labels = self._generate_longitudinal_labels(users_df, transactions_df)
        
        # Save collected data
        dataset = {
            'transactions': transactions_df.to_dict(orient='records'),
            'users': users_df.to_dict(orient='records'),
            'labels': labels,
            'metadata': {
                'collection_date': datetime.now().isoformat(),
                'start_date': self.start_date,
                'end_date': self.end_date,
                'airdrop_rounds': self.airdrop_rounds,
                'protocols': self.protocols,
                'total_transactions': len(transactions_df),
                'total_users': len(users_df),
                'hunter_addresses': len([u for u in labels.values() if u == 1])
            }
        }
        
        dataset_file = os.path.join(self.data_path, f'optimism_defi_{self.split}.json')
        os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
        
        with open(dataset_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Saved {len(transactions_df)} transactions to {dataset_file}")
        
        # Load the saved data
        self.load_raw_data()
    
    def _collect_round_data(self, round_num: int, start_date: datetime, end_date: datetime) -> List[OptimismDeFiTransaction]:
        """Collect transaction data for a specific airdrop round."""
        transactions = []
        
        # In practice, this would use Alchemy RPC or subgraph data
        # For now, return empty list as this requires complex RPC calls
        print(f"Real data collection for round {round_num} not yet implemented")
        
        return transactions
    
    def _generate_demonstration_data(self) -> None:
        """Generate demonstration data showing longitudinal farming evolution."""
        print("Generating demonstration Optimism longitudinal data...")
        
        # Generate more users to show evolution across rounds
        num_users = 1500
        base_transactions_per_user = 12
        
        users = []
        transactions = []
        labels = {}
        
        for i in range(num_users):
            user_id = f"optimism_user_{i:04d}"
            
            # Determine if user is a hunter and their sophistication level
            hunter_type = self._determine_hunter_type(i)
            is_hunter = hunter_type != 'legitimate'
            
            users.append({'user_id': user_id, 'hunter_type': hunter_type})
            labels[user_id] = 1 if is_hunter else 0
            
            # Generate transactions across multiple rounds
            user_transactions = self._generate_longitudinal_user_transactions(
                user_id, base_transactions_per_user, hunter_type
            )
            transactions.extend(user_transactions)
        
        # Convert to DataFrames
        users_df = pd.DataFrame(users)
        transactions_df = transactions_to_dataframe(transactions)
        
        # Save demonstration data
        dataset = {
            'transactions': transactions_df.to_dict(orient='records'),
            'users': users_df.to_dict(orient='records'),
            'labels': labels,
            'metadata': {
                'data_type': 'demonstration',
                'generation_date': datetime.now().isoformat(),
                'total_transactions': len(transactions_df),
                'total_users': len(users_df),
                'airdrop_rounds': self.airdrop_rounds,
                'hunter_types': users_df['hunter_type'].value_counts().to_dict()
            }
        }
        
        dataset_file = os.path.join(self.data_path, f'optimism_defi_{self.split}.json')
        os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
        
        with open(dataset_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Generated demonstration dataset with {len(transactions)} transactions")
        
        # Load the generated data
        self.load_raw_data()
    
    def _determine_hunter_type(self, user_index: int) -> str:
        """Determine hunter sophistication type for longitudinal analysis."""
        # Different types of farming behavior that evolved over time
        if user_index < 100:
            return 'early_naive_hunter'      # Simple farming in early rounds
        elif user_index < 200:
            return 'evolved_hunter'          # Adapted strategies across rounds
        elif user_index < 250:
            return 'sophisticated_hunter'    # Advanced multi-round strategies
        elif user_index < 300:
            return 'late_entry_hunter'       # Started farming in later rounds
        else:
            return 'legitimate'              # Genuine users
    
    def _generate_longitudinal_user_transactions(self, user_id: str, base_count: int, hunter_type: str) -> List[OptimismDeFiTransaction]:
        """Generate transactions showing evolution across airdrop rounds."""
        transactions = []
        
        for round_num in self.airdrop_rounds:
            if round_num not in self.airdrop_round_dates:
                continue
            
            # Determine activity level for this round based on hunter type
            round_tx_count = self._get_round_activity_level(hunter_type, round_num)
            
            if round_tx_count == 0:
                continue
            
            # Generate transactions for this round
            round_transactions = self._generate_round_transactions(
                user_id, round_num, round_tx_count, hunter_type
            )
            transactions.extend(round_transactions)
        
        return transactions
    
    def _get_round_activity_level(self, hunter_type: str, round_num: int) -> int:
        """Determine activity level for a specific round based on hunter evolution."""
        if hunter_type == 'legitimate':
            # Legitimate users have consistent but lower activity
            return np.random.poisson(3) if np.random.random() < 0.6 else 0
        
        elif hunter_type == 'early_naive_hunter':
            # High activity in early rounds, learns to avoid detection
            if round_num <= 2:
                return np.random.poisson(8)
            else:
                return np.random.poisson(2)  # Reduced activity after learning
        
        elif hunter_type == 'evolved_hunter':
            # Adapts strategy across rounds
            if round_num == 1:
                return np.random.poisson(5)  # Moderate start
            elif round_num == 2:
                return np.random.poisson(10)  # Increased after learning
            else:
                return np.random.poisson(6)  # Stable sophisticated strategy
        
        elif hunter_type == 'sophisticated_hunter':
            # Consistent sophisticated approach across all rounds
            return np.random.poisson(7)
        
        elif hunter_type == 'late_entry_hunter':
            # Only active in later rounds
            if round_num <= 2:
                return 0
            else:
                return np.random.poisson(9)  # High activity when they start
        
        return 0
    
    def _generate_round_transactions(self, user_id: str, round_num: int, tx_count: int, hunter_type: str) -> List[OptimismDeFiTransaction]:
        """Generate transactions for a specific airdrop round."""
        transactions = []
        
        # Time window around airdrop
        airdrop_timestamp = self.airdrop_round_dates[round_num]
        airdrop_date = datetime.fromtimestamp(airdrop_timestamp)
        
        # Pre-airdrop activity window (90 days before)
        start_window = airdrop_date - timedelta(days=90)
        end_window = airdrop_date - timedelta(days=1)  # Stop before airdrop
        
        for i in range(tx_count):
            # Generate timestamp within window
            time_range = (end_window - start_window).total_seconds()
            random_offset = np.random.uniform(0, time_range)
            tx_timestamp = start_window + timedelta(seconds=random_offset)
            
            # Choose protocol based on round and hunter sophistication
            protocol = self._choose_protocol_for_round(round_num, hunter_type)
            
            # Generate transaction details
            tx_type, value_usd = self._generate_optimism_transaction_details(protocol, hunter_type)
            
            # L2-specific gas fees (very low)
            gas_fee = np.random.uniform(0.0001, 0.001)  # ETH
            
            tx = OptimismDeFiTransaction(
                user_id=user_id,
                timestamp=tx_timestamp.timestamp(),
                chain_id='optimism',
                transaction_type=tx_type,
                value_usd=value_usd,
                gas_fee=gas_fee,
                signature=f"optimism_r{round_num}_{user_id}_{i}_{int(tx_timestamp.timestamp())}",
                block_number=1000000 + round_num * 1000000 + i * 100,
                protocol=protocol,
                # Add round-specific metadata
                token_in='USDC' if tx_type == 'swap' else None,
                token_out='OP' if tx_type == 'swap' else None
            )
            
            transactions.append(tx)
        
        return transactions
    
    def _choose_protocol_for_round(self, round_num: int, hunter_type: str) -> str:
        """Choose protocol based on round number and hunter sophistication."""
        if hunter_type == 'legitimate':
            # Legitimate users have diverse protocol usage
            return np.random.choice(self.protocols)
        
        # Hunters adapt their protocol choice over rounds
        if round_num == 1:
            # Early round: focus on basic DEX activity
            return np.random.choice(['uniswap_v3'], p=[1.0])
        elif round_num == 2:
            # Second round: discovered more protocols
            return np.random.choice(['uniswap_v3', 'synthetix'], p=[0.7, 0.3])
        elif round_num == 3:
            # Third round: more sophisticated strategies
            return np.random.choice(['uniswap_v3', 'synthetix', 'aave'], p=[0.5, 0.3, 0.2])
        else:
            # Later rounds: full ecosystem exploitation
            return np.random.choice(self.protocols, p=[0.4, 0.25, 0.2, 0.15])
    
    def _generate_optimism_transaction_details(self, protocol: str, hunter_type: str) -> Tuple[str, float]:
        """Generate protocol-specific transaction details."""
        if protocol == 'uniswap_v3':
            tx_type = 'swap'
            if hunter_type == 'legitimate':
                value_usd = np.random.lognormal(4, 1.2)  # Natural distribution
            else:
                value_usd = np.random.choice([100, 250, 500, 1000]) * np.random.uniform(0.9, 1.1)
        
        elif protocol == 'synthetix':
            tx_type = np.random.choice(['swap', 'mint'])
            value_usd = np.random.uniform(50, 2000)
        
        elif protocol == 'aave':
            tx_type = np.random.choice(['lend', 'borrow'])
            value_usd = np.random.uniform(500, 5000)
        
        elif protocol == 'velodrome':
            tx_type = np.random.choice(['swap', 'add_liquidity'])
            value_usd = np.random.uniform(100, 1500)
        
        else:
            tx_type = 'swap'
            value_usd = np.random.uniform(100, 1000)
        
        return tx_type, value_usd
    
    def _generate_users_from_transactions(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate users DataFrame from collected transactions."""
        users = transactions_df['user_id'].unique()
        users_df = pd.DataFrame({'user_id': users})
        return users_df
    
    def _generate_longitudinal_labels(self, users_df: pd.DataFrame, transactions_df: pd.DataFrame) -> Dict[str, int]:
        """Generate labels based on longitudinal analysis."""
        labels = {}
        
        for user_id in users_df['user_id']:
            user_txs = transactions_df[transactions_df['user_id'] == user_id]
            
            # Analyze cross-round behavior patterns
            is_hunter = self._analyze_longitudinal_patterns(user_txs)
            labels[user_id] = 1 if is_hunter else 0
        
        return labels
    
    def _analyze_longitudinal_patterns(self, user_txs: pd.DataFrame) -> bool:
        """Analyze user behavior across multiple rounds to detect farming."""
        if len(user_txs) == 0:
            return False
        
        # Check for suspicious patterns:
        # 1. Activity clustering around multiple airdrop dates
        # 2. Sudden changes in transaction patterns between rounds
        # 3. Artificial protocol diversity increases
        
        activity_around_airdrops = 0
        
        for airdrop_timestamp in self.airdrop_events:
            airdrop_date = datetime.fromtimestamp(airdrop_timestamp)
            
            # Check activity in 30 days before airdrop
            window_start = airdrop_date - timedelta(days=30)
            window_end = airdrop_date
            
            window_activity = user_txs[
                (user_txs['timestamp'] >= window_start.timestamp()) &
                (user_txs['timestamp'] <= window_end.timestamp())
            ]
            
            if len(window_activity) > 0:
                activity_around_airdrops += 1
        
        # If user is active around multiple airdrops, likely a hunter
        return activity_around_airdrops >= 2
    
    def load_raw_data(self) -> None:
        """Load raw Optimism blockchain data from saved files."""
        dataset_file = os.path.join(self.data_path, f'optimism_defi_{self.split}.json')
        
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        
        # Load components
        self.transactions = pd.DataFrame(dataset['transactions'])
        self.users = pd.DataFrame(dataset['users'])
        self.labels = dataset['labels']
        
        print(f"Loaded Optimism dataset: {len(self.transactions)} transactions, {len(self.users)} users")
        
        if 'hunter_types' in dataset.get('metadata', {}):
            hunter_types = dataset['metadata']['hunter_types']
            print(f"Hunter type distribution: {hunter_types}")
    
    def extract_defi_features(self, user_id: str) -> Dict[str, torch.Tensor]:
        """Extract Optimism DeFi-specific features for hunter detection."""
        user_txs = self.transactions[self.transactions['user_id'] == user_id]
        
        if len(user_txs) == 0:
            return {}
        
        # Cross-protocol features
        cross_protocol_features = self.extract_cross_protocol_features(user_id)
        
        # Optimism-specific features
        optimism_features = self._extract_optimism_specific_features(user_txs)
        
        # Longitudinal features (unique to multi-round analysis)
        longitudinal_features = self._extract_longitudinal_features(user_txs)
        
        # Combine features
        return {**cross_protocol_features, **optimism_features, **longitudinal_features}
    
    def _extract_optimism_specific_features(self, user_txs: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Extract features specific to Optimism ecosystem."""
        features = {}
        
        # L2 transaction patterns (high frequency, low cost)
        if len(user_txs) > 1:
            timestamps = user_txs['timestamp'].sort_values()
            time_intervals = np.diff(timestamps)
            
            # L2 allows for very frequent transactions
            high_frequency_ratio = (time_intervals < 300).mean()  # Within 5 minutes
            features['high_frequency_ratio'] = torch.tensor(high_frequency_ratio, dtype=torch.float32)
        
        # Protocol evolution patterns
        protocols_used = user_txs['protocol'].nunique()
        features['protocol_diversity'] = torch.tensor(protocols_used, dtype=torch.float32)
        
        # Value distribution analysis (hunters often use round numbers)
        values = user_txs['value_usd']
        round_value_ratio = 0.0
        if len(values) > 0:
            # Check for suspiciously round values
            round_values = values[values % 100 == 0]  # Exact hundreds
            round_value_ratio = len(round_values) / len(values)
        
        features['round_value_ratio'] = torch.tensor(round_value_ratio, dtype=torch.float32)
        
        return features
    
    def _extract_longitudinal_features(self, user_txs: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Extract features based on behavior across multiple airdrop rounds."""
        features = {}
        
        # Activity around each airdrop round
        round_activity = []
        
        for round_num in self.airdrop_rounds:
            if round_num not in self.airdrop_round_dates:
                continue
                
            airdrop_timestamp = self.airdrop_round_dates[round_num]
            airdrop_date = datetime.fromtimestamp(airdrop_timestamp)
            
            # Count activity in 60 days before airdrop
            window_start = airdrop_date - timedelta(days=60)
            window_end = airdrop_date
            
            round_txs = user_txs[
                (user_txs['timestamp'] >= window_start.timestamp()) &
                (user_txs['timestamp'] <= window_end.timestamp())
            ]
            
            round_activity.append(len(round_txs))
        
        # Cross-round pattern analysis
        if len(round_activity) > 1:
            activity_variance = np.var(round_activity)
            activity_trend = np.corrcoef(range(len(round_activity)), round_activity)[0, 1]
            
            features['cross_round_variance'] = torch.tensor(activity_variance, dtype=torch.float32)
            features['activity_trend'] = torch.tensor(activity_trend if not np.isnan(activity_trend) else 0.0, dtype=torch.float32)
        
        # Number of rounds with activity
        active_rounds = sum(1 for activity in round_activity if activity > 0)
        features['active_round_count'] = torch.tensor(active_rounds, dtype=torch.float32)
        
        return features
    
    def build_protocol_interaction_graph(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Build user graph based on Optimism protocol interactions with longitudinal analysis."""
        user_pairs = defaultdict(list)
        
        # Group users by round-specific protocol interactions
        for round_num in self.airdrop_rounds:
            if round_num not in self.airdrop_round_dates:
                continue
            
            airdrop_timestamp = self.airdrop_round_dates[round_num]
            airdrop_date = datetime.fromtimestamp(airdrop_timestamp)
            
            # Focus on pre-airdrop period
            window_start = airdrop_date - timedelta(days=90)
            window_end = airdrop_date
            
            round_txs = self.transactions[
                (self.transactions['timestamp'] >= window_start.timestamp()) &
                (self.transactions['timestamp'] <= window_end.timestamp())
            ]
            
            # Create edges between users active in same round + protocol
            for protocol in self.protocols:
                protocol_users = round_txs[round_txs['protocol'] == protocol]['user_id'].unique()
                
                for i, user1 in enumerate(protocol_users):
                    for user2 in protocol_users[i+1:]:
                        user_pairs[(user1, user2)].append(f"round{round_num}_{protocol}")
        
        # Build edge index and features
        edges = []
        edge_features = []
        
        user_to_idx = {user: idx for idx, user in enumerate(self.users['user_id'])}
        
        for (user1, user2), shared_interactions in user_pairs.items():
            if user1 in user_to_idx and user2 in user_to_idx:
                edges.extend([(user_to_idx[user1], user_to_idx[user2]),
                            (user_to_idx[user2], user_to_idx[user1])])
                
                # Compute edge features including longitudinal aspects
                edge_feature = self._compute_longitudinal_edge_features(user1, user2, shared_interactions)
                edge_features.extend([edge_feature, edge_feature])
        
        if not edges:
            num_users = len(user_to_idx)
            edges = [(i, i) for i in range(num_users)]
            edge_features = [torch.zeros(64) for _ in range(num_users)]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_features = torch.stack(edge_features)
        
        return edge_index, {'edge_features': edge_features}
    
    def _compute_longitudinal_edge_features(self, user1: str, user2: str, shared_interactions: List[str]) -> torch.Tensor:
        """Compute edge features including longitudinal patterns."""
        user1_txs = self.transactions[self.transactions['user_id'] == user1]
        user2_txs = self.transactions[self.transactions['user_id'] == user2]
        
        # Basic interaction features
        num_shared_interactions = len(shared_interactions)
        
        # Cross-round coordination score
        cross_round_coordination = 0.0
        rounds_with_coordination = 0
        
        for round_num in self.airdrop_rounds:
            round_interactions = [i for i in shared_interactions if f"round{round_num}" in i]
            if len(round_interactions) > 0:
                rounds_with_coordination += 1
        
        if len(self.airdrop_rounds) > 0:
            cross_round_coordination = rounds_with_coordination / len(self.airdrop_rounds)
        
        # Temporal synchronization across rounds
        temporal_sync = self._compute_cross_round_temporal_sync(user1_txs, user2_txs)
        
        # Protocol evolution similarity
        protocol_evolution_sim = self._compute_protocol_evolution_similarity(user1_txs, user2_txs)
        
        # Combine features
        features = torch.tensor([
            num_shared_interactions,
            cross_round_coordination,
            temporal_sync,
            protocol_evolution_sim
        ], dtype=torch.float32)
        
        # Pad to standard size
        padded_features = torch.zeros(64)
        padded_features[:min(len(features), 64)] = features[:64]
        
        return padded_features
    
    def _compute_cross_round_temporal_sync(self, user1_txs: pd.DataFrame, user2_txs: pd.DataFrame) -> float:
        """Compute temporal synchronization across multiple airdrop rounds."""
        sync_scores = []
        
        for round_num in self.airdrop_rounds:
            if round_num not in self.airdrop_round_dates:
                continue
            
            airdrop_timestamp = self.airdrop_round_dates[round_num]
            airdrop_date = datetime.fromtimestamp(airdrop_timestamp)
            
            window_start = airdrop_date - timedelta(days=60)
            window_end = airdrop_date
            
            round_txs1 = user1_txs[
                (user1_txs['timestamp'] >= window_start.timestamp()) &
                (user1_txs['timestamp'] <= window_end.timestamp())
            ]
            
            round_txs2 = user2_txs[
                (user2_txs['timestamp'] >= window_start.timestamp()) &
                (user2_txs['timestamp'] <= window_end.timestamp())
            ]
            
            if len(round_txs1) > 0 and len(round_txs2) > 0:
                # Simple synchronization: check if both users were active in same week
                round_sync = self._compute_weekly_activity_overlap(round_txs1, round_txs2)
                sync_scores.append(round_sync)
        
        return np.mean(sync_scores) if sync_scores else 0.0
    
    def _compute_weekly_activity_overlap(self, txs1: pd.DataFrame, txs2: pd.DataFrame) -> float:
        """Compute weekly activity overlap between two users."""
        # Convert timestamps to week numbers
        weeks1 = set((txs1['timestamp'] // (7 * 24 * 3600)).astype(int))
        weeks2 = set((txs2['timestamp'] // (7 * 24 * 3600)).astype(int))
        
        if not weeks1 or not weeks2:
            return 0.0
        
        overlap = len(weeks1 & weeks2)
        total_weeks = len(weeks1 | weeks2)
        
        return overlap / total_weeks if total_weeks > 0 else 0.0
    
    def _compute_protocol_evolution_similarity(self, user1_txs: pd.DataFrame, user2_txs: pd.DataFrame) -> float:
        """Compute how similarly users evolved their protocol usage across rounds."""
        user1_evolution = {}
        user2_evolution = {}
        
        for round_num in self.airdrop_rounds:
            if round_num not in self.airdrop_round_dates:
                continue
            
            airdrop_timestamp = self.airdrop_round_dates[round_num]
            airdrop_date = datetime.fromtimestamp(airdrop_timestamp)
            
            window_start = airdrop_date - timedelta(days=60)
            window_end = airdrop_date
            
            round_txs1 = user1_txs[
                (user1_txs['timestamp'] >= window_start.timestamp()) &
                (user1_txs['timestamp'] <= window_end.timestamp())
            ]
            
            round_txs2 = user2_txs[
                (user2_txs['timestamp'] >= window_start.timestamp()) &
                (user2_txs['timestamp'] <= window_end.timestamp())
            ]
            
            user1_evolution[round_num] = set(round_txs1['protocol'].unique()) if len(round_txs1) > 0 else set()
            user2_evolution[round_num] = set(round_txs2['protocol'].unique()) if len(round_txs2) > 0 else set()
        
        # Compute similarity in protocol adoption patterns
        similarities = []
        
        for round_num in self.airdrop_rounds:
            if round_num in user1_evolution and round_num in user2_evolution:
                protocols1 = user1_evolution[round_num]
                protocols2 = user2_evolution[round_num]
                
                if protocols1 or protocols2:
                    jaccard_sim = len(protocols1 & protocols2) / len(protocols1 | protocols2)
                    similarities.append(jaccard_sim)
        
        return np.mean(similarities) if similarities else 0.0