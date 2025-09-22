"""
Pure cryptocurrency dataset base class for DeFi/DEX airdrop hunter detection.

This extends the unified interface for pure crypto markets (no NFTs),
focusing on protocol interactions, liquidity provision, and cross-chain DeFi activities.
Supports our primary research targets: Arbitrum, Jupiter Solana, and Optimism.
"""

import torch
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from abc import abstractmethod
from collections import defaultdict
import warnings

from .base_dataset import BaseTemporalGraphDataset


class PureCryptoDataset(BaseTemporalGraphDataset):
    """
    Abstract base class for pure cryptocurrency DeFi/DEX datasets.
    
    Handles transaction types like swaps, liquidity provision, lending/borrowing,
    and staking across various DeFi protocols. Focuses on temporal patterns
    that indicate airdrop farming behavior.
    """
    
    def __init__(self, 
                 data_path: str,
                 split: str = 'train',
                 blockchain: str = 'arbitrum',
                 protocols: Optional[List[str]] = None,
                 min_transaction_value_usd: float = 1.0,
                 include_mev_features: bool = True,
                 cross_protocol_analysis: bool = True,
                 **kwargs):
        """
        Args:
            data_path: Path to dataset directory
            split: 'train', 'val', or 'test'
            blockchain: Target blockchain ('arbitrum', 'solana', 'optimism', etc.)
            protocols: List of DeFi protocols to include (None = all available)
            min_transaction_value_usd: Minimum transaction value in USD
            include_mev_features: Whether to include MEV/arbitrage features
            cross_protocol_analysis: Whether to analyze cross-protocol interactions
        """
        super().__init__(data_path, split, **kwargs)
        
        self.blockchain = blockchain
        self.protocols = protocols or self._get_default_protocols()
        self.min_transaction_value_usd = min_transaction_value_usd
        self.include_mev_features = include_mev_features
        self.cross_protocol_analysis = cross_protocol_analysis
        
        # DeFi-specific transaction types
        self.defi_transaction_types = {
            'swap': 0,
            'add_liquidity': 1,
            'remove_liquidity': 2,
            'lend': 3,
            'borrow': 4,
            'repay': 5,
            'stake': 6,
            'unstake': 7,
            'claim_rewards': 8,
            'liquidation': 9,
            'flashloan': 10,
            'bridge': 11
        }
        
        # Protocol categorization for feature engineering
        self.protocol_categories = {
            'dex': ['uniswap_v3', 'sushiswap', 'camelot', 'jupiter', 'raydium', 'orca'],
            'lending': ['aave', 'compound', 'kamino', 'solend'],
            'derivatives': ['gmx', 'drift', 'mango'],
            'yield': ['yearn', 'marinade', 'lido'],
            'bridge': ['hop', 'stargate', 'wormhole']
        }
        
        # Initialize DeFi-specific features
        self.defi_features = None
        self.protocol_interactions = None
        self.cross_protocol_patterns = None
    
    def _get_default_protocols(self) -> List[str]:
        """Get default protocols for each blockchain."""
        protocol_map = {
            'arbitrum': ['uniswap_v3', 'gmx', 'camelot', 'sushiswap', 'aave'],
            'solana': ['jupiter', 'raydium', 'orca', 'drift', 'kamino', 'marinade'],
            'optimism': ['uniswap_v3', 'synthetix', 'aave', 'velodrome'],
            'ethereum': ['uniswap_v3', 'aave', 'compound', 'yearn']
        }
        return protocol_map.get(self.blockchain, [])
    
    def extract_nft_features(self, nft_id: str) -> Dict[str, torch.Tensor]:
        """
        Pure crypto datasets don't have NFT features.
        Return empty dict to maintain interface compatibility.
        """
        return {}
    
    @abstractmethod
    def extract_defi_features(self, user_id: str) -> Dict[str, torch.Tensor]:
        """
        Extract DeFi-specific transaction features for airdrop hunter detection.
        
        Should include:
        - Cross-protocol interaction patterns
        - Liquidity provision strategies
        - Gas optimization behaviors
        - Volume manipulation indicators
        - Temporal clustering around airdrop events
        """
        pass
    
    @abstractmethod
    def build_protocol_interaction_graph(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Build user interaction graph based on DeFi protocol usage.
        
        Creates edges between users based on:
        - Shared protocol interactions
        - Similar transaction patterns
        - Coordinated liquidity provision
        - Cross-protocol arbitrage activities
        """
        pass
    
    def extract_transaction_features(self, user_id: str) -> Dict[str, torch.Tensor]:
        """
        Extract comprehensive transaction features including DeFi-specific patterns.
        
        Combines base transaction features with DeFi-specific features
        for comprehensive airdrop hunter detection.
        """
        # Get base transaction features
        base_features = self._extract_base_transaction_features(user_id)
        
        # Get DeFi-specific features
        defi_features = self.extract_defi_features(user_id)
        
        # Combine features
        combined_features = {**base_features, **defi_features}
        
        return combined_features
    
    def build_user_graph(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Build user interaction graph using DeFi protocol interactions.
        
        For pure crypto datasets, we use protocol-based relationships
        rather than NFT marketplace relationships.
        """
        return self.build_protocol_interaction_graph()
    
    def _extract_base_transaction_features(self, user_id: str) -> Dict[str, torch.Tensor]:
        """Extract base transaction features adapted for DeFi."""
        user_txs = self.transactions[self.transactions['user_id'] == user_id].copy()
        user_txs = user_txs.sort_values('timestamp')
        
        # Limit sequence length
        if len(user_txs) > self.max_sequence_length:
            user_txs = user_txs.tail(self.max_sequence_length)
        
        seq_len = len(user_txs)
        if seq_len == 0:
            return self._get_empty_features()
        
        # Basic transaction features
        values_usd = user_txs['value_usd'].values
        gas_fees = user_txs['gas_fee'].values
        timestamps = user_txs['timestamp'].values
        
        # DeFi transaction type encoding
        tx_types = user_txs['transaction_type']
        tx_type_ids = [self.defi_transaction_types.get(t, 0) for t in tx_types]
        
        # Protocol encoding
        protocols = user_txs['protocol']
        protocol_to_id = {p: i for i, p in enumerate(self.protocols)}
        protocol_ids = [protocol_to_id.get(p, 0) for p in protocols]
        
        # Temporal features
        time_deltas = np.diff(timestamps, prepend=timestamps[0])
        time_of_day = (timestamps % (24 * 3600)) / (24 * 3600)
        day_of_week = ((timestamps // (24 * 3600)) % 7) / 7
        
        # Value-based features (using USD values instead of NFT prices)
        value_ratios = np.ones(seq_len)
        if seq_len > 1:
            value_ratios[1:] = values_usd[1:] / (values_usd[:-1] + 1e-8)
        
        # Gas efficiency (value per gas)
        gas_efficiency = values_usd / (gas_fees + 1e-8)
        
        # Cumulative features
        cumulative_volume = np.cumsum(values_usd)
        rolling_avg_value = pd.Series(values_usd).rolling(window=5, min_periods=1).mean().values
        
        # Protocol diversity features
        protocol_diversity = []
        for i in range(seq_len):
            window_start = max(0, i - 6)  # 7-transaction window
            window_protocols = protocols.iloc[window_start:i+1]
            unique_protocols = window_protocols.nunique()
            protocol_diversity.append(unique_protocols)
        protocol_diversity = np.array(protocol_diversity)
        
        # Transaction frequency
        transaction_frequency = np.ones(seq_len)
        if seq_len > 1:
            for i in range(1, seq_len):
                window_start = max(0, i - 6)
                window_txs = timestamps[window_start:i+1]
                time_span = timestamps[i] - timestamps[window_start] + 1
                transaction_frequency[i] = len(window_txs) / (time_span / (24 * 3600))
        
        # Pad sequences
        def pad_sequence(arr):
            padded = np.zeros(self.max_sequence_length)
            padded[:len(arr)] = arr
            return torch.tensor(padded, dtype=torch.float32)
        
        return {
            'values_usd': pad_sequence(np.log1p(values_usd)),
            'gas_fees': pad_sequence(np.log1p(gas_fees)),
            'timestamps': pad_sequence(timestamps),
            'transaction_types': pad_sequence(tx_type_ids),
            'protocol_ids': pad_sequence(protocol_ids),
            'time_deltas': pad_sequence(np.log1p(time_deltas)),
            'time_of_day': pad_sequence(time_of_day),
            'day_of_week': pad_sequence(day_of_week),
            'value_ratios': pad_sequence(np.log(value_ratios + 1e-8)),
            'gas_efficiency': pad_sequence(np.log1p(gas_efficiency)),
            'cumulative_volume': pad_sequence(np.log1p(cumulative_volume)),
            'rolling_avg_value': pad_sequence(np.log1p(rolling_avg_value)),
            'protocol_diversity': pad_sequence(protocol_diversity),
            'transaction_frequency': pad_sequence(transaction_frequency),
            'sequence_length': torch.tensor(seq_len, dtype=torch.long)
        }
    
    def _get_empty_features(self) -> Dict[str, torch.Tensor]:
        """Return empty features when user has no transactions."""
        feature_names = [
            'values_usd', 'gas_fees', 'timestamps', 'transaction_types', 'protocol_ids',
            'time_deltas', 'time_of_day', 'day_of_week', 'value_ratios', 'gas_efficiency',
            'cumulative_volume', 'rolling_avg_value', 'protocol_diversity', 'transaction_frequency'
        ]
        
        empty_features = {}
        for name in feature_names:
            empty_features[name] = torch.zeros(self.max_sequence_length, dtype=torch.float32)
        
        empty_features['sequence_length'] = torch.tensor(0, dtype=torch.long)
        return empty_features
    
    def extract_cross_protocol_features(self, user_id: str) -> Dict[str, torch.Tensor]:
        """
        Extract features that capture cross-protocol interaction patterns.
        
        These are particularly important for identifying sophisticated
        airdrop hunters who coordinate activities across multiple protocols.
        """
        user_txs = self.transactions[self.transactions['user_id'] == user_id]
        
        if len(user_txs) == 0:
            return {'cross_protocol_score': torch.tensor(0.0)}
        
        # Protocol usage patterns
        protocol_counts = user_txs['protocol'].value_counts()
        protocol_diversity = len(protocol_counts)
        protocol_entropy = -(protocol_counts / len(user_txs) * np.log(protocol_counts / len(user_txs) + 1e-8)).sum()
        
        # Category diversity (DEX, lending, derivatives, etc.)
        categories_used = set()
        for protocol in user_txs['protocol'].unique():
            for category, protocols in self.protocol_categories.items():
                if protocol in protocols:
                    categories_used.add(category)
        category_diversity = len(categories_used)
        
        # Temporal clustering analysis
        protocol_timing_score = self._compute_protocol_timing_score(user_txs)
        
        # MEV/arbitrage indicators
        mev_score = 0.0
        if self.include_mev_features:
            mev_score = self._compute_mev_score(user_txs)
        
        return {
            'protocol_diversity': torch.tensor(protocol_diversity, dtype=torch.float32),
            'protocol_entropy': torch.tensor(protocol_entropy, dtype=torch.float32),
            'category_diversity': torch.tensor(category_diversity, dtype=torch.float32),
            'protocol_timing_score': torch.tensor(protocol_timing_score, dtype=torch.float32),
            'mev_score': torch.tensor(mev_score, dtype=torch.float32),
            'cross_protocol_score': torch.tensor(
                protocol_diversity * protocol_entropy * category_diversity, 
                dtype=torch.float32
            )
        }
    
    def _compute_protocol_timing_score(self, user_txs: pd.DataFrame) -> float:
        """
        Compute how clustered a user's protocol interactions are in time.
        
        Hunters often have bursts of activity across multiple protocols
        in short time windows (farming behavior).
        """
        if len(user_txs) < 2:
            return 0.0
        
        # Group transactions by protocol
        protocol_groups = user_txs.groupby('protocol')['timestamp']
        
        # Compute time clustering score
        clustering_scores = []
        for protocol, timestamps in protocol_groups:
            if len(timestamps) >= 2:
                # Compute coefficient of variation of time intervals
                time_diffs = np.diff(sorted(timestamps))
                if len(time_diffs) > 0:
                    cv = np.std(time_diffs) / (np.mean(time_diffs) + 1e-8)
                    clustering_scores.append(1.0 / (1.0 + cv))  # Higher score for more clustered
        
        return np.mean(clustering_scores) if clustering_scores else 0.0
    
    def _compute_mev_score(self, user_txs: pd.DataFrame) -> float:
        """
        Compute MEV/arbitrage behavior score.
        
        Looks for patterns like:
        - Very fast transactions (indicating bot behavior)
        - High gas fees (priority for MEV)
        - Specific transaction patterns (sandwich attacks, arbitrage)
        """
        if len(user_txs) < 2:
            return 0.0
        
        # Fast transaction patterns
        timestamps = sorted(user_txs['timestamp'])
        time_diffs = np.diff(timestamps)
        fast_tx_ratio = np.mean(time_diffs < 60)  # Transactions within 1 minute
        
        # High gas usage patterns
        gas_fees = user_txs['gas_fee']
        high_gas_ratio = np.mean(gas_fees > gas_fees.quantile(0.9))
        
        # MEV-specific transaction patterns (simplified)
        mev_types = ['swap', 'flashloan', 'liquidation']
        mev_tx_ratio = np.mean(user_txs['transaction_type'].isin(mev_types))
        
        # Combine scores
        mev_score = (fast_tx_ratio * 0.4 + high_gas_ratio * 0.3 + mev_tx_ratio * 0.3)
        
        return mev_score
    
    def get_protocol_statistics(self) -> Dict[str, Any]:
        """Get statistics about protocol usage in the dataset."""
        if self.transactions is None:
            return {}
        
        stats = {}
        
        # Protocol distribution
        protocol_counts = self.transactions['protocol'].value_counts()
        stats['protocol_distribution'] = protocol_counts.to_dict()
        
        # Transaction type distribution
        tx_type_counts = self.transactions['transaction_type'].value_counts()
        stats['transaction_type_distribution'] = tx_type_counts.to_dict()
        
        # Cross-protocol users
        user_protocol_counts = self.transactions.groupby('user_id')['protocol'].nunique()
        stats['avg_protocols_per_user'] = user_protocol_counts.mean()
        stats['multi_protocol_users'] = (user_protocol_counts > 1).sum()
        stats['single_protocol_users'] = (user_protocol_counts == 1).sum()
        
        # Value statistics
        stats['total_volume_usd'] = self.transactions['value_usd'].sum()
        stats['avg_transaction_value_usd'] = self.transactions['value_usd'].mean()
        stats['median_transaction_value_usd'] = self.transactions['value_usd'].median()
        
        return stats
    
    def verify_data_integrity(self) -> bool:
        """Verify that DeFi data is complete and valid."""
        if self.transactions is None or len(self.transactions) == 0:
            return False
        
        required_columns = [
            'user_id', 'timestamp', 'value_usd', 'gas_fee', 
            'transaction_type', 'protocol'
        ]
        
        # Check required columns exist
        for col in required_columns:
            if col not in self.transactions.columns:
                warnings.warn(f"Missing required column: {col}")
                return False
        
        # Check data validity
        if self.transactions['value_usd'].isna().any():
            warnings.warn("Found NaN values in value_usd")
            return False
        
        if (self.transactions['value_usd'] < 0).any():
            warnings.warn("Found negative values in value_usd")
            return False
        
        if self.transactions['timestamp'].isna().any():
            warnings.warn("Found NaN values in timestamp")
            return False
        
        # Check protocol validity
        unknown_protocols = set(self.transactions['protocol'].unique()) - set(self.protocols)
        if unknown_protocols:
            warnings.warn(f"Found unknown protocols: {unknown_protocols}")
        
        # Check transaction type validity
        unknown_tx_types = set(self.transactions['transaction_type'].unique()) - set(self.defi_transaction_types.keys())
        if unknown_tx_types:
            warnings.warn(f"Found unknown transaction types: {unknown_tx_types}")
        
        return True