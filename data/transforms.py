"""
Data transformation utilities for temporal graph datasets.

Provides specialized transforms for different model types and
blockchain data preprocessing needs.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import random
from datetime import datetime, timedelta


class TemporalGraphTransform:
    """
    Transform for Temporal Graph Transformer model.
    
    Focuses on temporal sequence processing and behavioral change detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_sequence_length = config.get('max_sequence_length', 100)
        self.normalize_timestamps = config.get('normalize_timestamps', True)
        self.augment_sequences = config.get('augment_sequences', False)
        
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply temporal graph transforms to a sample."""
        transformed = sample.copy()
        
        # Apply temporal sequence transforms
        transformed = self._transform_temporal_features(transformed)
        
        # Apply timestamp normalization
        if self.normalize_timestamps:
            transformed = self._normalize_timestamps(transformed)
        
        # Apply sequence augmentation during training
        if self.augment_sequences:
            transformed = self._augment_temporal_sequence(transformed)
        
        return transformed
    
    def _transform_temporal_features(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform transaction features for temporal modeling."""
        tx_features = sample['transaction_features']
        
        # Ensure all features are properly shaped
        for key, values in tx_features.items():
            if not isinstance(values, torch.Tensor):
                tx_features[key] = torch.tensor(values, dtype=torch.float32)
        
        # Add derived temporal features
        if 'timestamps' in sample:
            timestamps = sample['timestamps']
            
            # Add time-based cyclical features
            tx_features.update(self._add_cyclical_features(timestamps))
            
            # Add sequence position encoding
            tx_features['position_encoding'] = self._create_position_encoding(len(timestamps))
        
        sample['transaction_features'] = tx_features
        return sample
    
    def _normalize_timestamps(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize timestamps to relative values."""
        if 'timestamps' in sample:
            timestamps = sample['timestamps']
            
            # Convert to relative time from first transaction
            if len(timestamps) > 0:
                first_timestamp = timestamps[0]
                relative_timestamps = timestamps - first_timestamp
                sample['timestamps'] = relative_timestamps
        
        return sample
    
    def _add_cyclical_features(self, timestamps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Add cyclical time features (hour of day, day of week, etc.)."""
        cyclical_features = {}
        
        # Hour of day (0-23)
        hours = (timestamps % (24 * 3600)) / 3600
        cyclical_features['hour_sin'] = torch.sin(2 * np.pi * hours / 24)
        cyclical_features['hour_cos'] = torch.cos(2 * np.pi * hours / 24)
        
        # Day of week (0-6)
        days = ((timestamps // (24 * 3600)) % 7)
        cyclical_features['day_sin'] = torch.sin(2 * np.pi * days / 7)
        cyclical_features['day_cos'] = torch.cos(2 * np.pi * days / 7)
        
        # Day of month (1-31)
        # Simplified approximation
        day_of_month = ((timestamps // (24 * 3600)) % 30) + 1
        cyclical_features['month_day_sin'] = torch.sin(2 * np.pi * day_of_month / 31)
        cyclical_features['month_day_cos'] = torch.cos(2 * np.pi * day_of_month / 31)
        
        return cyclical_features
    
    def _create_position_encoding(self, seq_len: int) -> torch.Tensor:
        """Create positional encoding for sequence elements."""
        positions = torch.arange(seq_len, dtype=torch.float32)
        
        # Pad to max length
        padded_positions = torch.zeros(self.max_sequence_length)
        padded_positions[:seq_len] = positions
        
        return padded_positions
    
    def _augment_temporal_sequence(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply data augmentation to temporal sequences."""
        if random.random() < 0.3:  # 30% chance of augmentation
            # Choose augmentation type
            aug_type = random.choice(['noise', 'dropout', 'time_shift'])
            
            if aug_type == 'noise':
                sample = self._add_feature_noise(sample)
            elif aug_type == 'dropout':
                sample = self._apply_temporal_dropout(sample)
            elif aug_type == 'time_shift':
                sample = self._apply_time_shift(sample)
        
        return sample
    
    def _add_feature_noise(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Add small noise to transaction features."""
        tx_features = sample['transaction_features']
        
        # Add noise to price-related features
        noise_features = ['prices', 'gas_fees', 'cumulative_volume']
        for feature in noise_features:
            if feature in tx_features:
                noise = torch.normal(0, 0.01, tx_features[feature].shape)
                tx_features[feature] = tx_features[feature] + noise
        
        return sample
    
    def _apply_temporal_dropout(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Randomly drop some transactions from the sequence."""
        seq_len = sample['transaction_features']['sequence_length'].item()
        
        if seq_len > 5:  # Only apply if sequence is long enough
            # Keep 80-95% of transactions
            keep_ratio = random.uniform(0.8, 0.95)
            keep_count = max(5, int(seq_len * keep_ratio))
            
            # Randomly select transactions to keep
            keep_indices = sorted(random.sample(range(seq_len), keep_count))
            
            # Update all sequence features
            tx_features = sample['transaction_features']
            for key, values in tx_features.items():
                if key != 'sequence_length' and len(values.shape) > 0:
                    if values.shape[0] >= seq_len:
                        # Create new tensor with kept values
                        new_values = torch.zeros_like(values)
                        for i, idx in enumerate(keep_indices):
                            new_values[i] = values[idx]
                        tx_features[key] = new_values
            
            # Update sequence length
            tx_features['sequence_length'] = torch.tensor(keep_count, dtype=torch.long)
            
            # Update timestamps and attention mask
            if 'timestamps' in sample:
                new_timestamps = torch.zeros_like(sample['timestamps'])
                for i, idx in enumerate(keep_indices):
                    new_timestamps[i] = sample['timestamps'][idx]
                sample['timestamps'] = new_timestamps
            
            if 'attention_mask' in sample:
                new_mask = torch.zeros_like(sample['attention_mask'])
                new_mask[:keep_count] = True
                sample['attention_mask'] = new_mask
        
        return sample
    
    def _apply_time_shift(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply small random time shifts to timestamps."""
        if 'timestamps' in sample:
            timestamps = sample['timestamps']
            seq_len = sample['transaction_features']['sequence_length'].item()
            
            # Add random time shifts (±10% of average time delta)
            if seq_len > 1:
                time_diffs = torch.diff(timestamps[:seq_len])
                avg_time_diff = time_diffs.mean().item()
                
                # Generate random shifts
                max_shift = 0.1 * avg_time_diff
                shifts = torch.normal(0, max_shift/3, (seq_len,))
                
                # Apply shifts while maintaining order
                shifted_timestamps = timestamps.clone()
                for i in range(seq_len):
                    if i > 0:
                        shifted_timestamps[i] = max(
                            shifted_timestamps[i-1] + 1,  # Maintain order
                            timestamps[i] + shifts[i]
                        )
                
                sample['timestamps'] = shifted_timestamps
        
        return sample


class ARTEMISTransform:
    """
    Transform for ARTEMIS baseline model.
    
    Focuses on static graph features and manual feature engineering.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.normalize_features = config.get('normalize_features', True)
        
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply ARTEMIS transforms to a sample."""
        transformed = sample.copy()
        
        # Extract manual features from transaction data
        transformed = self._extract_manual_features(transformed)
        
        # Simplify graph structure (ARTEMIS uses simpler graphs)
        transformed = self._simplify_graph_structure(transformed)
        
        return transformed
    
    def _extract_manual_features(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ARTEMIS-style manual features."""
        tx_features = sample['transaction_features']
        timestamps = sample['timestamps']
        attention_mask = sample.get('attention_mask', torch.ones_like(timestamps, dtype=torch.bool))
        
        # Get valid sequence length
        seq_len = attention_mask.sum().item()
        
        if seq_len == 0:
            manual_features = torch.zeros(50)  # Standard ARTEMIS feature size
        else:
            # Extract price statistics
            prices = tx_features['prices'][:seq_len]
            gas_fees = tx_features['gas_fees'][:seq_len] if 'gas_fees' in tx_features else torch.zeros(seq_len)
            
            # Basic statistics
            price_stats = self._compute_price_statistics(prices)
            gas_stats = self._compute_gas_statistics(gas_fees)
            temporal_stats = self._compute_temporal_statistics(timestamps[:seq_len])
            behavioral_stats = self._compute_behavioral_statistics(tx_features, seq_len)
            
            # Market manipulation features
            manipulation_stats = self._compute_manipulation_features(prices, timestamps[:seq_len])
            
            # Combine all features
            manual_features = torch.cat([
                price_stats,
                gas_stats,
                temporal_stats,
                behavioral_stats,
                manipulation_stats
            ])
        
        # Ensure standard size
        if len(manual_features) > 50:
            manual_features = manual_features[:50]
        elif len(manual_features) < 50:
            padded = torch.zeros(50)
            padded[:len(manual_features)] = manual_features
            manual_features = padded
        
        sample['manual_features'] = manual_features
        return sample
    
    def _compute_price_statistics(self, prices: torch.Tensor) -> torch.Tensor:
        """Compute price-related statistics."""
        if len(prices) == 0:
            return torch.zeros(10)
        
        stats = torch.tensor([
            prices.mean().item(),
            prices.std().item() if len(prices) > 1 else 0.0,
            prices.median().item(),
            prices.max().item(),
            prices.min().item(),
            (prices.max() - prices.min()).item(),  # Range
            prices.quantile(0.25).item(),  # Q1
            prices.quantile(0.75).item(),  # Q3
            (prices > prices.mean()).float().mean().item(),  # Fraction above mean
            torch.log(prices + 1e-8).std().item() if len(prices) > 1 else 0.0  # Log price volatility
        ])
        
        return stats
    
    def _compute_gas_statistics(self, gas_fees: torch.Tensor) -> torch.Tensor:
        """Compute gas-related statistics."""
        if len(gas_fees) == 0:
            return torch.zeros(5)
        
        stats = torch.tensor([
            gas_fees.mean().item(),
            gas_fees.std().item() if len(gas_fees) > 1 else 0.0,
            gas_fees.max().item(),
            gas_fees.min().item(),
            (gas_fees > gas_fees.mean()).float().mean().item()  # Fraction above mean
        ])
        
        return stats
    
    def _compute_temporal_statistics(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Compute temporal pattern statistics."""
        if len(timestamps) <= 1:
            return torch.zeros(10)
        
        # Time deltas
        time_diffs = torch.diff(timestamps)
        
        # Activity patterns
        hours = (timestamps % (24 * 3600)) / 3600
        hour_distribution = torch.bincount((hours % 24).long(), minlength=24).float()
        hour_entropy = -torch.sum(hour_distribution * torch.log(hour_distribution + 1e-8)).item()
        
        # Day patterns
        days = ((timestamps // (24 * 3600)) % 7).long()
        day_distribution = torch.bincount(days, minlength=7).float()
        day_entropy = -torch.sum(day_distribution * torch.log(day_distribution + 1e-8)).item()
        
        stats = torch.tensor([
            time_diffs.mean().item(),
            time_diffs.std().item(),
            time_diffs.median().item(),
            time_diffs.min().item(),
            time_diffs.max().item(),
            hour_entropy,
            day_entropy,
            (timestamps[-1] - timestamps[0]).item(),  # Total time span
            len(timestamps) / ((timestamps[-1] - timestamps[0]) / (24 * 3600) + 1),  # Average txs per day
            (time_diffs < time_diffs.median()).float().mean().item()  # Fraction of short intervals
        ])
        
        return stats
    
    def _compute_behavioral_statistics(self, tx_features: Dict[str, torch.Tensor], seq_len: int) -> torch.Tensor:
        """Compute behavioral pattern statistics."""
        if seq_len == 0:
            return torch.zeros(10)
        
        # Transaction frequency
        tx_freq = tx_features.get('transaction_frequency', torch.zeros(seq_len))[:seq_len]
        
        # Price ratios (relative changes)
        price_ratios = tx_features.get('price_ratios', torch.zeros(seq_len))[:seq_len]
        
        # Volume patterns
        cumulative_vol = tx_features.get('cumulative_volume', torch.zeros(seq_len))[:seq_len]
        
        stats = torch.tensor([
            tx_freq.mean().item(),
            tx_freq.std().item() if len(tx_freq) > 1 else 0.0,
            price_ratios.mean().item(),
            price_ratios.std().item() if len(price_ratios) > 1 else 0.0,
            (price_ratios > 0).float().mean().item(),  # Fraction of price increases
            cumulative_vol[-1].item() if len(cumulative_vol) > 0 else 0.0,  # Total volume
            seq_len,  # Transaction count
            (tx_freq > tx_freq.mean()).float().mean().item(),  # High activity periods
            torch.abs(price_ratios).mean().item(),  # Average price change magnitude
            (torch.abs(price_ratios) > 0.1).float().mean().item()  # Large price changes
        ])
        
        return stats
    
    def _compute_manipulation_features(self, prices: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """Compute market manipulation detection features."""
        if len(prices) == 0:
            return torch.zeros(15)
        
        # Benford's Law analysis
        first_digits = []
        for price in prices:
            price_int = int(price.item())
            if price_int > 0:
                first_digit = int(str(price_int)[0])
                first_digits.append(first_digit)
        
        if first_digits:
            digit_counts = torch.bincount(torch.tensor(first_digits), minlength=10)[1:10]
            digit_distribution = digit_counts.float() / digit_counts.sum()
            
            # Expected Benford distribution
            benford_expected = torch.log10(1 + 1/torch.arange(1, 10, dtype=torch.float32))
            benford_deviation = torch.sum(torch.abs(digit_distribution - benford_expected)).item()
        else:
            benford_deviation = 0.0
        
        # Price rounding analysis
        rounded_prices = torch.round(prices)
        rounding_score = (prices == rounded_prices).float().mean().item()
        
        # Wash trading indicators
        if len(prices) > 1:
            price_volatility = prices.std().item() / (prices.mean().item() + 1e-8)
            consecutive_same_prices = 0
            for i in range(1, len(prices)):
                if torch.abs(prices[i] - prices[i-1]) < 1e-6:
                    consecutive_same_prices += 1
            same_price_ratio = consecutive_same_prices / max(len(prices) - 1, 1)
        else:
            price_volatility = 0.0
            same_price_ratio = 0.0
        
        # Volume concentration
        if len(timestamps) > 1:
            time_windows = (timestamps - timestamps[0]) // (3600)  # 1-hour windows
            volume_per_window = torch.bincount(time_windows.long(), weights=prices)
            volume_concentration = (volume_per_window.max() / volume_per_window.sum()).item()
        else:
            volume_concentration = 1.0
        
        stats = torch.tensor([
            benford_deviation,
            rounding_score,
            price_volatility,
            same_price_ratio,
            volume_concentration,
            # Add more manipulation indicators
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Placeholder for additional features
        ])
        
        return stats[:15]  # Ensure exactly 15 features
    
    def _simplify_graph_structure(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify graph structure for ARTEMIS (which uses simpler GCNs)."""
        # ARTEMIS uses the same graph structure but with different processing
        # No major changes needed here
        return sample


class AirdropEventProcessor:
    """
    Processor for airdrop event data and behavioral change detection.
    """
    
    def __init__(self, window_days: int = 7):
        self.window_days = window_days
        self.window_seconds = window_days * 24 * 3600
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process airdrop events for a sample."""
        if 'airdrop_events' not in sample or len(sample['airdrop_events']) == 0:
            return sample
        
        timestamps = sample['timestamps']
        airdrop_events = sample['airdrop_events']
        
        # Create airdrop period masks
        airdrop_masks = self._create_airdrop_masks(timestamps, airdrop_events)
        
        # Add behavioral change features
        change_features = self._extract_behavioral_change_features(sample, airdrop_masks)
        
        sample['airdrop_masks'] = airdrop_masks
        sample['behavioral_change_features'] = change_features
        
        return sample
    
    def _create_airdrop_masks(self, timestamps: torch.Tensor, 
                            airdrop_events: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create masks for different airdrop periods."""
        masks = {}
        
        # Pre-airdrop period (window before each event)
        pre_airdrop_mask = torch.zeros_like(timestamps, dtype=torch.bool)
        
        # Post-airdrop period (window after each event)
        post_airdrop_mask = torch.zeros_like(timestamps, dtype=torch.bool)
        
        # During airdrop period (small window around events)
        during_airdrop_mask = torch.zeros_like(timestamps, dtype=torch.bool)
        
        for event_time in airdrop_events:
            # Pre-airdrop: window_days before event
            pre_start = event_time - self.window_seconds
            pre_end = event_time
            pre_airdrop_mask |= (timestamps >= pre_start) & (timestamps < pre_end)
            
            # During airdrop: ±1 day around event
            during_start = event_time - 24 * 3600
            during_end = event_time + 24 * 3600
            during_airdrop_mask |= (timestamps >= during_start) & (timestamps <= during_end)
            
            # Post-airdrop: window_days after event
            post_start = event_time
            post_end = event_time + self.window_seconds
            post_airdrop_mask |= (timestamps > post_start) & (timestamps <= post_end)
        
        masks['pre_airdrop'] = pre_airdrop_mask
        masks['during_airdrop'] = during_airdrop_mask
        masks['post_airdrop'] = post_airdrop_mask
        
        return masks
    
    def _extract_behavioral_change_features(self, sample: Dict[str, Any], 
                                          airdrop_masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract behavioral change features around airdrop events."""
        tx_features = sample['transaction_features']
        
        # Get price and frequency data
        prices = tx_features['prices']
        tx_freq = tx_features.get('transaction_frequency', torch.zeros_like(prices))
        
        features = []
        
        # Compare pre vs post airdrop behavior
        pre_mask = airdrop_masks['pre_airdrop']
        post_mask = airdrop_masks['post_airdrop']
        
        if pre_mask.sum() > 0 and post_mask.sum() > 0:
            # Price behavior changes
            pre_avg_price = prices[pre_mask].mean()
            post_avg_price = prices[post_mask].mean()
            price_change_ratio = (post_avg_price / (pre_avg_price + 1e-8)).item()
            
            # Frequency changes
            pre_avg_freq = tx_freq[pre_mask].mean()
            post_avg_freq = tx_freq[post_mask].mean()
            freq_change_ratio = (post_avg_freq / (pre_avg_freq + 1e-8)).item()
            
            # Volume changes
            pre_volume = prices[pre_mask].sum()
            post_volume = prices[post_mask].sum()
            volume_change_ratio = (post_volume / (pre_volume + 1e-8)).item()
            
            features.extend([price_change_ratio, freq_change_ratio, volume_change_ratio])
        else:
            features.extend([1.0, 1.0, 1.0])  # No change indicators
        
        # Activity during airdrop periods
        during_mask = airdrop_masks['during_airdrop']
        if during_mask.sum() > 0:
            during_activity = during_mask.float().mean().item()
            during_avg_price = prices[during_mask].mean().item()
        else:
            during_activity = 0.0
            during_avg_price = 0.0
        
        features.extend([during_activity, during_avg_price])
        
        return torch.tensor(features, dtype=torch.float32)


class NFTFeatureExtractor:
    """
    Extract and process NFT multimodal features.
    """
    
    def __init__(self, visual_model_name: str = 'vit', text_model_name: str = 'bert'):
        self.visual_model_name = visual_model_name
        self.text_model_name = text_model_name
        
        # In practice, these would load actual pre-trained models
        self.visual_extractor = None
        self.text_extractor = None
    
    def extract_visual_features(self, nft_data: Dict[str, Any]) -> torch.Tensor:
        """Extract visual features from NFT image."""
        # Placeholder implementation
        # In practice, this would process actual images through ViT
        if 'image_url' in nft_data and nft_data['image_url']:
            # Simulate processing image through ViT
            return torch.randn(768)  # ViT-base feature dimension
        else:
            return torch.zeros(768)
    
    def extract_text_features(self, nft_data: Dict[str, Any]) -> torch.Tensor:
        """Extract textual features from NFT metadata."""
        # Placeholder implementation
        # In practice, this would process text through BERT
        text_content = str(nft_data.get('name', '')) + ' ' + str(nft_data.get('description', ''))
        if text_content.strip():
            # Simulate processing text through BERT
            return torch.randn(768)  # BERT-base feature dimension
        else:
            return torch.zeros(768)
    
    def extract_metadata_features(self, nft_data: Dict[str, Any]) -> torch.Tensor:
        """Extract features from NFT metadata attributes."""
        features = []
        
        # Basic metadata
        collection_size = nft_data.get('collection_size', 1)
        features.append(np.log1p(collection_size))
        
        rarity_rank = nft_data.get('rarity_rank', collection_size)
        rarity_score = 1.0 - (rarity_rank / collection_size)
        features.append(rarity_score)
        
        # Attribute counts
        attributes = nft_data.get('attributes', [])
        features.append(len(attributes))
        
        # Creator reputation (placeholder)
        creator_reputation = nft_data.get('creator_reputation', 0.5)
        features.append(creator_reputation)
        
        return torch.tensor(features, dtype=torch.float32)