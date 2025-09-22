"""
Transaction Sequence Transformer for modeling temporal user behavior patterns.

This module implements the first level of our hierarchical architecture,
focusing on capturing temporal patterns within individual user's transaction sequences.
It uses our novel Functional Time Encoding to detect behavioral changes around airdrop events.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import math

# Try relative import first, fall back to absolute import
try:
    from ..utils.time_encoding import (
        FunctionalTimeEncoding, 
        BehaviorChangeTimeEncoding,
        create_time_mask
    )
except ImportError:
    from utils.time_encoding import (
        FunctionalTimeEncoding, 
        BehaviorChangeTimeEncoding,
        create_time_mask
    )


class TransactionFeatureEmbedding(nn.Module):
    """
    Embeds transaction features into a continuous representation.
    Handles both numerical and categorical transaction attributes.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        d_model = config['d_model']
        
        # Numerical feature projections
        self.value_proj = nn.Linear(1, d_model // 4)  # Transaction value
        self.gas_proj = nn.Linear(1, d_model // 8)    # Gas fees
        self.volume_proj = nn.Linear(1, d_model // 8)  # Volume metrics
        
        # Categorical embeddings
        self.tx_type_embedding = nn.Embedding(
            config.get('num_tx_types', 10), d_model // 4
        )
        self.nft_collection_embedding = nn.Embedding(
            config.get('num_collections', 1000), d_model // 4
        )
        
        # NFT-specific features (if available)
        if config.get('use_nft_features', True):
            self.nft_visual_proj = nn.Linear(
                config.get('nft_visual_dim', 768), d_model // 8
            )
            self.nft_text_proj = nn.Linear(
                config.get('nft_text_dim', 768), d_model // 8
            )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, transaction_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            transaction_features: Dictionary containing transaction attributes
            
        Returns:
            embedded_features: Tensor of shape (batch_size, seq_len, d_model)
        """
        embeddings = []
        
        # Numerical features
        if 'value' in transaction_features:
            value_emb = self.value_proj(
                transaction_features['value'].unsqueeze(-1)
            )
            embeddings.append(value_emb)
            
        if 'gas_fee' in transaction_features:
            gas_emb = self.gas_proj(
                transaction_features['gas_fee'].unsqueeze(-1)
            )
            embeddings.append(gas_emb)
            
        if 'volume' in transaction_features:
            volume_emb = self.volume_proj(
                transaction_features['volume'].unsqueeze(-1)
            )
            embeddings.append(volume_emb)
        
        # Categorical features
        if 'tx_type' in transaction_features:
            type_emb = self.tx_type_embedding(transaction_features['tx_type'])
            embeddings.append(type_emb)
            
        if 'nft_collection' in transaction_features:
            collection_emb = self.nft_collection_embedding(
                transaction_features['nft_collection']
            )
            embeddings.append(collection_emb)
        
        # NFT multimodal features
        if self.config.get('use_nft_features', True):
            if 'nft_visual' in transaction_features:
                visual_emb = self.nft_visual_proj(transaction_features['nft_visual'])
                embeddings.append(visual_emb)
                
            if 'nft_text' in transaction_features:
                text_emb = self.nft_text_proj(transaction_features['nft_text'])
                embeddings.append(text_emb)
        
        # Concatenate all embeddings
        if embeddings:
            combined = torch.cat(embeddings, dim=-1)
            
            # Project to correct dimension if needed
            if combined.size(-1) != self.config['d_model']:
                proj = nn.Linear(combined.size(-1), self.config['d_model']).to(combined.device)
                combined = proj(combined)
                
            return self.layer_norm(combined)
        else:
            # Fallback to zeros if no features available
            batch_size, seq_len = list(transaction_features.values())[0].shape[:2]
            return torch.zeros(batch_size, seq_len, self.config['d_model'])


class ChangePointAttention(nn.Module):
    """
    Specialized attention mechanism for detecting behavioral change points.
    Focuses on transitions in user behavior patterns around airdrop events.
    """
    
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0
        
        # Attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Change detection weights
        self.change_detector = nn.Linear(d_model, 1)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor, 
                timestamps: torch.Tensor,
                airdrop_events: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            timestamps: Transaction timestamps
            airdrop_events: Optional airdrop event timestamps
            
        Returns:
            attended_output: Attended sequence representations
            change_scores: Change point scores for each position
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply temporal mask if needed
        if timestamps is not None:
            time_mask = create_time_mask(timestamps, window_size=86400.0 * 7)  # 7 days
            scores.masked_fill_(~time_mask.unsqueeze(1), float('-inf'))
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        attended = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.out_proj(attended)
        
        # Compute change point scores
        change_scores = self.change_detector(output).squeeze(-1)
        
        return output, change_scores


class TransactionSequenceTransformer(nn.Module):
    """
    Main Transaction Sequence Transformer module.
    
    This implements Level 1 of our hierarchical architecture, modeling temporal patterns
    within individual user's transaction sequences using:
    1. Functional time encoding for behavioral rhythm detection
    2. Multi-head self-attention for sequence modeling
    3. Change point detection for identifying airdrop hunting behavior
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        d_model = config['d_model']
        
        # Transaction feature embedding
        self.feature_embedding = TransactionFeatureEmbedding(config)
        
        # Temporal encoding
        if config.get('use_behavior_change_encoding', True):
            self.time_encoding = BehaviorChangeTimeEncoding(d_model)
        else:
            self.time_encoding = FunctionalTimeEncoding(d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=config.get('num_heads', 8),
                dim_feedforward=config.get('dim_feedforward', d_model * 4),
                dropout=config.get('dropout', 0.1),
                activation='relu',
                batch_first=True
            )
            for _ in range(config.get('num_layers', 4))
        ])
        
        # Change point attention
        self.change_attention = ChangePointAttention(
            d_model, 
            num_heads=config.get('change_attention_heads', 4)
        )
        
        # Sequence pooling strategies
        self.pooling_strategy = config.get('pooling', 'attention')
        if self.pooling_strategy == 'attention':
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=config.get('pool_heads', 4),
                batch_first=True
            )
            self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Output projections
        self.sequence_proj = nn.Linear(d_model, d_model)
        self.change_proj = nn.Linear(d_model, 1)
        
        # Layer normalization
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, 
                transaction_features: Dict[str, torch.Tensor],
                timestamps: torch.Tensor,
                airdrop_events: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            transaction_features: Dictionary of transaction attributes
            timestamps: Transaction timestamps of shape (batch_size, seq_len)
            airdrop_events: Optional airdrop event timestamps
            attention_mask: Optional attention mask for padding
            
        Returns:
            Dictionary containing:
                - sequence_embedding: User-level temporal representation
                - change_scores: Behavioral change scores
                - attention_weights: Attention weights for interpretability
        """
        batch_size, seq_len = timestamps.shape
        
        # 1. Embed transaction features
        feature_emb = self.feature_embedding(transaction_features)
        
        # 2. Add temporal encoding
        time_emb = self.time_encoding(timestamps, airdrop_events)
        
        # Combine features and time
        x = feature_emb + time_emb
        
        # 3. Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, src_key_padding_mask=attention_mask)
        
        # 4. Apply change point attention
        x, change_scores = self.change_attention(x, timestamps, airdrop_events)
        
        # 5. Pool sequence to user-level representation
        if self.pooling_strategy == 'mean':
            if attention_mask is not None:
                # Masked mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
                x_masked = x.masked_fill(mask_expanded, 0.0)
                sequence_emb = x_masked.sum(dim=1) / (~attention_mask).sum(dim=1, keepdim=True)
            else:
                sequence_emb = x.mean(dim=1)
                
        elif self.pooling_strategy == 'max':
            if attention_mask is not None:
                x_masked = x.masked_fill(attention_mask.unsqueeze(-1), float('-inf'))
                sequence_emb = x_masked.max(dim=1)[0]
            else:
                sequence_emb = x.max(dim=1)[0]
                
        elif self.pooling_strategy == 'attention':
            # Attention pooling
            query = self.pool_query.expand(batch_size, 1, -1)
            pooled, attention_weights = self.attention_pool(
                query, x, x, 
                key_padding_mask=attention_mask
            )
            sequence_emb = pooled.squeeze(1)
        else:
            # Last token pooling
            if attention_mask is not None:
                # Get last non-padded token
                lengths = (~attention_mask).sum(dim=1) - 1
                sequence_emb = x[torch.arange(batch_size), lengths]
            else:
                sequence_emb = x[:, -1]
        
        # 6. Final projections
        sequence_emb = self.final_norm(self.sequence_proj(sequence_emb))
        
        # 7. Aggregate change scores
        if attention_mask is not None:
            change_mask = ~attention_mask
            masked_scores = change_scores.masked_fill(attention_mask, 0.0)
            aggregated_change = masked_scores.sum(dim=1) / change_mask.sum(dim=1)
        else:
            aggregated_change = change_scores.mean(dim=1)
        
        return {
            'sequence_embedding': sequence_emb,
            'change_scores': aggregated_change,
            'temporal_sequence': x,  # For visualization/analysis
            'individual_change_scores': change_scores
        }
    
    def get_attention_weights(self, 
                            transaction_features: Dict[str, torch.Tensor],
                            timestamps: torch.Tensor) -> torch.Tensor:
        """
        Extract attention weights for interpretability analysis.
        """
        with torch.no_grad():
            outputs = self.forward(transaction_features, timestamps)
            return outputs.get('attention_weights', None)


def create_transaction_sequence_config(
    d_model: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    dropout: float = 0.1,
    use_nft_features: bool = True,
    pooling: str = 'attention'
) -> Dict:
    """
    Create a configuration dictionary for TransactionSequenceTransformer.
    
    Args:
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_nft_features: Whether to use NFT multimodal features
        pooling: Pooling strategy ('mean', 'max', 'attention', 'last')
        
    Returns:
        Configuration dictionary
    """
    return {
        'd_model': d_model,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'dim_feedforward': d_model * 4,
        'dropout': dropout,
        'use_nft_features': use_nft_features,
        'use_behavior_change_encoding': True,
        'pooling': pooling,
        'change_attention_heads': 4,
        'pool_heads': 4,
        'num_tx_types': 10,
        'num_collections': 1000,
        'nft_visual_dim': 768,
        'nft_text_dim': 768
    }