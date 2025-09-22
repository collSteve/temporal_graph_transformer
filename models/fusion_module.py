"""
Temporal-Graph Fusion Module for combining temporal and structural representations.

This module implements Level 3 of our hierarchical architecture, responsible for:
1. Fusing temporal sequence representations with graph structure representations
2. Detecting behavioral changes around airdrop events  
3. Global market pattern detection
4. Final classification of airdrop hunters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math

from ..utils.attention import CrossAttention, GlobalAttentionPooling, AdaptiveAttention


class BehaviorChangeScorer(nn.Module):
    """
    Specialized module for scoring behavioral changes around airdrop events.
    Uses attention over temporal windows to identify suspicious behavior transitions.
    """
    
    def __init__(self, d_model: int, window_size: int = 7):
        super().__init__()
        
        self.d_model = d_model
        self.window_size = window_size  # Days before/after airdrop
        
        # Multi-scale temporal attention
        self.short_term_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, batch_first=True
        )
        self.long_term_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=4, batch_first=True
        )
        
        # Change detection networks
        self.pre_airdrop_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4)
        )
        
        self.post_airdrop_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4)
        )
        
        # Change magnitude scorer
        self.change_scorer = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Temporal consistency scorer
        self.consistency_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def create_airdrop_mask(self, 
                          timestamps: torch.Tensor,
                          airdrop_events: torch.Tensor,
                          window_days: float = 7.0) -> torch.Tensor:
        """
        Create attention mask for airdrop periods.
        
        Args:
            timestamps: Transaction timestamps of shape (batch_size, seq_len)
            airdrop_events: Airdrop announcement timestamps
            window_days: Window size in days around airdrop events
            
        Returns:
            mask: Boolean mask indicating airdrop periods
        """
        window_seconds = window_days * 24 * 3600
        batch_size, seq_len = timestamps.shape
        
        # Expand dimensions for broadcasting
        times_expanded = timestamps.unsqueeze(-1)  # (batch_size, seq_len, 1)
        events_expanded = airdrop_events.unsqueeze(0).unsqueeze(0)  # (1, 1, num_events)
        
        # Check if any timestamp is within window of any airdrop event
        distances = torch.abs(times_expanded - events_expanded)
        within_window = distances <= window_seconds
        airdrop_mask = within_window.any(dim=-1)  # (batch_size, seq_len)
        
        return airdrop_mask
    
    def forward(self, 
                temporal_sequence: torch.Tensor,
                timestamps: torch.Tensor,
                airdrop_events: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            temporal_sequence: Temporal representations from sequence transformer
            timestamps: Transaction timestamps
            airdrop_events: Optional airdrop event timestamps
            
        Returns:
            Dictionary containing behavioral change scores and analysis
        """
        batch_size, seq_len, d_model = temporal_sequence.shape
        
        outputs = {}
        
        if airdrop_events is not None:
            # Create airdrop period mask
            airdrop_mask = self.create_airdrop_mask(timestamps, airdrop_events)
            
            # Separate pre and post airdrop behaviors
            pre_airdrop = temporal_sequence.clone()
            post_airdrop = temporal_sequence.clone()
            
            # Mask out post-airdrop for pre-airdrop analysis
            pre_airdrop[airdrop_mask] = 0
            # Mask out pre-airdrop for post-airdrop analysis  
            post_airdrop[~airdrop_mask] = 0
            
            # Encode pre and post airdrop behaviors
            pre_encoded = self.pre_airdrop_encoder(pre_airdrop.mean(dim=1))  # (batch_size, d_model//4)
            post_encoded = self.post_airdrop_encoder(post_airdrop.mean(dim=1))  # (batch_size, d_model//4)
            
            # Compute change magnitude
            behavior_change = torch.cat([pre_encoded, post_encoded], dim=-1)
            change_score = self.change_scorer(behavior_change)  # (batch_size, 1)
            
            outputs['change_score'] = change_score.squeeze(-1)
            outputs['pre_airdrop_behavior'] = pre_encoded
            outputs['post_airdrop_behavior'] = post_encoded
            outputs['airdrop_mask'] = airdrop_mask
        else:
            # No airdrop events provided, score general consistency
            outputs['change_score'] = torch.zeros(batch_size, device=temporal_sequence.device)
        
        # Temporal consistency scoring (regardless of airdrop events)
        consistency_score = self.consistency_scorer(temporal_sequence.mean(dim=1))
        outputs['consistency_score'] = consistency_score.squeeze(-1)
        
        # Short-term vs long-term behavior comparison
        if seq_len >= 10:
            # Split into short-term (recent) and long-term (historical)
            split_point = seq_len // 2
            short_term = temporal_sequence[:, split_point:]
            long_term = temporal_sequence[:, :split_point]
            
            # Attention-based comparison
            short_attended, _ = self.short_term_attention(short_term, short_term, short_term)
            long_attended, _ = self.long_term_attention(long_term, long_term, long_term)
            
            # Compute behavioral drift
            short_repr = short_attended.mean(dim=1)
            long_repr = long_attended.mean(dim=1)
            behavioral_drift = F.cosine_similarity(short_repr, long_repr, dim=-1)
            
            outputs['behavioral_drift'] = behavioral_drift
        else:
            outputs['behavioral_drift'] = torch.ones(batch_size, device=temporal_sequence.device)
        
        return outputs


class MarketPatternDetector(nn.Module):
    """
    Detects market-wide manipulation patterns across multiple users.
    Uses global attention to identify coordinated activities.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        self.d_model = d_model
        
        # Global market attention
        self.market_attention = GlobalAttentionPooling(d_model, num_heads=8)
        
        # Pattern detection networks
        self.wash_trading_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.coordination_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.market_manipulation_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                user_embeddings: torch.Tensor,
                user_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            user_embeddings: User embeddings of shape (batch_size, num_users, d_model)
            user_mask: Optional mask for valid users
            
        Returns:
            Dictionary containing market pattern detection results
        """
        # Global market representation
        market_repr, attention_weights = self.market_attention(
            user_embeddings, attention_mask=user_mask
        )
        
        # Detect different types of market manipulation
        wash_trading_score = self.wash_trading_detector(market_repr)
        coordination_score = self.coordination_detector(market_repr)
        manipulation_score = self.market_manipulation_detector(market_repr)
        
        return {
            'market_representation': market_repr,
            'wash_trading_score': wash_trading_score.squeeze(-1),
            'coordination_score': coordination_score.squeeze(-1),
            'manipulation_score': manipulation_score.squeeze(-1),
            'market_attention_weights': attention_weights
        }


class TemporalGraphFusion(nn.Module):
    """
    Main Temporal-Graph Fusion module that combines all components.
    
    This implements Level 3 of our hierarchical architecture, responsible for:
    1. Cross-modal fusion of temporal and graph representations
    2. Behavioral change detection and scoring
    3. Market-wide pattern detection
    4. Final classification decisions
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        d_model = config['d_model']
        
        # Cross-modal attention for temporal-graph fusion
        self.temporal_to_graph = CrossAttention(
            d_model=d_model,
            num_heads=config.get('fusion_heads', 8),
            dropout=config.get('dropout', 0.1)
        )
        
        self.graph_to_temporal = CrossAttention(
            d_model=d_model,
            num_heads=config.get('fusion_heads', 8),
            dropout=config.get('dropout', 0.1)
        )
        
        # Adaptive attention for combining different representations
        self.adaptive_fusion = AdaptiveAttention(
            d_model=d_model,
            num_attention_types=3  # temporal, graph, cross-modal
        )
        
        # Behavioral change detection
        self.behavior_change_scorer = BehaviorChangeScorer(
            d_model=d_model,
            window_size=config.get('airdrop_window_days', 7)
        )
        
        # Market pattern detection
        self.market_detector = MarketPatternDetector(d_model)
        
        # Feature fusion layers
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),  # temporal + graph + cross-modal
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model + 4, d_model // 2),  # +4 for behavioral scores
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.3)),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.2)),
            nn.Linear(d_model // 4, 2)  # hunter vs legitimate
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                temporal_embeddings: torch.Tensor,
                graph_embeddings: torch.Tensor,
                temporal_sequences: torch.Tensor,
                timestamps: torch.Tensor,
                airdrop_events: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            temporal_embeddings: User-level temporal representations (batch_size, d_model)
            graph_embeddings: User-level graph representations (batch_size, d_model)  
            temporal_sequences: Full temporal sequences (batch_size, seq_len, d_model)
            timestamps: Transaction timestamps (batch_size, seq_len)
            airdrop_events: Optional airdrop event timestamps
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing classification results and analysis
        """
        batch_size = temporal_embeddings.shape[0]
        
        # 1. Cross-modal attention fusion
        # Temporal attending to graph structure
        temp_to_graph, temp_graph_attn = self.temporal_to_graph(
            query=temporal_embeddings.unsqueeze(1),
            key=graph_embeddings.unsqueeze(1),
            value=graph_embeddings.unsqueeze(1)
        )
        temp_to_graph = temp_to_graph.squeeze(1)
        
        # Graph attending to temporal patterns
        graph_to_temp, graph_temp_attn = self.graph_to_temporal(
            query=graph_embeddings.unsqueeze(1),
            key=temporal_embeddings.unsqueeze(1),
            value=temporal_embeddings.unsqueeze(1)
        )
        graph_to_temp = graph_to_temp.squeeze(1)
        
        # 2. Adaptive fusion of multiple representations
        fusion_inputs = [temporal_embeddings, graph_embeddings, temp_to_graph]
        context = torch.stack(fusion_inputs, dim=1)  # (batch_size, 3, d_model)
        fused_representation = self.adaptive_fusion(fusion_inputs, context)
        
        # 3. Alternative fusion strategy (concatenation + projection)
        concatenated = torch.cat([temporal_embeddings, graph_embeddings, temp_to_graph], dim=-1)
        feature_fused = self.feature_fusion(concatenated)
        
        # 4. Behavioral change analysis
        behavior_analysis = self.behavior_change_scorer(
            temporal_sequences, timestamps, airdrop_events
        )
        
        # 5. Market pattern detection (if batch represents multiple users)
        if temporal_embeddings.dim() == 3:  # (batch_size, num_users, d_model)
            market_analysis = self.market_detector(temporal_embeddings)
        else:
            # Single user per batch, create dummy market analysis
            market_analysis = {
                'market_representation': feature_fused.mean(dim=0, keepdim=True),
                'wash_trading_score': torch.zeros(1, device=feature_fused.device),
                'coordination_score': torch.zeros(1, device=feature_fused.device),
                'manipulation_score': torch.zeros(1, device=feature_fused.device)
            }
        
        # 6. Combine behavioral scores for classification
        behavioral_features = torch.stack([
            behavior_analysis['change_score'],
            behavior_analysis['consistency_score'],
            behavior_analysis['behavioral_drift'],
            market_analysis['manipulation_score'].expand(batch_size)
        ], dim=-1)  # (batch_size, 4)
        
        # 7. Final classification
        classification_input = torch.cat([feature_fused, behavioral_features], dim=-1)
        logits = self.classifier(classification_input)
        
        # 8. Confidence estimation
        confidence = self.confidence_estimator(feature_fused)
        
        # 9. Prepare outputs
        outputs = {
            'logits': logits,
            'probabilities': F.softmax(logits, dim=-1),
            'confidence': confidence.squeeze(-1),
            'fused_representation': feature_fused,
            'behavioral_scores': behavioral_features,
            'behavior_analysis': behavior_analysis,
            'market_analysis': market_analysis
        }
        
        if return_attention:
            outputs['attention_weights'] = {
                'temporal_to_graph': temp_graph_attn,
                'graph_to_temporal': graph_temp_attn
            }
        
        return outputs
    
    def predict(self, 
               temporal_embeddings: torch.Tensor,
               graph_embeddings: torch.Tensor,
               temporal_sequences: torch.Tensor,
               timestamps: torch.Tensor,
               airdrop_events: Optional[torch.Tensor] = None,
               threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Make predictions with confidence thresholding.
        
        Args:
            threshold: Confidence threshold for predictions
            
        Returns:
            Dictionary containing predictions and confidence scores
        """
        with torch.no_grad():
            outputs = self.forward(
                temporal_embeddings, graph_embeddings, 
                temporal_sequences, timestamps, airdrop_events
            )
            
            probabilities = outputs['probabilities']
            confidence = outputs['confidence']
            
            # Apply confidence thresholding
            high_confidence_mask = confidence > threshold
            predictions = probabilities.argmax(dim=-1)
            
            # Set low-confidence predictions to uncertain (-1)
            predictions[~high_confidence_mask] = -1
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'confidence': confidence,
                'high_confidence_mask': high_confidence_mask,
                'behavioral_scores': outputs['behavioral_scores']
            }


def create_fusion_config(
    d_model: int = 256,
    fusion_heads: int = 8,
    dropout: float = 0.1,
    airdrop_window_days: int = 7
) -> Dict:
    """
    Create configuration for TemporalGraphFusion.
    
    Returns:
        Configuration dictionary
    """
    return {
        'd_model': d_model,
        'fusion_heads': fusion_heads,
        'dropout': dropout,
        'airdrop_window_days': airdrop_window_days
    }