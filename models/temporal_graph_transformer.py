"""
Main Temporal Graph Transformer architecture for airdrop hunter detection.

This module implements the complete hierarchical architecture that combines:
Level 1: Transaction Sequence Transformer (temporal patterns)
Level 2: Graph Structure Transformer (graph relationships)  
Level 3: Temporal-Graph Fusion (behavioral change detection + classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any
import warnings

from .temporal_encoder import TransactionSequenceTransformer, create_transaction_sequence_config
from .graph_encoder import GraphStructureTransformer, create_graph_structure_config
from .fusion_module import TemporalGraphFusion, create_fusion_config


class TemporalGraphTransformer(nn.Module):
    """
    Complete Temporal Graph Transformer for airdrop hunter detection.
    
    This model implements a novel hierarchical architecture that addresses
    the key limitations of ARTEMIS:
    
    1. Temporal Dynamics: Models behavioral changes around airdrop events
    2. Global Attention: Unlimited receptive field vs ARTEMIS's 3-hop limitation
    3. Learned Features: End-to-end learning vs manual feature engineering
    4. Multi-Scale Patterns: Transaction → User → Market level analysis
    
    Architecture:
    Level 1: Transaction Sequence Transformer
        - Functional time encoding for behavioral rhythm detection
        - Self-attention over chronological transaction sequences
        - Change point detection for behavioral transitions
        
    Level 2: Graph Structure Transformer  
        - Multi-modal edge features (visual + textual + transaction)
        - Structural positional encoding (centrality measures)
        - Graph attention with unlimited receptive field
        
    Level 3: Temporal-Graph Fusion
        - Cross-modal attention between temporal and graph representations
        - Behavioral change scoring around airdrop events
        - Market-wide pattern detection
        - Final classification with confidence estimation
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.d_model = config['d_model']
        
        # Validate configuration
        self._validate_config(config)
        
        # Level 1: Transaction Sequence Transformer
        temporal_config = create_transaction_sequence_config(
            d_model=config['d_model'],
            num_layers=config.get('temporal_layers', 4),
            num_heads=config.get('temporal_heads', 8),
            dropout=config.get('dropout', 0.1),
            use_nft_features=config.get('use_nft_features', True),
            pooling=config.get('temporal_pooling', 'attention')
        )
        self.temporal_encoder = TransactionSequenceTransformer(temporal_config)
        
        # Level 2: Graph Structure Transformer
        graph_config = create_graph_structure_config(
            d_model=config['d_model'],
            num_layers=config.get('graph_layers', 6),
            num_heads=config.get('graph_heads', 8),
            dropout=config.get('dropout', 0.1),
            nft_visual_dim=config.get('nft_visual_dim', 768),
            nft_text_dim=config.get('nft_text_dim', 768),
            transaction_dim=config.get('transaction_dim', 64),
            market_features_dim=config.get('market_features_dim', 32)
        )
        self.graph_encoder = GraphStructureTransformer(graph_config)
        
        # Level 3: Temporal-Graph Fusion
        fusion_config = create_fusion_config(
            d_model=config['d_model'],
            fusion_heads=config.get('fusion_heads', 8),
            dropout=config.get('dropout', 0.1),
            airdrop_window_days=config.get('airdrop_window_days', 7)
        )
        self.fusion_module = TemporalGraphFusion(fusion_config)
        
        # Model initialization
        self.apply(self._init_weights)
        
        # Training state
        self.training_step = 0
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate model configuration."""
        required_keys = ['d_model']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
                
        if config['d_model'] % 8 != 0:
            warnings.warn("d_model should be divisible by 8 for optimal attention performance")
            
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0.0)
            torch.nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, 
                batch: Dict[str, Any],
                return_attention: bool = False,
                return_intermediate: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete hierarchical architecture.
        
        Args:
            batch: Dictionary containing:
                - 'transaction_features': Dict of transaction attributes per user
                - 'timestamps': Transaction timestamps (batch_size, seq_len)
                - 'node_features': Initial node features (num_nodes, node_dim)
                - 'edge_index': Graph edge indices (2, num_edges)
                - 'edge_features': Dict of multi-modal edge features
                - 'airdrop_events': Optional airdrop event timestamps
                - 'user_indices': Mapping from batch to graph nodes
                - 'attention_mask': Optional padding mask for transactions
                
            return_attention: Whether to return attention weights for interpretability
            return_intermediate: Whether to return intermediate representations
            
        Returns:
            Dictionary containing:
                - 'logits': Classification logits (batch_size, 2)
                - 'probabilities': Class probabilities (batch_size, 2)  
                - 'confidence': Prediction confidence scores (batch_size,)
                - 'behavioral_scores': Behavioral change scores (batch_size, 4)
                - Additional analysis and attention weights if requested
        """
        # Extract batch components
        transaction_features = batch['transaction_features']
        timestamps = batch['timestamps']
        node_features = batch['node_features']
        edge_index = batch['edge_index']
        edge_features = batch['edge_features']
        airdrop_events = batch.get('airdrop_events', None)
        user_indices = batch.get('user_indices', None)
        attention_mask = batch.get('attention_mask', None)
        
        batch_size = timestamps.shape[0]
        
        # Level 1: Transaction Sequence Transformer
        temporal_outputs = self.temporal_encoder(
            transaction_features=transaction_features,
            timestamps=timestamps,
            airdrop_events=airdrop_events,
            attention_mask=attention_mask
        )
        
        temporal_embeddings = temporal_outputs['sequence_embedding']  # (batch_size, d_model)
        temporal_sequences = temporal_outputs['temporal_sequence']    # (batch_size, seq_len, d_model)
        
        # Level 2: Graph Structure Transformer
        graph_outputs = self.graph_encoder(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            return_attention=return_attention
        )
        
        graph_node_embeddings = graph_outputs['node_embeddings']  # (num_nodes, d_model)
        graph_global_embedding = graph_outputs['graph_embedding']  # (d_model,)
        
        # Map graph node embeddings to batch users
        if user_indices is not None:
            # Extract embeddings for users in current batch
            graph_embeddings = graph_node_embeddings[user_indices]  # (batch_size, d_model)
        else:
            # Assume first batch_size nodes correspond to batch users
            graph_embeddings = graph_node_embeddings[:batch_size]
            
        # Level 3: Temporal-Graph Fusion
        fusion_outputs = self.fusion_module(
            temporal_embeddings=temporal_embeddings,
            graph_embeddings=graph_embeddings,
            temporal_sequences=temporal_sequences,
            timestamps=timestamps,
            airdrop_events=airdrop_events,
            return_attention=return_attention
        )
        
        # Prepare main outputs
        outputs = {
            'logits': fusion_outputs['logits'],
            'probabilities': fusion_outputs['probabilities'],
            'confidence': fusion_outputs['confidence'],
            'behavioral_scores': fusion_outputs['behavioral_scores'],
            'fused_representation': fusion_outputs['fused_representation']
        }
        
        # Add intermediate representations if requested
        if return_intermediate:
            outputs['intermediate'] = {
                'temporal_embeddings': temporal_embeddings,
                'graph_embeddings': graph_embeddings,
                'temporal_sequences': temporal_sequences,
                'graph_node_embeddings': graph_node_embeddings,
                'graph_global_embedding': graph_global_embedding,
                'behavior_analysis': fusion_outputs['behavior_analysis'],
                'market_analysis': fusion_outputs['market_analysis']
            }
            
        # Add attention weights if requested
        if return_attention:
            attention_weights = {}
            
            # Temporal attention weights
            if hasattr(self.temporal_encoder, 'get_attention_weights'):
                attention_weights['temporal'] = self.temporal_encoder.get_attention_weights(
                    transaction_features, timestamps
                )
                
            # Graph attention weights
            if 'attention_weights' in graph_outputs:
                attention_weights['graph'] = graph_outputs['attention_weights']
                
            # Fusion attention weights
            if 'attention_weights' in fusion_outputs:
                attention_weights['fusion'] = fusion_outputs['attention_weights']
                
            outputs['attention_weights'] = attention_weights
            
        return outputs
    
    def predict(self, 
               batch: Dict[str, Any],
               threshold: float = 0.5,
               return_confidence: bool = True) -> Dict[str, torch.Tensor]:
        """
        Make predictions with optional confidence thresholding.
        
        Args:
            batch: Input batch dictionary
            threshold: Confidence threshold for predictions
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary containing predictions and optionally confidence scores
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch)
            
            probabilities = outputs['probabilities']
            confidence = outputs['confidence']
            
            # Get predicted classes
            predictions = probabilities.argmax(dim=-1)  # (batch_size,)
            hunter_probabilities = probabilities[:, 1]   # Probability of being a hunter
            
            # Apply confidence thresholding if specified
            if threshold > 0:
                high_confidence_mask = confidence > threshold
                low_confidence_predictions = predictions.clone()
                low_confidence_predictions[~high_confidence_mask] = -1  # Uncertain
            else:
                high_confidence_mask = torch.ones_like(confidence, dtype=torch.bool)
                low_confidence_predictions = predictions
            
            result = {
                'predictions': predictions,
                'hunter_probabilities': hunter_probabilities,
                'high_confidence_predictions': low_confidence_predictions,
                'high_confidence_mask': high_confidence_mask
            }
            
            if return_confidence:
                result['confidence'] = confidence
                result['behavioral_scores'] = outputs['behavioral_scores']
                
            return result
    
    def get_embeddings(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Extract learned embeddings for analysis and visualization.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Dictionary containing various learned representations
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch, return_intermediate=True)
            
            return {
                'temporal_embeddings': outputs['intermediate']['temporal_embeddings'],
                'graph_embeddings': outputs['intermediate']['graph_embeddings'],
                'fused_embeddings': outputs['fused_representation'],
                'behavioral_scores': outputs['behavioral_scores']
            }
    
    def analyze_behavior_changes(self, 
                               batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Detailed analysis of behavioral changes for interpretability.
        
        Args:
            batch: Input batch dictionary with airdrop events
            
        Returns:
            Dictionary containing behavioral change analysis
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch, return_intermediate=True)
            behavior_analysis = outputs['intermediate']['behavior_analysis']
            
            return {
                'change_scores': behavior_analysis['change_score'],
                'consistency_scores': behavior_analysis['consistency_score'],
                'behavioral_drift': behavior_analysis['behavioral_drift'],
                'airdrop_mask': behavior_analysis.get('airdrop_mask', None),
                'pre_airdrop_behavior': behavior_analysis.get('pre_airdrop_behavior', None),
                'post_airdrop_behavior': behavior_analysis.get('post_airdrop_behavior', None)
            }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of model architecture and parameters."""
        
        def count_parameters(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            'total_parameters': count_parameters(self),
            'temporal_encoder_parameters': count_parameters(self.temporal_encoder),
            'graph_encoder_parameters': count_parameters(self.graph_encoder),
            'fusion_module_parameters': count_parameters(self.fusion_module),
            'd_model': self.d_model,
            'config': self.config
        }
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, 
                       device: Optional[torch.device] = None) -> 'TemporalGraphTransformer':
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to load model on
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract config and create model
        config = checkpoint['config']
        model = cls(config)
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        
        # Load training state if available
        if 'training_step' in checkpoint:
            model.training_step = checkpoint['training_step']
            
        return model
    
    def save_checkpoint(self, checkpoint_path: str, 
                       optimizer_state: Optional[Dict] = None,
                       scheduler_state: Optional[Dict] = None,
                       **kwargs) -> None:
        """
        Save model checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
            optimizer_state: Optional optimizer state dict
            scheduler_state: Optional scheduler state dict
            **kwargs: Additional items to save
        """
        checkpoint = {
            'config': self.config,
            'state_dict': self.state_dict(),
            'training_step': self.training_step,
            **kwargs
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state'] = optimizer_state
            
        if scheduler_state is not None:
            checkpoint['scheduler_state'] = scheduler_state
            
        torch.save(checkpoint, checkpoint_path)


def create_model_config(
    d_model: int = 256,
    temporal_layers: int = 4,
    graph_layers: int = 6,
    temporal_heads: int = 8,
    graph_heads: int = 8,
    fusion_heads: int = 8,
    dropout: float = 0.1,
    use_nft_features: bool = True,
    airdrop_window_days: int = 7,
    temporal_pooling: str = 'attention'
) -> Dict[str, Any]:
    """
    Create a complete model configuration.
    
    Args:
        d_model: Model dimension
        temporal_layers: Number of temporal transformer layers
        graph_layers: Number of graph transformer layers
        temporal_heads: Number of attention heads in temporal encoder
        graph_heads: Number of attention heads in graph encoder
        fusion_heads: Number of attention heads in fusion module
        dropout: Dropout rate
        use_nft_features: Whether to use NFT multimodal features
        airdrop_window_days: Window size for airdrop event detection
        temporal_pooling: Pooling strategy for temporal sequences
        
    Returns:
        Complete model configuration dictionary
    """
    return {
        # Model architecture
        'd_model': d_model,
        'temporal_layers': temporal_layers,
        'graph_layers': graph_layers,
        'temporal_heads': temporal_heads,
        'graph_heads': graph_heads,
        'fusion_heads': fusion_heads,
        
        # Training parameters
        'dropout': dropout,
        
        # Feature configuration
        'use_nft_features': use_nft_features,
        'nft_visual_dim': 768,  # ViT features
        'nft_text_dim': 768,    # BERT features
        'transaction_dim': 64,
        'market_features_dim': 32,
        
        # Temporal configuration
        'temporal_pooling': temporal_pooling,
        'airdrop_window_days': airdrop_window_days,
        
        # Graph configuration
        'max_distance': 10,
        'global_heads': 4
    }