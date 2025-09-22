"""
ARTEMIS baseline implementation for comparison with our Temporal Graph Transformer.

This module replicates the ARTEMIS architecture from the original paper:
- 3-layer Graph Convolutional Network
- Manual feature engineering (price manipulation detection)
- Multi-modal NFT features (ViT + BERT)
- Static analysis without temporal dynamics

This serves as our baseline to demonstrate improvements of the Temporal Graph Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.nn import MessagePassing
from typing import Dict, Optional, Tuple, List, Any
import math
import numpy as np


class ArtemisFirstLayerConv(MessagePassing):
    """
    ARTEMIS's first layer that combines node and edge features.
    Replicates the original implementation from the paper.
    """
    
    def __init__(self, in_node_channels: int, in_edge_channels: int, 
                 out_channels: int, aggr: str = 'mean'):
        super(ArtemisFirstLayerConv, self).__init__(aggr=aggr)
        
        self.lin = nn.Linear(in_node_channels + in_edge_channels, out_channels)
        self.aggr = aggr

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features of shape (num_nodes, in_node_channels)
            edge_index: Edge indices of shape (2, num_edges)
            edge_attr: Edge features of shape (num_edges, in_edge_channels)
            
        Returns:
            Updated node features after first layer convolution
        """
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), 
                            x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Combine neighbor node features with edge features."""
        return self.lin(torch.cat([x_j, edge_attr], dim=-1))

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Apply activation after aggregation."""
        return F.relu(aggr_out)


class ArtemisManualFeatures(nn.Module):
    """
    Manual feature engineering from ARTEMIS paper.
    
    Implements:
    1. Benford's Law features for price manipulation detection
    2. Price rounding detection features
    3. Asset turnover features
    4. Wallet activity features
    """
    
    def __init__(self, feature_dim: int = 32):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Feature projection layers
        self.benford_proj = nn.Linear(9, feature_dim // 4)  # 9 digits
        self.rounding_proj = nn.Linear(10, feature_dim // 4)  # 0-9 endings
        self.turnover_proj = nn.Linear(4, feature_dim // 4)  # Turnover metrics
        self.activity_proj = nn.Linear(6, feature_dim // 4)  # Activity metrics
        
    def compute_benford_features(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Compute Benford's Law features from transaction prices.
        
        Args:
            prices: Transaction prices
            
        Returns:
            Benford's law deviation features
        """
        # Convert prices to strings and extract first digits
        price_strings = prices.abs().int().cpu().numpy().astype(str)
        first_digits = [int(s[0]) if s != '0' else 1 for s in price_strings]
        
        # Count frequency of each digit (1-9)
        digit_counts = torch.zeros(9, device=prices.device)
        for digit in first_digits:
            if 1 <= digit <= 9:
                digit_counts[digit - 1] += 1
        
        # Normalize to frequencies
        total_count = len(first_digits)
        if total_count > 0:
            digit_freqs = digit_counts / total_count
        else:
            digit_freqs = torch.ones(9, device=prices.device) / 9
        
        # Expected Benford's Law frequencies
        expected_freqs = torch.tensor([
            math.log10((d + 1) / d) for d in range(1, 10)
        ], device=prices.device)
        
        # Deviation from Benford's Law
        benford_deviation = torch.abs(digit_freqs - expected_freqs)
        
        return benford_deviation
    
    def compute_rounding_features(self, prices: torch.Tensor) -> torch.Tensor:
        """
        Compute price rounding features.
        
        Args:
            prices: Transaction prices
            
        Returns:
            Price rounding pattern features
        """
        # Extract last digits
        price_ints = prices.abs().int()
        last_digits = price_ints % 10
        
        # Count frequency of each last digit (0-9)
        digit_counts = torch.zeros(10, device=prices.device)
        for i in range(10):
            digit_counts[i] = (last_digits == i).float().sum()
        
        # Normalize to frequencies
        total_count = len(prices)
        if total_count > 0:
            digit_freqs = digit_counts / total_count
        else:
            digit_freqs = torch.ones(10, device=prices.device) / 10
        
        return digit_freqs
    
    def forward(self, transaction_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            transaction_data: Dictionary containing:
                - 'prices': Transaction prices
                - 'volumes': Transaction volumes  
                - 'holding_times': NFT holding durations
                - 'unique_addresses': Number of unique addresses interacted with
                - 'transaction_counts': Number of transactions
                - 'contract_calls': Number of contract interactions
                
        Returns:
            Manual features of shape (batch_size, feature_dim)
        """
        batch_size = list(transaction_data.values())[0].shape[0]
        device = list(transaction_data.values())[0].device
        
        features = []
        
        # 1. Benford's Law features
        if 'prices' in transaction_data:
            benford_feats = []
            for i in range(batch_size):
                user_prices = transaction_data['prices'][i]
                benford_feat = self.compute_benford_features(user_prices)
                benford_feats.append(benford_feat)
            benford_features = torch.stack(benford_feats)
            benford_projected = self.benford_proj(benford_features)
            features.append(benford_projected)
        
        # 2. Price rounding features
        if 'prices' in transaction_data:
            rounding_feats = []
            for i in range(batch_size):
                user_prices = transaction_data['prices'][i]
                rounding_feat = self.compute_rounding_features(user_prices)
                rounding_feats.append(rounding_feat)
            rounding_features = torch.stack(rounding_feats)
            rounding_projected = self.rounding_proj(rounding_features)
            features.append(rounding_projected)
        
        # 3. Asset turnover features
        turnover_feats = []
        if 'holding_times' in transaction_data:
            avg_holding = transaction_data['holding_times'].mean(dim=-1, keepdim=True)
            min_holding = transaction_data['holding_times'].min(dim=-1, keepdim=True)[0]
            max_holding = transaction_data['holding_times'].max(dim=-1, keepdim=True)[0]
            holding_std = transaction_data['holding_times'].std(dim=-1, keepdim=True)
            turnover_raw = torch.cat([avg_holding, min_holding, max_holding, holding_std], dim=-1)
        else:
            turnover_raw = torch.zeros(batch_size, 4, device=device)
        turnover_projected = self.turnover_proj(turnover_raw)
        features.append(turnover_projected)
        
        # 4. Wallet activity features
        activity_feats = []
        activity_raw = torch.zeros(batch_size, 6, device=device)
        if 'unique_addresses' in transaction_data:
            activity_raw[:, 0:1] = transaction_data['unique_addresses'].unsqueeze(-1)
        if 'transaction_counts' in transaction_data:
            activity_raw[:, 1:2] = transaction_data['transaction_counts'].unsqueeze(-1)
        if 'contract_calls' in transaction_data:
            activity_raw[:, 2:3] = transaction_data['contract_calls'].unsqueeze(-1)
        if 'volumes' in transaction_data:
            activity_raw[:, 3:4] = transaction_data['volumes'].mean(dim=-1, keepdim=True)
            activity_raw[:, 4:5] = transaction_data['volumes'].std(dim=-1, keepdim=True)
            activity_raw[:, 5:6] = transaction_data['volumes'].sum(dim=-1, keepdim=True)
        
        activity_projected = self.activity_proj(activity_raw)
        features.append(activity_projected)
        
        # Concatenate all features
        if features:
            return torch.cat(features, dim=-1)
        else:
            return torch.zeros(batch_size, self.feature_dim, device=device)


class ArtemisMultimodalModule(nn.Module):
    """
    ARTEMIS's multimodal module for NFT feature extraction.
    
    Combines ViT visual features and BERT textual features using attention.
    Replicates the approach from the original paper.
    """
    
    def __init__(self, 
                 visual_dim: int = 768,
                 text_dim: int = 768,
                 output_dim: int = 256):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.output_dim = output_dim
        
        # Feature projections
        self.visual_proj = nn.Linear(visual_dim, output_dim // 2)
        self.text_proj = nn.Linear(text_dim, output_dim // 2)
        
        # Attention mechanism (as described in ARTEMIS paper)
        self.visual_attention = nn.Linear(output_dim // 2, 1)
        self.text_attention = nn.Linear(output_dim // 2, 1)
        
        # Fusion layer
        self.fusion = nn.Linear(output_dim, output_dim)
        
    def forward(self, visual_features: torch.Tensor, 
                text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: ViT features of shape (num_nfts, visual_dim)
            text_features: BERT features of shape (num_nfts, text_dim)
            
        Returns:
            Fused multimodal features of shape (num_nfts, output_dim)
        """
        # Project features
        visual_proj = F.relu(self.visual_proj(visual_features))
        text_proj = F.relu(self.text_proj(text_features))
        
        # Compute attention weights
        visual_attn = torch.softmax(self.visual_attention(visual_proj), dim=0)
        text_attn = torch.softmax(self.text_attention(text_proj), dim=0)
        
        # Apply attention
        visual_attended = visual_attn * visual_proj
        text_attended = text_attn * text_proj
        
        # Concatenate and fuse
        combined = torch.cat([visual_attended, text_attended], dim=-1)
        fused = self.fusion(combined)
        
        return fused


class ARTEMISBaseline(nn.Module):
    """
    Complete ARTEMIS baseline implementation.
    
    Replicates the original ARTEMIS architecture:
    1. First layer: Combines node and edge features
    2. GraphSAGE layers: 2 additional GCN layers (3 total)
    3. Manual feature engineering
    4. Multimodal NFT features
    5. MLP classifier
    
    This serves as our baseline for comparison.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        
        # Model dimensions
        in_node_channels = config.get('in_node_channels', 64)
        in_edge_channels = config.get('in_edge_channels', 256)
        hidden_channels = config.get('hidden_channels', 256)
        
        # ARTEMIS architecture
        self.conv1 = ArtemisFirstLayerConv(
            in_node_channels, 
            in_edge_channels, 
            hidden_channels
        )
        self.bn1 = BatchNorm(hidden_channels)
        
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        
        # Manual feature engineering
        self.manual_features = ArtemisManualFeatures(
            feature_dim=config.get('manual_feature_dim', 32)
        )
        
        # Multimodal NFT features
        self.multimodal_module = ArtemisMultimodalModule(
            visual_dim=config.get('nft_visual_dim', 768),
            text_dim=config.get('nft_text_dim', 768),
            output_dim=config.get('nft_feature_dim', 256)
        )
        
        # Final MLP classifier
        classifier_input_dim = (
            hidden_channels +  # Graph features
            in_node_channels +  # Original node features (residual)
            config.get('manual_feature_dim', 32)  # Manual features
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2)  # Binary classification
        )
        
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ARTEMIS architecture.
        
        Args:
            batch: Input batch containing:
                - 'node_features': Node features (num_nodes, node_dim)
                - 'edge_index': Edge indices (2, num_edges)  
                - 'edge_features': Multi-modal edge features
                - 'transaction_data': Manual feature data
                - 'user_indices': Batch user indices
                
        Returns:
            Dictionary containing logits and probabilities
        """
        # Extract batch components
        node_features = batch['node_features']
        edge_index = batch['edge_index']
        edge_features = batch['edge_features']
        transaction_data = batch.get('transaction_data', {})
        user_indices = batch.get('user_indices', None)
        
        batch_size = len(user_indices) if user_indices is not None else node_features.shape[0]
        
        # Process multimodal edge features
        if isinstance(edge_features, dict):
            # Combine visual and text features
            if 'nft_visual' in edge_features and 'nft_text' in edge_features:
                multimodal_edges = self.multimodal_module(
                    edge_features['nft_visual'],
                    edge_features['nft_text']
                )
            else:
                # Fallback to available features
                available_features = []
                for key, value in edge_features.items():
                    available_features.append(value)
                if available_features:
                    multimodal_edges = torch.cat(available_features, dim=-1)
                else:
                    multimodal_edges = torch.zeros(
                        edge_index.shape[1], 
                        self.config.get('in_edge_channels', 256),
                        device=edge_index.device
                    )
        else:
            multimodal_edges = edge_features
            
        # Store initial embeddings for residual connection
        initial_embedding = node_features
        
        # First layer with edge features
        x = self.conv1(node_features, edge_index, multimodal_edges)
        x = F.relu(self.bn1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second GraphSAGE layer
        x = self.conv2(x, edge_index)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Third GraphSAGE layer
        x = self.conv3(x, edge_index)
        x = F.relu(self.bn3(x))
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Extract user embeddings
        if user_indices is not None:
            user_graph_features = x[user_indices]
            user_initial_features = initial_embedding[user_indices]
        else:
            user_graph_features = x[:batch_size]
            user_initial_features = initial_embedding[:batch_size]
        
        # Manual feature engineering
        if transaction_data:
            manual_feats = self.manual_features(transaction_data)
        else:
            # Use provided manual features if no transaction data
            manual_feats = batch.get('manual_features', torch.zeros(batch_size, self.config['manual_feature_dim']))
        
        # Combine all features (following ARTEMIS approach)
        combined_features = torch.cat([
            user_graph_features,
            user_initial_features,
            manual_feats
        ], dim=1)
        
        # Final classification
        logits = self.classifier(combined_features)
        probabilities = F.softmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'graph_embeddings': user_graph_features,
            'manual_features': manual_feats
        }
    
    def predict(self, batch: Dict[str, Any], 
                threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        """Make predictions with ARTEMIS baseline."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch)
            
            probabilities = outputs['probabilities']
            predictions = probabilities.argmax(dim=-1)
            hunter_probabilities = probabilities[:, 1]
            
            return {
                'predictions': predictions,
                'hunter_probabilities': hunter_probabilities,
                'probabilities': probabilities
            }


def create_artemis_config(
    in_node_channels: int = 64,
    in_edge_channels: int = 256,
    hidden_channels: int = 256,
    manual_feature_dim: int = 32,
    nft_visual_dim: int = 768,
    nft_text_dim: int = 768,
    nft_feature_dim: int = 256
) -> Dict[str, Any]:
    """
    Create configuration for ARTEMIS baseline.
    
    Returns:
        ARTEMIS configuration dictionary
    """
    return {
        'in_node_channels': in_node_channels,
        'in_edge_channels': in_edge_channels,
        'hidden_channels': hidden_channels,
        'manual_feature_dim': manual_feature_dim,
        'nft_visual_dim': nft_visual_dim,
        'nft_text_dim': nft_text_dim,
        'nft_feature_dim': nft_feature_dim
    }