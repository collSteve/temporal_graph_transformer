"""
Graph Structure Transformer for modeling user-to-user interactions and NFT trading relationships.

This module implements the second level of our hierarchical architecture,
focusing on graph structure patterns and multi-modal edge features.
It incorporates ARTEMIS's multimodal NFT features while adding flexible attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math
import networkx as nx
import numpy as np

# Try relative import first, fall back to absolute import
try:
    from ..utils.attention import GraphStructureAttention, CrossAttention
except ImportError:
    from utils.attention import GraphStructureAttention, CrossAttention


class MultiModalEdgeEncoder(nn.Module):
    """
    Encodes multi-modal edge features combining:
    1. NFT visual features (ViT embeddings)
    2. NFT textual features (BERT embeddings) 
    3. Transaction features (price, volume, gas, etc.)
    
    This preserves ARTEMIS's multimodal approach while making it more flexible.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        d_model = config['d_model']
        
        # NFT visual feature projection (ViT features)
        self.visual_proj = nn.Sequential(
            nn.Linear(config.get('nft_visual_dim', 768), d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(d_model // 2, d_model // 4)
        )
        
        # NFT textual feature projection (BERT features) 
        self.text_proj = nn.Sequential(
            nn.Linear(config.get('nft_text_dim', 768), d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(d_model // 2, d_model // 4)
        )
        
        # Transaction feature projection
        self.transaction_proj = nn.Sequential(
            nn.Linear(config.get('transaction_dim', 64), d_model // 4),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1))
        )
        
        # Market manipulation features (from ARTEMIS)
        self.market_features_proj = nn.Sequential(
            nn.Linear(config.get('market_features_dim', 32), d_model // 4),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1))
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1))
        )
        
        # Cross-modal attention for feature interaction
        self.cross_modal_attention = CrossAttention(
            d_model=d_model // 4,
            num_heads=4,
            dropout=config.get('dropout', 0.1)
        )
        
    def forward(self, edge_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            edge_features: Dictionary containing:
                - 'nft_visual': ViT features of shape (num_edges, visual_dim)
                - 'nft_text': BERT features of shape (num_edges, text_dim)
                - 'transaction': Transaction features of shape (num_edges, tx_dim)
                - 'market_features': Market manipulation features
                
        Returns:
            edge_embeddings: Multi-modal edge embeddings of shape (num_edges, d_model)
        """
        embeddings = []
        
        # Process each modality
        if 'nft_visual' in edge_features:
            visual_emb = self.visual_proj(edge_features['nft_visual'])
            embeddings.append(visual_emb)
            
        if 'nft_text' in edge_features:
            text_emb = self.text_proj(edge_features['nft_text'])
            embeddings.append(text_emb)
            
        if 'transaction' in edge_features:
            tx_emb = self.transaction_proj(edge_features['transaction'])
            embeddings.append(tx_emb)
            
        if 'market_features' in edge_features:
            market_emb = self.market_features_proj(edge_features['market_features'])
            embeddings.append(market_emb)
        
        # Cross-modal attention between visual and text features
        if len(embeddings) >= 2:
            # Apply cross-attention between first two modalities
            attended_emb, _ = self.cross_modal_attention(
                query=embeddings[0].unsqueeze(1),
                key=embeddings[1].unsqueeze(1),
                value=embeddings[1].unsqueeze(1)
            )
            embeddings[0] = attended_emb.squeeze(1)
        
        # Concatenate all embeddings
        if embeddings:
            combined = torch.cat(embeddings, dim=-1)
        else:
            # Fallback for missing features
            num_edges = list(edge_features.values())[0].shape[0]
            combined = torch.zeros(num_edges, self.config['d_model'])
        
        # Fusion
        return self.fusion(combined)


class CentralityEncoding(nn.Module):
    """
    Encodes graph centrality measures as positional encoding.
    Captures the importance and connectivity patterns of nodes.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        self.d_model = d_model
        
        # Centrality measure embeddings
        self.degree_embedding = nn.Embedding(1000, d_model // 4)  # Max degree 1000
        self.pagerank_proj = nn.Linear(1, d_model // 4)
        self.clustering_proj = nn.Linear(1, d_model // 4)
        self.betweenness_proj = nn.Linear(1, d_model // 4)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, graph_metrics: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            graph_metrics: Dictionary containing centrality measures:
                - 'degree': Node degrees
                - 'pagerank': PageRank scores
                - 'clustering': Clustering coefficients
                - 'betweenness': Betweenness centrality
                
        Returns:
            centrality_encoding: Centrality-based positional encoding
        """
        embeddings = []
        
        if 'degree' in graph_metrics:
            # Clamp degree to max embedding size
            degree_clamped = torch.clamp(graph_metrics['degree'], 0, 999)
            degree_emb = self.degree_embedding(degree_clamped)
            embeddings.append(degree_emb)
            
        if 'pagerank' in graph_metrics:
            pagerank_emb = self.pagerank_proj(graph_metrics['pagerank'].unsqueeze(-1))
            embeddings.append(pagerank_emb)
            
        if 'clustering' in graph_metrics:
            clustering_emb = self.clustering_proj(graph_metrics['clustering'].unsqueeze(-1))
            embeddings.append(clustering_emb)
            
        if 'betweenness' in graph_metrics:
            betweenness_emb = self.betweenness_proj(graph_metrics['betweenness'].unsqueeze(-1))
            embeddings.append(betweenness_emb)
        
        # Concatenate and normalize
        if embeddings:
            combined = torch.cat(embeddings, dim=-1)
            return self.layer_norm(combined)
        else:
            # Fallback
            num_nodes = list(graph_metrics.values())[0].shape[0]
            return torch.zeros(num_nodes, self.d_model)


class DistanceEncoding(nn.Module):
    """
    Encodes shortest path distances between nodes as positional encoding.
    Helps the transformer understand graph topology.
    """
    
    def __init__(self, d_model: int, max_distance: int = 10):
        super().__init__()
        
        self.d_model = d_model
        self.max_distance = max_distance
        
        # Distance embedding
        self.distance_embedding = nn.Embedding(max_distance + 1, d_model)
        
    def forward(self, distance_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distance_matrix: Shortest path distances of shape (num_nodes, num_nodes)
            
        Returns:
            distance_encoding: Distance-based encoding for each node pair
        """
        # Clamp distances to max value
        distances_clamped = torch.clamp(distance_matrix, 0, self.max_distance)
        
        # Embed distances
        distance_emb = self.distance_embedding(distances_clamped)
        
        return distance_emb


class GraphTransformerLayer(nn.Module):
    """
    Single layer of the Graph Transformer with structural awareness.
    Combines self-attention with graph structure information.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        d_model = config['d_model']
        
        # Graph structure attention
        self.graph_attention = GraphStructureAttention(
            d_model=d_model,
            num_heads=config.get('num_heads', 8),
            use_edge_features=True
        )
        
        # Standard transformer components
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, config.get('dim_feedforward', d_model * 4)),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(config.get('dim_feedforward', d_model * 4), d_model),
            nn.Dropout(config.get('dropout', 0.1))
        )
        
    def forward(self, 
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: Node features of shape (num_nodes, d_model)
            edge_index: Edge indices of shape (2, num_edges)
            edge_features: Edge features of shape (num_edges, d_model)
            
        Returns:
            updated_features: Updated node features after graph attention
        """
        # Graph attention with residual connection
        attended = self.graph_attention(node_features, edge_index, edge_features)
        x = self.norm1(node_features + attended)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class GraphStructureTransformer(nn.Module):
    """
    Main Graph Structure Transformer module.
    
    This implements Level 2 of our hierarchical architecture, modeling graph structure
    patterns with multi-modal edge features and structural positional encoding.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        d_model = config['d_model']
        
        # Multi-modal edge encoder
        self.edge_encoder = MultiModalEdgeEncoder(config)
        
        # Structural positional encodings
        self.centrality_encoder = CentralityEncoding(d_model)
        self.distance_encoder = DistanceEncoding(
            d_model, 
            max_distance=config.get('max_distance', 10)
        )
        
        # Initial node feature projection
        self.node_proj = nn.Linear(
            config.get('input_node_dim', d_model), 
            d_model
        )
        
        # Graph transformer layers
        self.graph_layers = nn.ModuleList([
            GraphTransformerLayer(config)
            for _ in range(config.get('num_layers', 6))
        ])
        
        # Global attention pooling
        try:
            from ..utils.attention import GlobalAttentionPooling
        except ImportError:
            from utils.attention import GlobalAttentionPooling
        self.global_pooling = GlobalAttentionPooling(
            d_model=d_model,
            num_heads=config.get('global_heads', 4)
        )
        
        # Output projections
        self.node_output_proj = nn.Linear(d_model, d_model)
        self.graph_output_proj = nn.Linear(d_model, d_model)
        
        # Layer normalization
        self.final_norm = nn.LayerNorm(d_model)
        
    def compute_graph_metrics(self, edge_index: torch.Tensor, 
                            num_nodes: int) -> Dict[str, torch.Tensor]:
        """
        Compute graph centrality metrics for positional encoding.
        
        Args:
            edge_index: Edge indices
            num_nodes: Number of nodes
            
        Returns:
            Dictionary of centrality measures
        """
        # Convert to NetworkX for efficient centrality computation
        edge_list = edge_index.t().cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_list)
        
        # Compute centrality measures
        try:
            degree_centrality = nx.degree_centrality(G)
            pagerank = nx.pagerank(G, max_iter=100)
            clustering = nx.clustering(G)
            betweenness = nx.betweenness_centrality(G, k=min(100, num_nodes))
        except:
            # Fallback for disconnected or problematic graphs
            degree_centrality = {i: 0.0 for i in range(num_nodes)}
            pagerank = {i: 1.0/num_nodes for i in range(num_nodes)}
            clustering = {i: 0.0 for i in range(num_nodes)}
            betweenness = {i: 0.0 for i in range(num_nodes)}
        
        # Convert to tensors
        device = edge_index.device
        
        metrics = {
            'degree': torch.tensor([
                G.degree(i) for i in range(num_nodes)
            ], device=device),
            'pagerank': torch.tensor([
                pagerank[i] for i in range(num_nodes)
            ], device=device),
            'clustering': torch.tensor([
                clustering[i] for i in range(num_nodes)
            ], device=device),
            'betweenness': torch.tensor([
                betweenness[i] for i in range(num_nodes)
            ], device=device)
        }
        
        return metrics
    
    def forward(self, 
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_features: Dict[str, torch.Tensor],
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            node_features: Initial node features of shape (num_nodes, input_dim)
            edge_index: Edge indices of shape (2, num_edges)
            edge_features: Dictionary of multi-modal edge features
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - 'node_embeddings': Updated node embeddings
                - 'graph_embedding': Global graph representation
                - 'attention_weights': Optional attention weights
        """
        num_nodes = node_features.shape[0]
        
        # 1. Project initial node features
        x = self.node_proj(node_features)
        
        # 2. Encode multi-modal edge features
        edge_emb = self.edge_encoder(edge_features)
        
        # 3. Add structural positional encoding
        graph_metrics = self.compute_graph_metrics(edge_index, num_nodes)
        centrality_encoding = self.centrality_encoder(graph_metrics)
        x = x + centrality_encoding
        
        # 4. Apply graph transformer layers
        attention_weights = []
        for layer in self.graph_layers:
            x = layer(x, edge_index, edge_emb)
            
            # Store attention weights if requested
            if return_attention:
                # This would require modifying the layer to return attention weights
                pass
        
        # 5. Global pooling for graph-level representation
        graph_emb, global_attention = self.global_pooling(x.unsqueeze(0))
        graph_emb = graph_emb.squeeze(0)
        
        # 6. Final projections
        node_embeddings = self.final_norm(self.node_output_proj(x))
        graph_embedding = self.graph_output_proj(graph_emb)
        
        outputs = {
            'node_embeddings': node_embeddings,
            'graph_embedding': graph_embedding,
            'global_attention_weights': global_attention
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
            
        return outputs
    
    def get_node_importance(self, 
                          node_features: torch.Tensor,
                          edge_index: torch.Tensor,
                          edge_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get node importance scores for interpretability.
        
        Returns:
            importance_scores: Tensor of shape (num_nodes,)
        """
        with torch.no_grad():
            outputs = self.forward(node_features, edge_index, edge_features)
            global_attention = outputs['global_attention_weights']
            return global_attention.squeeze(0)


def create_graph_structure_config(
    d_model: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    dropout: float = 0.1,
    nft_visual_dim: int = 768,
    nft_text_dim: int = 768,
    transaction_dim: int = 64,
    market_features_dim: int = 32
) -> Dict:
    """
    Create configuration for GraphStructureTransformer.
    
    Returns:
        Configuration dictionary
    """
    return {
        'd_model': d_model,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'dim_feedforward': d_model * 4,
        'dropout': dropout,
        'global_heads': 4,
        'max_distance': 10,
        'input_node_dim': d_model,
        'nft_visual_dim': nft_visual_dim,
        'nft_text_dim': nft_text_dim,
        'transaction_dim': transaction_dim,
        'market_features_dim': market_features_dim
    }