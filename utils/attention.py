"""
Custom attention mechanisms for Temporal Graph Transformer.

This module implements specialized attention mechanisms including:
1. Cross-attention for temporal-graph fusion
2. Global attention pooling for market-wide pattern detection
3. Graph-aware attention with structural biases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for fusing temporal and graph representations.
    Allows temporal patterns to attend to graph structure and vice versa.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        
        assert d_model % num_heads == 0
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor, 
                value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model)
            key: Key tensor of shape (batch_size, seq_len_k, d_model)
            value: Value tensor of shape (batch_size, seq_len_v, d_model)
            attn_mask: Optional attention mask
            
        Returns:
            output: Cross-attended output
            attention_weights: Attention weights for interpretability
        """
        batch_size, seq_len_q, d_model = query.shape
        seq_len_k = key.shape[1]
        
        # Project to Q, K, V
        Q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        K = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len_q, head_dim)
        K = K.transpose(1, 2)  # (batch_size, num_heads, seq_len_k, head_dim)
        V = V.transpose(1, 2)  # (batch_size, num_heads, seq_len_v, head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attn_mask is not None:
            scores.masked_fill_(attn_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention
        attended = torch.matmul(attn_weights, V)
        
        # Reshape and project
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, d_model
        )
        output = self.proj_dropout(self.out_proj(attended))
        
        return output, attn_weights.mean(dim=1)  # Average over heads for interpretability


class GlobalAttentionPooling(nn.Module):
    """
    Global attention pooling for aggregating information across the entire graph.
    Useful for detecting market-wide manipulation patterns.
    """
    
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Learnable global query
        self.global_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, 
                node_embeddings: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_embeddings: Node embeddings of shape (batch_size, num_nodes, d_model)
            attention_mask: Optional mask for invalid nodes
            
        Returns:
            global_representation: Global graph representation
            attention_weights: Attention weights showing important nodes
        """
        batch_size, num_nodes, d_model = node_embeddings.shape
        
        # Expand global query for batch
        query = self.global_query.expand(batch_size, 1, -1)
        
        # Global attention
        global_rep, attn_weights = self.attention(
            query, 
            node_embeddings, 
            node_embeddings,
            key_padding_mask=attention_mask
        )
        
        # Project output
        global_rep = self.output_proj(global_rep.squeeze(1))
        
        return global_rep, attn_weights.squeeze(1)


class GraphStructureAttention(nn.Module):
    """
    Graph-aware attention that incorporates structural information.
    Uses graph topology to bias attention weights.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, use_edge_features: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_edge_features = use_edge_features
        
        assert d_model % num_heads == 0
        
        # Standard attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Edge feature projection
        if use_edge_features:
            self.edge_proj = nn.Linear(d_model, num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Scaling
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, 
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None,
                num_nodes: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            node_features: Node features of shape (num_nodes, d_model)
            edge_index: Edge indices of shape (2, num_edges)
            edge_features: Optional edge features of shape (num_edges, d_model)
            num_nodes: Number of nodes (for batch processing)
            
        Returns:
            updated_features: Updated node features after graph attention
        """
        if num_nodes is None:
            num_nodes = node_features.shape[0]
        
        # Project to Q, K, V
        Q = self.q_proj(node_features).view(num_nodes, self.num_heads, self.head_dim)
        K = self.k_proj(node_features).view(num_nodes, self.num_heads, self.head_dim)
        V = self.v_proj(node_features).view(num_nodes, self.num_heads, self.head_dim)
        
        # Initialize attention scores
        row, col = edge_index[0], edge_index[1]
        
        # Compute attention scores for edges
        edge_scores = (Q[row] * K[col]).sum(dim=-1) * self.scale  # (num_edges, num_heads)
        
        # Add edge feature bias if available
        if self.use_edge_features and edge_features is not None:
            edge_bias = self.edge_proj(edge_features)  # (num_edges, num_heads)
            edge_scores = edge_scores + edge_bias
        
        # Apply softmax per source node
        edge_scores_softmax = torch.zeros_like(edge_scores)
        for head in range(self.num_heads):
            edge_scores_softmax[:, head] = self._softmax_per_node(
                edge_scores[:, head], row, num_nodes
            )
        
        # Apply attention to values
        attended_values = edge_scores_softmax.unsqueeze(-1) * V[col]  # (num_edges, num_heads, head_dim)
        
        # Aggregate by target node
        output = torch.zeros(num_nodes, self.num_heads, self.head_dim, 
                           device=node_features.device, dtype=node_features.dtype)
        
        for head in range(self.num_heads):
            output[:, head] = self._scatter_add(
                attended_values[:, head], row, num_nodes
            )
        
        # Reshape and project
        output = output.view(num_nodes, self.d_model)
        output = self.out_proj(output)
        
        return output
    
    def _softmax_per_node(self, scores: torch.Tensor, 
                         index: torch.Tensor, 
                         num_nodes: int) -> torch.Tensor:
        """Apply softmax per source node for attention normalization."""
        
        # Create softmax per unique index
        unique_indices, inverse_indices = torch.unique(index, return_inverse=True)
        softmax_scores = torch.zeros_like(scores)
        
        for i, idx in enumerate(unique_indices):
            mask = index == idx
            if mask.sum() > 0:
                softmax_scores[mask] = F.softmax(scores[mask], dim=0)
        
        return softmax_scores
    
    def _scatter_add(self, values: torch.Tensor, 
                    index: torch.Tensor, 
                    num_nodes: int) -> torch.Tensor:
        """Scatter-add operation for aggregating values by index."""
        
        output = torch.zeros(num_nodes, values.shape[-1], 
                           device=values.device, dtype=values.dtype)
        
        for i in range(num_nodes):
            mask = index == i
            if mask.sum() > 0:
                output[i] = values[mask].sum(dim=0)
        
        return output


class AdaptiveAttention(nn.Module):
    """
    Adaptive attention that learns to weight different attention mechanisms.
    Useful for combining multiple types of attention (temporal, structural, global).
    """
    
    def __init__(self, d_model: int, num_attention_types: int = 3):
        super().__init__()
        
        self.d_model = d_model
        self.num_attention_types = num_attention_types
        
        # Attention type weights
        self.attention_weights = nn.Parameter(
            torch.ones(num_attention_types) / num_attention_types
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_attention_types),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, 
                attention_outputs: list,
                context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attention_outputs: List of attention outputs to combine
            context: Context tensor for adaptive weighting
            
        Returns:
            combined_output: Adaptively weighted combination
        """
        # Compute adaptive weights
        adaptive_weights = self.gate(context.mean(dim=1))  # (batch_size, num_attention_types)
        
        # Combine global and adaptive weights
        final_weights = adaptive_weights * self.attention_weights.unsqueeze(0)
        
        # Weighted combination
        combined = torch.zeros_like(attention_outputs[0])
        for i, output in enumerate(attention_outputs):
            # Reshape weight for proper broadcasting
            weight = final_weights[:, i].unsqueeze(-1)  # (batch_size, 1)
            while weight.dim() < output.dim():
                weight = weight.unsqueeze(-1)
            combined += weight * output
        
        return combined


class TemporalGraphAttention(nn.Module):
    """
    Specialized attention for temporal graphs that considers both time and structure.
    """
    
    def __init__(self, d_model: int, num_heads: int = 8, temporal_weight: float = 0.3):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.temporal_weight = temporal_weight
        
        # Separate attention for temporal and structural information
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.structural_attention = GraphStructureAttention(
            d_model=d_model,
            num_heads=num_heads
        )
        
        # Fusion mechanism
        self.fusion = nn.Linear(d_model * 2, d_model)
        
    def forward(self, 
                temporal_features: torch.Tensor,
                structural_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            temporal_features: Temporal features from sequence transformer
            structural_features: Structural node features
            edge_index: Graph edge indices
            edge_features: Optional edge features
            
        Returns:
            fused_features: Combined temporal-structural representation
        """
        # Temporal self-attention
        temp_attended, _ = self.temporal_attention(
            temporal_features, temporal_features, temporal_features
        )
        
        # Structural attention
        struct_attended = self.structural_attention(
            structural_features, edge_index, edge_features
        )
        
        # Ensure same batch dimension
        if temp_attended.dim() == 3 and struct_attended.dim() == 2:
            struct_attended = struct_attended.unsqueeze(0).expand(
                temp_attended.shape[0], -1, -1
            )
        
        # Weighted fusion
        temporal_weighted = self.temporal_weight * temp_attended
        structural_weighted = (1 - self.temporal_weight) * struct_attended
        
        # Concatenate and fuse
        combined = torch.cat([temporal_weighted, structural_weighted], dim=-1)
        fused = self.fusion(combined)
        
        return fused