"""
Temporal encoding modules for blockchain transaction sequences.

This module implements our novel Functional Time Encoding approach,
which combines sinusoidal encoding with learnable projections to capture
both periodic and task-specific temporal patterns in user behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class FunctionalTimeEncoding(nn.Module):
    """
    Novel functional time encoding that combines:
    1. Sinusoidal encoding for capturing periodic patterns
    2. Learnable projections for task-specific temporal patterns
    3. Time delta encoding for behavioral rhythm detection
    
    This is superior to standard positional encoding for behavioral analysis
    as it can learn transaction timing patterns specific to airdrop hunting.
    """
    
    def __init__(self, d_model: int, max_timescale: float = 10000.0):
        super().__init__()
        
        self.d_model = d_model
        self.max_timescale = max_timescale
        
        # Learnable frequency parameters for sinusoidal encoding
        self.frequency_embedding = nn.Parameter(
            torch.randn(d_model // 4) * 0.02
        )
        
        # Learnable projection for task-specific patterns
        self.time_proj = nn.Linear(1, d_model // 4)
        
        # Time delta projection for behavioral rhythm
        self.delta_proj = nn.Linear(1, d_model // 4)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: Tensor of shape (batch_size, seq_len) or (seq_len,) with transaction timestamps
            
        Returns:
            time_encoding: Tensor of shape (batch_size, seq_len, d_model)
        """
        # Handle both 1D and 2D input tensors
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(0)  # Add batch dimension
        batch_size, seq_len = timestamps.shape
        device = timestamps.device
        
        # 1. Convert to relative time since first transaction
        relative_times = timestamps - timestamps[:, 0:1]  # (batch_size, seq_len)
        
        # 2. Compute time deltas (time since previous transaction)
        time_deltas = torch.diff(
            timestamps, 
            prepend=timestamps[:, 0:1], 
            dim=1
        )  # (batch_size, seq_len)
        
        # 3. Sinusoidal encoding with learnable frequencies
        angles = relative_times.unsqueeze(-1) * self.frequency_embedding.unsqueeze(0).unsqueeze(0)
        sin_emb = torch.sin(angles)  # (batch_size, seq_len, d_model//4)
        cos_emb = torch.cos(angles)  # (batch_size, seq_len, d_model//4)
        
        # 4. Learnable projection of absolute time
        time_emb = self.time_proj(relative_times.unsqueeze(-1))  # (batch_size, seq_len, d_model//4)
        
        # 5. Time delta encoding for rhythm patterns
        delta_emb = self.delta_proj(time_deltas.unsqueeze(-1))  # (batch_size, seq_len, d_model//4)
        
        # 6. Concatenate all components
        full_encoding = torch.cat([sin_emb, cos_emb, time_emb, delta_emb], dim=-1)
        
        # 7. Layer normalization for training stability
        return self.layer_norm(full_encoding)


class SinusoidalTimeEncoding(nn.Module):
    """
    Standard sinusoidal time encoding for comparison with our functional approach.
    Based on the original Transformer positional encoding but adapted for continuous time.
    """
    
    def __init__(self, d_model: int, max_timescale: float = 10000.0):
        super().__init__()
        
        self.d_model = d_model
        self.max_timescale = max_timescale
        
        # Pre-compute frequency dividers
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(np.log(max_timescale) / d_model)
        )
        self.register_buffer('div_term', div_term)
        
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: Tensor of shape (batch_size, seq_len)
            
        Returns:
            time_encoding: Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = timestamps.shape
        
        # Convert to relative time
        relative_times = timestamps - timestamps[:, 0:1]
        
        # Expand for broadcasting
        time_expanded = relative_times.unsqueeze(-1)  # (batch_size, seq_len, 1)
        div_expanded = self.div_term.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model//2)
        
        # Compute sinusoidal encoding
        angles = time_expanded * div_expanded  # (batch_size, seq_len, d_model//2)
        
        # Interleave sin and cos
        encoding = torch.zeros(batch_size, seq_len, self.d_model, device=timestamps.device)
        encoding[:, :, 0::2] = torch.sin(angles)
        encoding[:, :, 1::2] = torch.cos(angles)
        
        return encoding


class AdaptiveTimeEncoding(nn.Module):
    """
    Adaptive time encoding that learns to weight different temporal scales.
    Useful for capturing both short-term (minutes/hours) and long-term (days/weeks) patterns.
    """
    
    def __init__(self, d_model: int, num_scales: int = 4):
        super().__init__()
        
        self.d_model = d_model
        self.num_scales = num_scales
        
        # Different time scales (seconds, minutes, hours, days)
        self.time_scales = nn.Parameter(
            torch.tensor([1.0, 60.0, 3600.0, 86400.0])[:num_scales]
        )
        
        # Scale-specific encoders
        self.scale_encoders = nn.ModuleList([
            SinusoidalTimeEncoding(d_model // num_scales)
            for _ in range(num_scales)
        ])
        
        # Attention weights for different scales
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            batch_first=True
        )
        
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: Tensor of shape (batch_size, seq_len)
            
        Returns:
            time_encoding: Tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = timestamps.shape
        
        # Encode at different time scales
        scale_encodings = []
        for i, encoder in enumerate(self.scale_encoders):
            scaled_times = timestamps / self.time_scales[i]
            scale_enc = encoder(scaled_times)
            scale_encodings.append(scale_enc)
        
        # Concatenate scale encodings
        multi_scale_encoding = torch.cat(scale_encodings, dim=-1)
        
        # Apply self-attention to learn scale importance
        attended_encoding, _ = self.scale_attention(
            multi_scale_encoding,
            multi_scale_encoding, 
            multi_scale_encoding
        )
        
        return attended_encoding


class BehaviorChangeTimeEncoding(nn.Module):
    """
    Specialized time encoding that emphasizes periods around airdrop announcements.
    This helps the model focus on behavioral changes that indicate hunting activity.
    """
    
    def __init__(self, d_model: int, change_window: float = 86400.0 * 7):  # 7 days
        super().__init__()
        
        self.d_model = d_model
        self.change_window = change_window
        
        # Base time encoding - use 3/4 of dimensions
        self.base_encoding = FunctionalTimeEncoding(d_model * 3 // 4)
        
        # Change proximity encoding - use 1/4 of dimensions
        self.change_proj = nn.Linear(1, d_model // 4)
        
        # Final projection to ensure exact d_model dimensions
        self.final_proj = nn.Linear(d_model, d_model)
        
        # Learnable change emphasis
        self.change_emphasis = nn.Parameter(torch.tensor(2.0))
        
    def forward(self, timestamps: torch.Tensor, 
                airdrop_events: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            timestamps: Tensor of shape (batch_size, seq_len)
            airdrop_events: Tensor of airdrop announcement timestamps
            
        Returns:
            time_encoding: Tensor of shape (batch_size, seq_len, d_model)
        """
        # Base temporal encoding
        base_enc = self.base_encoding(timestamps)
        
        if airdrop_events is not None and len(airdrop_events) > 0:
            # Compute distance to nearest airdrop event
            batch_size, seq_len = timestamps.shape
            
            # Expand dimensions for broadcasting
            times_expanded = timestamps.unsqueeze(-1)  # (batch_size, seq_len, 1)
            events_expanded = airdrop_events.unsqueeze(0).unsqueeze(0)  # (1, 1, num_events)
            
            # Distance to nearest airdrop
            distances = torch.abs(times_expanded - events_expanded)
            if distances.shape[-1] > 0:  # Check if there are any events
                min_distances = torch.min(distances, dim=-1)[0]  # (batch_size, seq_len)
                
                # Proximity encoding (closer to airdrop = higher value)
                proximity = torch.exp(-min_distances / self.change_window)
                change_enc = self.change_proj(proximity.unsqueeze(-1))
            else:
                change_enc = torch.zeros(batch_size, seq_len, self.d_model // 4, device=timestamps.device)
            
            # Combine base encoding with change emphasis
            combined = torch.cat([base_enc, change_enc * self.change_emphasis], dim=-1)
        else:
            # No airdrop events provided, just use base encoding
            batch_size, seq_len = timestamps.shape
            padding = torch.zeros(batch_size, seq_len, self.d_model // 4, device=timestamps.device)
            combined = torch.cat([base_enc, padding], dim=-1)
        
        # Project to exact d_model dimensions
        return self.final_proj(combined)


# Utility functions for time encoding
def create_time_mask(timestamps: torch.Tensor, 
                    window_size: float = 3600.0) -> torch.Tensor:
    """
    Create attention mask for temporal windows.
    
    Args:
        timestamps: Tensor of shape (batch_size, seq_len)
        window_size: Size of attention window in seconds
        
    Returns:
        mask: Boolean tensor of shape (batch_size, seq_len, seq_len)
    """
    batch_size, seq_len = timestamps.shape
    
    # Compute pairwise time differences
    time_diff = timestamps.unsqueeze(-1) - timestamps.unsqueeze(-2)
    time_diff = torch.abs(time_diff)
    
    # Create mask for attention window
    mask = time_diff <= window_size
    
    return mask


def normalize_timestamps(timestamps: torch.Tensor) -> Tuple[torch.Tensor, dict]:
    """
    Normalize timestamps to improve training stability.
    
    Args:
        timestamps: Raw timestamps
        
    Returns:
        normalized_timestamps: Normalized timestamps
        normalization_stats: Statistics for denormalization
    """
    min_time = timestamps.min()
    max_time = timestamps.max()
    time_range = max_time - min_time
    
    # Normalize to [0, 1] range
    normalized = (timestamps - min_time) / (time_range + 1e-8)
    
    stats = {
        'min_time': min_time,
        'max_time': max_time, 
        'time_range': time_range
    }
    
    return normalized, stats