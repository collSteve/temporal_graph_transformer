"""
Temporal Graph Transformer for airdrop hunter detection.

This package provides a novel hierarchical architecture that combines temporal sequence 
modeling with graph neural networks to detect airdrop hunting behavior in blockchain ecosystems.

Main components:
- Transaction Sequence Transformer (Level 1): Temporal pattern modeling
- Graph Structure Transformer (Level 2): Graph relationship modeling  
- Temporal-Graph Fusion (Level 3): Cross-modal fusion and classification

Key innovations:
- Functional time encoding for behavioral rhythm detection
- Multi-modal edge features (NFT visual + textual + transaction)
- Unlimited receptive field graph attention
- End-to-end learning without manual feature engineering
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__email__ = "research@example.com"

# Main model exports
from .models.temporal_graph_transformer import TemporalGraphTransformer
from .models.temporal_encoder import (
    TransactionSequenceTransformer,
    create_transaction_sequence_config
)
from .models.graph_encoder import (
    GraphStructureTransformer, 
    create_graph_structure_config
)
from .models.artemis_baseline import ARTEMISBaseline

# Data exports  
from .data.base_dataset import BaseTemporalGraphDataset
from .data.solana_dataset import SolanaNFTDataset

# Utility exports
from .utils.loss_functions import TemporalGraphLoss
from .utils.time_encoding import (
    FunctionalTimeEncoding,
    BehaviorChangeTimeEncoding
)

__all__ = [
    # Models
    "TemporalGraphTransformer",
    "TransactionSequenceTransformer", 
    "GraphStructureTransformer",
    "ARTEMISBaseline",
    
    # Data
    "BaseTemporalGraphDataset",
    "SolanaNFTDataset",
    
    # Utils
    "TemporalGraphLoss",
    "FunctionalTimeEncoding",
    "BehaviorChangeTimeEncoding",
    
    # Config functions
    "create_transaction_sequence_config",
    "create_graph_structure_config",
]