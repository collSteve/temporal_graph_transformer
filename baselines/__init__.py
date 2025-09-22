"""
Baseline Methods for Temporal Graph Transformer

Collection of state-of-the-art baseline methods for airdrop hunter detection:
- TrustaLabs Framework (Industry Standard)
- Subgraph Feature Propagation (Academic SOTA)
- Enhanced Graph Neural Networks (GAT, GraphSAGE, SybilGAT)
- Traditional ML approaches (LightGBM, Random Forest)
"""

from .base_interface import BaselineMethodInterface
from .trustalab_framework import TrustaLabFramework
from .subgraph_propagation import SubgraphFeaturePropagation
from .enhanced_gnns import EnhancedGNNBaseline
from .traditional_ml import LightGBMBaseline, RandomForestBaseline
from .temporal_graph_transformer_baseline import TemporalGraphTransformerBaseline

__all__ = [
    'BaselineMethodInterface',
    'TrustaLabFramework',
    'SubgraphFeaturePropagation',
    'EnhancedGNNBaseline',
    'LightGBMBaseline',
    'RandomForestBaseline',
    'TemporalGraphTransformerBaseline'
]