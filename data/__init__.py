"""
Data preprocessing pipeline for temporal graph datasets.

This module provides a unified interface for loading and preprocessing various
blockchain datasets for airdrop hunter detection, supporting both our
Temporal Graph Transformer and the ARTEMIS baseline.

Supports both NFT marketplaces and pure cryptocurrency (DeFi) datasets:
- NFT Markets: Solana (Magic Eden), Ethereum (Blur), L2 networks
- Pure Crypto Markets: Arbitrum DeFi, Jupiter Solana, Optimism L2
"""

from .base_dataset import BaseTemporalGraphDataset

# NFT marketplace datasets
from .solana_dataset import SolanaNFTDataset  
from .ethereum_dataset import EthereumNFTDataset
from .l2_dataset import L2NetworkDataset

# Pure cryptocurrency (DeFi) datasets
from .pure_crypto_dataset import PureCryptoDataset
from .arbitrum_dataset import ArbitrumDeFiDataset
from .jupiter_dataset import JupiterSolanaDataset

# Transaction schema and validation
from .transaction_schema import (
    BaseTransaction,
    ArbitrumDeFiTransaction,
    SolanaDeFiTransaction,
    OptimismDeFiTransaction,
    TransactionSchemaValidator,
    create_transaction_from_dict,
    transactions_to_dataframe
)

# Core processing components
from .preprocessing import TemporalGraphPreprocessor
from .data_loader import UnifiedDataLoader, create_data_loaders
from .transforms import (
    TemporalGraphTransform,
    ARTEMISTransform,
    AirdropEventProcessor,
    NFTFeatureExtractor
)

__all__ = [
    # Base classes
    'BaseTemporalGraphDataset',
    'PureCryptoDataset',
    
    # NFT marketplace datasets
    'SolanaNFTDataset',
    'EthereumNFTDataset', 
    'L2NetworkDataset',
    
    # Pure crypto (DeFi) datasets
    'ArbitrumDeFiDataset',
    'JupiterSolanaDataset',
    
    # Transaction schema
    'BaseTransaction',
    'ArbitrumDeFiTransaction',
    'SolanaDeFiTransaction',
    'OptimismDeFiTransaction',
    'TransactionSchemaValidator',
    'create_transaction_from_dict',
    'transactions_to_dataframe',
    
    # Processing components
    'TemporalGraphPreprocessor',
    'UnifiedDataLoader',
    'create_data_loaders',
    'TemporalGraphTransform',
    'ARTEMISTransform',
    'AirdropEventProcessor',
    'NFTFeatureExtractor'
]