# Changelog

All notable changes to the Temporal Graph Transformer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-09-21

### Added
- Initial implementation of Temporal Graph Transformer architecture
- Novel Functional Time Encoding for blockchain temporal patterns
- Behavioral Change Time Encoding for airdrop event detection
- Multi-modal graph neural network with NFT visual + textual features
- Three-level hierarchical architecture:
  - Level 1: Transaction Sequence Transformer
  - Level 2: Graph Structure Transformer  
  - Level 3: Temporal-Graph Fusion
- ARTEMIS baseline implementation for comparison
- Multi-task loss function with InfoNCE, Focal, Temporal Consistency, and Behavioral Change losses
- Unified dataset interface for blockchain data
- Solana NFT marketplace dataset support
- Comprehensive testing suite
- Training and evaluation scripts
- Documentation and examples

### Technical Features
- Cross-modal attention fusion between temporal and graph representations
- Adaptive attention mechanisms with unlimited receptive field
- Confidence estimation and uncertainty quantification
- Synthetic data generation for testing and prototyping
- End-to-end gradient-based learning

### Infrastructure
- Proper Python package structure with setup.py and pyproject.toml
- GitHub-ready repository structure
- Comprehensive testing with pytest
- CI/CD configuration templates
- Documentation with README and examples

### Performance
- Successfully passes all unit and integration tests
- Handles synthetic data with 1000+ users and 20k+ transactions
- Memory-efficient implementation with proper batching
- GPU-accelerated training support