# Temporal Graph Transformer for Airdrop Hunter Detection

A novel hierarchical architecture that combines temporal sequence modeling with graph neural networks to detect airdrop hunting behavior in blockchain ecosystems.

## 🏗️ Architecture

Our Temporal Graph Transformer (TGT) implements a 3-level hierarchical architecture:

### Level 1: Transaction Sequence Transformer
- **Functional Time Encoding**: Novel approach combining sinusoidal encoding with learnable projections
- **Behavioral Change Detection**: Specialized attention mechanism for identifying behavioral transitions
- **Transaction Feature Embedding**: Multi-modal transaction attributes (price, gas, volume, type)

### Level 2: Graph Structure Transformer  
- **Multi-modal Edge Features**: NFT visual + textual + transaction features
- **Structural Positional Encoding**: Centrality measures and graph topology
- **Unlimited Receptive Field**: Overcomes ARTEMIS's 3-hop limitation

### Level 3: Temporal-Graph Fusion
- **Cross-modal Attention**: Fusion between temporal and graph representations
- **Behavioral Scoring**: Market-wide pattern detection around airdrop events
- **Confidence Estimation**: Uncertainty quantification for predictions

## 🔬 Key Innovations

1. **Functional Time Encoding**: Captures both periodic patterns and task-specific temporal dynamics
2. **Behavioral Change Time Encoding**: Emphasizes periods around airdrop announcements
3. **Multi-task Loss Function**: InfoNCE + Focal + Temporal Consistency + Behavioral Change
4. **End-to-End Learning**: No manual feature engineering required

## 📊 Advantages over ARTEMIS

| Aspect | ARTEMIS | Our TGT |
|--------|---------|---------|
| **Temporal Modeling** | Manual features | Learned temporal patterns |
| **Receptive Field** | 3-hop limitation | Unlimited attention |
| **Feature Engineering** | Manual | End-to-end learning |
| **Behavioral Changes** | Static analysis | Dynamic change detection |
| **Multi-modal Data** | Limited | NFT visual + text + transaction |

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv temporal_graph_transformer_env
source temporal_graph_transformer_env/bin/activate  # On Windows: Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from temporal_graph_transformer.models.temporal_graph_transformer import TemporalGraphTransformer
from temporal_graph_transformer.data.solana_dataset import SolanaNFTDataset

# Create dataset
dataset = SolanaNFTDataset(
    data_path="./data/demo",
    split='train',
    marketplace='magic_eden'
)

# Initialize model
config = {
    'd_model': 256,
    'temporal_config': {...},
    'graph_config': {...},
    'num_classes': 2
}
model = TemporalGraphTransformer(config)

# Train or inference
sample = dataset[0]
outputs = model(sample)
```

### Run Demo

```bash
cd examples/
python simple_demo.py
```

## 📁 Project Structure

```
temporal_graph_transformer/
├── temporal_graph_transformer/          # Main package
│   ├── models/                         # Neural network architectures
│   │   ├── temporal_encoder.py        # Transaction sequence transformer
│   │   ├── graph_encoder.py           # Graph structure transformer  
│   │   ├── fusion_module.py           # Temporal-graph fusion
│   │   ├── temporal_graph_transformer.py  # Complete model
│   │   └── artemis_baseline.py        # ARTEMIS implementation
│   ├── data/                          # Data loading and processing
│   │   ├── base_dataset.py           # Unified dataset interface
│   │   ├── solana_dataset.py         # Solana NFT marketplace data
│   │   └── data_utils.py             # Preprocessing utilities
│   └── utils/                         # Utility modules
│       ├── time_encoding.py          # Temporal encoding implementations
│       ├── attention.py              # Attention mechanisms
│       ├── loss_functions.py         # Multi-task loss functions
│       └── metrics.py               # Evaluation metrics
├── examples/                          # Usage examples and demos
│   ├── simple_demo.py               # Basic demo
│   └── demo_preprocessing_pipeline.py  # Data preprocessing demo
├── tests/                            # Unit tests
├── scripts/                          # Training and evaluation scripts
├── experiments/                      # Experimental configurations
├── docs/                            # Documentation
└── README.md                        # This file
```

## 🔧 Model Components

### Temporal Encoder
- `TransactionSequenceTransformer`: Main temporal modeling component
- `FunctionalTimeEncoding`: Novel time encoding approach
- `BehaviorChangeTimeEncoding`: Airdrop-aware temporal encoding
- `ChangePointAttention`: Behavioral transition detection

### Graph Encoder  
- `GraphStructureTransformer`: Graph neural network component
- `MultiModalEdgeEncoder`: NFT + transaction edge features
- `StructuralPositionalEncoding`: Graph topology encoding
- `AdaptiveGraphAttention`: Attention with unlimited receptive field

### Fusion Module
- `TemporalGraphFusion`: Cross-modal attention fusion
- `BehavioralScoring`: Market-wide pattern detection
- `ConfidenceEstimation`: Uncertainty quantification

## 📈 Datasets

### Supported Blockchains
- **Solana**: NFT marketplaces (Magic Eden, Solanart)
- **Ethereum**: ERC-721 transactions (extensible)

### Data Format
Each user sample contains:
- `transaction_features`: Temporal transaction sequence
- `node_features`: User graph node attributes  
- `edge_index`: Graph connectivity
- `edge_features`: Multi-modal edge attributes
- `timestamps`: Transaction timing
- `airdrop_events`: Known airdrop announcements
- `labels`: Ground truth (0=legitimate, 1=hunter)

## 🧪 Testing

The implementation has been thoroughly tested:

```bash
# Run all tests (requires virtual environment)
source temporal_graph_transformer_env/bin/activate
python -m pytest tests/

# Run individual component tests
python tests/test_temporal_encoder.py
python tests/test_graph_encoder.py
python tests/test_loss_functions.py
```

## 📚 Research Context

This work addresses limitations in existing airdrop hunting detection:

1. **ARTEMIS (WWW 2024)**: Manual feature engineering, 3-hop limitation, static analysis
2. **Our Contribution**: End-to-end learning, unlimited attention, dynamic behavioral modeling

### Key Technical Innovations
- Functional time encoding for blockchain temporal patterns
- Behavioral change detection around airdrop events  
- Multi-modal graph neural networks for NFT + transaction data
- Hierarchical attention with cross-modal fusion

## 🤝 Contributing

We welcome contributions! Please see our contribution guidelines for:
- Code style and testing requirements
- Adding new blockchain dataset support
- Extending the model architecture
- Improving documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@article{temporal_graph_transformer_2024,
  title={Temporal Graph Transformer for Airdrop Hunter Detection in Blockchain Ecosystems},
  author={[Authors]},
  journal={[Venue]},
  year={2024}
}
```

## 🔗 References

- ARTEMIS: [Detecting Airdrop Hunters in NFT Markets using Graph Neural Networks](https://github.com/StackPie71/ARTEMIS-WWW2024)
- Transformer Architecture: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Graph Neural Networks: [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)