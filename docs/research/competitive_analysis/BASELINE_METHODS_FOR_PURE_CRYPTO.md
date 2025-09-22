# Baseline Methods for Pure Crypto Market Airdrop Hunter Detection

## 🎯 The Baseline Problem

You're absolutely correct - **ARTEMIS cannot serve as a proper baseline for pure crypto markets** because:

1. **NFT-Specific Features**: ARTEMIS relies heavily on NFT visual (ViT) and textual (BERT) features
2. **Multimodal Architecture**: Core components designed for image + text metadata processing
3. **Market Context**: Trained specifically on Blur NFT marketplace data
4. **Feature Engineering**: Manual features tailored to NFT trading patterns

Using ARTEMIS for pure crypto (DeFi/DEX) would be like using an image classifier for text analysis - technically possible but fundamentally mismatched.

---

## 🏆 **Appropriate Baselines for Pure Crypto Markets**

### **Tier 1: Direct Sybil Detection Methods**

#### **1. TrustaLabs Sybil Detection Framework** [Primary Baseline]
- **Paper**: "Fighting Sybils in Airdrops" + industry implementation
- **Approach**: Two-stage AI/ML framework specifically for airdrop hunters
- **Features**: 
  - Star-like divergence/convergence attack patterns
  - Tree-structured attack detection
  - Chain-like attack identification
  - Behavioral similarity analysis
- **Performance**: Real-world deployment (Linea airdrop: 654K Sybils flagged)
- **Advantage**: Actually designed for airdrop hunter detection
- **Code**: Available on GitHub (TrustaLabs/Airdrop-Sybil-Identification)

#### **2. Subgraph-based Feature Propagation (2025)**
- **Paper**: "Detecting Sybil Addresses in Blockchain Airdrops: A Subgraph-based Feature Propagation and Fusion Approach"
- **Approach**: Two-layer deep transaction subgraph construction
- **Features**:
  - Lifecycle-based event operation features
  - Amount and network structure features
  - Feature propagation and fusion
- **Performance**: 0.9+ precision, recall, F1, AUC on 193K addresses
- **Dataset**: 23,240 Sybil addresses identified
- **Advantage**: Latest academic SOTA for airdrop detection

### **Tier 2: Graph Neural Network Baselines**

#### **3. SybilGAT (Graph Attention Networks)**
- **Paper**: "Sybil Detection using Graph Neural Networks"
- **Approach**: Graph Attention Networks with dynamic attention weighting
- **Architecture**: Multi-layer GAT with attention mechanisms
- **Features**: Node embeddings, graph topology, attention weights
- **Performance**: Superior to basic GCN methods
- **Advantage**: Specialized for Sybil detection in social/transaction networks

#### **4. Traditional Graph Neural Networks**

**A. Graph Convolutional Networks (GCN)**
- **Approach**: Basic neighbor aggregation with neural networks
- **Features**: Node features + graph structure
- **Performance**: Competitive baseline for graph-based tasks
- **Implementation**: PyTorch Geometric, DGL

**B. GraphSAGE**
- **Approach**: Sampling and aggregating from node neighborhoods
- **Features**: Inductive learning, scalable to large graphs
- **Performance**: Strong baseline for large-scale graphs
- **Advantage**: Handles unseen nodes (new addresses)

**C. Graph Attention Networks (GAT)**
- **Approach**: Attention mechanism for neighbor weighting
- **Features**: Learned attention weights + node features
- **Performance**: Often outperforms GCN
- **Advantage**: Interpretable attention patterns

### **Tier 3: Traditional Machine Learning Baselines**

#### **5. Ensemble Methods**

**LightGBM + Feature Engineering**
- **Approach**: Gradient boosting with hand-crafted features
- **Features**: 
  - Transaction frequency patterns
  - Amount distributions
  - Timing analysis
  - Network centrality measures
- **Performance**: Strong baseline (subgraph paper shows 0.9+ metrics)
- **Advantage**: Interpretable, fast training

**Random Forest**
- **Approach**: Ensemble of decision trees
- **Features**: Traditional financial + network features
- **Performance**: Robust baseline for comparison
- **Advantage**: Feature importance analysis

#### **6. Classical Graph Analysis**

**Community Detection Methods**
- **Louvain Algorithm**: As used in Arbitrum analysis
- **Leiden Algorithm**: Improved community detection
- **Features**: Graph clustering + statistical analysis
- **Performance**: Industry-standard for address clustering
- **Advantage**: Established baseline, no ML required

---

## 📊 **Recommended Baseline Strategy**

### **Primary Baselines (Must Include)**

#### **1. TrustaLabs Framework (Industry SOTA)**
- **Why**: Only production-deployed airdrop hunter detection system
- **Implementation**: Reproduce their 4-pattern detection + behavior analysis
- **Fair Comparison**: Same problem domain (airdrop hunters)
- **Expected Performance**: High - real-world validated

#### **2. Subgraph Feature Propagation (Academic SOTA)**
- **Why**: Latest academic paper specifically for airdrop detection
- **Implementation**: Two-layer subgraph + feature propagation
- **Fair Comparison**: Same input data (blockchain transactions)
- **Expected Performance**: Very high (0.9+ metrics reported)

#### **3. Enhanced GNN Baselines**
```python
Baseline Architecture Options:
├── Basic GCN: Standard graph convolution
├── GraphSAGE: Scalable inductive learning  
├── GAT: Attention-based aggregation
└── Temporal GCN: Add temporal edges for fairness
```

### **Secondary Baselines (For Completeness)**

#### **4. Traditional ML + Feature Engineering**
- **LightGBM**: With comprehensive feature engineering
- **Random Forest**: Standard ensemble baseline
- **Features**: Transaction patterns, network metrics, timing analysis

#### **5. Community Detection + Classification**
- **Two-stage approach**: Louvain clustering → ML classification
- **Features**: Cluster membership + individual behavior
- **Advantage**: Matches some industry practices

---

## ⚖️ **Fair Comparison Framework**

### **Input Data Standardization**
```
All methods receive identical inputs:
├── Transaction graphs (edges = transactions)
├── Node features (address statistics)
├── Temporal information (timestamps)
├── Ground truth labels (hunter vs legitimate)
└── NO NFT metadata (fair for pure crypto)
```

### **Evaluation Protocol**
1. **Same datasets**: Arbitrum, Jupiter, Optimism
2. **Same splits**: Train/validation/test
3. **Same metrics**: Precision, Recall, F1, AUC
4. **Same preprocessing**: Feature normalization, graph construction
5. **Cross-validation**: Multiple random seeds

### **Expected Performance Hierarchy**
```
Anticipated Performance (F1 Score):
├── Our Temporal Graph Transformer: 0.85-0.95 (goal)
├── Subgraph Feature Propagation: 0.80-0.90 (current SOTA)
├── TrustaLabs Framework: 0.75-0.85 (industry validated)
├── Enhanced GNNs (GAT/GraphSAGE): 0.70-0.80
├── Basic GCN: 0.65-0.75
├── LightGBM + Features: 0.70-0.80
└── Community Detection: 0.60-0.70 (baseline)
```

---

## 🔬 **Implementation Strategy**

### **Phase 1: Reproduce SOTA Baselines**
1. **TrustaLabs**: Implement 4-pattern detection + behavior analysis
2. **Subgraph Propagation**: Two-layer subgraph + feature fusion
3. **Enhanced GNNs**: GAT, GraphSAGE, temporal variants

### **Phase 2: Fair Evaluation**
1. **Standardized datasets**: Same Arbitrum/Jupiter data
2. **Ablation studies**: Show contribution of each component
3. **Statistical testing**: Significance of improvements

### **Phase 3: Analysis & Insights**
1. **Failure case analysis**: Where do baselines fail?
2. **Temporal advantage**: Show benefit of behavioral change detection
3. **Cross-chain validation**: Generalization across ecosystems

---

## 💡 **Key Advantages Over Baselines**

### **Our Temporal Graph Transformer Innovations**

#### **1. Behavioral Change Detection**
- **Baselines**: Static pattern analysis
- **Ours**: Dynamic behavioral change around airdrop events
- **Advantage**: Captures hunting behavior evolution

#### **2. End-to-End Learning**
- **Baselines**: Manual feature engineering
- **Ours**: Learned temporal + graph representations
- **Advantage**: Adapts to new hunting strategies

#### **3. Hierarchical Architecture**
- **Baselines**: Single-level analysis
- **Ours**: 3-level temporal-graph fusion
- **Advantage**: Captures patterns at multiple scales

#### **4. Cross-Chain Generalization**
- **Baselines**: Single-platform training
- **Ours**: Multi-ecosystem validation
- **Advantage**: Generalizable insights

---

## 🎯 **Final Recommendation**

### **Primary Baselines to Implement**

1. **TrustaLabs Framework** - Industry standard
2. **Subgraph Feature Propagation** - Academic SOTA  
3. **Graph Attention Networks (GAT)** - Strong GNN baseline
4. **LightGBM + Features** - Traditional ML baseline

### **Why Not ARTEMIS?**
- **Unfair comparison**: NFT vs pure crypto markets
- **Architecture mismatch**: Multimodal vs unimodal
- **Different problem**: NFT trading vs DeFi farming
- **Better alternatives**: Direct airdrop detection methods available

This baseline strategy ensures fair, comprehensive evaluation while highlighting our temporal innovations' unique value in pure cryptocurrency markets.