#!/usr/bin/env python3
"""
Unit tests for graph encoder components.

Tests:
1. MultiModalEdgeEncoder - edge feature processing
2. StructuralPositionalEncoding - graph topology encoding 
3. GraphStructureTransformer - complete graph modeling
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add test utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.test_config import print_test_header, print_test_result, print_test_section, get_graph_config

# Import components to test
from models.graph_encoder import (
    MultiModalEdgeEncoder,
    StructuralPositionalEncoding,
    GraphStructureTransformer,
    create_graph_structure_config
)

def test_multimodal_edge_encoder():
    """Test MultiModalEdgeEncoder component."""
    print_test_section("Multi-Modal Edge Encoder Tests")
    
    config = get_graph_config()
    config.update({
        'nft_visual_dim': 64,
        'nft_text_dim': 64,
        'transaction_dim': 32,
        'market_features_dim': 16
    })
    
    # Test 1: Basic edge encoding
    try:
        encoder = MultiModalEdgeEncoder(config)
        num_edges = 20
        
        # Create mock edge features
        edge_features = {
            'nft_visual': torch.randn(num_edges, 64),
            'nft_text': torch.randn(num_edges, 64),
            'transaction_features': torch.randn(num_edges, 32),
            'market_features': torch.randn(num_edges, 16)
        }
        
        output = encoder(edge_features)
        expected_shape = (num_edges, config['d_model'])
        
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains infinite values"
        
        print_test_result("Basic edge encoding", True, f"Shape: {output.shape}")
        
    except Exception as e:
        print_test_result("Basic edge encoding", False, str(e))
        return False
    
    # Test 2: Partial features
    try:
        partial_features = {
            'nft_visual': torch.randn(num_edges, 64),
            'transaction_features': torch.randn(num_edges, 32)
            # Missing nft_text and market_features
        }
        
        output_partial = encoder(partial_features)
        assert output_partial.shape == expected_shape, "Partial features failed"
        
        print_test_result("Partial features", True, "Handles missing modalities")
        
    except Exception as e:
        print_test_result("Partial features", False, str(e))
        return False
    
    # Test 3: Empty features
    try:
        empty_features = {}
        output_empty = encoder(empty_features)
        
        assert output_empty.shape == expected_shape, "Empty features shape incorrect"
        
        print_test_result("Empty features fallback", True, "Graceful degradation")
        
    except Exception as e:
        print_test_result("Empty features fallback", False, str(e))
        return False
    
    return True

def test_structural_positional_encoding():
    """Test StructuralPositionalEncoding component."""
    print_test_section("Structural Positional Encoding Tests")
    
    config = get_graph_config()
    d_model = config['d_model']
    
    # Test 1: Basic positional encoding
    try:
        encoder = StructuralPositionalEncoding(d_model)
        num_nodes = 15
        
        # Create mock graph structure
        edge_index = torch.randint(0, num_nodes, (2, 40))  # 40 edges
        
        output = encoder(edge_index, num_nodes)
        expected_shape = (num_nodes, d_model)
        
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains infinite values"
        
        print_test_result("Basic positional encoding", True, f"Shape: {output.shape}")
        
    except Exception as e:
        print_test_result("Basic positional encoding", False, str(e))
        return False
    
    # Test 2: Different graph topologies
    try:
        # Test with different graph structures
        results = {}
        
        # Dense graph
        dense_edges = torch.combinations(torch.arange(num_nodes), r=2).T
        dense_output = encoder(dense_edges, num_nodes)
        results['dense'] = dense_output.shape
        
        # Sparse graph  
        sparse_edges = torch.randint(0, num_nodes, (2, 10))
        sparse_output = encoder(sparse_edges, num_nodes)
        results['sparse'] = sparse_output.shape
        
        # Linear chain
        chain_edges = torch.stack([
            torch.arange(num_nodes - 1),
            torch.arange(1, num_nodes)
        ])
        chain_output = encoder(chain_edges, num_nodes)
        results['chain'] = chain_output.shape
        
        # All should have same shape
        assert all(shape == expected_shape for shape in results.values()), "Different topologies produce different shapes"
        
        # But different values
        assert not torch.allclose(dense_output, sparse_output, atol=1e-3), "Dense and sparse too similar"
        
        print_test_result("Different graph topologies", True, f"Handles: {list(results.keys())}")
        
    except Exception as e:
        print_test_result("Different graph topologies", False, str(e))
        return False
    
    return True

def test_graph_structure_transformer():
    """Test complete GraphStructureTransformer."""
    print_test_section("Graph Structure Transformer Tests")
    
    # Test 1: Full model functionality
    try:
        config = create_graph_structure_config(
            d_model=64,
            num_layers=3,
            num_heads=4,
            dropout=0.1
        )
        
        model = GraphStructureTransformer(config)
        num_nodes = 20
        num_edges = 50
        
        # Create mock data
        node_features = torch.randn(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_features = {
            'edge_features': torch.randn(num_edges, 64)
        }
        
        model.eval()
        with torch.no_grad():
            outputs = model(node_features, edge_index, edge_features)
        
        # Check outputs
        assert 'node_embeddings' in outputs, "Missing node_embeddings"
        assert 'graph_embedding' in outputs, "Missing graph_embedding"
        assert 'attention_weights' in outputs, "Missing attention_weights"
        
        node_emb = outputs['node_embeddings']
        graph_emb = outputs['graph_embedding']
        
        assert node_emb.shape == (num_nodes, 64), f"Node embeddings shape: {node_emb.shape}"
        assert graph_emb.shape == (64,), f"Graph embedding shape: {graph_emb.shape}"
        
        print_test_result("Full model forward", True, f"Nodes: {node_emb.shape}, Graph: {graph_emb.shape}")
        
    except Exception as e:
        print_test_result("Full model forward", False, str(e))
        return False
    
    # Test 2: Different pooling strategies
    try:
        pooling_strategies = ['mean', 'max', 'attention', 'add']
        results = {}
        
        for pooling in pooling_strategies:
            pool_config = config.copy()
            pool_config['global_pool'] = pooling
            pool_model = GraphStructureTransformer(pool_config)
            
            pool_model.eval()
            with torch.no_grad():
                pool_outputs = pool_model(node_features, edge_index, edge_features)
            
            results[pooling] = pool_outputs['graph_embedding'].shape
            assert pool_outputs['graph_embedding'].shape == (64,), f"Pooling {pooling} failed"
        
        print_test_result("Different pooling strategies", True, f"All strategies work: {list(results.keys())}")
        
    except Exception as e:
        print_test_result("Different pooling strategies", False, str(e))
        return False
    
    # Test 3: Variable graph sizes
    try:
        sizes = [5, 15, 25]
        results = {}
        
        for size in sizes:
            test_nodes = torch.randn(size, 64)
            test_edges = torch.randint(0, size, (2, size * 2))
            test_edge_features = {
                'edge_features': torch.randn(size * 2, 64)
            }
            
            model.eval()
            with torch.no_grad():
                size_outputs = model(test_nodes, test_edges, test_edge_features)
            
            results[size] = size_outputs['node_embeddings'].shape[0]
            assert size_outputs['node_embeddings'].shape == (size, 64), f"Size {size} failed"
            assert size_outputs['graph_embedding'].shape == (64,), f"Graph embedding size {size} failed"
        
        print_test_result("Variable graph sizes", True, f"Handled sizes: {list(results.keys())}")
        
    except Exception as e:
        print_test_result("Variable graph sizes", False, str(e))
        return False
    
    # Test 4: Gradient flow
    try:
        model.train()
        outputs = model(node_features, edge_index, edge_features)
        
        # Simple loss for gradient test
        loss = outputs['node_embeddings'].sum() + outputs['graph_embedding'].sum()
        loss.backward()
        
        # Check gradients exist
        has_gradients = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for param in model.parameters()
        )
        
        assert has_gradients, "No gradients found"
        print_test_result("Gradient flow", True, "Gradients computed successfully")
        
    except Exception as e:
        print_test_result("Gradient flow", False, str(e))
        return False
    
    return True

def run_all_tests():
    """Run all graph encoder tests."""
    print_test_header("Graph Encoder Components")
    
    results = {
        'multimodal_edge_encoder': test_multimodal_edge_encoder(),
        'structural_positional_encoding': test_structural_positional_encoding(),
        'graph_structure_transformer': test_graph_structure_transformer()
    }
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n📊 GRAPH ENCODER TEST SUMMARY:")
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   ❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All graph encoder tests PASSED!")
        return True
    else:
        print("⚠️  Some graph encoder tests FAILED!")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)