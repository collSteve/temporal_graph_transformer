"""
Test configuration and utilities for the Temporal Graph Transformer test suite.

This module provides common test configurations, fixtures, and utilities
used across all test files.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test configurations
TEST_CONFIG = {
    'temporal': {
        'd_model': 64,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.1,
        'max_seq_len': 20
    },
    'graph': {
        'd_model': 64,
        'num_layers': 3,
        'num_heads': 4,
        'dropout': 0.1
    },
    'data': {
        'batch_size': 2,
        'num_users': 10,
        'max_transactions': 15,
        'num_collections': 5
    }
}

def get_test_data_config():
    """Get standard test data configuration."""
    return TEST_CONFIG['data'].copy()

def get_temporal_config():
    """Get standard temporal model configuration."""
    return TEST_CONFIG['temporal'].copy()

def get_graph_config():
    """Get standard graph model configuration."""
    return TEST_CONFIG['graph'].copy()

def print_test_header(test_name: str):
    """Print formatted test header."""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING: {test_name}")
    print(f"{'='*60}")

def print_test_result(test_name: str, success: bool, details: str = ""):
    """Print formatted test result."""
    status = "✅ PASSED" if success else "❌ FAILED"
    print(f"{status}: {test_name}")
    if details:
        print(f"   📋 {details}")

def print_test_section(section_name: str):
    """Print test section separator."""
    print(f"\n📂 {section_name}")
    print("-" * 40)