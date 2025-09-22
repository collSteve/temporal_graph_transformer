"""
Evaluation Framework for Temporal Graph Transformer

Comprehensive evaluation and benchmarking suite for airdrop hunter detection:
- Cross-validation framework with multiple strategies
- Statistical significance testing
- Performance benchmarking and comparison
- Result analysis and visualization tools
"""

from .cross_validation import CrossValidationFramework, CrossValidationResult
from .benchmarking_suite import ComprehensiveBenchmarkingSuite, BenchmarkingExperiment, run_full_benchmark

__all__ = [
    'CrossValidationFramework',
    'CrossValidationResult',
    'ComprehensiveBenchmarkingSuite',
    'BenchmarkingExperiment',
    'run_full_benchmark'
]