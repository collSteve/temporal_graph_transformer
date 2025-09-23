"""
Evaluation Framework for Temporal Graph Transformer

Comprehensive evaluation and benchmarking suite for airdrop hunter detection:
- Cross-validation framework with multiple strategies
- Statistical significance testing
- Performance benchmarking and comparison
- Result analysis and visualization tools
- Phase 4: Experimental validation and interpretability analysis
"""

# Core evaluation framework (Phases 1-3)
from .cross_validation import CrossValidationFramework, CrossValidationResult
from .benchmarking_suite import ComprehensiveBenchmarkingSuite, BenchmarkingExperiment, run_full_benchmark

# Phase 4: Experimental validation framework
from .phase4_experimental_framework import (
    ExperimentalConfig,
    ComprehensiveEvaluationRunner,
    CrossChainGeneralizationAnalyzer,
    run_phase4_comprehensive_evaluation,
    run_phase4_cross_chain_analysis
)

# Phase 4: Temporal and failure analysis
from .temporal_failure_analysis import (
    TemporalPatternAnalyzer,
    FailureCaseAnalyzer,
    generate_temporal_analysis_report,
    generate_failure_analysis_report
)

# Phase 4: Ablation and interpretability analysis
from .ablation_interpretability import (
    AblationStudyFramework,
    InterpretabilityAnalyzer,
    run_complete_phase4_analysis
)

__all__ = [
    # Core evaluation framework
    'CrossValidationFramework',
    'CrossValidationResult',
    'ComprehensiveBenchmarkingSuite',
    'BenchmarkingExperiment',
    'run_full_benchmark',
    
    # Phase 4: Experimental framework
    'ExperimentalConfig',
    'ComprehensiveEvaluationRunner',
    'CrossChainGeneralizationAnalyzer',
    'run_phase4_comprehensive_evaluation',
    'run_phase4_cross_chain_analysis',
    
    # Phase 4: Temporal and failure analysis
    'TemporalPatternAnalyzer',
    'FailureCaseAnalyzer',
    'generate_temporal_analysis_report',
    'generate_failure_analysis_report',
    
    # Phase 4: Ablation and interpretability
    'AblationStudyFramework',
    'InterpretabilityAnalyzer',
    'run_complete_phase4_analysis'
]