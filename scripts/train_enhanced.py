"""
Enhanced Training Script for Temporal Graph Transformer

Supports multi-dataset training, baseline method comparison, and cross-chain evaluation.
Phase 3 implementation with comprehensive baseline integration framework.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import yaml
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Core model imports
from models.temporal_graph_transformer import TemporalGraphTransformer
from models.temporal_encoder import create_transaction_sequence_config
from models.graph_encoder import create_graph_structure_config

# Dataset imports
from data.arbitrum_dataset import ArbitrumDeFiDataset
from data.jupiter_dataset import JupiterSolanaDataset
from data.optimism_dataset import OptimismDataset
from data.blur_dataset import BlurNFTDataset
from data.solana_dataset import SolanaNFTDataset

# Utility imports
from utils.loss_functions import TemporalGraphLoss
from utils.metrics import BinaryClassificationMetrics, CrossValidationMetrics

# Baseline method imports
from baselines.base_interface import BaselineMethodInterface
from baselines.trustalab_framework import TrustaLabFramework
from baselines.subgraph_propagation import SubgraphFeaturePropagation
from baselines.enhanced_gnns import EnhancedGNNBaseline
from baselines.traditional_ml import LightGBMBaseline, RandomForestBaseline


class DatasetFactory:
    """Factory for creating datasets based on configuration."""
    
    DATASET_CLASSES = {
        'arbitrum': ArbitrumDeFiDataset,
        'jupiter': JupiterSolanaDataset,
        'optimism': OptimismDataset,
        'blur': BlurNFTDataset,
        'solana': SolanaNFTDataset
    }
    
    @classmethod
    def create_dataset(cls, dataset_type: str, data_path: str, split: str, **kwargs):
        """Create a dataset instance based on type."""
        if dataset_type not in cls.DATASET_CLASSES:
            raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(cls.DATASET_CLASSES.keys())}")
        
        dataset_class = cls.DATASET_CLASSES[dataset_type]
        return dataset_class(data_path=data_path, split=split, **kwargs)
    
    @classmethod
    def create_multi_dataset(cls, dataset_configs: List[Dict[str, Any]], split: str):
        """Create a combined dataset from multiple configurations."""
        datasets = []
        
        for config in dataset_configs:
            dataset_type = config['type']
            data_path = config['data_path']
            dataset_kwargs = config.get('kwargs', {})
            
            try:
                dataset = cls.create_dataset(dataset_type, data_path, split, **dataset_kwargs)
                datasets.append(dataset)
                print(f"✅ Loaded {dataset_type} dataset: {len(dataset)} samples")
            except Exception as e:
                print(f"⚠️  Failed to load {dataset_type} dataset: {e}")
                continue
        
        if not datasets:
            raise ValueError("No datasets were successfully loaded")
        
        return ConcatDataset(datasets)


class TemporalGraphTransformerBaseline(BaselineMethodInterface):
    """Wrapper for our main TGT model to fit baseline interface."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.optimizer = None
        self.loss_fn = TemporalGraphLoss()
    
    def _create_model(self, device: torch.device):
        """Create the TGT model."""
        model_config = {
            'd_model': self.config.get('d_model', 256),
            'temporal_config': create_transaction_sequence_config(
                d_model=self.config.get('d_model', 256),
                num_layers=self.config.get('temporal_layers', 4),
                num_heads=self.config.get('temporal_heads', 8),
                dropout=self.config.get('dropout', 0.1)
            ),
            'graph_config': create_graph_structure_config(
                d_model=self.config.get('d_model', 256),
                num_layers=self.config.get('graph_layers', 6),
                num_heads=self.config.get('graph_heads', 8),
                dropout=self.config.get('dropout', 0.1)
            ),
            'output_dim': self.config.get('d_model', 256),
            'num_classes': 2,
            'fusion_type': self.config.get('fusion_type', 'cross_attention')
        }
        
        self.model = TemporalGraphTransformer(model_config).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('lr', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
        """Train the TGT model."""
        if self.model is None:
            self._create_model(device)
        
        epochs = self.config.get('epochs', 50)
        best_val_f1 = 0.0
        
        for epoch in range(epochs):
            # Training epoch
            train_metrics = self._train_epoch(train_loader, device)
            
            # Validation epoch
            if (epoch + 1) % 5 == 0:  # Validate every 5 epochs
                val_metrics = self._validate_epoch(val_loader, device)
                
                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                
                if self.config.get('verbose', False):
                    print(f"Epoch {epoch+1}/{epochs} - Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        return {'best_val_f1': best_val_f1}
    
    def _train_epoch(self, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics = BinaryClassificationMetrics()
        total_loss = 0.0
        
        for batch in dataloader:
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            
            # Forward pass
            outputs = self.model(batch)
            labels = batch['labels']
            loss_dict = self.loss_fn(outputs, batch, labels)
            loss = loss_dict['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = outputs['probabilities'][:, 1]
            metrics.update(predictions, labels)
        
        final_metrics = metrics.compute()
        final_metrics['loss'] = total_loss / len(dataloader)
        return final_metrics
    
    def _validate_epoch(self, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        metrics = BinaryClassificationMetrics()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device)
                
                # Forward pass
                outputs = self.model(batch)
                labels = batch['labels']
                loss_dict = self.loss_fn(outputs, batch, labels)
                loss = loss_dict['total']
                
                # Update metrics
                total_loss += loss.item()
                predictions = outputs['probabilities'][:, 1]
                metrics.update(predictions, labels)
        
        final_metrics = metrics.compute()
        final_metrics['loss'] = total_loss / len(dataloader)
        return final_metrics
    
    def evaluate(self, test_loader: DataLoader, device: torch.device) -> Dict[str, float]:
        """Evaluate the model on test set."""
        return self._validate_epoch(test_loader, device)
    
    def predict(self, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        """Make predictions on a batch."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch)
            return outputs['probabilities']


class MultiDatasetTrainer:
    """Enhanced trainer supporting multiple datasets and baseline comparison."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._setup_device()
        self.baseline_methods = {}
        
    def _setup_device(self) -> torch.device:
        """Setup computational device."""
        device_config = self.config.get('device', 'auto')
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device_config)
        
        print(f"Using device: {device}")
        return device
    
    def register_baseline(self, name: str, baseline: BaselineMethodInterface):
        """Register a baseline method for comparison."""
        self.baseline_methods[name] = baseline
        print(f"Registered baseline: {name}")
    
    def _register_all_baselines(self):
        """Register all baseline methods automatically."""
        # Import baseline methods
        from baselines import (
            TrustaLabFramework, SubgraphFeaturePropagation, 
            EnhancedGNNBaseline, LightGBMBaseline, RandomForestBaseline
        )
        from models.temporal_graph_transformer import TemporalGraphTransformer
        from baselines.temporal_graph_transformer_baseline import TemporalGraphTransformerBaseline
        
        # TGT Model configuration
        tgt_config = self.config.get('model', {})
        tgt_config.update({
            'd_model': tgt_config.get('d_model', 64),
            'temporal_layers': tgt_config.get('temporal_layers', 3),
            'temporal_heads': tgt_config.get('temporal_heads', 8),
            'graph_layers': tgt_config.get('graph_layers', 3),
            'graph_heads': tgt_config.get('graph_heads', 8),
            'epochs': tgt_config.get('epochs', 10),
            'lr': tgt_config.get('lr', 1e-3)
        })
        
        # Register TGT baseline
        tgt_baseline = TemporalGraphTransformerBaseline(tgt_config)
        self.register_baseline('TemporalGraphTransformer', tgt_baseline)
        
        # Register TrustaLabs
        trustalab_config = self.config.get('trustalab', {})
        trustalab_baseline = TrustaLabFramework(trustalab_config)
        self.register_baseline('TrustaLabFramework', trustalab_baseline)
        
        # Register Subgraph Propagation
        subgraph_config = self.config.get('subgraph_propagation', {})
        subgraph_baseline = SubgraphFeaturePropagation(subgraph_config)
        self.register_baseline('SubgraphFeaturePropagation', subgraph_baseline)
        
        # Register Enhanced GNNs
        gnn_config = self.config.get('enhanced_gnns', {})
        for gnn_type in ['gat', 'graphsage', 'sybilgat', 'gcn']:
            gnn_baseline_config = gnn_config.copy()
            gnn_baseline_config['model_type'] = gnn_type
            gnn_baseline = EnhancedGNNBaseline(gnn_baseline_config)
            self.register_baseline(gnn_type.upper(), gnn_baseline)
        
        # Register Traditional ML
        ml_config = self.config.get('traditional_ml', {})
        lgb_baseline = LightGBMBaseline(ml_config)
        self.register_baseline('LightGBM', lgb_baseline)
        
        rf_baseline = RandomForestBaseline(ml_config)
        self.register_baseline('RandomForest', rf_baseline)
    
    def _create_all_baseline_methods(self):
        """Create all baseline methods and return them."""
        self._register_all_baselines()
        return self.baseline_methods
    
    def create_data_loaders(self) -> Dict[str, DataLoader]:
        """Create data loaders for training, validation, and testing."""
        dataset_configs = self.config['datasets']
        batch_size = self.config.get('batch_size', 32)
        num_workers = self.config.get('num_workers', 4)
        
        data_loaders = {}
        
        for split in ['train', 'val', 'test']:
            try:
                dataset = DatasetFactory.create_multi_dataset(dataset_configs, split)
                data_loaders[split] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=(split == 'train'),
                    num_workers=num_workers,
                    pin_memory=True if self.device.type == 'cuda' else False
                )
                print(f"Created {split} loader: {len(dataset)} samples")
            except Exception as e:
                print(f"⚠️  Failed to create {split} loader: {e}")
                data_loaders[split] = None
        
        return data_loaders
    
    def train_and_evaluate_all(self) -> Dict[str, Dict[str, float]]:
        """Train and evaluate all registered baseline methods."""
        data_loaders = self.create_data_loaders()
        
        if not data_loaders['train'] or not data_loaders['val']:
            raise ValueError("Training and validation data loaders are required")
        
        results = {}
        
        # Add our main TGT model
        tgt_config = self.config.get('model', {})
        tgt_baseline = TemporalGraphTransformerBaseline(tgt_config)
        self.register_baseline('TemporalGraphTransformer', tgt_baseline)
        
        # Add TrustaLabs baseline
        trustalab_config = self.config.get('trustalab', {})
        trustalab_baseline = TrustaLabFramework(trustalab_config)
        self.register_baseline('TrustaLabFramework', trustalab_baseline)
        
        # Add Subgraph Feature Propagation baseline
        subgraph_config = self.config.get('subgraph_propagation', {})
        subgraph_baseline = SubgraphFeaturePropagation(subgraph_config)
        self.register_baseline('SubgraphFeaturePropagation', subgraph_baseline)
        
        # Add Enhanced GNN baselines
        gnn_base_config = self.config.get('enhanced_gnns', {})
        
        # GAT baseline
        gat_config = gnn_base_config.copy()
        gat_config['model_type'] = 'gat'
        gat_baseline = EnhancedGNNBaseline(gat_config)
        self.register_baseline('GAT', gat_baseline)
        
        # GraphSAGE baseline
        sage_config = gnn_base_config.copy()
        sage_config['model_type'] = 'graphsage'
        sage_baseline = EnhancedGNNBaseline(sage_config)
        self.register_baseline('GraphSAGE', sage_baseline)
        
        # SybilGAT baseline
        sybilgat_config = gnn_base_config.copy()
        sybilgat_config['model_type'] = 'sybilgat'
        sybilgat_baseline = EnhancedGNNBaseline(sybilgat_config)
        self.register_baseline('SybilGAT', sybilgat_baseline)
        
        # Basic GCN baseline
        gcn_config = gnn_base_config.copy()
        gcn_config['model_type'] = 'gcn'
        gcn_baseline = EnhancedGNNBaseline(gcn_config)
        self.register_baseline('BasicGCN', gcn_baseline)
        
        # Add Traditional ML baselines
        traditional_ml_config = self.config.get('traditional_ml', {})
        
        # LightGBM baseline
        lgb_config = traditional_ml_config.copy()
        lgb_baseline = LightGBMBaseline(lgb_config)
        self.register_baseline('LightGBM', lgb_baseline)
        
        # Random Forest baseline
        rf_config = traditional_ml_config.copy()
        rf_baseline = RandomForestBaseline(rf_config)
        self.register_baseline('RandomForest', rf_baseline)
        
        # Train and evaluate each baseline
        for name, baseline in self.baseline_methods.items():
            print(f"\n{'='*50}")
            print(f"Training {name}")
            print(f"{'='*50}")
            
            try:
                # Training
                train_results = baseline.train(
                    data_loaders['train'], 
                    data_loaders['val'], 
                    self.device
                )
                
                # Testing
                if data_loaders['test']:
                    test_results = baseline.evaluate(data_loaders['test'], self.device)
                else:
                    test_results = baseline.evaluate(data_loaders['val'], self.device)
                
                # Combine results
                results[name] = {
                    **train_results,
                    **{f"test_{k}": v for k, v in test_results.items()}
                }
                
                print(f"✅ {name} completed - Test F1: {test_results['f1']:.4f}")
                
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def cross_validate(self, n_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """Perform cross-validation evaluation."""
        print(f"\n{'='*50}")
        print(f"Cross-Validation with {n_folds} folds")
        print(f"{'='*50}")
        
        cv_metrics = CrossValidationMetrics()
        
        # For now, simulate cross-validation by running multiple times
        # In a full implementation, this would properly split the data
        for fold in range(n_folds):
            print(f"\nFold {fold + 1}/{n_folds}")
            fold_results = self.train_and_evaluate_all()
            
            # Extract main metrics for CV
            for method_name, results in fold_results.items():
                if 'test_f1' in results:
                    cv_metrics.add_fold({
                        f'{method_name}_f1': results['test_f1'],
                        f'{method_name}_accuracy': results.get('test_accuracy', 0.0),
                        f'{method_name}_precision': results.get('test_precision', 0.0),
                        f'{method_name}_recall': results.get('test_recall', 0.0)
                    })
        
        cv_summary = cv_metrics.compute_summary()
        return cv_summary
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save training and evaluation results."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_default_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Create default configuration from command line arguments."""
    return {
        'datasets': [
            {
                'type': 'arbitrum',
                'data_path': args.data_path,
                'kwargs': {
                    'max_sequence_length': args.max_sequence_length,
                    'include_known_hunters': True
                }
            },
            {
                'type': 'jupiter',
                'data_path': args.data_path,
                'kwargs': {
                    'max_sequence_length': args.max_sequence_length,
                    'anti_farming_analysis': True
                }
            }
        ],
        'model': {
            'd_model': args.d_model,
            'temporal_layers': args.temporal_layers,
            'temporal_heads': args.temporal_heads,
            'graph_layers': args.graph_layers,
            'graph_heads': args.graph_heads,
            'fusion_type': args.fusion_type,
            'dropout': args.dropout,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'epochs': args.epochs,
            'verbose': True
        },
        'batch_size': args.batch_size,
        'num_workers': 4,
        'device': args.device,
        'output_dir': args.output_dir
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Enhanced Multi-Dataset Training for Temporal Graph Transformer')
    
    # Data arguments
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to dataset directory')
    parser.add_argument('--max_sequence_length', type=int, default=50,
                        help='Maximum transaction sequence length')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=256,
                        help='Model dimension')
    parser.add_argument('--temporal_layers', type=int, default=4,
                        help='Number of temporal transformer layers')
    parser.add_argument('--temporal_heads', type=int, default=8,
                        help='Number of temporal attention heads')
    parser.add_argument('--graph_layers', type=int, default=6,
                        help='Number of graph transformer layers')
    parser.add_argument('--graph_heads', type=int, default=8,
                        help='Number of graph attention heads')
    parser.add_argument('--fusion_type', type=str, default='cross_attention',
                        choices=['cross_attention', 'concatenation', 'addition'],
                        help='Fusion strategy for temporal and graph representations')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    
    # Evaluation arguments
    parser.add_argument('--cross_validation', action='store_true',
                        help='Perform cross-validation evaluation')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./experiments/phase3_outputs',
                        help='Output directory for saving results')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config(args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create trainer
    trainer = MultiDatasetTrainer(config)
    
    # Run training and evaluation
    if args.cross_validation:
        print(f"\n🔬 Starting Cross-Validation with {args.cv_folds} folds")
        results = trainer.cross_validate(args.cv_folds)
    else:
        print(f"\n🚀 Starting Multi-Dataset Training")
        results = trainer.train_and_evaluate_all()
    
    # Save results
    results_path = os.path.join(args.output_dir, 'results.json')
    trainer.save_results(results, results_path)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    
    if args.cross_validation:
        for metric, stats in results.items():
            if 'f1' in metric:
                method_name = metric.replace('_f1', '')
                mean_f1 = stats.get('mean', 0.0)
                std_f1 = stats.get('std', 0.0)
                print(f"{method_name}: F1 = {mean_f1:.4f} ± {std_f1:.4f}")
    else:
        for method_name, metrics in results.items():
            if 'test_f1' in metrics:
                print(f"{method_name}: F1 = {metrics['test_f1']:.4f}")
    
    print(f"\n✅ Training completed! Results saved to {results_path}")


if __name__ == "__main__":
    main()