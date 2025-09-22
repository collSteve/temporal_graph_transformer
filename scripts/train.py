"""Training script for Temporal Graph Transformer."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.temporal_graph_transformer import TemporalGraphTransformer
from models.temporal_encoder import create_transaction_sequence_config
from models.graph_encoder import create_graph_structure_config
from data.solana_dataset import SolanaNFTDataset
from utils.loss_functions import TemporalGraphLoss
from utils.metrics import BinaryClassificationMetrics


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Create model configuration from command line arguments."""
    return {
        'd_model': args.d_model,
        'temporal_config': create_transaction_sequence_config(
            d_model=args.d_model,
            num_layers=args.temporal_layers,
            num_heads=args.temporal_heads,
            dropout=args.dropout
        ),
        'graph_config': create_graph_structure_config(
            d_model=args.d_model,
            num_layers=args.graph_layers,
            num_heads=args.graph_heads,
            dropout=args.dropout
        ),
        'output_dim': args.d_model,
        'num_classes': 2,
        'fusion_type': args.fusion_type
    }


def train_epoch(model: nn.Module, 
                dataloader: DataLoader, 
                loss_fn: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    metrics = BinaryClassificationMetrics()
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        
        # Forward pass
        outputs = model(batch)
        
        # Compute loss
        labels = batch['labels']
        loss_dict = loss_fn(outputs, batch, labels)
        loss = loss_dict['total']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        predictions = outputs['probabilities'][:, 1]  # Hunter probability
        metrics.update(predictions, labels)
        
        # Update progress bar
        current_metrics = metrics.compute()
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{current_metrics['accuracy']:.4f}",
            'f1': f"{current_metrics['f1']:.4f}"
        })
    
    # Compute final metrics
    final_metrics = metrics.compute()
    final_metrics['loss'] = total_loss / len(dataloader)
    
    return final_metrics


def validate_epoch(model: nn.Module, 
                  dataloader: DataLoader, 
                  loss_fn: nn.Module,
                  device: torch.device) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    metrics = BinaryClassificationMetrics()
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        
        for batch in progress_bar:
            # Move batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Compute loss
            labels = batch['labels']
            loss_dict = loss_fn(outputs, batch, labels)
            loss = loss_dict['total']
            
            # Update metrics
            total_loss += loss.item()
            predictions = outputs['probabilities'][:, 1]
            metrics.update(predictions, labels)
            
            # Update progress bar
            current_metrics = metrics.compute()
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{current_metrics['accuracy']:.4f}",
                'f1': f"{current_metrics['f1']:.4f}"
            })
    
    # Compute final metrics
    final_metrics = metrics.compute()
    final_metrics['loss'] = total_loss / len(dataloader)
    
    return final_metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Temporal Graph Transformer')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--marketplace', type=str, default='magic_eden',
                        help='NFT marketplace to use')
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
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./experiments/outputs',
                        help='Output directory for saving models and logs')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save model every N epochs')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Evaluate every N epochs')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_dict = vars(args)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = SolanaNFTDataset(
        data_path=args.data_path,
        split='train',
        marketplace=args.marketplace,
        max_sequence_length=args.max_sequence_length
    )
    
    val_dataset = SolanaNFTDataset(
        data_path=args.data_path,
        split='val',
        marketplace=args.marketplace,
        max_sequence_length=args.max_sequence_length
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print("Creating model...")
    model_config = create_model_config(args)
    model = TemporalGraphTransformer(model_config)
    model = model.to(device)
    
    # Create loss function and optimizer
    loss_fn = TemporalGraphLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    # Training loop
    print("Starting training...")
    best_val_f1 = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        
        # Validate
        if (epoch + 1) % args.eval_every == 0:
            val_metrics = validate_epoch(model, val_loader, loss_fn, device)
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_f1': best_val_f1,
                    'config': model_config
                }, os.path.join(args.output_dir, 'best_model.pt'))
                print(f"Saved new best model with F1: {best_val_f1:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pt'))
        
        # Update learning rate
        scheduler.step()
    
    print(f"\nTraining completed! Best validation F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()