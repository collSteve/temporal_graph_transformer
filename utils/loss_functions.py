"""
Multi-task loss functions for Temporal Graph Transformer training.

This module implements sophisticated loss functions that go beyond simple
classification to encourage the model to learn robust behavioral representations:

1. Classification Loss: Standard cross-entropy for hunter vs legitimate
2. Contrastive Loss: Self-supervised learning to reduce labeling dependency  
3. Temporal Consistency Loss: Encourages smooth temporal patterns
4. Behavioral Change Loss: Emphasizes detection of sudden behavior changes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class InfoNCE(nn.Module):
    """
    InfoNCE contrastive loss for self-supervised representation learning.
    
    This loss encourages similar users (same class) to have similar embeddings
    while pushing dissimilar users (different classes) apart in embedding space.
    Reduces dependency on labeled data.
    """
    
    def __init__(self, temperature: float = 0.1, negative_mode: str = 'unpaired'):
        super().__init__()
        
        self.temperature = temperature
        self.negative_mode = negative_mode  # 'unpaired' or 'paired'
        
    def forward(self, 
                embeddings: torch.Tensor,
                labels: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            embeddings: User embeddings of shape (batch_size, embedding_dim)
            labels: Binary labels (0=legitimate, 1=hunter) of shape (batch_size,)
            mask: Optional mask for valid samples
            
        Returns:
            contrastive_loss: InfoNCE loss value
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device
        
        if mask is None:
            mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create positive pairs mask (same class)
        labels_expanded = labels.unsqueeze(1).expand(batch_size, batch_size)
        positive_mask = (labels_expanded == labels_expanded.T) & mask.unsqueeze(1) & mask.unsqueeze(0)
        
        # Remove self-similarity
        positive_mask.fill_diagonal_(False)
        
        # Create negative pairs mask (different class)
        negative_mask = (labels_expanded != labels_expanded.T) & mask.unsqueeze(1) & mask.unsqueeze(0)
        
        # Compute contrastive loss
        if positive_mask.sum() == 0:
            # No positive pairs, return zero loss
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # For each anchor, compute loss
        losses = []
        for i in range(batch_size):
            if not mask[i]:
                continue
                
            pos_similarities = similarity_matrix[i][positive_mask[i]]
            neg_similarities = similarity_matrix[i][negative_mask[i]]
            
            if len(pos_similarities) == 0:
                continue
                
            # InfoNCE loss for this anchor
            pos_exp = torch.exp(pos_similarities)
            neg_exp = torch.exp(neg_similarities)
            
            # Numerator: sum of positive similarities
            numerator = pos_exp.sum()
            
            # Denominator: positive + negative similarities
            denominator = pos_exp.sum() + neg_exp.sum()
            
            # Loss for this anchor
            anchor_loss = -torch.log(numerator / (denominator + 1e-8))
            losses.append(anchor_loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        return torch.stack(losses).mean()


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss to encourage smooth behavioral patterns.
    
    Penalizes sudden, unexplained changes in user behavior that are not
    associated with airdrop events. This helps the model learn that 
    legitimate users have more consistent behavior over time.
    """
    
    def __init__(self, smoothness_weight: float = 1.0, change_threshold: float = 0.5):
        super().__init__()
        
        self.smoothness_weight = smoothness_weight
        self.change_threshold = change_threshold
        
    def forward(self, 
                temporal_sequence: torch.Tensor,
                timestamps: torch.Tensor,
                airdrop_events: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            temporal_sequence: Temporal embeddings of shape (batch_size, seq_len, d_model)
            timestamps: Transaction timestamps of shape (batch_size, seq_len)
            airdrop_events: Optional airdrop event timestamps
            labels: Optional labels to weight loss differently for each class
            
        Returns:
            consistency_loss: Temporal consistency loss value
        """
        batch_size, seq_len, d_model = temporal_sequence.shape
        
        if seq_len < 2:
            return torch.tensor(0.0, device=temporal_sequence.device, requires_grad=True)
        
        # Compute temporal gradients (changes between consecutive time steps)
        temporal_diffs = torch.diff(temporal_sequence, dim=1)  # (batch_size, seq_len-1, d_model)
        time_diffs = torch.diff(timestamps, dim=1)  # (batch_size, seq_len-1)
        
        # Normalize by time differences to account for irregular sampling
        time_diffs = torch.clamp(time_diffs.abs(), min=1.0, max=86400.0)  # Clamp to reasonable range (1 sec to 1 day)
        normalized_diffs = temporal_diffs / time_diffs.unsqueeze(-1)
        
        # Compute smoothness loss (L2 norm of temporal gradients)
        smoothness_loss = torch.norm(normalized_diffs, dim=-1).mean() * 0.1  # Scale down to prevent explosion
        
        # If airdrop events are provided, allow larger changes near those events
        if airdrop_events is not None:
            airdrop_tolerance_mask = self._create_airdrop_tolerance_mask(
                timestamps[:, 1:], airdrop_events, window_days=7.0
            )
            
            # Reduce penalty for changes near airdrop events
            change_penalties = torch.norm(normalized_diffs, dim=-1)
            change_penalties = change_penalties * (~airdrop_tolerance_mask).float()
            smoothness_loss = change_penalties.mean()
        
        # Weight loss differently for hunters vs legitimate users
        if labels is not None:
            # Legitimate users (label=0) should have higher consistency
            # Hunters (label=1) are allowed more variability
            class_weights = torch.where(labels == 0, 1.5, 0.5)  # Higher weight for legitimate users
            
            # Expand weights to sequence dimension
            expanded_weights = class_weights.unsqueeze(1).expand(-1, seq_len-1)
            smoothness_loss = (smoothness_loss * expanded_weights).mean()
        
        return self.smoothness_weight * smoothness_loss
    
    def _create_airdrop_tolerance_mask(self, 
                                     timestamps: torch.Tensor,
                                     airdrop_events: torch.Tensor,
                                     window_days: float = 7.0) -> torch.Tensor:
        """Create mask for periods around airdrop events where changes are tolerated."""
        window_seconds = window_days * 24 * 3600
        batch_size, seq_len = timestamps.shape
        
        # Expand dimensions for broadcasting
        times_expanded = timestamps.unsqueeze(-1)  # (batch_size, seq_len, 1)
        events_expanded = airdrop_events.unsqueeze(0).unsqueeze(0)  # (1, 1, num_events)
        
        # Check if any timestamp is within window of any airdrop event
        distances = torch.abs(times_expanded - events_expanded)
        within_window = distances <= window_seconds
        tolerance_mask = within_window.any(dim=-1)  # (batch_size, seq_len)
        
        return tolerance_mask


class BehavioralChangeLoss(nn.Module):
    """
    Behavioral change loss to emphasize detection of airdrop-related behavior changes.
    
    This loss encourages the model to detect significant behavioral changes
    around airdrop events, which is a key indicator of hunting behavior.
    """
    
    def __init__(self, change_weight: float = 1.0, margin: float = 0.2):
        super().__init__()
        
        self.change_weight = change_weight
        self.margin = margin
        
    def forward(self, 
                change_scores: torch.Tensor,
                labels: torch.Tensor,
                confidence: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            change_scores: Behavioral change scores of shape (batch_size,)
            labels: Binary labels (0=legitimate, 1=hunter)
            confidence: Optional confidence scores for weighting
            
        Returns:
            change_loss: Behavioral change loss value
        """
        # Hunters should have high change scores, legitimate users should have low scores
        target_scores = labels.float()  # 0 for legitimate, 1 for hunters
        
        # Margin-based loss: encourage separation between classes
        hunter_scores = change_scores[labels == 1]
        legit_scores = change_scores[labels == 0]
        
        if len(hunter_scores) == 0 or len(legit_scores) == 0:
            # If we don't have both classes, use MSE loss instead
            target_scores = labels.float()  # 0 for legitimate, 1 for hunters
            loss = F.mse_loss(change_scores, target_scores)
        else:
            # Use all pairs for ranking loss
            num_pairs = min(len(hunter_scores), len(legit_scores))
            hunter_subset = hunter_scores[:num_pairs]
            legit_subset = legit_scores[:num_pairs]
            
            loss = F.margin_ranking_loss(
                hunter_subset,  # Hunter scores (should be high)
                legit_subset,   # Legitimate scores (should be low)
                torch.ones(num_pairs, device=change_scores.device),
                margin=self.margin,
                reduction='mean'
            )
        
        # Weight by confidence if available
        if confidence is not None:
            # Higher confidence predictions should contribute more to loss
            loss = loss * confidence.mean()
        
        return self.change_weight * loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance in airdrop hunter detection.
    
    Focuses learning on hard-to-classify examples and reduces the impact
    of easy examples. Particularly useful when hunters are rare.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model logits of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size,)
            
        Returns:
            focal_loss: Focal loss value
        """
        # Compute cross-entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Compute focal weight
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ConfidenceRegularizationLoss(nn.Module):
    """
    Confidence regularization loss to encourage well-calibrated predictions.
    
    Penalizes overconfident predictions on difficult examples and encourages
    the model to express uncertainty when appropriate.
    """
    
    def __init__(self, regularization_weight: float = 0.1):
        super().__init__()
        
        self.regularization_weight = regularization_weight
        
    def forward(self, 
                confidence: torch.Tensor,
                probabilities: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            confidence: Model confidence scores of shape (batch_size,)
            probabilities: Class probabilities of shape (batch_size, num_classes)
            labels: Ground truth labels of shape (batch_size,)
            
        Returns:
            regularization_loss: Confidence regularization loss
        """
        # Get predicted probabilities for true class
        true_class_probs = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        # Confidence should be calibrated with actual correctness
        # High confidence should correspond to high true class probability
        calibration_loss = F.mse_loss(confidence, true_class_probs)
        
        # Encourage moderate confidence (avoid overconfidence)
        overconfidence_penalty = torch.relu(confidence - 0.9).mean()
        
        total_loss = calibration_loss + 0.1 * overconfidence_penalty
        
        return self.regularization_weight * total_loss


class TemporalGraphLoss(nn.Module):
    """
    Combined multi-task loss for Temporal Graph Transformer training.
    
    Integrates multiple loss components to encourage learning of robust
    behavioral representations beyond simple classification.
    """
    
    def __init__(self, 
                 classification_weight: float = 1.0,
                 contrastive_weight: float = 0.3,
                 temporal_consistency_weight: float = 0.1,
                 behavioral_change_weight: float = 0.2,
                 confidence_regularization_weight: float = 0.05,
                 use_focal_loss: bool = True,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super().__init__()
        
        # Loss weights
        self.classification_weight = classification_weight
        self.contrastive_weight = contrastive_weight
        self.temporal_consistency_weight = temporal_consistency_weight
        self.behavioral_change_weight = behavioral_change_weight
        self.confidence_regularization_weight = confidence_regularization_weight
        
        # Loss components
        if use_focal_loss:
            self.classification_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.classification_loss = nn.CrossEntropyLoss()
            
        self.contrastive_loss = InfoNCE(temperature=0.1)
        self.temporal_consistency_loss = TemporalConsistencyLoss()
        self.behavioral_change_loss = BehavioralChangeLoss()
        self.confidence_regularization_loss = ConfidenceRegularizationLoss()
        
    def forward(self, 
                outputs: Dict[str, torch.Tensor],
                batch: Dict[str, torch.Tensor],
                labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Model outputs dictionary containing:
                - 'logits': Classification logits
                - 'probabilities': Class probabilities
                - 'confidence': Prediction confidence
                - 'behavioral_scores': Behavioral change scores
                - 'fused_representation': Final embeddings
                - 'intermediate': Intermediate representations
            batch: Input batch dictionary
            labels: Ground truth labels
            
        Returns:
            Dictionary containing total loss and individual loss components
        """
        device = labels.device
        loss_dict = {}
        
        # 1. Classification loss
        cls_loss = self.classification_loss(outputs['logits'], labels)
        loss_dict['classification'] = cls_loss
        
        # 2. Contrastive loss (if we have embeddings)
        if 'fused_representation' in outputs:
            contrastive_loss = self.contrastive_loss(
                outputs['fused_representation'], 
                labels
            )
            loss_dict['contrastive'] = contrastive_loss
        else:
            loss_dict['contrastive'] = torch.tensor(0.0, device=device)
        
        # 3. Temporal consistency loss (if we have temporal sequences)
        if 'intermediate' in outputs and 'temporal_sequences' in outputs['intermediate']:
            temporal_loss = self.temporal_consistency_loss(
                outputs['intermediate']['temporal_sequences'],
                batch['timestamps'],
                batch.get('airdrop_events', None),
                labels
            )
            loss_dict['temporal_consistency'] = temporal_loss
        else:
            loss_dict['temporal_consistency'] = torch.tensor(0.0, device=device)
        
        # 4. Behavioral change loss (if we have change scores)
        if 'behavioral_scores' in outputs:
            change_loss = self.behavioral_change_loss(
                outputs['behavioral_scores'][:, 0],  # First behavioral score is change score
                labels,
                outputs.get('confidence', None)
            )
            loss_dict['behavioral_change'] = change_loss
        else:
            loss_dict['behavioral_change'] = torch.tensor(0.0, device=device)
        
        # 5. Confidence regularization loss
        if 'confidence' in outputs:
            conf_reg_loss = self.confidence_regularization_loss(
                outputs['confidence'],
                outputs['probabilities'],
                labels
            )
            loss_dict['confidence_regularization'] = conf_reg_loss
        else:
            loss_dict['confidence_regularization'] = torch.tensor(0.0, device=device)
        
        # Compute total weighted loss
        total_loss = (
            self.classification_weight * loss_dict['classification'] +
            self.contrastive_weight * loss_dict['contrastive'] +
            self.temporal_consistency_weight * loss_dict['temporal_consistency'] +
            self.behavioral_change_weight * loss_dict['behavioral_change'] +
            self.confidence_regularization_weight * loss_dict['confidence_regularization']
        )
        
        loss_dict['total'] = total_loss
        
        return loss_dict
    
    def update_weights(self, epoch: int, total_epochs: int) -> None:
        """
        Update loss weights during training (curriculum learning).
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of training epochs
        """
        # Gradually increase contrastive learning weight
        progress = epoch / total_epochs
        self.contrastive_weight = 0.1 + 0.2 * progress
        
        # Reduce temporal consistency weight as model learns
        self.temporal_consistency_weight = 0.2 * (1 - progress)
        
    def get_current_weights(self) -> Dict[str, float]:
        """Get current loss weights for logging."""
        return {
            'classification_weight': self.classification_weight,
            'contrastive_weight': self.contrastive_weight,
            'temporal_consistency_weight': self.temporal_consistency_weight,
            'behavioral_change_weight': self.behavioral_change_weight,
            'confidence_regularization_weight': self.confidence_regularization_weight
        }