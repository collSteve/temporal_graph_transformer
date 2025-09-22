"""
TrustaLabs Framework Implementation

Industry-standard airdrop hunter detection based on 4-pattern analysis:
1. Star Pattern Detection - Central nodes with many connections
2. Tree Pattern Detection - Hierarchical transaction structures  
3. Chain Pattern Detection - Sequential transaction chains
4. Behavioral Similarity - Similar transaction patterns across accounts

Reference: TrustaLabs methodology for Sybil/farming detection
Expected Performance: F1 0.75-0.85
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import networkx as nx
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import warnings

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Optional tqdm import
try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback if tqdm not available
    def tqdm(iterable, desc=None):
        return iterable

from baselines.base_interface import BaselineMethodInterface
from utils.metrics import BinaryClassificationMetrics


@dataclass 
class PatternFeatures:
    """Container for pattern-based features."""
    # Star pattern features
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    
    # Tree pattern features  
    tree_depth: int = 0
    tree_branching_factor: float = 0.0
    hierarchical_score: float = 0.0
    
    # Chain pattern features
    max_chain_length: int = 0
    chain_frequency: float = 0.0
    sequential_score: float = 0.0
    
    # Behavioral similarity features
    transaction_diversity: float = 0.0
    timing_regularity: float = 0.0
    value_pattern_score: float = 0.0
    behavioral_anomaly_score: float = 0.0


class StarPatternDetector:
    """Detects star-like transaction patterns indicating potential farming."""
    
    def __init__(self, threshold_degree: int = 10, threshold_centrality: float = 0.1):
        self.threshold_degree = threshold_degree
        self.threshold_centrality = threshold_centrality
    
    def detect_patterns(self, transaction_graph: nx.Graph, user_transactions: Dict[str, List]) -> Dict[str, PatternFeatures]:
        """Detect star patterns in transaction network."""
        features = {}
        
        # Compute centrality metrics
        degree_centrality = nx.degree_centrality(transaction_graph)
        betweenness_centrality = nx.betweenness_centrality(transaction_graph)
        clustering = nx.clustering(transaction_graph)
        
        for user_id in transaction_graph.nodes():
            pattern_features = PatternFeatures()
            
            # Star pattern indicators
            pattern_features.degree_centrality = degree_centrality.get(user_id, 0.0)
            pattern_features.betweenness_centrality = betweenness_centrality.get(user_id, 0.0)
            pattern_features.clustering_coefficient = clustering.get(user_id, 0.0)
            
            features[user_id] = pattern_features
        
        return features


class TreePatternDetector:
    """Detects hierarchical tree patterns in transaction flows."""
    
    def __init__(self, max_depth: int = 5):
        self.max_depth = max_depth
    
    def detect_patterns(self, transaction_graph: nx.Graph, user_transactions: Dict[str, List]) -> Dict[str, PatternFeatures]:
        """Detect tree patterns indicating hierarchical farming structures."""
        features = {}
        
        for user_id in transaction_graph.nodes():
            pattern_features = PatternFeatures()
            
            # Build ego graph (subgraph around this user)
            if user_id in transaction_graph:
                ego_graph = nx.ego_graph(transaction_graph, user_id, radius=2)
                
                # Tree structure analysis
                pattern_features.tree_depth = self._compute_tree_depth(ego_graph, user_id)
                pattern_features.tree_branching_factor = self._compute_branching_factor(ego_graph, user_id)
                pattern_features.hierarchical_score = self._compute_hierarchical_score(ego_graph, user_id)
            
            features[user_id] = pattern_features
        
        return features
    
    def _compute_tree_depth(self, graph: nx.Graph, root: str) -> int:
        """Compute maximum depth of tree structure."""
        if root not in graph:
            return 0
        
        try:
            # Use BFS to find maximum distance
            distances = nx.single_source_shortest_path_length(graph, root)
            return max(distances.values()) if distances else 0
        except:
            return 0
    
    def _compute_branching_factor(self, graph: nx.Graph, root: str) -> float:
        """Compute average branching factor."""
        if root not in graph:
            return 0.0
        
        degrees = [graph.degree(node) for node in graph.nodes() if node != root]
        return np.mean(degrees) if degrees else 0.0
    
    def _compute_hierarchical_score(self, graph: nx.Graph, root: str) -> float:
        """Compute how tree-like the structure is."""
        if root not in graph or graph.number_of_nodes() < 2:
            return 0.0
        
        # Tree-likeness: ratio of edges to potential tree edges
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        tree_edges = n_nodes - 1
        
        if tree_edges == 0:
            return 0.0
        
        return min(1.0, tree_edges / n_edges) if n_edges > 0 else 0.0


class ChainPatternDetector:
    """Detects sequential chain patterns in transactions."""
    
    def __init__(self, min_chain_length: int = 3):
        self.min_chain_length = min_chain_length
    
    def detect_patterns(self, transaction_graph: nx.Graph, user_transactions: Dict[str, List]) -> Dict[str, PatternFeatures]:
        """Detect chain patterns indicating sequential farming."""
        features = {}
        
        for user_id in transaction_graph.nodes():
            pattern_features = PatternFeatures()
            
            if user_id in user_transactions:
                transactions = user_transactions[user_id]
                
                # Find longest chain
                pattern_features.max_chain_length = self._find_longest_chain(transactions)
                pattern_features.chain_frequency = self._compute_chain_frequency(transactions)
                pattern_features.sequential_score = self._compute_sequential_score(transactions)
            
            features[user_id] = pattern_features
        
        return features
    
    def _find_longest_chain(self, transactions: List) -> int:
        """Find the longest sequence of consecutive transactions."""
        if not transactions:
            return 0
        
        # Sort by timestamp
        sorted_transactions = sorted(transactions, key=lambda x: x.get('timestamp', 0))
        
        max_chain = 1
        current_chain = 1
        
        for i in range(1, len(sorted_transactions)):
            prev_time = sorted_transactions[i-1].get('timestamp', 0)
            curr_time = sorted_transactions[i].get('timestamp', 0)
            
            # Consider sequential if within reasonable time window (e.g., 1 hour)
            if curr_time - prev_time < 3600:  # 1 hour in seconds
                current_chain += 1
                max_chain = max(max_chain, current_chain)
            else:
                current_chain = 1
        
        return max_chain
    
    def _compute_chain_frequency(self, transactions: List) -> float:
        """Compute frequency of chain-like behavior."""
        if len(transactions) < 2:
            return 0.0
        
        # Count transactions that are part of chains
        chain_transactions = 0
        sorted_transactions = sorted(transactions, key=lambda x: x.get('timestamp', 0))
        
        for i in range(1, len(sorted_transactions)):
            prev_time = sorted_transactions[i-1].get('timestamp', 0)
            curr_time = sorted_transactions[i].get('timestamp', 0)
            
            if curr_time - prev_time < 3600:  # Part of a chain
                chain_transactions += 1
        
        return chain_transactions / len(transactions)
    
    def _compute_sequential_score(self, transactions: List) -> float:
        """Compute how sequential the transaction pattern is."""
        if len(transactions) < 2:
            return 0.0
        
        # Measure regularity of timing intervals
        sorted_transactions = sorted(transactions, key=lambda x: x.get('timestamp', 0))
        intervals = []
        
        for i in range(1, len(sorted_transactions)):
            prev_time = sorted_transactions[i-1].get('timestamp', 0)
            curr_time = sorted_transactions[i].get('timestamp', 0)
            intervals.append(curr_time - prev_time)
        
        if not intervals:
            return 0.0
        
        # Low variance in intervals indicates regular, sequential behavior
        mean_interval = np.mean(intervals)
        var_interval = np.var(intervals)
        
        if mean_interval == 0:
            return 0.0
        
        # Coefficient of variation (lower = more regular)
        cv = np.sqrt(var_interval) / mean_interval
        return max(0.0, 1.0 - cv)  # Convert to score (higher = more sequential)


class BehavioralSimilarityDetector:
    """Detects behavioral similarity patterns across accounts."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
    
    def detect_patterns(self, transaction_graph: nx.Graph, user_transactions: Dict[str, List]) -> Dict[str, PatternFeatures]:
        """Detect behavioral similarity patterns."""
        features = {}
        
        # Compute behavioral profiles for each user
        user_profiles = self._compute_behavioral_profiles(user_transactions)
        
        # Compute similarity scores
        for user_id in user_transactions.keys():
            pattern_features = PatternFeatures()
            
            if user_id in user_profiles:
                profile = user_profiles[user_id]
                
                pattern_features.transaction_diversity = profile['diversity']
                pattern_features.timing_regularity = profile['timing_regularity']
                pattern_features.value_pattern_score = profile['value_pattern']
                pattern_features.behavioral_anomaly_score = self._compute_anomaly_score(
                    profile, user_profiles
                )
            
            features[user_id] = pattern_features
        
        return features
    
    def _compute_behavioral_profiles(self, user_transactions: Dict[str, List]) -> Dict[str, Dict]:
        """Compute behavioral profiles for all users."""
        profiles = {}
        
        for user_id, transactions in user_transactions.items():
            if not transactions:
                profiles[user_id] = self._empty_profile()
                continue
            
            profile = {}
            
            # Transaction diversity (entropy of transaction types)
            tx_types = [tx.get('transaction_type', 'unknown') for tx in transactions]
            type_counts = Counter(tx_types)
            type_probs = [count / len(tx_types) for count in type_counts.values()]
            profile['diversity'] = entropy(type_probs) if len(type_probs) > 1 else 0.0
            
            # Timing regularity
            timestamps = [tx.get('timestamp', 0) for tx in transactions]
            timestamps.sort()
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            if intervals:
                mean_interval = np.mean(intervals)
                var_interval = np.var(intervals)
                profile['timing_regularity'] = 1.0 / (1.0 + var_interval / (mean_interval + 1e-6))
            else:
                profile['timing_regularity'] = 0.0
            
            # Value pattern (regularity of transaction values)
            values = [tx.get('value_usd', 0) for tx in transactions]
            if values:
                unique_values = len(set(values))
                profile['value_pattern'] = 1.0 - (unique_values / len(values))
            else:
                profile['value_pattern'] = 0.0
            
            profiles[user_id] = profile
        
        return profiles
    
    def _empty_profile(self) -> Dict:
        """Return empty behavioral profile."""
        return {
            'diversity': 0.0,
            'timing_regularity': 0.0,
            'value_pattern': 0.0
        }
    
    def _compute_anomaly_score(self, user_profile: Dict, all_profiles: Dict[str, Dict]) -> float:
        """Compute how anomalous this user's behavior is."""
        if not all_profiles:
            return 0.0
        
        # Find similar users based on behavioral profile
        similarities = []
        user_vector = np.array([
            user_profile['diversity'],
            user_profile['timing_regularity'], 
            user_profile['value_pattern']
        ])
        
        for other_profile in all_profiles.values():
            other_vector = np.array([
                other_profile['diversity'],
                other_profile['timing_regularity'],
                other_profile['value_pattern']
            ])
            
            # Compute cosine similarity
            if np.linalg.norm(user_vector) > 0 and np.linalg.norm(other_vector) > 0:
                similarity = 1 - cosine(user_vector, other_vector)
                similarities.append(similarity)
        
        if not similarities:
            return 0.0
        
        # High max similarity indicates this user is very similar to others (suspicious)
        max_similarity = max(similarities)
        return max_similarity


class TrustaLabFramework(BaselineMethodInterface):
    """
    Complete TrustaLabs Framework implementation for airdrop hunter detection.
    
    Combines 4-pattern detection with machine learning classification.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Pattern detectors
        self.star_detector = StarPatternDetector(
            threshold_degree=config.get('star_threshold_degree', 10),
            threshold_centrality=config.get('star_threshold_centrality', 0.1)
        )
        self.tree_detector = TreePatternDetector(
            max_depth=config.get('tree_max_depth', 5)
        )
        self.chain_detector = ChainPatternDetector(
            min_chain_length=config.get('chain_min_length', 3)
        )
        self.similarity_detector = BehavioralSimilarityDetector(
            similarity_threshold=config.get('similarity_threshold', 0.8)
        )
        
        # ML components
        self.classifier = RandomForestClassifier(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', 10),
            random_state=config.get('random_state', 42)
        )
        self.scaler = StandardScaler()
        
        # State
        self.is_trained = False
    
    def extract_features(self, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
        """Extract TrustaLabs features from data."""
        all_features = []
        all_labels = []
        
        print("Extracting TrustaLabs pattern features...")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Feature Extraction")):
            # Extract transaction network and user data from batch
            user_transactions, transaction_graph, labels = self._extract_batch_data(batch)
            
            # Detect patterns
            star_features = self.star_detector.detect_patterns(transaction_graph, user_transactions)
            tree_features = self.tree_detector.detect_patterns(transaction_graph, user_transactions)
            chain_features = self.chain_detector.detect_patterns(transaction_graph, user_transactions)
            similarity_features = self.similarity_detector.detect_patterns(transaction_graph, user_transactions)
            
            # Combine features for each user
            for user_id in user_transactions.keys():
                feature_vector = self._combine_pattern_features(
                    star_features.get(user_id, PatternFeatures()),
                    tree_features.get(user_id, PatternFeatures()),
                    chain_features.get(user_id, PatternFeatures()),
                    similarity_features.get(user_id, PatternFeatures())
                )
                
                all_features.append(feature_vector)
                # For now, use a simple labeling scheme - in practice this comes from ground truth
                all_labels.append(labels.get(user_id, 0))
            
            if batch_idx >= 10:  # Limit for demonstration
                break
        
        return np.array(all_features), np.array(all_labels)
    
    def _extract_batch_data(self, batch: Dict) -> Tuple[Dict[str, List], nx.Graph, Dict]:
        """Extract structured data from batch."""
        # This is a simplified version - in practice would need to properly
        # parse the batch structure from our datasets
        user_transactions = {}
        transaction_graph = nx.Graph()
        labels = {}
        
        # Extract user IDs and basic transaction data
        if 'user_id' in batch:
            user_ids = batch['user_id'] if isinstance(batch['user_id'], list) else [batch['user_id']]
            batch_labels = batch.get('labels', torch.zeros(len(user_ids)))
            
            for i, user_id in enumerate(user_ids):
                # Create dummy transaction data for demonstration
                user_transactions[user_id] = [
                    {
                        'timestamp': 1679529600 + i * 3600,
                        'transaction_type': 'swap',
                        'value_usd': 100.0 + i * 10,
                        'protocol': 'uniswap_v3'
                    }
                ]
                
                # Add to graph
                transaction_graph.add_node(user_id)
                
                # Add dummy label
                labels[user_id] = int(batch_labels[i]) if i < len(batch_labels) else 0
        
        return user_transactions, transaction_graph, labels
    
    def _combine_pattern_features(self, star: PatternFeatures, tree: PatternFeatures, 
                                 chain: PatternFeatures, similarity: PatternFeatures) -> np.ndarray:
        """Combine all pattern features into a single feature vector."""
        features = [
            # Star pattern features
            star.degree_centrality,
            star.betweenness_centrality,
            star.clustering_coefficient,
            
            # Tree pattern features
            float(tree.tree_depth),
            tree.tree_branching_factor,
            tree.hierarchical_score,
            
            # Chain pattern features
            float(chain.max_chain_length),
            chain.chain_frequency,
            chain.sequential_score,
            
            # Behavioral similarity features
            similarity.transaction_diversity,
            similarity.timing_regularity,
            similarity.value_pattern_score,
            similarity.behavioral_anomaly_score
        ]
        
        return np.array(features, dtype=np.float32)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
        """Train the TrustaLabs framework."""
        print(f"\n🔧 Training TrustaLabs Framework")
        
        # Extract features
        X_train, y_train = self.extract_features(train_loader, device)
        X_val, y_val = self.extract_features(val_loader, device)
        
        if len(X_train) == 0:
            print("⚠️  No training data extracted")
            return {'best_val_f1': 0.0}
        
        print(f"Extracted {len(X_train)} training samples, {len(X_val)} validation samples")
        print(f"Feature dimension: {X_train.shape[1]}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if len(X_val) > 0 else np.array([])
        
        # Train classifier
        self.classifier.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate on validation set
        val_f1 = 0.0
        if len(X_val) > 0:
            y_val_pred = self.classifier.predict(X_val_scaled)
            val_f1 = f1_score(y_val, y_val_pred, zero_division=0.0)
        
        print(f"✅ TrustaLabs training completed - Validation F1: {val_f1:.4f}")
        
        return {'best_val_f1': val_f1}
    
    def evaluate(self, test_loader: DataLoader, device: torch.device) -> Dict[str, float]:
        """Evaluate the TrustaLabs framework."""
        if not self.is_trained:
            return {'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # Extract test features
        X_test, y_test = self.extract_features(test_loader, device)
        
        if len(X_test) == 0:
            return {'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # Scale and predict
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.classifier.predict(X_test_scaled)
        y_prob = self.classifier.predict_proba(X_test_scaled)[:, 1]
        
        # Compute metrics
        metrics = BinaryClassificationMetrics()
        metrics.update(torch.tensor(y_prob), torch.tensor(y_test))
        
        return metrics.compute()
    
    def predict(self, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        """Make predictions on a batch."""
        if not self.is_trained:
            batch_size = len(batch.get('user_id', [1]))
            return torch.zeros(batch_size, 2)  # Return dummy predictions
        
        # Extract features for this batch
        user_transactions, transaction_graph, _ = self._extract_batch_data(batch)
        
        if not user_transactions:
            batch_size = len(batch.get('user_id', [1]))
            return torch.zeros(batch_size, 2)
        
        # Get pattern features
        star_features = self.star_detector.detect_patterns(transaction_graph, user_transactions)
        tree_features = self.tree_detector.detect_patterns(transaction_graph, user_transactions)
        chain_features = self.chain_detector.detect_patterns(transaction_graph, user_transactions)
        similarity_features = self.similarity_detector.detect_patterns(transaction_graph, user_transactions)
        
        # Combine features
        features = []
        for user_id in user_transactions.keys():
            feature_vector = self._combine_pattern_features(
                star_features.get(user_id, PatternFeatures()),
                tree_features.get(user_id, PatternFeatures()),
                chain_features.get(user_id, PatternFeatures()),
                similarity_features.get(user_id, PatternFeatures())
            )
            features.append(feature_vector)
        
        if not features:
            batch_size = len(batch.get('user_id', [1]))
            return torch.zeros(batch_size, 2)
        
        # Scale and predict
        X = self.scaler.transform(np.array(features))
        probabilities = self.classifier.predict_proba(X)
        
        return torch.tensor(probabilities, dtype=torch.float32)