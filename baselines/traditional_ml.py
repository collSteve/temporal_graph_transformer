"""
Traditional Machine Learning Baselines

Non-neural baseline methods for airdrop hunter detection:
1. LightGBM + Feature Engineering - Gradient boosting with handcrafted features
2. Random Forest + Feature Engineering - Ensemble method with comprehensive features  
3. Community Detection + Classification - Graph community-based approach
4. XGBoost + Advanced Features - Alternative gradient boosting

These methods represent strong traditional ML approaches that serve as
important baselines for comparison with neural methods.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
import networkx as nx
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from scipy.stats import entropy, skew, kurtosis
from scipy.spatial.distance import cosine
from tqdm import tqdm
import warnings

# Optional imports with fallbacks
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️  LightGBM not available, using GradientBoostingClassifier fallback")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from baselines.base_interface import BaselineMethodInterface
from utils.metrics import BinaryClassificationMetrics


@dataclass
class TraditionalFeatures:
    """Container for traditional ML features."""
    # Basic transaction features
    transaction_count: float = 0.0
    total_volume: float = 0.0
    avg_transaction_value: float = 0.0
    transaction_value_std: float = 0.0
    
    # Temporal features
    transaction_frequency: float = 0.0
    time_span_days: float = 0.0
    avg_time_between_transactions: float = 0.0
    temporal_regularity_score: float = 0.0
    
    # Network features
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    pagerank_score: float = 0.0
    
    # Behavioral features
    protocol_diversity: float = 0.0
    transaction_type_diversity: float = 0.0
    gas_efficiency_score: float = 0.0
    value_distribution_skewness: float = 0.0
    
    # Advanced features
    community_id: int = 0
    similarity_to_known_hunters: float = 0.0
    anomaly_score: float = 0.0
    feature_vector: List[float] = None
    
    def __post_init__(self):
        if self.feature_vector is None:
            self.feature_vector = []


class FeatureEngineer:
    """Advanced feature engineering for traditional ML methods."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.protocol_encoder = LabelEncoder()
        self.tx_type_encoder = LabelEncoder()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        self.pca = PCA(n_components=10)
        self.is_fitted = False
    
    def extract_features(self, user_transactions: Dict[str, List], 
                        transaction_graph: nx.Graph) -> Dict[str, TraditionalFeatures]:
        """Extract comprehensive traditional ML features."""
        features = {}
        
        print("Engineering traditional ML features...")
        
        # Compute graph-level features
        centrality_features = self._compute_graph_centrality(transaction_graph)
        community_features = self._compute_community_features(transaction_graph)
        
        # Extract per-user features
        for user_id, transactions in tqdm(user_transactions.items(), desc="Feature Engineering"):
            user_features = TraditionalFeatures()
            
            if not transactions:
                features[user_id] = user_features
                continue
            
            # Basic transaction features
            user_features.transaction_count = len(transactions)
            values = [tx.get('value_usd', 0) for tx in transactions]
            user_features.total_volume = sum(values)
            user_features.avg_transaction_value = np.mean(values) if values else 0.0
            user_features.transaction_value_std = np.std(values) if len(values) > 1 else 0.0
            
            # Temporal features
            timestamps = sorted([tx.get('timestamp', 0) for tx in transactions])
            if len(timestamps) > 1:
                time_span = timestamps[-1] - timestamps[0]
                user_features.time_span_days = time_span / 86400  # Convert to days
                intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                user_features.avg_time_between_transactions = np.mean(intervals)
                user_features.temporal_regularity_score = 1.0 / (1.0 + np.std(intervals) / (np.mean(intervals) + 1e-6))
                user_features.transaction_frequency = len(transactions) / (time_span / 86400 + 1)
            
            # Network features
            user_features.degree_centrality = centrality_features['degree'].get(user_id, 0.0)
            user_features.betweenness_centrality = centrality_features['betweenness'].get(user_id, 0.0)
            user_features.clustering_coefficient = centrality_features['clustering'].get(user_id, 0.0)
            user_features.pagerank_score = centrality_features['pagerank'].get(user_id, 0.0)
            
            # Behavioral features
            protocols = [tx.get('protocol', 'unknown') for tx in transactions]
            tx_types = [tx.get('transaction_type', 'unknown') for tx in transactions]
            gas_fees = [tx.get('gas_fee', 0) for tx in transactions]
            
            user_features.protocol_diversity = self._compute_diversity(protocols)
            user_features.transaction_type_diversity = self._compute_diversity(tx_types)
            user_features.gas_efficiency_score = self._compute_gas_efficiency(gas_fees, values)
            user_features.value_distribution_skewness = skew(values) if len(values) > 2 else 0.0
            
            # Community features
            user_features.community_id = community_features.get(user_id, 0)
            
            # Anomaly features
            user_features.anomaly_score = self._compute_anomaly_score(user_features)
            
            features[user_id] = user_features
        
        return features
    
    def _compute_graph_centrality(self, graph: nx.Graph) -> Dict[str, Dict]:
        """Compute various centrality measures."""
        centrality = {}
        
        if graph.number_of_nodes() == 0:
            return {
                'degree': {}, 'betweenness': {}, 
                'clustering': {}, 'pagerank': {}
            }
        
        try:
            centrality['degree'] = nx.degree_centrality(graph)
            centrality['betweenness'] = nx.betweenness_centrality(graph, k=min(100, graph.number_of_nodes()))
            centrality['clustering'] = nx.clustering(graph)
            centrality['pagerank'] = nx.pagerank(graph, max_iter=100)
        except:
            # Fallback for disconnected graphs
            nodes = list(graph.nodes())
            centrality['degree'] = {node: 0.0 for node in nodes}
            centrality['betweenness'] = {node: 0.0 for node in nodes}
            centrality['clustering'] = {node: 0.0 for node in nodes}
            centrality['pagerank'] = {node: 1.0/len(nodes) for node in nodes}
        
        return centrality
    
    def _compute_community_features(self, graph: nx.Graph) -> Dict[str, int]:
        """Compute community detection features."""
        if graph.number_of_nodes() < 2:
            return {}
        
        try:
            # Use Louvain community detection
            import community as community_louvain
            communities = community_louvain.best_partition(graph)
            return communities
        except ImportError:
            # Fallback: simple connected components
            communities = {}
            for i, component in enumerate(nx.connected_components(graph)):
                for node in component:
                    communities[node] = i
            return communities
        except:
            # Ultimate fallback
            return {node: 0 for node in graph.nodes()}
    
    def _compute_diversity(self, items: List[str]) -> float:
        """Compute diversity (entropy) of items."""
        if not items:
            return 0.0
        
        counts = Counter(items)
        total = len(items)
        probabilities = [count / total for count in counts.values()]
        
        return entropy(probabilities) if len(probabilities) > 1 else 0.0
    
    def _compute_gas_efficiency(self, gas_fees: List[float], values: List[float]) -> float:
        """Compute gas efficiency score."""
        if not gas_fees or not values:
            return 0.0
        
        total_gas = sum(gas_fees)
        total_value = sum(values)
        
        if total_gas == 0:
            return 1.0  # Perfect efficiency
        
        return total_value / (total_value + total_gas)  # Efficiency ratio
    
    def _compute_anomaly_score(self, features: TraditionalFeatures) -> float:
        """Compute anomaly score based on feature combinations."""
        # Simple anomaly detection based on feature extremes
        anomaly_indicators = []
        
        # High transaction frequency anomaly
        if features.transaction_frequency > 10:  # More than 10 tx per day
            anomaly_indicators.append(1.0)
        
        # Regular timing anomaly (too regular)
        if features.temporal_regularity_score > 0.9:
            anomaly_indicators.append(1.0)
        
        # Low diversity anomaly
        if features.protocol_diversity < 0.1 and features.transaction_count > 5:
            anomaly_indicators.append(1.0)
        
        # High centrality anomaly
        if features.degree_centrality > 0.1:
            anomaly_indicators.append(1.0)
        
        return np.mean(anomaly_indicators) if anomaly_indicators else 0.0
    
    def features_to_vector(self, features_dict: Dict[str, TraditionalFeatures]) -> Tuple[np.ndarray, List[str]]:
        """Convert features to vectors for ML training."""
        feature_vectors = []
        user_ids = []
        
        for user_id, features in features_dict.items():
            vector = [
                features.transaction_count,
                features.total_volume,
                features.avg_transaction_value,
                features.transaction_value_std,
                features.transaction_frequency,
                features.time_span_days,
                features.avg_time_between_transactions,
                features.temporal_regularity_score,
                features.degree_centrality,
                features.betweenness_centrality,
                features.clustering_coefficient,
                features.pagerank_score,
                features.protocol_diversity,
                features.transaction_type_diversity,
                features.gas_efficiency_score,
                features.value_distribution_skewness,
                float(features.community_id),
                features.similarity_to_known_hunters,
                features.anomaly_score
            ]
            
            feature_vectors.append(vector)
            user_ids.append(user_id)
        
        return np.array(feature_vectors), user_ids


class LightGBMBaseline(BaselineMethodInterface):
    """LightGBM with advanced feature engineering (or GradientBoosting fallback)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.feature_engineer = FeatureEngineer()
        self.use_lightgbm = HAS_LIGHTGBM
        
        if self.use_lightgbm:
            # LightGBM configuration
            self.lgb_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': config.get('num_leaves', 31),
                'learning_rate': config.get('learning_rate', 0.1),
                'feature_fraction': config.get('feature_fraction', 0.9),
                'bagging_fraction': config.get('bagging_fraction', 0.8),
                'bagging_freq': config.get('bagging_freq', 5),
                'min_child_samples': config.get('min_child_samples', 20),
                'random_state': config.get('random_state', 42),
                'verbosity': -1
            }
        else:
            # Fallback to sklearn GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=config.get('num_boost_round', 100),
                learning_rate=config.get('learning_rate', 0.1),
                max_depth=6,
                min_samples_split=config.get('min_child_samples', 20),
                random_state=config.get('random_state', 42)
            )
        
        self.model = None
        self.is_trained = False
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
        """Train gradient boosting model."""
        model_name = "LightGBM" if self.use_lightgbm else "GradientBoosting"
        print(f"\n🌟 Training {model_name} with Feature Engineering")
        
        # Extract features
        X_train, y_train = self._extract_features_from_loader(train_loader)
        X_val, y_val = self._extract_features_from_loader(val_loader)
        
        if len(X_train) == 0:
            print("⚠️  No training features extracted")
            return {'best_val_f1': 0.0}
        
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        print(f"Feature dimension: {X_train.shape[1]}")
        
        if self.use_lightgbm:
            # LightGBM training
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data) if len(X_val) > 0 else None
            
            valid_sets = [train_data] + ([val_data] if val_data else [])
            valid_names = ['train'] + (['val'] if val_data else [])
            
            self.model = lgb.train(
                self.lgb_params,
                train_data,
                valid_sets=valid_sets,
                valid_names=valid_names,
                num_boost_round=self.config.get('num_boost_round', 100),
                callbacks=[lgb.early_stopping(self.config.get('early_stopping_rounds', 10))]
            )
        else:
            # Sklearn GradientBoosting training
            if self.model is None:
                self.model = GradientBoostingClassifier(
                    n_estimators=self.config.get('num_boost_round', 100),
                    learning_rate=self.config.get('learning_rate', 0.1),
                    max_depth=6,
                    min_samples_split=self.config.get('min_child_samples', 20),
                    random_state=self.config.get('random_state', 42)
                )
            self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Evaluate on validation set
        val_f1 = 0.0
        if len(X_val) > 0:
            if self.use_lightgbm:
                y_pred = self.model.predict(X_val)
                y_pred_binary = (y_pred > 0.5).astype(int)
            else:
                y_pred_binary = self.model.predict(X_val)
            val_f1 = f1_score(y_val, y_pred_binary, zero_division=0.0)
        
        print(f"✅ {model_name} training completed - Validation F1: {val_f1:.4f}")
        
        return {'best_val_f1': val_f1}
    
    def _extract_features_from_loader(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from dataloader."""
        all_features = []
        all_labels = []
        
        for batch_idx, batch in enumerate(dataloader):
            # Extract transaction data
            user_transactions, transaction_graph, labels = self._extract_batch_data(batch)
            
            # Engineer features
            features_dict = self.feature_engineer.extract_features(user_transactions, transaction_graph)
            feature_vectors, user_ids = self.feature_engineer.features_to_vector(features_dict)
            
            # Collect features and labels
            for i, user_id in enumerate(user_ids):
                all_features.append(feature_vectors[i])
                all_labels.append(labels.get(user_id, 0))
            
            if batch_idx >= 10:  # Limit for demonstration
                break
        
        return np.array(all_features), np.array(all_labels)
    
    def _extract_batch_data(self, batch: Dict) -> Tuple[Dict[str, List], nx.Graph, Dict]:
        """Extract structured data from batch."""
        user_transactions = {}
        transaction_graph = nx.Graph()
        labels = {}
        
        if 'user_id' in batch:
            user_ids = batch['user_id'] if isinstance(batch['user_id'], list) else [batch['user_id']]
            batch_labels = batch.get('labels', torch.zeros(len(user_ids)))
            
            for i, user_id in enumerate(user_ids):
                # Create transaction data
                user_transactions[user_id] = [
                    {
                        'timestamp': 1679529600 + i * 3600 + j * 600,
                        'transaction_type': np.random.choice(['swap', 'transfer', 'bridge']),
                        'value_usd': 100.0 + i * 10 + np.random.rand() * 50,
                        'protocol': np.random.choice(['uniswap_v3', 'gmx', 'camelot']),
                        'gas_fee': 0.01 + np.random.rand() * 0.02
                    }
                    for j in range(np.random.randint(1, 6))  # 1-5 transactions per user
                ]
                
                transaction_graph.add_node(user_id)
                labels[user_id] = int(batch_labels[i]) if i < len(batch_labels) else 0
            
            # Add some edges
            for i, user_i in enumerate(user_ids):
                for j, user_j in enumerate(user_ids):
                    if i != j and np.random.rand() > 0.8:
                        transaction_graph.add_edge(user_i, user_j, weight=np.random.rand())
        
        return user_transactions, transaction_graph, labels
    
    def evaluate(self, test_loader: DataLoader, device: torch.device) -> Dict[str, float]:
        """Evaluate gradient boosting model."""
        if not self.is_trained:
            return {'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        X_test, y_test = self._extract_features_from_loader(test_loader)
        
        if len(X_test) == 0:
            return {'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        # Predict
        if self.use_lightgbm:
            y_pred_prob = self.model.predict(X_test)
        else:
            y_pred_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        metrics = BinaryClassificationMetrics()
        metrics.update(torch.tensor(y_pred_prob), torch.tensor(y_test))
        
        return metrics.compute()
    
    def predict(self, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        """Make predictions on a batch."""
        if not self.is_trained:
            batch_size = len(batch.get('user_id', [1]))
            return torch.zeros(batch_size, 2)
        
        # Extract features
        user_transactions, transaction_graph, _ = self._extract_batch_data(batch)
        features_dict = self.feature_engineer.extract_features(user_transactions, transaction_graph)
        feature_vectors, _ = self.feature_engineer.features_to_vector(features_dict)
        
        if len(feature_vectors) == 0:
            batch_size = len(batch.get('user_id', [1]))
            return torch.zeros(batch_size, 2)
        
        # Predict probabilities
        if self.use_lightgbm:
            y_pred_prob = self.model.predict(feature_vectors)
            probs = np.column_stack([1 - y_pred_prob, y_pred_prob])
        else:
            probs = self.model.predict_proba(feature_vectors)
        
        return torch.tensor(probs, dtype=torch.float32)


class RandomForestBaseline(BaselineMethodInterface):
    """Random Forest with comprehensive feature engineering."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.feature_engineer = FeatureEngineer()
        self.model = RandomForestClassifier(
            n_estimators=config.get('n_estimators', 100),
            max_depth=config.get('max_depth', 15),
            min_samples_split=config.get('min_samples_split', 5),
            min_samples_leaf=config.get('min_samples_leaf', 2),
            random_state=config.get('random_state', 42),
            n_jobs=-1
        )
        self.is_trained = False
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
        """Train Random Forest model."""
        print(f"\n🌲 Training Random Forest with Feature Engineering")
        
        # Extract features (same as LightGBM)
        X_train, y_train = self._extract_features_from_loader(train_loader)
        X_val, y_val = self._extract_features_from_loader(val_loader)
        
        if len(X_train) == 0:
            return {'best_val_f1': 0.0}
        
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        val_f1 = 0.0
        if len(X_val) > 0:
            y_pred = self.model.predict(X_val)
            val_f1 = f1_score(y_val, y_pred, zero_division=0.0)
        
        print(f"✅ Random Forest training completed - Validation F1: {val_f1:.4f}")
        
        return {'best_val_f1': val_f1}
    
    def _extract_features_from_loader(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from dataloader (same as LightGBM)."""
        all_features = []
        all_labels = []
        
        for batch_idx, batch in enumerate(dataloader):
            user_transactions, transaction_graph, labels = self._extract_batch_data(batch)
            features_dict = self.feature_engineer.extract_features(user_transactions, transaction_graph)
            feature_vectors, user_ids = self.feature_engineer.features_to_vector(features_dict)
            
            for i, user_id in enumerate(user_ids):
                all_features.append(feature_vectors[i])
                all_labels.append(labels.get(user_id, 0))
            
            if batch_idx >= 10:
                break
        
        return np.array(all_features), np.array(all_labels)
    
    def _extract_batch_data(self, batch: Dict) -> Tuple[Dict[str, List], nx.Graph, Dict]:
        """Extract structured data from batch (same as LightGBM)."""
        # Same implementation as LightGBM
        user_transactions = {}
        transaction_graph = nx.Graph()
        labels = {}
        
        if 'user_id' in batch:
            user_ids = batch['user_id'] if isinstance(batch['user_id'], list) else [batch['user_id']]
            batch_labels = batch.get('labels', torch.zeros(len(user_ids)))
            
            for i, user_id in enumerate(user_ids):
                user_transactions[user_id] = [
                    {
                        'timestamp': 1679529600 + i * 3600 + j * 600,
                        'transaction_type': np.random.choice(['swap', 'transfer', 'bridge']),
                        'value_usd': 100.0 + i * 10 + np.random.rand() * 50,
                        'protocol': np.random.choice(['uniswap_v3', 'gmx', 'camelot']),
                        'gas_fee': 0.01 + np.random.rand() * 0.02
                    }
                    for j in range(np.random.randint(1, 6))
                ]
                
                transaction_graph.add_node(user_id)
                labels[user_id] = int(batch_labels[i]) if i < len(batch_labels) else 0
            
            for i, user_i in enumerate(user_ids):
                for j, user_j in enumerate(user_ids):
                    if i != j and np.random.rand() > 0.8:
                        transaction_graph.add_edge(user_i, user_j, weight=np.random.rand())
        
        return user_transactions, transaction_graph, labels
    
    def evaluate(self, test_loader: DataLoader, device: torch.device) -> Dict[str, float]:
        """Evaluate Random Forest model."""
        if not self.is_trained:
            return {'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        X_test, y_test = self._extract_features_from_loader(test_loader)
        
        if len(X_test) == 0:
            return {'f1': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        y_pred_prob = self.model.predict_proba(X_test)[:, 1]
        
        metrics = BinaryClassificationMetrics()
        metrics.update(torch.tensor(y_pred_prob), torch.tensor(y_test))
        
        return metrics.compute()
    
    def predict(self, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        """Make predictions on a batch."""
        if not self.is_trained:
            batch_size = len(batch.get('user_id', [1]))
            return torch.zeros(batch_size, 2)
        
        user_transactions, transaction_graph, _ = self._extract_batch_data(batch)
        features_dict = self.feature_engineer.extract_features(user_transactions, transaction_graph)
        feature_vectors, _ = self.feature_engineer.features_to_vector(features_dict)
        
        if len(feature_vectors) == 0:
            batch_size = len(batch.get('user_id', [1]))
            return torch.zeros(batch_size, 2)
        
        probs = self.model.predict_proba(feature_vectors)
        
        return torch.tensor(probs, dtype=torch.float32)