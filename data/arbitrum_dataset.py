"""
Arbitrum DeFi dataset implementation for airdrop hunter detection.

Integrates with The Graph Protocol and Alchemy RPC to collect real transaction data
from major Arbitrum protocols. Includes documented hunter addresses from the 
$3.3M ARB token consolidation event.

Primary protocols: Uniswap V3, GMX, Camelot, SushiSwap, Aave
Target period: January 1, 2023 - March 23, 2023 (pre-airdrop)
"""

import torch
import pandas as pd
import numpy as np
import json
import os
import requests
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from collections import defaultdict
import warnings

from .pure_crypto_dataset import PureCryptoDataset
from .transaction_schema import ArbitrumDeFiTransaction, TransactionSchemaValidator, transactions_to_dataframe
from .preprocessing import TemporalGraphPreprocessor


class ArbitrumDeFiDataset(PureCryptoDataset):
    """
    Arbitrum DeFi dataset focusing on ARB airdrop hunter detection.
    
    This is our highest-priority dataset with documented hunter addresses
    and $3.3M consolidation evidence. Supports comprehensive analysis
    of pre-airdrop farming behavior across major Arbitrum protocols.
    """
    
    def __init__(self, 
                 data_path: str,
                 split: str = 'train',
                 protocols: Optional[List[str]] = None,
                 start_date: str = '2023-01-01',
                 end_date: str = '2023-03-22',  # Day before ARB airdrop
                 the_graph_api_key: Optional[str] = None,
                 alchemy_api_key: Optional[str] = None,
                 include_known_hunters: bool = True,
                 hunter_label_confidence: float = 0.95,
                 **kwargs):
        """
        Args:
            data_path: Path to dataset directory
            split: 'train', 'val', or 'test'
            protocols: List of protocols to include
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD) 
            the_graph_api_key: The Graph Protocol API key
            alchemy_api_key: Alchemy RPC API key
            include_known_hunters: Whether to include documented hunter addresses
            hunter_label_confidence: Confidence threshold for hunter labeling
        """
        
        # Set Arbitrum-specific defaults
        if protocols is None:
            protocols = ['uniswap_v3', 'gmx', 'camelot', 'sushiswap', 'aave']
        
        super().__init__(
            data_path=data_path,
            split=split,
            blockchain='arbitrum',
            protocols=protocols,
            **kwargs
        )
        
        self.start_date = start_date
        self.end_date = end_date
        self.the_graph_api_key = the_graph_api_key
        self.alchemy_api_key = alchemy_api_key
        self.include_known_hunters = include_known_hunters
        self.hunter_label_confidence = hunter_label_confidence
        
        # Arbitrum ARB airdrop event (March 23, 2023)
        self.airdrop_events = [1679529600]  # March 23, 2023 timestamp
        
        # Known hunter addresses from research documentation
        self.known_hunter_addresses = {
            '0xe1e271a26a42d00731caf4c7ab8ed1684510ab6e': {
                'tokens_claimed': 2100000,  # 2.1M ARB
                'source_addresses': 1200,
                'confidence': 1.0,
                'pattern': 'major_consolidator'
            },
            '0x770edb43ecc5bcbe6f7088e1049fc42b2d1b195c': {
                'tokens_claimed': 1190000,  # 1.19M ARB  
                'source_addresses': 1375,
                'confidence': 1.0,
                'pattern': 'major_consolidator'
            }
            # Additional hunter addresses would be added here from further research
        }
        
        # Protocol configurations with The Graph subgraph IDs
        self.protocol_configs = {
            'uniswap_v3': {
                'subgraph_id': 'ELUcwgpm14LKPLrBRuVvPvNKHQ9HvwmtKgKSH6123cr7',
                'entity_types': ['swaps', 'mints', 'burns'],
                'contract_address': '0x1F98431c8aD98523631AE4a59f267346ea31F984'
            },
            'gmx': {
                'subgraph_id': 'AYl4LsFThjAopwFaFqD9J5PMPQ2Hyz8fyGKk2S8PG2F3',
                'entity_types': ['positions', 'trades', 'liquidations'],
                'contract_address': '0x489ee077994B6658eAfA855C308275EAd8097C4A'
            },
            'camelot': {
                'subgraph_id': 'camelot-arbitrum',  # Hosted service
                'entity_types': ['swaps', 'pairs'],
                'contract_address': '0xc873fEcbd354f5A56E00E710B90EF4201db2448d'
            },
            'sushiswap': {
                'subgraph_id': 'sushiswap-arbitrum',  # Hosted service
                'entity_types': ['swaps', 'pairs'],
                'contract_address': '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506'
            },
            'aave': {
                'subgraph_id': 'aave-v3-arbitrum',
                'entity_types': ['deposits', 'withdraws', 'borrows', 'repays'],
                'contract_address': '0x794a61358D6845594F94dc1DB02A252b5b4814aD'
            }
        }
        
        # Transaction schema validator
        self.schema_validator = TransactionSchemaValidator()
        
        # Load dataset
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """Load the dataset, collecting from APIs if necessary."""
        dataset_file = os.path.join(self.data_path, f'arbitrum_defi_{self.split}.json')
        
        if os.path.exists(dataset_file) and not self._should_refresh_data():
            print(f"Loading existing Arbitrum dataset from {dataset_file}")
            self.load_raw_data()
        else:
            print(f"Dataset not found or needs refresh at {dataset_file}")
            if self.the_graph_api_key or self.alchemy_api_key:
                print("Collecting real data from Arbitrum APIs...")
                self.download_data()
            else:
                print("No API keys provided. Generating demonstration data...")
                print("To collect real data, provide the_graph_api_key and/or alchemy_api_key")
                self._generate_demonstration_data()
    
    def _should_refresh_data(self) -> bool:
        """Check if data should be refreshed based on age and completeness."""
        dataset_file = os.path.join(self.data_path, f'arbitrum_defi_{self.split}.json')
        
        if not os.path.exists(dataset_file):
            return True
        
        # Check file age (refresh if older than 7 days)
        file_age = time.time() - os.path.getmtime(dataset_file)
        if file_age > 7 * 24 * 3600:  # 7 days
            return True
        
        # Check data completeness
        try:
            with open(dataset_file, 'r') as f:
                data = json.load(f)
            
            if len(data.get('transactions', [])) < 1000:  # Minimum expected transactions
                return True
        except:
            return True
        
        return False
    
    def download_data(self) -> None:
        """Download real Arbitrum DeFi data from The Graph and Alchemy."""
        print("Starting Arbitrum DeFi data collection...")
        
        all_transactions = []
        
        # Collect data from each protocol
        for protocol in self.protocols:
            print(f"Collecting {protocol} data...")
            try:
                protocol_transactions = self._collect_protocol_data(protocol)
                all_transactions.extend(protocol_transactions)
                print(f"Collected {len(protocol_transactions)} transactions from {protocol}")
                
                # Rate limiting
                time.sleep(1)  # 1 second between protocol queries
                
            except Exception as e:
                print(f"Error collecting {protocol} data: {e}")
                continue
        
        if not all_transactions:
            print("No data collected from APIs, falling back to demonstration data")
            self._generate_demonstration_data()
            return
        
        # Convert to DataFrame and validate
        transactions_df = transactions_to_dataframe(all_transactions)
        
        # Validate schema
        is_valid, errors = self.schema_validator.validate_dataframe(transactions_df)
        if not is_valid:
            print(f"Data validation errors: {errors}")
            # Continue with warnings
        
        # Generate users and labels
        users_df = self._generate_users_from_transactions(transactions_df)
        labels = self._generate_labels_from_known_hunters(users_df)
        
        # Save collected data
        dataset = {
            'transactions': transactions_df.to_dict(orient='records'),
            'users': users_df.to_dict(orient='records'),
            'labels': labels,
            'metadata': {
                'collection_date': datetime.now().isoformat(),
                'start_date': self.start_date,
                'end_date': self.end_date,
                'protocols': self.protocols,
                'total_transactions': len(transactions_df),
                'total_users': len(users_df),
                'hunter_addresses': len([u for u in labels.values() if u == 1])
            }
        }
        
        dataset_file = os.path.join(self.data_path, f'arbitrum_defi_{self.split}.json')
        os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
        
        with open(dataset_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Saved {len(transactions_df)} transactions to {dataset_file}")
        
        # Load the saved data
        self.load_raw_data()
    
    def _collect_protocol_data(self, protocol: str) -> List[ArbitrumDeFiTransaction]:
        """Collect transaction data for a specific protocol."""
        if protocol not in self.protocol_configs:
            print(f"Unknown protocol: {protocol}")
            return []
        
        config = self.protocol_configs[protocol]
        
        if self.the_graph_api_key and 'subgraph_id' in config:
            return self._collect_from_the_graph(protocol, config)
        elif self.alchemy_api_key:
            return self._collect_from_alchemy(protocol, config)
        else:
            print(f"No suitable API key for {protocol}")
            return []
    
    def _collect_from_the_graph(self, protocol: str, config: Dict) -> List[ArbitrumDeFiTransaction]:
        """Collect data from The Graph Protocol subgraph."""
        transactions = []
        
        # Convert dates to timestamps
        start_timestamp = int(datetime.strptime(self.start_date, '%Y-%m-%d').timestamp())
        end_timestamp = int(datetime.strptime(self.end_date, '%Y-%m-%d').timestamp())
        
        # Build GraphQL query based on protocol
        if protocol == 'uniswap_v3':
            query = self._build_uniswap_v3_query(start_timestamp, end_timestamp)
        elif protocol == 'gmx':
            query = self._build_gmx_query(start_timestamp, end_timestamp)
        elif protocol in ['camelot', 'sushiswap']:
            query = self._build_dex_query(start_timestamp, end_timestamp)
        elif protocol == 'aave':
            query = self._build_aave_query(start_timestamp, end_timestamp)
        else:
            print(f"No query builder for {protocol}")
            return []
        
        # Execute query
        try:
            subgraph_id = config['subgraph_id']
            
            # Use gateway if we have a proper subgraph ID, otherwise hosted service
            if len(subgraph_id) > 40:  # Gateway subgraph ID
                url = f"https://gateway.thegraph.com/api/{self.the_graph_api_key}/subgraphs/id/{subgraph_id}"
            else:  # Hosted service
                url = f"https://api.thegraph.com/subgraphs/name/{subgraph_id}"
            
            response = requests.post(url, json={'query': query}, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'errors' in data:
                print(f"GraphQL errors for {protocol}: {data['errors']}")
                return []
            
            # Parse response into transactions
            transactions = self._parse_the_graph_response(protocol, data['data'])
            
        except Exception as e:
            print(f"Error querying The Graph for {protocol}: {e}")
            return []
        
        return transactions
    
    def _build_uniswap_v3_query(self, start_timestamp: int, end_timestamp: int) -> str:
        """Build GraphQL query for Uniswap V3 swaps."""
        return f"""
        {{
          swaps(
            first: 1000
            where: {{
              timestamp_gte: "{start_timestamp}"
              timestamp_lte: "{end_timestamp}"
              amountUSD_gte: "1.0"
            }}
            orderBy: timestamp
            orderDirection: asc
          ) {{
            id
            transaction {{
              id
              blockNumber
              gasUsed
              gasPrice
            }}
            timestamp
            sender
            recipient
            amount0
            amount1
            amountUSD
            token0 {{
              symbol
              decimals
            }}
            token1 {{
              symbol
              decimals
            }}
            pool {{
              id
              feeTier
            }}
            sqrtPriceX96
            tick
          }}
        }}
        """
    
    def _build_gmx_query(self, start_timestamp: int, end_timestamp: int) -> str:
        """Build GraphQL query for GMX positions and trades."""
        return f"""
        {{
          trades(
            first: 1000
            where: {{
              timestamp_gte: "{start_timestamp}"
              timestamp_lte: "{end_timestamp}"
            }}
            orderBy: timestamp
            orderDirection: asc
          ) {{
            id
            account
            indexToken
            collateralToken
            size
            sizeDelta
            collateralDelta
            fee
            isLong
            timestamp
            transaction {{
              id
              blockNumber
            }}
          }}
        }}
        """
    
    def _build_dex_query(self, start_timestamp: int, end_timestamp: int) -> str:
        """Build GraphQL query for generic DEX swaps (Camelot, SushiSwap)."""
        return f"""
        {{
          swaps(
            first: 1000
            where: {{
              timestamp_gte: "{start_timestamp}"
              timestamp_lte: "{end_timestamp}"
            }}
            orderBy: timestamp
            orderDirection: asc
          ) {{
            id
            transaction {{
              id
              blockNumber
            }}
            pair {{
              id
              token0 {{
                symbol
              }}
              token1 {{
                symbol
              }}
            }}
            sender
            to
            amount0In
            amount0Out
            amount1In
            amount1Out
            amountUSD
            timestamp
          }}
        }}
        """
    
    def _build_aave_query(self, start_timestamp: int, end_timestamp: int) -> str:
        """Build GraphQL query for Aave lending activities."""
        return f"""
        {{
          deposits(
            first: 500
            where: {{
              timestamp_gte: "{start_timestamp}"
              timestamp_lte: "{end_timestamp}"
            }}
            orderBy: timestamp
            orderDirection: asc
          ) {{
            id
            user {{
              id
            }}
            reserve {{
              symbol
            }}
            amount
            timestamp
            transaction {{
              id
              blockNumber
            }}
          }}
          
          withdraws(
            first: 500
            where: {{
              timestamp_gte: "{start_timestamp}"
              timestamp_lte: "{end_timestamp}"
            }}
            orderBy: timestamp
            orderDirection: asc
          ) {{
            id
            user {{
              id
            }}
            reserve {{
              symbol
            }}
            amount
            timestamp
            transaction {{
              id
              blockNumber
            }}
          }}
        }}
        """
    
    def _parse_the_graph_response(self, protocol: str, data: Dict) -> List[ArbitrumDeFiTransaction]:
        """Parse The Graph response into standardized transactions."""
        transactions = []
        
        if protocol == 'uniswap_v3' and 'swaps' in data:
            for swap in data['swaps']:
                tx = ArbitrumDeFiTransaction(
                    user_id=swap['sender'],
                    timestamp=float(swap['timestamp']),
                    transaction_type='swap',
                    value_usd=float(swap['amountUSD']),
                    gas_fee=self._estimate_gas_fee(swap.get('transaction', {})),
                    signature=swap['transaction']['id'],
                    block_number=int(swap['transaction']['blockNumber']),
                    protocol='uniswap_v3',
                    token_in=swap['token0']['symbol'],
                    token_out=swap['token1']['symbol'],
                    amount_in=float(swap['amount0']),
                    amount_out=float(swap['amount1']),
                    pool_address=swap['pool']['id']
                )
                transactions.append(tx)
        
        elif protocol == 'gmx' and 'trades' in data:
            for trade in data['trades']:
                tx = ArbitrumDeFiTransaction(
                    user_id=trade['account'],
                    timestamp=float(trade['timestamp']),
                    transaction_type='swap',  # GMX perpetual trade
                    value_usd=float(trade['sizeDelta']) / 1e30,  # Convert from GMX units
                    gas_fee=self._estimate_gas_fee(trade.get('transaction', {})),
                    signature=trade['transaction']['id'],
                    block_number=int(trade['transaction']['blockNumber']),
                    protocol='gmx',
                    position_size=float(trade['size']) / 1e30,
                    leverage=float(trade['sizeDelta']) / max(float(trade['collateralDelta']), 1e-8),
                    is_long=trade['isLong'],
                    collateral_token=trade['collateralToken'],
                    index_token=trade['indexToken']
                )
                transactions.append(tx)
        
        elif protocol in ['camelot', 'sushiswap'] and 'swaps' in data:
            for swap in data['swaps']:
                tx = ArbitrumDeFiTransaction(
                    user_id=swap['sender'],
                    timestamp=float(swap['timestamp']),
                    transaction_type='swap',
                    value_usd=float(swap.get('amountUSD', 0)),
                    gas_fee=self._estimate_gas_fee(swap.get('transaction', {})),
                    signature=swap['transaction']['id'],
                    block_number=int(swap['transaction']['blockNumber']),
                    protocol=protocol,
                    token_in=swap['pair']['token0']['symbol'],
                    token_out=swap['pair']['token1']['symbol'],
                    amount_in=float(swap['amount0In']) + float(swap['amount0Out']),
                    amount_out=float(swap['amount1In']) + float(swap['amount1Out'])
                )
                transactions.append(tx)
        
        elif protocol == 'aave':
            # Handle deposits
            for deposit in data.get('deposits', []):
                tx = ArbitrumDeFiTransaction(
                    user_id=deposit['user']['id'],
                    timestamp=float(deposit['timestamp']),
                    transaction_type='lend',
                    value_usd=float(deposit['amount']),  # Would need price conversion
                    gas_fee=self._estimate_gas_fee(deposit.get('transaction', {})),
                    signature=deposit['transaction']['id'],
                    block_number=int(deposit['transaction']['blockNumber']),
                    protocol='aave',
                    asset_address=deposit['reserve']['symbol'],
                    amount=float(deposit['amount'])
                )
                transactions.append(tx)
            
            # Handle withdraws
            for withdraw in data.get('withdraws', []):
                tx = ArbitrumDeFiTransaction(
                    user_id=withdraw['user']['id'],
                    timestamp=float(withdraw['timestamp']),
                    transaction_type='remove_liquidity',
                    value_usd=float(withdraw['amount']),  # Would need price conversion
                    gas_fee=self._estimate_gas_fee(withdraw.get('transaction', {})),
                    signature=withdraw['transaction']['id'],
                    block_number=int(withdraw['transaction']['blockNumber']),
                    protocol='aave',
                    asset_address=withdraw['reserve']['symbol'],
                    amount=float(withdraw['amount'])
                )
                transactions.append(tx)
        
        return transactions
    
    def _estimate_gas_fee(self, transaction_data: Dict) -> float:
        """Estimate gas fee from transaction data."""
        # This is a simplified estimation
        # In practice, you'd need to get gas_used * gas_price from RPC
        gas_used = transaction_data.get('gasUsed', 100000)  # Default estimate
        gas_price = transaction_data.get('gasPrice', 1e9)   # Default 1 gwei
        
        # Convert to ETH (gas_used * gas_price / 1e18)
        gas_fee_eth = (gas_used * gas_price) / 1e18
        return gas_fee_eth
    
    def _collect_from_alchemy(self, protocol: str, config: Dict) -> List[ArbitrumDeFiTransaction]:
        """Collect data from Alchemy RPC (fallback method)."""
        # This would implement direct RPC calls to get transaction data
        # For now, return empty list as this is more complex to implement
        print(f"Alchemy RPC collection for {protocol} not yet implemented")
        return []
    
    def _generate_demonstration_data(self) -> None:
        """Generate demonstration data when APIs are not available."""
        print("Generating demonstration Arbitrum DeFi data...")
        
        # Generate synthetic transactions
        num_users = 1000
        transactions_per_user = 20
        
        users = []
        transactions = []
        labels = {}
        
        # Include known hunter addresses
        hunter_addresses = list(self.known_hunter_addresses.keys())
        
        for i in range(num_users):
            if i < len(hunter_addresses):
                user_id = hunter_addresses[i]
                is_hunter = True
            else:
                user_id = f"arbitrum_user_{i:04d}"
                is_hunter = np.random.random() < 0.1  # 10% hunters
            
            users.append({'user_id': user_id})
            labels[user_id] = 1 if is_hunter else 0
            
            # Generate transactions for this user
            user_transactions = self._generate_user_transactions(
                user_id, transactions_per_user, is_hunter
            )
            transactions.extend(user_transactions)
        
        # Convert to DataFrames
        users_df = pd.DataFrame(users)
        transactions_df = transactions_to_dataframe(transactions)
        
        # Save demonstration data
        dataset = {
            'transactions': transactions_df.to_dict(orient='records'),
            'users': users_df.to_dict(orient='records'),
            'labels': labels,
            'metadata': {
                'data_type': 'demonstration',
                'generation_date': datetime.now().isoformat(),
                'total_transactions': len(transactions_df),
                'total_users': len(users_df),
                'hunter_addresses': len([u for u in labels.values() if u == 1])
            }
        }
        
        dataset_file = os.path.join(self.data_path, f'arbitrum_defi_{self.split}.json')
        os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
        
        with open(dataset_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Generated demonstration dataset with {len(transactions)} transactions")
        
        # Load the generated data
        self.load_raw_data()
    
    def _generate_user_transactions(self, user_id: str, num_transactions: int, is_hunter: bool) -> List[ArbitrumDeFiTransaction]:
        """Generate synthetic transactions for a user."""
        transactions = []
        
        # Time range for transactions
        start_time = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_time = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        for i in range(num_transactions):
            # Generate timestamp
            if is_hunter:
                # Hunters cluster activity around airdrop announcement
                # Focus activity in final weeks before airdrop
                time_weight = np.random.beta(1, 3)  # Skew toward end
            else:
                time_weight = np.random.random()  # Uniform distribution
            
            timestamp = start_time + timedelta(
                seconds=time_weight * (end_time - start_time).total_seconds()
            )
            
            # Choose protocol (hunters favor certain protocols)
            if is_hunter:
                protocol = np.random.choice(
                    self.protocols, 
                    p=[0.4, 0.3, 0.15, 0.1, 0.05]  # Prefer Uniswap V3, GMX
                )
            else:
                protocol = np.random.choice(self.protocols)
            
            # Generate transaction type based on protocol
            if protocol in ['uniswap_v3', 'camelot', 'sushiswap']:
                tx_type = 'swap'
                tokens = ['USDC', 'WETH', 'ARB', 'WBTC', 'DAI']
                token_in = np.random.choice(tokens)
                token_out = np.random.choice([t for t in tokens if t != token_in])
            elif protocol == 'gmx':
                tx_type = 'swap'  # Perpetual trade
                token_in = 'USDC'
                token_out = np.random.choice(['WETH', 'WBTC'])
            elif protocol == 'aave':
                tx_type = np.random.choice(['lend', 'remove_liquidity'])
                token_in = np.random.choice(['USDC', 'WETH', 'DAI'])
                token_out = token_in
            
            # Generate transaction value
            if is_hunter:
                # Hunters often use specific amounts to qualify for airdrops
                value_usd = np.random.choice([100, 250, 500, 1000, 2500]) * np.random.uniform(0.8, 1.2)
            else:
                value_usd = np.random.lognormal(5, 1.5)  # More natural distribution
            
            # Generate gas fee (Arbitrum has low fees)
            gas_fee = np.random.uniform(0.001, 0.01)  # ETH
            
            tx = ArbitrumDeFiTransaction(
                user_id=user_id,
                timestamp=timestamp.timestamp(),
                chain_id='arbitrum',  # Required parameter
                transaction_type=tx_type,
                value_usd=value_usd,
                gas_fee=gas_fee,
                signature=f"arbitrum_tx_{user_id}_{i}_{int(timestamp.timestamp())}",
                block_number=17000000 + i * 100,  # Rough Arbitrum block numbers
                protocol=protocol,
                token_in=token_in if 'token_in' in locals() else None,
                token_out=token_out if 'token_out' in locals() else None,
                amount_in=value_usd / 2000 if token_in == 'WETH' else value_usd,  # Rough price conversion
                amount_out=value_usd / 2000 if token_out == 'WETH' else value_usd
            )
            
            transactions.append(tx)
        
        return transactions
    
    def _generate_users_from_transactions(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate users DataFrame from collected transactions."""
        users = transactions_df['user_id'].unique()
        users_df = pd.DataFrame({'user_id': users})
        return users_df
    
    def _generate_labels_from_known_hunters(self, users_df: pd.DataFrame) -> Dict[str, int]:
        """Generate labels based on known hunter addresses."""
        labels = {}
        
        for user_id in users_df['user_id']:
            if user_id in self.known_hunter_addresses:
                hunter_info = self.known_hunter_addresses[user_id]
                if hunter_info['confidence'] >= self.hunter_label_confidence:
                    labels[user_id] = 1  # Hunter
                else:
                    labels[user_id] = 0  # Uncertain, default to legitimate
            else:
                labels[user_id] = 0  # Legitimate (default)
        
        return labels
    
    def load_raw_data(self) -> None:
        """Load raw blockchain data from saved files."""
        dataset_file = os.path.join(self.data_path, f'arbitrum_defi_{self.split}.json')
        
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        
        with open(dataset_file, 'r') as f:
            dataset = json.load(f)
        
        # Load components
        self.transactions = pd.DataFrame(dataset['transactions'])
        self.users = pd.DataFrame(dataset['users'])
        self.labels = dataset['labels']
        
        # Set airdrop events
        self.airdrop_events = self.airdrop_events
        
        print(f"Loaded Arbitrum dataset: {len(self.transactions)} transactions, {len(self.users)} users")
    
    def extract_defi_features(self, user_id: str) -> Dict[str, torch.Tensor]:
        """Extract Arbitrum DeFi-specific features for hunter detection."""
        user_txs = self.transactions[self.transactions['user_id'] == user_id]
        
        if len(user_txs) == 0:
            return {}
        
        # Cross-protocol features
        cross_protocol_features = self.extract_cross_protocol_features(user_id)
        
        # Arbitrum-specific features
        arbitrum_features = self._extract_arbitrum_specific_features(user_txs)
        
        # Combine features
        return {**cross_protocol_features, **arbitrum_features}
    
    def _extract_arbitrum_specific_features(self, user_txs: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Extract features specific to Arbitrum ecosystem."""
        features = {}
        
        # L2 gas efficiency patterns
        if 'l1_gas_used' in user_txs.columns and user_txs['l1_gas_used'].notna().any():
            l1_gas = user_txs['l1_gas_used'].fillna(0)
            l2_gas = user_txs['l2_gas_used'].fillna(0)
            total_gas = l1_gas + l2_gas
            
            features['l1_gas_ratio'] = torch.tensor(
                l1_gas.sum() / max(total_gas.sum(), 1e-8), dtype=torch.float32
            )
            features['gas_optimization_score'] = torch.tensor(
                (total_gas < total_gas.quantile(0.25)).mean(), dtype=torch.float32
            )
        
        # Protocol interaction patterns
        protocol_counts = user_txs['protocol'].value_counts()
        
        # Uniswap V3 specific patterns
        if 'uniswap_v3' in protocol_counts:
            uniswap_txs = user_txs[user_txs['protocol'] == 'uniswap_v3']
            if len(uniswap_txs) > 0 and 'price_impact' in uniswap_txs.columns:
                avg_price_impact = uniswap_txs['price_impact'].fillna(0).mean()
                features['uniswap_price_impact'] = torch.tensor(avg_price_impact, dtype=torch.float32)
        
        # GMX perpetual patterns
        if 'gmx' in protocol_counts:
            gmx_txs = user_txs[user_txs['protocol'] == 'gmx']
            if len(gmx_txs) > 0:
                avg_leverage = gmx_txs['leverage'].fillna(1.0).mean()
                long_ratio = gmx_txs['is_long'].fillna(False).mean()
                features['gmx_avg_leverage'] = torch.tensor(avg_leverage, dtype=torch.float32)
                features['gmx_long_ratio'] = torch.tensor(long_ratio, dtype=torch.float32)
        
        # Bridge activity (indicating multi-chain farming)
        bridge_txs = user_txs[user_txs['transaction_type'] == 'bridge']
        features['bridge_activity'] = torch.tensor(len(bridge_txs), dtype=torch.float32)
        
        return features
    
    def build_protocol_interaction_graph(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Build user graph based on Arbitrum protocol interactions."""
        # Implementation similar to base class but with Arbitrum-specific relationships
        user_pairs = defaultdict(list)
        
        # Group users by protocol interactions
        for protocol in self.protocols:
            protocol_txs = self.transactions[self.transactions['protocol'] == protocol]
            protocol_users = protocol_txs['user_id'].unique()
            
            # Create edges between users who used the same protocol
            for i, user1 in enumerate(protocol_users):
                for user2 in protocol_users[i+1:]:
                    user_pairs[(user1, user2)].append(protocol)
        
        # Build edge index and features
        edges = []
        edge_features = []
        
        user_to_idx = {user: idx for idx, user in enumerate(self.users['user_id'])}
        
        for (user1, user2), shared_protocols in user_pairs.items():
            if user1 in user_to_idx and user2 in user_to_idx:
                edges.extend([(user_to_idx[user1], user_to_idx[user2]),
                            (user_to_idx[user2], user_to_idx[user1])])
                
                # Compute edge features
                edge_feature = self._compute_protocol_edge_features(user1, user2, shared_protocols)
                edge_features.extend([edge_feature, edge_feature])
        
        if not edges:
            num_users = len(user_to_idx)
            edges = [(i, i) for i in range(num_users)]
            edge_features = [torch.zeros(64) for _ in range(num_users)]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_features = torch.stack(edge_features)
        
        return edge_index, {'edge_features': edge_features}
    
    def _compute_protocol_edge_features(self, user1: str, user2: str, shared_protocols: List[str]) -> torch.Tensor:
        """Compute edge features based on shared protocol usage."""
        user1_txs = self.transactions[self.transactions['user_id'] == user1]
        user2_txs = self.transactions[self.transactions['user_id'] == user2]
        
        # Protocol overlap features
        num_shared_protocols = len(shared_protocols)
        protocol_similarity = num_shared_protocols / len(self.protocols)
        
        # Temporal correlation
        temporal_correlation = 0.0
        if len(user1_txs) > 0 and len(user2_txs) > 0:
            time_overlap = self._compute_temporal_overlap(user1_txs, user2_txs)
            temporal_correlation = time_overlap
        
        # Volume correlation
        volume_correlation = 0.0
        for protocol in shared_protocols:
            p1_volume = user1_txs[user1_txs['protocol'] == protocol]['value_usd'].sum()
            p2_volume = user2_txs[user2_txs['protocol'] == protocol]['value_usd'].sum()
            
            if p1_volume > 0 and p2_volume > 0:
                volume_ratio = min(p1_volume, p2_volume) / max(p1_volume, p2_volume)
                volume_correlation += volume_ratio
        
        volume_correlation = volume_correlation / max(len(shared_protocols), 1)
        
        # Combine features
        features = torch.tensor([
            num_shared_protocols,
            protocol_similarity,
            temporal_correlation,
            volume_correlation
        ], dtype=torch.float32)
        
        # Pad to standard size
        padded_features = torch.zeros(64)
        padded_features[:min(len(features), 64)] = features[:64]
        
        return padded_features
    
    def _compute_temporal_overlap(self, user1_txs: pd.DataFrame, user2_txs: pd.DataFrame) -> float:
        """Compute temporal overlap between two users' activity."""
        min1, max1 = user1_txs['timestamp'].min(), user1_txs['timestamp'].max()
        min2, max2 = user2_txs['timestamp'].min(), user2_txs['timestamp'].max()
        
        overlap_start = max(min1, min2)
        overlap_end = min(max1, max2)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_duration = overlap_end - overlap_start
        total_duration = max(max1, max2) - min(min1, min2)
        
        return overlap_duration / max(total_duration, 1.0)