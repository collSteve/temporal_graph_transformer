"""
Jupiter Solana DeFi dataset implementation for airdrop hunter detection.

Integrates with Jupiter API and Helius RPC to collect real transaction data
from Solana DeFi ecosystem. Focuses on Jupiter DEX aggregator and other
major Solana protocols around the JUP airdrop event.

Primary protocols: Jupiter, Raydium, Orca, Drift, Kamino, Marinade
Target period: October 1, 2023 - January 30, 2024 (pre-JUP airdrop)
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
import base64

from .pure_crypto_dataset import PureCryptoDataset
from .transaction_schema import SolanaDeFiTransaction, TransactionSchemaValidator, transactions_to_dataframe
from .preprocessing import TemporalGraphPreprocessor


class JupiterSolanaDataset(PureCryptoDataset):
    """
    Jupiter Solana DeFi dataset focusing on JUP airdrop hunter detection.
    
    Analyzes sophisticated anti-farming measures implemented by Jupiter
    and patterns around the 955,000+ eligible wallets for JUP distribution.
    """
    
    def __init__(self, 
                 data_path: str,
                 split: str = 'train',
                 protocols: Optional[List[str]] = None,
                 start_date: str = '2023-10-01',
                 end_date: str = '2024-01-30',  # Day before JUP airdrop
                 jupiter_api_key: Optional[str] = None,
                 helius_api_key: Optional[str] = None,
                 quicknode_api_key: Optional[str] = None,
                 include_stablecoin_farming: bool = True,
                 anti_farming_analysis: bool = True,
                 **kwargs):
        """
        Args:
            data_path: Path to dataset directory
            split: 'train', 'val', or 'test'
            protocols: List of Solana protocols to include
            start_date: Start date for data collection (YYYY-MM-DD)
            end_date: End date for data collection (YYYY-MM-DD)
            jupiter_api_key: Jupiter API key (if available)
            helius_api_key: Helius RPC API key
            quicknode_api_key: QuickNode RPC API key
            include_stablecoin_farming: Whether to include stablecoin farming patterns
            anti_farming_analysis: Whether to analyze Jupiter's anti-farming measures
        """
        
        # Set Solana-specific defaults
        if protocols is None:
            protocols = ['jupiter', 'raydium', 'orca', 'drift', 'kamino', 'marinade']
        
        super().__init__(
            data_path=data_path,
            split=split,
            blockchain='solana',
            protocols=protocols,
            min_transaction_value_usd=0.1,  # Lower threshold for Solana
            **kwargs
        )
        
        self.start_date = start_date
        self.end_date = end_date
        self.jupiter_api_key = jupiter_api_key
        self.helius_api_key = helius_api_key
        self.quicknode_api_key = quicknode_api_key
        self.include_stablecoin_farming = include_stablecoin_farming
        self.anti_farming_analysis = anti_farming_analysis
        
        # Jupiter JUP airdrop event (January 31, 2024)
        self.airdrop_events = [1706659200]  # January 31, 2024 timestamp
        
        # Jupiter anti-farming criteria (from research documentation)
        self.jupiter_anti_farming_criteria = {
            'min_interactions': 3,  # Required minimum interactions
            'stablecoin_penalty': 0.5,  # Reduced weight for stablecoin pairs
            'volume_threshold': 1000,  # USD minimum volume
            'unique_days_threshold': 7,  # Minimum activity days
            'penalized_pairs': ['USDC-USDT', 'USDC-DAI', 'USDT-DAI']  # Stablecoin pairs
        }
        
        # Solana program addresses for major protocols
        self.protocol_configs = {
            'jupiter': {
                'program_id': 'JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4',
                'instruction_types': ['sharedAccountsRoute', 'route'],
                'api_endpoint': 'https://quote-api.jup.ag/v6',
                'swap_endpoint': 'https://quote-api.jup.ag/v6/swap'
            },
            'raydium': {
                'program_id': '675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8',
                'instruction_types': ['swap', 'addLiquidity', 'removeLiquidity'],
                'amm_program': '5quBtoiQqy7JBxHPCcJoKddhEhKyCnKNvkqGp6Wgje9h'
            },
            'orca': {
                'program_id': '9W959DqEETiGZocYWCQPaJ6sD9kmkKGSqrVUCL6zN1D',
                'instruction_types': ['swap', 'increaseLiquidity', 'decreaseLiquidity'],
                'whirlpool_program': 'whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc'
            },
            'drift': {
                'program_id': 'dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH',
                'instruction_types': ['openPosition', 'closePosition', 'placeOrder']
            },
            'kamino': {
                'program_id': 'KLend2g3cP87fffoy8q1mQqGKjrxjC8boSyAYavgmjD',
                'instruction_types': ['deposit', 'withdraw', 'borrow', 'repay']
            },
            'marinade': {
                'program_id': '8szGkuLTAux9XMgZ2vtY39jVSowEcpBfFfD8hXSEqdGC',
                'instruction_types': ['deposit', 'liquidUnstake', 'delayedUnstake']
            }
        }
        
        # Common Solana token mints
        self.token_mints = {
            'SOL': 'So11111111111111111111111111111111111111112',
            'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
            'USDT': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
            'RAY': '4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R',
            'ORCA': 'orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE',
            'MNDE': 'MNDEFzGvMt87ueuHvVU9VcTqsAP5b3fTGPSAAxzvtUmV',
            'JUP': 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
            'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263'
        }
        
        # Transaction schema validator
        self.schema_validator = TransactionSchemaValidator()
        
        # Load dataset
        self._load_dataset()
    
    def _load_dataset(self) -> None:
        """Load the dataset, collecting from APIs if necessary."""
        dataset_file = os.path.join(self.data_path, f'jupiter_solana_{self.split}.json')
        
        if os.path.exists(dataset_file) and not self._should_refresh_data():
            print(f"Loading existing Jupiter Solana dataset from {dataset_file}")
            self.load_raw_data()
        else:
            print(f"Dataset not found or needs refresh at {dataset_file}")
            if self.helius_api_key or self.quicknode_api_key:
                print("Collecting real data from Solana APIs...")
                self.download_data()
            else:
                print("No RPC API keys provided. Generating demonstration data...")
                print("To collect real data, provide helius_api_key or quicknode_api_key")
                self._generate_demonstration_data()
    
    def _should_refresh_data(self) -> bool:
        """Check if data should be refreshed based on age and completeness."""
        dataset_file = os.path.join(self.data_path, f'jupiter_solana_{self.split}.json')
        
        if not os.path.exists(dataset_file):
            return True
        
        # Check file age (refresh if older than 7 days)
        file_age = time.time() - os.path.getmtime(dataset_file)
        if file_age > 7 * 24 * 3600:
            return True
        
        # Check data completeness
        try:
            with open(dataset_file, 'r') as f:
                data = json.load(f)
            
            if len(data.get('transactions', [])) < 500:  # Minimum expected
                return True
        except:
            return True
        
        return False
    
    def download_data(self) -> None:
        """Download real Solana DeFi data from Jupiter API and Helius RPC."""
        print("Starting Jupiter Solana data collection...")
        
        all_transactions = []
        
        # Collect data from each protocol
        for protocol in self.protocols:
            print(f"Collecting {protocol} data...")
            try:
                protocol_transactions = self._collect_protocol_data(protocol)
                all_transactions.extend(protocol_transactions)
                print(f"Collected {len(protocol_transactions)} transactions from {protocol}")
                
                # Rate limiting for Solana RPC
                time.sleep(0.5)  # 500ms between protocol queries
                
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
        
        # Generate users and labels
        users_df = self._generate_users_from_transactions(transactions_df)
        labels = self._generate_labels_from_jupiter_criteria(users_df, transactions_df)
        
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
                'eligible_addresses': len([u for u in labels.values() if u == 0]),
                'ineligible_addresses': len([u for u in labels.values() if u == 1])
            }
        }
        
        dataset_file = os.path.join(self.data_path, f'jupiter_solana_{self.split}.json')
        os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
        
        with open(dataset_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Saved {len(transactions_df)} transactions to {dataset_file}")
        
        # Load the saved data
        self.load_raw_data()
    
    def _collect_protocol_data(self, protocol: str) -> List[SolanaDeFiTransaction]:
        """Collect transaction data for a specific Solana protocol."""
        if protocol not in self.protocol_configs:
            print(f"Unknown protocol: {protocol}")
            return []
        
        config = self.protocol_configs[protocol]
        
        if protocol == 'jupiter' and self.jupiter_api_key:
            return self._collect_jupiter_data(config)
        elif self.helius_api_key:
            return self._collect_from_helius(protocol, config)
        elif self.quicknode_api_key:
            return self._collect_from_quicknode(protocol, config)
        else:
            print(f"No suitable API key for {protocol}")
            return []
    
    def _collect_jupiter_data(self, config: Dict) -> List[SolanaDeFiTransaction]:
        """Collect data from Jupiter API."""
        transactions = []
        
        # Jupiter API doesn't provide historical data directly
        # We would need to use RPC calls to get historical transactions
        print("Jupiter API doesn't provide historical data. Using RPC fallback.")
        
        return self._collect_from_helius('jupiter', config)
    
    def _collect_from_helius(self, protocol: str, config: Dict) -> List[SolanaDeFiTransaction]:
        """Collect data from Helius RPC with enhanced APIs."""
        transactions = []
        
        if not self.helius_api_key:
            return []
        
        try:
            # Use Helius enhanced API to get transactions for a program
            program_id = config['program_id']
            
            url = f"https://api.helius.xyz/v0/addresses/{program_id}/transactions"
            
            params = {
                'api-key': self.helius_api_key,
                'limit': 1000,  # Maximum per request
                'type': 'SWAP',  # Filter for relevant transaction types
                'commitment': 'confirmed'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse Helius response
            for tx_data in data:
                try:
                    tx = self._parse_helius_transaction(protocol, tx_data)
                    if tx and self._is_in_date_range(tx.timestamp):
                        transactions.append(tx)
                except Exception as e:
                    print(f"Error parsing transaction: {e}")
                    continue
            
        except Exception as e:
            print(f"Error collecting from Helius for {protocol}: {e}")
        
        return transactions
    
    def _collect_from_quicknode(self, protocol: str, config: Dict) -> List[SolanaDeFiTransaction]:
        """Collect data from QuickNode RPC."""
        # QuickNode collection would be implemented here
        print(f"QuickNode collection for {protocol} not yet implemented")
        return []
    
    def _parse_helius_transaction(self, protocol: str, tx_data: Dict) -> Optional[SolanaDeFiTransaction]:
        """Parse Helius transaction data into standardized format."""
        try:
            # Extract basic transaction info
            signature = tx_data.get('signature', '')
            block_time = tx_data.get('blockTime', 0)
            
            # Extract account info (sender)
            accounts = tx_data.get('accountData', [])
            sender = accounts[0]['account'] if accounts else 'unknown'
            
            # Estimate transaction value (simplified)
            # In practice, you'd parse the instruction data properly
            sol_transfer = tx_data.get('nativeTransfers', [])
            value_sol = sum(transfer.get('amount', 0) for transfer in sol_transfer) / 1e9
            value_usd = value_sol * 100  # Rough SOL price estimate
            
            # Extract program-specific data
            instructions = tx_data.get('instructions', [])
            program_instruction = None
            
            for instruction in instructions:
                if instruction.get('programId') == self.protocol_configs[protocol]['program_id']:
                    program_instruction = instruction
                    break
            
            if not program_instruction:
                return None
            
            # Parse instruction data based on protocol
            instruction_data = self._parse_instruction_data(protocol, program_instruction)
            
            tx = SolanaDeFiTransaction(
                user_id=sender,
                timestamp=float(block_time),
                chain_id='solana',  # Required parameter
                transaction_type='swap',  # Default, would be refined based on instruction
                value_usd=value_usd,
                gas_fee=tx_data.get('fee', 5000) / 1e9,  # Convert lamports to SOL
                signature=signature,
                block_number=tx_data.get('slot', 0),
                protocol=protocol,
                program_id=self.protocol_configs[protocol]['program_id'],
                **instruction_data
            )
            
            return tx
            
        except Exception as e:
            print(f"Error parsing Helius transaction: {e}")
            return None
    
    def _parse_instruction_data(self, protocol: str, instruction: Dict) -> Dict[str, Any]:
        """Parse protocol-specific instruction data."""
        data = {}
        
        # This would contain protocol-specific parsing logic
        # For demonstration, return basic data
        
        if protocol == 'jupiter':
            data.update({
                'instruction_type': 'swap',
                'route_plan': [],  # Would parse the actual route
                'input_mint': self.token_mints.get('USDC'),
                'output_mint': self.token_mints.get('SOL')
            })
        elif protocol == 'raydium':
            data.update({
                'instruction_type': 'swap',
                'lp_token_mint': None
            })
        elif protocol == 'orca':
            data.update({
                'instruction_type': 'swap',
                'tick_lower': None,
                'tick_upper': None
            })
        
        return data
    
    def _is_in_date_range(self, timestamp: float) -> bool:
        """Check if timestamp is within collection date range."""
        start_ts = datetime.strptime(self.start_date, '%Y-%m-%d').timestamp()
        end_ts = datetime.strptime(self.end_date, '%Y-%m-%d').timestamp()
        
        return start_ts <= timestamp <= end_ts
    
    def _generate_demonstration_data(self) -> None:
        """Generate demonstration data for Jupiter Solana ecosystem."""
        print("Generating demonstration Jupiter Solana data...")
        
        # Generate more users due to Solana's high throughput
        num_users = 2000
        transactions_per_user = 15
        
        users = []
        transactions = []
        labels = {}
        
        for i in range(num_users):
            user_id = f"solana_user_{i:05d}"
            
            # Apply Jupiter's anti-farming criteria for labeling
            is_hunter = self._simulate_jupiter_hunter_behavior(i)
            
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
        
        # Apply Jupiter's anti-farming analysis if enabled
        if self.anti_farming_analysis:
            labels = self._apply_jupiter_anti_farming_criteria(users_df, transactions_df, labels)
        
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
                'eligible_addresses': len([u for u in labels.values() if u == 0]),
                'ineligible_addresses': len([u for u in labels.values() if u == 1]),
                'jupiter_criteria_applied': self.anti_farming_analysis
            }
        }
        
        dataset_file = os.path.join(self.data_path, f'jupiter_solana_{self.split}.json')
        os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
        
        with open(dataset_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Generated demonstration dataset with {len(transactions)} transactions")
        
        # Load the generated data
        self.load_raw_data()
    
    def _simulate_jupiter_hunter_behavior(self, user_index: int) -> bool:
        """Simulate whether a user exhibits Jupiter hunter behavior."""
        # Base probability of being a hunter
        base_hunter_prob = 0.15
        
        # Hunters are more likely to:
        # 1. Have exactly the minimum required interactions
        # 2. Focus on stablecoin pairs (despite penalties)
        # 3. Have activity clustering around snapshot dates
        
        return np.random.random() < base_hunter_prob
    
    def _generate_user_transactions(self, user_id: str, num_transactions: int, is_hunter: bool) -> List[SolanaDeFiTransaction]:
        """Generate synthetic transactions for a Solana user."""
        transactions = []
        
        # Time range for transactions
        start_time = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_time = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        # Adjust transaction count based on hunter behavior
        if is_hunter:
            # Hunters often aim for minimum qualifying interactions
            num_transactions = max(3, int(np.random.poisson(5)))
        else:
            # Legitimate users have more varied activity
            num_transactions = int(np.random.poisson(15))
        
        for i in range(num_transactions):
            # Generate timestamp
            if is_hunter:
                # Hunters cluster activity around Jupiter eligibility criteria
                # Focus on early period (before November 2, 2023 snapshot)
                snapshot_date = datetime(2023, 11, 2)
                if snapshot_date > start_time:
                    pre_snapshot_weight = 0.8
                    time_weight = np.random.beta(2, 1) * pre_snapshot_weight
                else:
                    time_weight = np.random.random()
            else:
                time_weight = np.random.random()
            
            timestamp = start_time + timedelta(
                seconds=time_weight * (end_time - start_time).total_seconds()
            )
            
            # Choose protocol (hunters favor Jupiter for airdrop eligibility)
            if is_hunter:
                protocol = np.random.choice(
                    self.protocols,
                    p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02]  # Heavily favor Jupiter
                )
            else:
                protocol = np.random.choice(self.protocols)
            
            # Generate transaction type and tokens
            tx_type = 'swap'  # Most common
            
            # Token selection (hunters favor stablecoin pairs despite penalties)
            if is_hunter and np.random.random() < 0.4:
                # Stablecoin farming (penalized by Jupiter)
                input_mint = self.token_mints['USDC']
                output_mint = self.token_mints['USDT']
            else:
                # More diverse token selection
                tokens = ['SOL', 'USDC', 'RAY', 'ORCA', 'MNDE']
                input_token = np.random.choice(tokens)
                output_token = np.random.choice([t for t in tokens if t != input_token])
                input_mint = self.token_mints[input_token]
                output_mint = self.token_mints[output_token]
            
            # Generate transaction value (lower due to Solana's low fees)
            if is_hunter:
                # Hunters often use minimum amounts to qualify
                value_usd = np.random.choice([10, 25, 50, 100]) * np.random.uniform(0.9, 1.1)
            else:
                value_usd = np.random.lognormal(3, 1.5)  # More natural distribution
            
            # Generate amounts
            if input_mint == self.token_mints['SOL']:
                in_amount = value_usd / 100  # Rough SOL price
                out_amount = value_usd
            else:
                in_amount = value_usd
                out_amount = value_usd / 100 if output_mint == self.token_mints['SOL'] else value_usd
            
            # Generate gas fee (very low on Solana)
            gas_fee = np.random.uniform(0.000005, 0.00001)  # 5-10 lamports
            
            # Generate route plan for Jupiter
            route_plan = []
            if protocol == 'jupiter':
                # Simulate multi-hop routing
                if np.random.random() < 0.3:  # 30% multi-hop
                    route_plan = [
                        {'swapInfo': {'label': 'Raydium'}},
                        {'swapInfo': {'label': 'Orca'}}
                    ]
            
            tx = SolanaDeFiTransaction(
                user_id=user_id,
                timestamp=timestamp.timestamp(),
                chain_id='solana',  # Required parameter
                transaction_type=tx_type,
                value_usd=value_usd,
                gas_fee=gas_fee,
                signature=f"solana_tx_{user_id}_{i}_{int(timestamp.timestamp())}",
                block_number=200000000 + i * 10,  # Rough Solana slot numbers
                protocol=protocol,
                program_id=self.protocol_configs[protocol]['program_id'],
                instruction_type='swap',
                route_plan=route_plan,
                input_mint=input_mint,
                output_mint=output_mint,
                in_amount=in_amount,
                out_amount=out_amount,
                compute_units_consumed=np.random.randint(10000, 50000),
                prioritization_fee=np.random.uniform(0, 0.001)
            )
            
            transactions.append(tx)
        
        return transactions
    
    def _apply_jupiter_anti_farming_criteria(self, users_df: pd.DataFrame, 
                                           transactions_df: pd.DataFrame, 
                                           initial_labels: Dict[str, int]) -> Dict[str, int]:
        """Apply Jupiter's anti-farming criteria to refine labels."""
        refined_labels = {}
        
        for user_id in users_df['user_id']:
            user_txs = transactions_df[transactions_df['user_id'] == user_id]
            
            # Apply Jupiter criteria
            meets_criteria = self._check_jupiter_eligibility(user_txs)
            
            if meets_criteria:
                refined_labels[user_id] = 0  # Eligible (legitimate)
            else:
                refined_labels[user_id] = 1  # Ineligible (hunter/insufficient activity)
        
        return refined_labels
    
    def _check_jupiter_eligibility(self, user_txs: pd.DataFrame) -> bool:
        """Check if user meets Jupiter airdrop eligibility criteria."""
        if len(user_txs) == 0:
            return False
        
        criteria = self.jupiter_anti_farming_criteria
        
        # Must have minimum interactions
        if len(user_txs) < criteria['min_interactions']:
            return False
        
        # Must meet volume threshold
        total_volume = user_txs['value_usd'].sum()
        if total_volume < criteria['volume_threshold']:
            return False
        
        # Must be active on minimum days
        unique_days = user_txs['timestamp'].apply(lambda x: datetime.fromtimestamp(x).date()).nunique()
        if unique_days < criteria['unique_days_threshold']:
            return False
        
        # Check stablecoin penalty
        stablecoin_txs = 0
        for _, tx in user_txs.iterrows():
            if self._is_stablecoin_pair(tx):
                stablecoin_txs += 1
        
        stablecoin_ratio = stablecoin_txs / len(user_txs)
        
        # Apply penalty for excessive stablecoin farming
        if stablecoin_ratio > 0.7:  # More than 70% stablecoin pairs
            adjusted_volume = total_volume * criteria['stablecoin_penalty']
            if adjusted_volume < criteria['volume_threshold']:
                return False
        
        return True
    
    def _is_stablecoin_pair(self, tx: pd.Series) -> bool:
        """Check if transaction involves penalized stablecoin pairs."""
        input_mint = tx.get('input_mint', '')
        output_mint = tx.get('output_mint', '')
        
        stablecoins = [
            self.token_mints['USDC'],
            self.token_mints['USDT']
        ]
        
        return input_mint in stablecoins and output_mint in stablecoins
    
    def _generate_users_from_transactions(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate users DataFrame from collected transactions."""
        users = transactions_df['user_id'].unique()
        users_df = pd.DataFrame({'user_id': users})
        return users_df
    
    def _generate_labels_from_jupiter_criteria(self, users_df: pd.DataFrame, 
                                             transactions_df: pd.DataFrame) -> Dict[str, int]:
        """Generate labels based on Jupiter eligibility criteria."""
        labels = {}
        
        for user_id in users_df['user_id']:
            user_txs = transactions_df[transactions_df['user_id'] == user_id]
            meets_criteria = self._check_jupiter_eligibility(user_txs)
            
            # 0 = eligible (legitimate), 1 = ineligible (hunter/insufficient)
            labels[user_id] = 0 if meets_criteria else 1
        
        return labels
    
    def load_raw_data(self) -> None:
        """Load raw Solana blockchain data from saved files."""
        dataset_file = os.path.join(self.data_path, f'jupiter_solana_{self.split}.json')
        
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
        
        print(f"Loaded Jupiter Solana dataset: {len(self.transactions)} transactions, {len(self.users)} users")
    
    def extract_defi_features(self, user_id: str) -> Dict[str, torch.Tensor]:
        """Extract Jupiter Solana-specific features for hunter detection."""
        user_txs = self.transactions[self.transactions['user_id'] == user_id]
        
        if len(user_txs) == 0:
            return {}
        
        # Cross-protocol features
        cross_protocol_features = self.extract_cross_protocol_features(user_id)
        
        # Solana-specific features
        solana_features = self._extract_solana_specific_features(user_txs)
        
        # Jupiter anti-farming features
        jupiter_features = self._extract_jupiter_anti_farming_features(user_txs)
        
        # Combine features
        return {**cross_protocol_features, **solana_features, **jupiter_features}
    
    def _extract_solana_specific_features(self, user_txs: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Extract features specific to Solana ecosystem."""
        features = {}
        
        # High-frequency trading patterns (enabled by fast blocks)
        if len(user_txs) > 1:
            timestamps = user_txs['timestamp'].sort_values()
            time_diffs = np.diff(timestamps)
            fast_tx_ratio = (time_diffs < 60).mean()  # Within 1 minute
            features['fast_transaction_ratio'] = torch.tensor(fast_tx_ratio, dtype=torch.float32)
        
        # Compute unit optimization patterns
        if 'compute_units_consumed' in user_txs.columns:
            compute_units = user_txs['compute_units_consumed'].fillna(30000)
            avg_compute_units = compute_units.mean()
            features['avg_compute_units'] = torch.tensor(np.log1p(avg_compute_units), dtype=torch.float32)
        
        # Priority fee patterns (MEV/bot behavior)
        if 'prioritization_fee' in user_txs.columns:
            priority_fees = user_txs['prioritization_fee'].fillna(0)
            high_priority_ratio = (priority_fees > 0.001).mean()
            features['high_priority_fee_ratio'] = torch.tensor(high_priority_ratio, dtype=torch.float32)
        
        # Multi-hop routing patterns (Jupiter specific)
        route_complexity = 0
        if 'route_plan' in user_txs.columns:
            for route_plan in user_txs['route_plan'].fillna('[]'):
                try:
                    if isinstance(route_plan, str):
                        route_data = json.loads(route_plan)
                    else:
                        route_data = route_plan
                    route_complexity += len(route_data)
                except:
                    continue
            
            avg_route_complexity = route_complexity / max(len(user_txs), 1)
            features['avg_route_complexity'] = torch.tensor(avg_route_complexity, dtype=torch.float32)
        
        return features
    
    def _extract_jupiter_anti_farming_features(self, user_txs: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Extract features based on Jupiter's anti-farming criteria."""
        features = {}
        
        # Stablecoin farming score
        stablecoin_txs = sum(1 for _, tx in user_txs.iterrows() if self._is_stablecoin_pair(tx))
        stablecoin_ratio = stablecoin_txs / max(len(user_txs), 1)
        features['stablecoin_farming_score'] = torch.tensor(stablecoin_ratio, dtype=torch.float32)
        
        # Activity day diversity
        unique_days = user_txs['timestamp'].apply(lambda x: datetime.fromtimestamp(x).date()).nunique()
        features['activity_day_diversity'] = torch.tensor(unique_days, dtype=torch.float32)
        
        # Volume concentration (hunters often use specific amounts)
        if len(user_txs) > 1:
            values = user_txs['value_usd']
            value_std = values.std()
            value_mean = values.mean()
            value_cv = value_std / max(value_mean, 1e-8)  # Coefficient of variation
            features['value_concentration'] = torch.tensor(1.0 / (1.0 + value_cv), dtype=torch.float32)
        
        # Eligibility score based on Jupiter criteria
        eligibility_score = 1.0 if self._check_jupiter_eligibility(user_txs) else 0.0
        features['jupiter_eligibility_score'] = torch.tensor(eligibility_score, dtype=torch.float32)
        
        return features
    
    def build_protocol_interaction_graph(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Build user graph based on Solana protocol interactions."""
        # Similar to Arbitrum but with Solana-specific relationships
        user_pairs = defaultdict(list)
        
        # Group users by protocol and token interactions
        for protocol in self.protocols:
            protocol_txs = self.transactions[self.transactions['protocol'] == protocol]
            
            # For DEX protocols, group by token pairs
            if protocol in ['jupiter', 'raydium', 'orca']:
                for input_mint in protocol_txs['input_mint'].unique():
                    for output_mint in protocol_txs['output_mint'].unique():
                        if input_mint != output_mint:
                            pair_txs = protocol_txs[
                                (protocol_txs['input_mint'] == input_mint) & 
                                (protocol_txs['output_mint'] == output_mint)
                            ]
                            pair_users = pair_txs['user_id'].unique()
                            
                            # Create edges between users trading the same pair
                            for i, user1 in enumerate(pair_users):
                                for user2 in pair_users[i+1:]:
                                    user_pairs[(user1, user2)].append(f"{protocol}_{input_mint}_{output_mint}")
        
        # Build edge index and features
        edges = []
        edge_features = []
        
        user_to_idx = {user: idx for idx, user in enumerate(self.users['user_id'])}
        
        for (user1, user2), shared_interactions in user_pairs.items():
            if user1 in user_to_idx and user2 in user_to_idx:
                edges.extend([(user_to_idx[user1], user_to_idx[user2]),
                            (user_to_idx[user2], user_to_idx[user1])])
                
                # Compute edge features
                edge_feature = self._compute_solana_edge_features(user1, user2, shared_interactions)
                edge_features.extend([edge_feature, edge_feature])
        
        if not edges:
            num_users = len(user_to_idx)
            edges = [(i, i) for i in range(num_users)]
            edge_features = [torch.zeros(64) for _ in range(num_users)]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_features = torch.stack(edge_features)
        
        return edge_index, {'edge_features': edge_features}
    
    def _compute_solana_edge_features(self, user1: str, user2: str, shared_interactions: List[str]) -> torch.Tensor:
        """Compute edge features based on shared Solana protocol interactions."""
        user1_txs = self.transactions[self.transactions['user_id'] == user1]
        user2_txs = self.transactions[self.transactions['user_id'] == user2]
        
        # Shared interaction features
        num_shared_interactions = len(shared_interactions)
        interaction_diversity = len(set(interaction.split('_')[0] for interaction in shared_interactions))
        
        # Temporal synchronization (important for coordinated farming)
        temporal_correlation = self._compute_temporal_correlation(user1_txs, user2_txs)
        
        # Similar transaction patterns
        pattern_similarity = self._compute_transaction_pattern_similarity(user1_txs, user2_txs)
        
        # Combine features
        features = torch.tensor([
            num_shared_interactions,
            interaction_diversity,
            temporal_correlation,
            pattern_similarity
        ], dtype=torch.float32)
        
        # Pad to standard size
        padded_features = torch.zeros(64)
        padded_features[:min(len(features), 64)] = features[:64]
        
        return padded_features
    
    def _compute_temporal_correlation(self, user1_txs: pd.DataFrame, user2_txs: pd.DataFrame) -> float:
        """Compute temporal correlation between users (coordinated activity indicator)."""
        if len(user1_txs) == 0 or len(user2_txs) == 0:
            return 0.0
        
        # Check for synchronized activity windows
        time_windows = []
        for timestamp in user1_txs['timestamp']:
            # Check if user2 has activity within 1 hour
            window_start = timestamp - 3600
            window_end = timestamp + 3600
            
            user2_in_window = user2_txs[
                (user2_txs['timestamp'] >= window_start) & 
                (user2_txs['timestamp'] <= window_end)
            ]
            
            if len(user2_in_window) > 0:
                time_windows.append(1)
            else:
                time_windows.append(0)
        
        return np.mean(time_windows) if time_windows else 0.0
    
    def _compute_transaction_pattern_similarity(self, user1_txs: pd.DataFrame, user2_txs: pd.DataFrame) -> float:
        """Compute similarity in transaction patterns."""
        if len(user1_txs) == 0 or len(user2_txs) == 0:
            return 0.0
        
        # Compare value distributions
        values1 = user1_txs['value_usd'].values
        values2 = user2_txs['value_usd'].values
        
        # Compute distribution similarity using Jensen-Shannon divergence
        try:
            hist1, _ = np.histogram(values1, bins=10, range=(1, 1000))
            hist2, _ = np.histogram(values2, bins=10, range=(1, 1000))
            
            # Normalize to probabilities
            p1 = hist1 / max(hist1.sum(), 1)
            p2 = hist2 / max(hist2.sum(), 1)
            
            # Jensen-Shannon divergence
            m = (p1 + p2) / 2
            js_div = 0.5 * np.sum(p1 * np.log(p1 / (m + 1e-8) + 1e-8)) + \
                     0.5 * np.sum(p2 * np.log(p2 / (m + 1e-8) + 1e-8))
            
            # Convert to similarity score
            similarity = 1.0 / (1.0 + js_div)
            
        except:
            similarity = 0.0
        
        return similarity