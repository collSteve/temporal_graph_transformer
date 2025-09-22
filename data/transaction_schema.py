"""
Unified transaction schema for cross-chain DeFi datasets.

Provides standardized transaction format across different blockchains
while preserving chain-specific features. Enables fair comparison
between Arbitrum, Solana, Optimism, and other supported chains.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import json


@dataclass
class BaseTransaction:
    """
    Base transaction schema shared across all blockchains.
    
    This ensures consistent features for cross-chain analysis
    while allowing chain-specific extensions.
    """
    user_id: str
    timestamp: float  # Unix timestamp
    chain_id: str  # 'arbitrum', 'solana', 'optimism', etc.
    transaction_type: str  # swap, add_liquidity, lend, etc.
    value_usd: float  # Standardized USD value
    gas_fee: float  # In native token (ETH, SOL, etc.)
    signature: str  # Transaction hash/signature
    block_number: Optional[int] = None
    transaction_index: Optional[int] = None
    protocol: Optional[str] = None  # uniswap_v3, gmx, jupiter, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'user_id': self.user_id,
            'timestamp': self.timestamp,
            'chain_id': self.chain_id,
            'transaction_type': self.transaction_type,
            'value_usd': self.value_usd,
            'gas_fee': self.gas_fee,
            'signature': self.signature,
            'block_number': self.block_number,
            'transaction_index': self.transaction_index,
            'protocol': self.protocol
        }


@dataclass
class ArbitrumDeFiTransaction(BaseTransaction):
    """
    Arbitrum-specific DeFi transaction with L2 and protocol-specific features.
    
    Supports major protocols: Uniswap V3, GMX, Camelot, SushiSwap, Aave
    """
    
    # DEX-specific fields (Uniswap V3, Camelot, SushiSwap)
    token_in: Optional[str] = None  # Token address or symbol
    token_out: Optional[str] = None
    amount_in: Optional[float] = None
    amount_out: Optional[float] = None
    price_impact: Optional[float] = None  # Percentage
    slippage: Optional[float] = None  # Percentage
    pool_address: Optional[str] = None
    
    # GMX-specific fields (derivatives/perpetuals)
    position_size: Optional[float] = None
    leverage: Optional[float] = None
    is_long: Optional[bool] = None
    collateral_token: Optional[str] = None
    index_token: Optional[str] = None
    
    # Lending-specific fields (Aave)
    asset_address: Optional[str] = None
    amount: Optional[float] = None
    interest_rate_mode: Optional[int] = None  # 1=stable, 2=variable
    
    # L2-specific fields
    l1_gas_used: Optional[float] = None  # L1 calldata cost
    l2_gas_used: Optional[float] = None  # L2 execution cost
    
    def __post_init__(self):
        """Set chain_id to arbitrum after initialization."""
        object.__setattr__(self, 'chain_id', 'arbitrum')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with Arbitrum-specific fields."""
        base_dict = super().to_dict()
        arbitrum_dict = {
            'token_in': self.token_in,
            'token_out': self.token_out,
            'amount_in': self.amount_in,
            'amount_out': self.amount_out,
            'price_impact': self.price_impact,
            'slippage': self.slippage,
            'pool_address': self.pool_address,
            'position_size': self.position_size,
            'leverage': self.leverage,
            'is_long': self.is_long,
            'collateral_token': self.collateral_token,
            'index_token': self.index_token,
            'asset_address': self.asset_address,
            'amount': self.amount,
            'interest_rate_mode': self.interest_rate_mode,
            'l1_gas_used': self.l1_gas_used,
            'l2_gas_used': self.l2_gas_used
        }
        return {**base_dict, **arbitrum_dict}


@dataclass
class SolanaDeFiTransaction(BaseTransaction):
    """
    Solana-specific DeFi transaction with high-frequency, low-cost features.
    
    Supports major protocols: Jupiter, Raydium, Orca, Drift, Kamino, Marinade
    """
    
    # Program-specific fields
    program_id: Optional[str] = None  # Solana program address
    instruction_type: Optional[str] = None  # Program-specific instruction
    
    # Jupiter aggregator fields
    route_plan: Optional[List[Dict]] = field(default_factory=list)  # Route through multiple DEXs
    input_mint: Optional[str] = None  # Token mint addresses
    output_mint: Optional[str] = None
    in_amount: Optional[float] = None
    out_amount: Optional[float] = None
    route_label: Optional[str] = None  # Jupiter route summary
    
    # Raydium/Orca LP fields
    lp_token_mint: Optional[str] = None
    base_token_amount: Optional[float] = None
    quote_token_amount: Optional[float] = None
    
    # Concentrated liquidity fields (Orca Whirlpools)
    tick_lower: Optional[int] = None
    tick_upper: Optional[int] = None
    liquidity_amount: Optional[float] = None
    
    # Drift/Mango derivatives fields
    market_index: Optional[int] = None
    order_type: Optional[str] = None
    side: Optional[str] = None  # long/short
    
    # Solana-specific fields
    compute_units_consumed: Optional[int] = None
    prioritization_fee: Optional[float] = None  # Additional fee for priority
    
    def __post_init__(self):
        """Set chain_id to solana after initialization."""
        object.__setattr__(self, 'chain_id', 'solana')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with Solana-specific fields."""
        base_dict = super().to_dict()
        solana_dict = {
            'program_id': self.program_id,
            'instruction_type': self.instruction_type,
            'route_plan': json.dumps(self.route_plan) if self.route_plan else None,
            'input_mint': self.input_mint,
            'output_mint': self.output_mint,
            'in_amount': self.in_amount,
            'out_amount': self.out_amount,
            'route_label': self.route_label,
            'lp_token_mint': self.lp_token_mint,
            'base_token_amount': self.base_token_amount,
            'quote_token_amount': self.quote_token_amount,
            'tick_lower': self.tick_lower,
            'tick_upper': self.tick_upper,
            'liquidity_amount': self.liquidity_amount,
            'market_index': self.market_index,
            'order_type': self.order_type,
            'side': self.side,
            'compute_units_consumed': self.compute_units_consumed,
            'prioritization_fee': self.prioritization_fee
        }
        return {**base_dict, **solana_dict}


@dataclass
class OptimismDeFiTransaction(BaseTransaction):
    """
    Optimism-specific DeFi transaction with L2 rollup features.
    
    Supports protocols: Uniswap V3, Synthetix, Aave, Velodrome, etc.
    """
    
    # Similar to Arbitrum but with Optimism-specific protocols
    token_in: Optional[str] = None
    token_out: Optional[str] = None
    amount_in: Optional[float] = None
    amount_out: Optional[float] = None
    price_impact: Optional[float] = None
    slippage: Optional[float] = None
    
    # Synthetix-specific fields
    synth_key: Optional[str] = None  # sUSD, sETH, etc.
    exchange_rate: Optional[float] = None
    
    # Velodrome-specific fields (ve(3,3) DEX)
    gauge_address: Optional[str] = None
    voting_power: Optional[float] = None
    bribes_claimed: Optional[float] = None
    
    # Optimism L2-specific
    l1_data_fee: Optional[float] = None  # L1 data publishing cost
    l2_execution_fee: Optional[float] = None  # L2 computation cost
    
    def __post_init__(self):
        """Set chain_id to optimism after initialization."""
        object.__setattr__(self, 'chain_id', 'optimism')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with Optimism-specific fields."""
        base_dict = super().to_dict()
        optimism_dict = {
            'token_in': self.token_in,
            'token_out': self.token_out,
            'amount_in': self.amount_in,
            'amount_out': self.amount_out,
            'price_impact': self.price_impact,
            'slippage': self.slippage,
            'synth_key': self.synth_key,
            'exchange_rate': self.exchange_rate,
            'gauge_address': self.gauge_address,
            'voting_power': self.voting_power,
            'bribes_claimed': self.bribes_claimed,
            'l1_data_fee': self.l1_data_fee,
            'l2_execution_fee': self.l2_execution_fee
        }
        return {**base_dict, **optimism_dict}


class TransactionSchemaValidator:
    """
    Validates and standardizes transactions across different blockchain schemas.
    
    Ensures data quality and consistency for cross-chain analysis.
    """
    
    def __init__(self):
        self.required_base_fields = [
            'user_id', 'timestamp', 'chain_id', 'transaction_type', 
            'value_usd', 'gas_fee', 'signature'
        ]
        
        self.valid_transaction_types = {
            'swap', 'add_liquidity', 'remove_liquidity', 'lend', 'borrow', 
            'repay', 'stake', 'unstake', 'claim_rewards', 'liquidation', 
            'flashloan', 'bridge', 'mint', 'burn'
        }
        
        self.valid_chains = {
            'arbitrum', 'solana', 'optimism', 'ethereum', 'polygon', 'avalanche'
        }
    
    def validate_transaction(self, transaction: BaseTransaction) -> bool:
        """Validate a single transaction against schema requirements."""
        # Check required fields
        if not transaction.user_id or not transaction.signature:
            return False
        
        if transaction.timestamp <= 0:
            return False
        
        if transaction.value_usd < 0:
            return False
        
        if transaction.gas_fee < 0:
            return False
        
        if transaction.chain_id not in self.valid_chains:
            return False
        
        if transaction.transaction_type not in self.valid_transaction_types:
            return False
        
        return True
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate a DataFrame of transactions."""
        errors = []
        
        # Check required columns
        missing_cols = set(self.required_base_fields) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check for null values in required fields
        for col in self.required_base_fields:
            if col in df.columns and df[col].isna().any():
                errors.append(f"Found null values in required column: {col}")
        
        # Check value constraints
        if 'value_usd' in df.columns and (df['value_usd'] < 0).any():
            errors.append("Found negative values in value_usd")
        
        if 'gas_fee' in df.columns and (df['gas_fee'] < 0).any():
            errors.append("Found negative values in gas_fee")
        
        if 'timestamp' in df.columns and (df['timestamp'] <= 0).any():
            errors.append("Found invalid timestamps")
        
        # Check enum constraints
        if 'chain_id' in df.columns:
            invalid_chains = set(df['chain_id'].unique()) - self.valid_chains
            if invalid_chains:
                errors.append(f"Found invalid chain_ids: {invalid_chains}")
        
        if 'transaction_type' in df.columns:
            invalid_types = set(df['transaction_type'].unique()) - self.valid_transaction_types
            if invalid_types:
                errors.append(f"Found invalid transaction_types: {invalid_types}")
        
        return len(errors) == 0, errors
    
    def standardize_dataframe(self, df: pd.DataFrame, chain_id: str) -> pd.DataFrame:
        """Standardize a DataFrame to the unified schema."""
        standardized = df.copy()
        
        # Ensure chain_id is set
        if 'chain_id' not in standardized.columns or standardized['chain_id'].isna().any():
            standardized['chain_id'] = chain_id
        
        # Convert timestamps to float if needed
        if 'timestamp' in standardized.columns:
            if standardized['timestamp'].dtype == 'object':
                # Try to parse datetime strings
                standardized['timestamp'] = pd.to_datetime(standardized['timestamp']).astype(int) / 10**9
        
        # Fill missing optional fields
        optional_fields = ['block_number', 'transaction_index', 'protocol']
        for field in optional_fields:
            if field not in standardized.columns:
                standardized[field] = None
        
        # Clean and validate transaction types
        if 'transaction_type' in standardized.columns:
            # Map common variations to standard types
            type_mapping = {
                'trade': 'swap',
                'deposit': 'add_liquidity',
                'withdraw': 'remove_liquidity',
                'supply': 'lend',
                'withdraw_collateral': 'remove_liquidity'
            }
            
            standardized['transaction_type'] = standardized['transaction_type'].replace(type_mapping)
        
        return standardized


def create_transaction_from_dict(data: Dict[str, Any]) -> BaseTransaction:
    """Create appropriate transaction object from dictionary data."""
    chain_id = data.get('chain_id', '').lower()
    
    if chain_id == 'arbitrum':
        return ArbitrumDeFiTransaction(**data)
    elif chain_id == 'solana':
        return SolanaDeFiTransaction(**data)
    elif chain_id == 'optimism':
        return OptimismDeFiTransaction(**data)
    else:
        return BaseTransaction(**data)


def transactions_to_dataframe(transactions: List[BaseTransaction]) -> pd.DataFrame:
    """Convert list of transaction objects to standardized DataFrame."""
    if not transactions:
        return pd.DataFrame()
    
    # Convert all transactions to dictionaries
    tx_dicts = [tx.to_dict() for tx in transactions]
    
    # Create DataFrame
    df = pd.DataFrame(tx_dicts)
    
    # Ensure consistent column order
    base_columns = [
        'user_id', 'timestamp', 'chain_id', 'transaction_type', 'value_usd', 
        'gas_fee', 'signature', 'block_number', 'transaction_index', 'protocol'
    ]
    
    # Reorder columns (base first, then chain-specific)
    available_base_cols = [col for col in base_columns if col in df.columns]
    chain_specific_cols = [col for col in df.columns if col not in base_columns]
    
    df = df[available_base_cols + chain_specific_cols]
    
    return df