# Solana DeFi Data Sources & Implementation Plan

## 🎯 Executive Summary

Solana represents our second-highest priority market due to its high transaction volume, low costs, and rich airdrop ecosystem. This document provides comprehensive data sources, API endpoints, and implementation strategies for collecting DeFi transaction data, particularly around Jupiter and other major airdrops.

---

## 📊 Major Airdrop Events

### **Jupiter (JUP) Airdrop - January 31, 2024**
- **Distribution**: 1 billion JUP tokens to 955,000+ wallets
- **Eligibility**: Platform interaction before November 2, 2023
- **Total Value**: ~$616M at peak prices
- **Unique Features**: 
  - Sophisticated scoring system to prevent farming
  - "Jupuary" annual airdrops (2025, 2026 confirmed)
  - Anti-bot measures and stablecoin pair scoring penalties

### **Other Major Solana Airdrops**
- **Kamino Finance**: Lending/borrowing protocol airdrop
- **Drift Protocol**: Perpetuals trading platform
- **Marinade**: Liquid staking protocol
- **Raydium**: DEX and AMM protocol
- **Orca**: Automated market maker

---

## 🔗 Primary Data Sources

### **1. Jupiter API (Primary DEX Aggregator)**

**Core Information**:
- **Program Address**: `JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4`
- **Official API**: `https://www.jupiterapi.com/`
- **TVL**: $2.5 billion
- **Volume**: $93 billion in spot trading (as of Nov 2024)

**API Features**:
- Real-time DEX aggregation across all Solana DEXs
- Route optimization for best price discovery
- Transaction serialization for direct blockchain submission
- WebSocket streaming for real-time swap monitoring

**Data Types**:
- Swap transactions with route details
- Token pair prices and liquidity
- Slippage and fee analysis
- Multi-DEX arbitrage opportunities

### **2. Solana RPC Providers**

#### **Helius (Recommended for Solana)**
- **Endpoint**: `https://mainnet.helius-rpc.com/?api-key=[KEY]`
- **Pricing**: $49/month (Developer - 10M credits), $755/month (Growth)
- **Features**: 
  - 1 RPC call = 1 credit (no multipliers)
  - Enhanced transaction APIs
  - Priority fee computation
  - Transaction decoding
- **Specialization**: Solana-native provider with superior performance

#### **QuickNode** 
- **Endpoint**: `https://[subdomain].solana-mainnet.quiknode.pro/[token]/`
- **Pricing**: $9-499/month base + usage
- **Features**: Jupiter API integration via Metis plugin
- **Rate Limits**: Variable by plan

#### **Alchemy**
- **Endpoint**: `https://solana-mainnet.g.alchemy.com/v2/[API-KEY]`
- **Pricing**: Free tier (12M transactions), $299+ for Scale
- **Features**: Enhanced APIs, NFT support, 99.9% uptime SLA

### **3. Specialized Solana APIs**

#### **Bitquery**
- **Features**: Fastest Solana WebSocket API
- **Data Types**: Real-time DEX trades, NFT transfers, instructions
- **Coverage**: Raydium, Jupiter, Orca, Phoenix
- **Benefits**: Zero delay real-time streaming

#### **Solscan API**
- **Features**: Transaction parsing and analytics
- **Data Types**: Account information, transaction history
- **Coverage**: Complete Solana blockchain data

---

## 🏪 DeFi Protocol Coverage

### **Jupiter DEX Aggregator**
- **Program**: `JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4`
- **Instructions**: `sharedAccountsRoute` for swaps
- **Data Access**: Real-time via Jupiter API + RPC calls
- **Advantages**: Aggregates all major DEXs, best price discovery

### **Raydium (AMM + DEX)**
- **Program**: `675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8`
- **Data Types**: LP positions, farming rewards, trading pairs
- **Access**: Direct RPC + Bitquery APIs

### **Orca (Concentrated Liquidity)**
- **Program**: `9W959DqEETiGZocYWCQPaJ6sD9kmkKGSqrVUCL6zN1D`
- **Data Types**: Whirlpool positions, concentrated liquidity
- **Access**: Direct RPC + specialized APIs

### **Phoenix (Orderbook DEX)**
- **Program**: `PhoeNiXZ8ByJGLkxNfZRnkUfjvmuYqLR89jjFHGqdXY`
- **Data Types**: Limit orders, market making, orderbook data
- **Access**: Direct RPC monitoring

### **Serum (Legacy but Historical)**
- **Program**: `9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin`
- **Data Types**: Historical orderbook trading
- **Access**: Archive RPC nodes

---

## 📈 Data Collection Strategy

### **Phase 1: Jupiter Airdrop Analysis**
**Time Range**: October 2023 - February 2024
**Focus**: Pre-airdrop farming detection

**Key Metrics**:
- Swap frequency and volume patterns
- Multi-token pair interactions
- Stablecoin farming detection (scored lower in Jupiter)
- Cross-DEX arbitrage activity
- Wallet clustering analysis

### **Phase 2: Multi-Protocol Farming Detection**
**Protocols**: Jupiter, Raydium, Orca, Drift, Kamino
**Focus**: Cross-protocol farming patterns

**Key Metrics**:
- Protocol interaction timing
- Liquidity provision patterns
- Yield farming optimization
- Gas efficiency strategies
- Multi-wallet coordination

### **Phase 3: Real-time Hunter Detection**
**Implementation**: Live monitoring system
**Focus**: Active airdrop farming identification

**Key Metrics**:
- New protocol interaction patterns
- Sudden volume increases
- Coordinated multi-wallet activity
- Unusual transaction timing

---

## 🛠️ Technical Implementation

### **Recommended Tech Stack**

1. **Primary Data Source**: Jupiter API + Helius RPC
   - Real-time swap monitoring
   - Complete transaction history
   - Solana-optimized performance

2. **Real-time Streaming**: Bitquery WebSocket
   - Zero-delay transaction monitoring
   - Multi-DEX coverage
   - Event-driven processing

3. **Archive Data**: Helius Enhanced APIs
   - Historical transaction parsing
   - Account activity tracking
   - Priority fee analysis

### **Sample API Implementations**

#### **Jupiter Swap Monitoring**
```python
import requests
import websocket

# Monitor Jupiter swaps in real-time
def monitor_jupiter_swaps():
    # Jupiter API for current data
    api_url = "https://www.jupiterapi.com/"
    
    # Bitquery WebSocket for real-time monitoring
    ws_url = "wss://streaming.bitquery.io/graphql"
    
    query = """
    subscription {
      Solana(network: solana) {
        Instructions(
          where: {
            Program: {Address: {is: "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4"}}
            Instruction: {Name: {is: "sharedAccountsRoute"}}
          }
        ) {
          Transaction {
            Signature
            Block {
              Time
            }
          }
          Program {
            Address
          }
          Accounts {
            Address
            Token {
              Symbol
              Decimals
            }
          }
        }
      }
    }
    """
```

#### **Multi-DEX Data Collection**
```python
# Collect data from multiple Solana DEXs
def collect_solana_dex_data(start_date, end_date):
    protocols = [
        "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",  # Jupiter
        "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",   # Raydium
        "9W959DqEETiGZocYWCQPaJ6sD9kmkKGSqrVUCL6zN1D",   # Orca
        "PhoeNiXZ8ByJGLkxNfZRnkUfjvmuYqLR89jjFHGqdXY"   # Phoenix
    ]
    
    for program in protocols:
        # Use Helius enhanced APIs for transaction parsing
        helius_url = f"https://api.helius.xyz/v0/transactions"
        
        params = {
            "api-key": "YOUR_API_KEY",
            "account": program,
            "type": "SWAP",
            "startTime": start_date,
            "endTime": end_date
        }
```

---

## 💰 Cost Analysis

### **Monthly Budget Recommendations**

#### **Development/Testing ($50-100/month)**
- **Helius Developer**: $49/month (10M credits)
- **Jupiter API**: Free tier sufficient
- **Bitquery**: Free tier for basic streaming

#### **Research Production ($200-400/month)**
- **Helius Growth**: Custom pricing for high volume
- **Helius Enhanced APIs**: Transaction parsing features
- **Archive data access**: Historical analysis

#### **Cost Comparison**
| Provider | Monthly Cost | Credits/Calls | Best For |
|----------|-------------|---------------|----------|
| **Helius** | $49-755 | 10M-∞ | Solana-specific features |
| **QuickNode** | $9-941 | Variable | Jupiter integration |
| **Alchemy** | $0-299+ | 12M-∞ | Multi-chain compatibility |
| **Bitquery** | $0-149 | 100K queries | Real-time streaming |

---

## 📋 Ground Truth Labeling

### **Jupiter Airdrop Hunter Patterns**
1. **Stablecoin Farming**: High stablecoin pair trading (scored lower)
2. **Volume Manipulation**: Large swaps just to meet thresholds
3. **Multi-wallet Coordination**: Synchronized activities across wallets
4. **Timing Exploitation**: Activity spike near snapshot dates

### **Legitimate User Indicators**
1. **Consistent Usage**: Regular trading over extended periods
2. **Diverse Interactions**: Multiple protocol engagement
3. **Organic Patterns**: Natural trading frequencies
4. **Real Value**: Meaningful transaction amounts

### **Data Sources for Labels**
- Jupiter's anti-bot scoring system results
- Community-identified farming addresses
- Cross-protocol farming patterns
- Wallet clustering analysis

---

## 🚀 Implementation Timeline

### **Week 1: Infrastructure Setup**
- Configure Helius RPC access
- Set up Jupiter API integration
- Test Bitquery WebSocket streaming
- Implement basic data collection

### **Week 2: Historical Data Collection**
- Collect Jupiter airdrop period data (Oct 2023 - Feb 2024)
- Analyze pre-airdrop farming patterns
- Map known farming addresses

### **Week 3: Multi-Protocol Integration**
- Add Raydium, Orca, Phoenix data
- Implement cross-protocol analysis
- Build wallet clustering algorithms

### **Week 4: Real-time System**
- Deploy live monitoring infrastructure
- Implement farming detection algorithms
- Create ground truth validation system

---

## 🎯 Success Metrics

1. **Data Volume**: 50M+ DeFi transactions
2. **Protocol Coverage**: 5+ major Solana DEXs
3. **Temporal Resolution**: ~400ms block time precision
4. **Hunter Detection**: 10,000+ farming addresses identified
5. **Real-time Capability**: <1 second detection latency

### **Advantages Over Other Chains**
- **High Frequency**: 400ms blocks = detailed temporal patterns
- **Low Costs**: $0.0001 per transaction = high volume farming
- **Rich Ecosystem**: Multiple protocols = diverse farming strategies
- **API Quality**: Solana-native providers = superior data access

This Solana dataset will complement our Arbitrum data to demonstrate cross-chain generalization of our temporal graph transformer approach.