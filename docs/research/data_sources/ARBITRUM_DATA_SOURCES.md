# Arbitrum ARB Airdrop Data Sources & Implementation Plan

## 🎯 Executive Summary

The Arbitrum $ARB airdrop (March 23, 2023) represents our highest-priority target for airdrop hunter detection research. This document provides comprehensive data sources, API endpoints, and implementation strategies for collecting transaction data around this major airdrop event.

---

## 📊 Airdrop Event Details

### **Core Event Information**
- **Date**: March 23, 2023
- **Total Distribution**: 1.162 billion ARB tokens (11.62% of total supply)
- **Eligible Addresses**: 625,143 wallet addresses (~28% of Arbitrum users)
- **Snapshot Date**: February 6, 2023
- **Criteria**: Minimum 3 of 6 qualifying actions required

### **Hunter Activity Identified**
- **Major Consolidation**: 1,496 wallets → 2 wallets (~$3.3M worth)
- **Sybil Communities**: ~4,000 communities controlling 150,000 addresses
- **Farming Scale**: 253M tokens (21% of user allocation) claimed by sybils
- **Specific Hunter Addresses**:
  - `0xe1e271a26a42d00731caf4c7ab8ed1684510ab6e`: 2.1M ARB from 1,200+ addresses
  - `0x770edb43ecc5bcbe6f7088e1049fc42b2d1b195c`: 1.19M ARB from 1,375 addresses

---

## 🔗 Primary Data Sources

### **1. The Graph Protocol Subgraphs**

**Arbitrum Integration**: Active since 2021, comprehensive subgraph support
**Key Endpoints**:
```
Arbitrum Hosted Service: https://api.thegraph.com/subgraphs/name/[subgraph-name]
Arbitrum Gateway: https://gateway.thegraph.com/api/[api-key]/subgraphs/id/[subgraph-id]
```

**Available Subgraphs**:
- **Uniswap V3 Arbitrum**: Trading pairs, liquidity, volume data
- **GMX Protocol**: Perpetuals, spot trading, leverage positions
- **Camelot DEX**: Native Arbitrum DEX transactions
- **SushiSwap**: Cross-chain DEX data on Arbitrum

### **2. RPC Providers**

**Alchemy**:
- Endpoint: `https://arb-mainnet.g.alchemy.com/v2/[API-KEY]`
- Rate Limits: 300M requests/month (paid plans)
- Features: Archive nodes, enhanced APIs, NFT support

**Infura**:
- Endpoint: `https://arbitrum-mainnet.infura.io/v3/[PROJECT-ID]`
- Rate Limits: 100K requests/day (free), unlimited (paid)
- Features: Archive access, WebSocket support

**QuickNode**:
- Endpoint: `https://[subdomain].arbitrum-mainnet.quiknode.pro/[token]/`
- Rate Limits: Variable by plan
- Features: Archive nodes, GraphQL, trace APIs

### **3. Blockchain Explorers**

**Arbiscan**:
- API: `https://api.arbiscan.io/api`
- Features: Transaction history, contract interactions, token transfers
- Rate Limits: 5 calls/second (free), higher for paid

**Arbitrum Block Explorer**:
- Direct blockchain data access
- Real-time transaction monitoring
- Contract verification and source code

---

## 🏪 DeFi Protocol Data Sources

### **GMX (Perpetuals & Spot)**
- **Contract Address**: `0x489ee077994B6658eAfA855C308275EAd8097C4A`
- **TVL**: ~$450M (V2) + ~$100M (V1)
- **Data Types**: Leverage positions, liquidations, fees, volume
- **Subgraph**: Available via The Graph Protocol
- **API**: Direct contract event monitoring

### **Camelot DEX (Native Arbitrum)**
- **Website**: `https://app.camelot.exchange/`
- **Features**: Ecosystem-native DEX, flexible liquidity
- **Token**: $GRAIL governance token
- **Data Types**: Swaps, liquidity provision, farming rewards
- **Access**: Smart contract events, potential API

### **Uniswap V3 (Cross-chain)**
- **Subgraph ID**: `DiYPVdygkfjDWhbxGSqAQxwBKmfKnkWQojqeM2rkLb3G`
- **Endpoint**: `https://gateway.thegraph.com/api/[KEY]/subgraphs/id/[ID]`
- **Data Types**: Swaps, pools, positions, fees, volume
- **Coverage**: All major trading pairs on Arbitrum

### **SushiSwap**
- **Deployment**: Multi-chain DEX on Arbitrum
- **Data Types**: Traditional AMM trading data
- **Access**: Subgraph via The Graph Protocol

---

## 📈 Data Collection Strategy

### **Phase 1: Historical Data Collection (Pre-Airdrop)**
**Time Range**: January 1, 2023 - March 22, 2023
**Focus**: Establish baseline behavior patterns

**Key Metrics**:
- Transaction frequency per address
- DEX interaction patterns
- Gas optimization strategies
- Cross-protocol usage
- Liquidity provision behavior

### **Phase 2: Airdrop Event Period**
**Time Range**: March 23, 2023 (24-48 hours)
**Focus**: Capture claiming behavior and immediate activity

**Key Metrics**:
- Claim transaction timing
- Immediate sell/hold behavior
- Multi-wallet coordination
- Gas price sensitivity

### **Phase 3: Post-Airdrop Analysis (Hunter Detection)**
**Time Range**: March 24, 2023 - June 2023
**Focus**: Identify farming patterns and consolidation

**Key Metrics**:
- Token consolidation patterns
- Address clustering analysis
- Cross-protocol farming continuation
- Behavioral pattern changes

---

## 🛠️ Technical Implementation

### **Recommended Tech Stack**

1. **Primary Data Source**: The Graph Protocol
   - Cost-effective for complex queries
   - Rich DEX trading data
   - Real-time indexing

2. **Backup RPC**: Alchemy or QuickNode
   - Direct blockchain access
   - Archive node support
   - Enhanced APIs

3. **Data Storage**: TimescaleDB
   - Time-series optimization
   - PostgreSQL compatibility
   - Efficient querying

4. **Processing**: Python + pandas
   - Rich data manipulation
   - Blockchain library support
   - ML/AI integration

### **Sample GraphQL Queries**

**Uniswap Trading Data**:
```graphql
{
  swaps(
    first: 1000
    where: {
      timestamp_gte: "1679529600"  # March 23, 2023
      timestamp_lte: "1679616000"  # March 24, 2023
    }
    orderBy: timestamp
    orderDirection: asc
  ) {
    id
    transaction {
      id
      blockNumber
      timestamp
    }
    sender
    recipient
    amount0
    amount1
    token0 {
      symbol
      decimals
    }
    token1 {
      symbol
      decimals
    }
  }
}
```

**GMX Position Data**:
```graphql
{
  positions(
    first: 1000
    where: {
      createdTimestamp_gte: "1679529600"
    }
  ) {
    id
    account
    collateralToken {
      symbol
    }
    indexToken {
      symbol
    }
    size
    collateral
    isLong
    createdTimestamp
  }
}
```

---

## 💰 Cost Analysis

### **API Costs (Monthly)**
- **The Graph Protocol**: $0-200 (generous free tier)
- **Alchemy**: $199-999 (comprehensive features)
- **Infura**: $50-1000 (standard RPC access)
- **QuickNode**: $9-299 (flexible pricing)

### **Recommended Budget**: $200-500/month
- Primary: The Graph Protocol + Alchemy
- Comprehensive data access
- Archive node support
- High rate limits

---

## 📋 Ground Truth Labeling Strategy

### **Confirmed Hunter Addresses** (From Public Analysis)

1. **Major Consolidators**:
   - `0xe1e271a26a42d00731caf4c7ab8ed1684510ab6e`
   - `0x770edb43ecc5bcbe6f7088e1049fc42b2d1b195c`

2. **Sybil Communities**: 4,000+ communities identified by X-explore

3. **Hop Protocol Blacklist**: Addresses disqualified during Hop bounty

### **Legitimate User Sampling**
- Random sampling from eligible addresses
- Exclude known hunter addresses
- Focus on consistent, organic usage patterns
- Validate with community reputation data

### **Labeling Criteria**
- **Hunter (1)**: Multi-wallet coordination, rapid consolidation, farming patterns
- **Legitimate (0)**: Organic usage, consistent behavior, single-wallet activity

---

## 🚀 Implementation Timeline

### **Week 1: Setup & Testing**
- Configure The Graph API access
- Set up Alchemy/QuickNode accounts
- Test rate limits with sample queries
- Implement basic data collection pipeline

### **Week 2: Historical Data Collection**
- Collect pre-airdrop data (Jan-March 2023)
- Focus on top DEX protocols (Uniswap, GMX, Camelot)
- Implement data validation and cleaning

### **Week 3: Airdrop Event Data**
- Collect airdrop claiming transactions
- Map known hunter addresses
- Analyze immediate post-airdrop behavior

### **Week 4: Data Processing & Analysis**
- Graph construction from transaction data
- Feature engineering for temporal patterns
- Initial hunter vs legitimate classification

---

## 🎯 Success Metrics

1. **Data Volume**: 10M+ transactions collected
2. **Address Coverage**: 100K+ unique addresses
3. **Hunter Identification**: 1,000+ confirmed hunter addresses
4. **Temporal Resolution**: Transaction-level timing data
5. **Protocol Coverage**: 5+ major DeFi protocols

This Arbitrum dataset will serve as our primary validation case for demonstrating superior performance over ARTEMIS baseline methods.