# 🪐 PAXIS Agent — Autonomous Quant & Multimodal Forex Trading System

> An autonomous Forex & Gold ($XAUUSD$) trading agent built on a **Deterministic-First, LLM-Last** architecture: Smart Money Concepts (SMC) quantitative engine, 4-Timeframe Analysis (4H/1H/15M/1M), Market Regime Detector, 18-Point Pre-Trade Validator, Multi-Factor Confluence Engine, Order Tracker with 5-Second Active Risk Management, and Causal Bar-by-Bar Backtesting Engine.

---

## 📖 Table of Contents
1. [Core Features](#-core-features)
2. [System Architecture (Pro Trader v2)](#-system-architecture-pro-trader-v2)
3. [6-Stage Pro Trader Pipeline](#-6-stage-pro-trader-pipeline)
4. [SMC Deep Zone Protection & Zone Context](#-smc-deep-zone-protection--zone-context)
5. [Backtesting & Historical Performance](#-backtesting--historical-performance)
6. [Tech Stack](#-tech-stack)
7. [Installation & Setup](#-installation--setup)
8. [Configuration (.env)](#-configuration-env)
9. [Running Live, Dry-Run & Backtester](#-running-live-dry-run--backtester)
10. [Remote MT5 Bridge Integration](#-remote-metatrader-5-mt5-bridge-integration)
11. [Unit Testing](#-unit-testing)

---

## ⚡ Core Features

- **⚡ Deterministic-First Architecture**: Over 85% of invalid setups are filtered out in sub-millisecond execution by mathematical SMC rules, eliminating price hallucinations and saving computational overhead.
- **📊 4-Timeframe SMC Engine**: Synchronizes 4H Macro Trend, 1H Intermediate Swing, 15M POI Setup, and 1M Micro Entry structure (Order Blocks, FVGs, Inducement Sweeps, CHoCH/BOS breaks, and 50% Equilibrium zones).
- **🛡️ SMC Deep Zone Protection**: Prevents high-risk entries at extremes (blocks `SHORT` below 25% of swing range in Deep Discount and blocks `LONG` above 75% in Deep Premium) directly at the Strategy Selection level.
- **🎯 7-Type Market Regime Classifier**: Identifies Trending, Volatile, Exhaustion, and Compression market regimes using ADX, Bollinger Band squeezes, and alignment matrices.
- **✅ 18-Point Pre-Trade Validator**: Enforces strict mandatory safety checks covering spread thresholds, reward-to-risk ratio ($\ge 1.5:1$), session windows, news blockouts, and daily drawdown limits.
- **⏱️ 5-Second Active Risk Management**: `OrderTracker` thread continuously monitors active positions every 5 seconds for breakeven ratchets, progressive profit-locking, and supervisory capital protection.
- **🔕 AutoTrading (10027) Suppression**: Gracefully suppresses persistent broker error spam for 120 seconds if MetaTrader 5 AutoTrading is disabled by the client.
- **📜 Causal Backtesting Engine**: Replays historical MT5 CSV data (4H, 1H, 15M, 1M) bar-by-bar with zero look-ahead bias and a default starting balance of **$100.00**.
- **📱 Telegram Remote Bot**: Full remote command and control (`/status`, `/pnl`, `/pause`, `/resume`, `/kill`, `/lot`) with automated position closure alerts.
- **🎛️ Modern Web Dashboard**: Dark glassmorphic React interface with real-time WebSockets, interactive charts, and system status monitoring.

---

## 🏗️ System Architecture (Pro Trader v2)

```
                    MT5 Multi-Timeframe Data Feed
                        (4H, 1H, 15M, 1M Candles)
                                    │
    ┌───────────────────────────────┴───────────────────────────────┐
    ▼                                                               ▼
1. Feature Engine                                         Order Tracker (Thread)
   • SMC Engine (OBs, FVGs, Sweeps)                         • Polls every 5s
   • Indicators (ADX, RSI, ATR)                             • Progressive Breakeven
   • Session Engine (Asia/London/NY)                        • Supervisory Capital Cap
    │                                                               │
    ▼                                                               ▼
2. Stage 1: Market Regime Detector (7 Types)              Remote MT5 Bridge
    │                                                     (HTTP Order Execution)
    ▼
3. Stage 2: Strategy Engine + Deep Zone Filter (<25% / >75%)
    │
    ▼
4. Stage 3: Candidate Trade Generator (Entry, SL, TP1-3, R:R)
    │
    ▼
5. Stage 4: 18-Point Pre-Trade Validator
    │
    ▼
6. Stage 5: Multi-Factor Confluence Engine (Score ≥ 0.65)
    │
    ▼
7. Stage 6: Risk Gate & Execution (Final Check & MT5 Order Dispatch)
```

---

## 🔄 6-Stage Pro Trader Pipeline

| Stage | Component | Output / Action |
|---|---|---|
| **1. Feature Engine** | `smc_engine.py`, `session_engine.py` | Computes swing breaks, active Order Blocks, Fair Value Gaps, Liquidity Sweeps, and Session POIs. |
| **2. Regime Detector** | `regime_detector.py` | Classifies 7 market regimes; outputs `NO_TRADE` in ranging or compressing markets. |
| **3. Strategy Engine** | `strategy_engine.py` | Selects valid strategy (*Inducement Sweep*, *OB Reaction*, *BOS Continuation*) and enforces SMC Deep Zone pre-filtering. |
| **4. Trade Generator** | `trade_generator.py` | Calculates exact Entry, SL, TP1, TP2, TP3, and Risk-to-Reward ratio ($\ge 1.5:1$). |
| **5. Validator** | `validator.py` | Validates 18 safety checks (spread, drawdown, news, session, zone bounds). |
| **6. Confluence Engine** | `confluence_engine.py` | Weighted scoring across 8 confluence factors (must meet threshold $\ge 0.65$). |

---

## 🛡️ SMC Deep Zone Protection & Zone Context

Selling at the bottom of a move or buying at the top is the #1 mistake in retail trading. PAXIS enforces institutional **Smart Money Concepts** range rules:

- **Deep Discount Zone (<25% of 1H Swing Range)**: `SHORT` entries are prohibited because price is reacting to macro demand/support. The Strategy Engine outputs a clear `HOLD` until price retraces above 25% or prints a bullish reversal setup.
- **Deep Premium Zone (>75% of 1H Swing Range)**: `LONG` entries are prohibited because price is reacting to macro supply/resistance.

---

## 📈 Backtesting & Historical Performance

PAXIS includes a standalone, causal backtester (`agent/backtest/engine.py`) and execution runner (`run_backtest.py`).

### Key Parameters:
- **Default Starting Balance**: **$100.00**
- **Risk Per Trade**: 1.0%
- **Simulated Spread**: 2.0 pips ($XAUUSD$)
- **Primary Timeframe**: 1H (with 4H, 15M, and 1M context)

### Run a Historical Backtest:
```bash
source venv/bin/activate
python run_backtest.py 1H
```

Results are printed to CLI and automatically saved to `backtest_results.json` and `backtest_trades.csv`.

---

## 💻 Tech Stack

### Backend
- **Core Engine**: Python 3.11 / 3.13
- **Quantitative Engine**: `pandas`, `numpy`, `pandas-ta-classic`
- **Execution & Data**: Custom MT5 Remote Bridge (HTTP REST + Sockets)
- **API Framework**: FastAPI + WebSockets
- **Task Orchestration**: APScheduler
- **Database**: SQLite with WAL mode (`paxis_trades.db`, `paxis_features.db`)
- **Logging**: Loguru

### Frontend Dashboard
- **Framework**: React 18 + Vite
- **UI Architecture**: Glassmorphic dark design system
- **Charts**: Recharts
- **Icons**: Lucide React

---

## 🛠️ Installation & Setup

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/your-username/trader-paxis.git
cd trader-paxis
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies (Optional Dashboard)
```bash
cd dashboard/frontend
npm install
npm run build
cd ../..
```

---

## ⚙️ Configuration (.env)

Create or configure your `.env` file:
```env
DRY_RUN=false
PRO_TRADER_MODE=true
TRADING_PAIRS=XAUUSD
TRADE_CYCLE_MINUTES=1
POSITION_POLL_SECONDS=5
CONFLUENCE_THRESHOLD=0.65
MIN_RR_RATIO=1.5
MAX_SPREAD_PIPS=3.0
MAX_DAILY_LOSS_USD=15.0

# Remote MT5 Bridge Config
MT5_REMOTE_IP=127.0.0.1
MT5_REMOTE_PORT=8001

# Telegram Control Bot
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

---

## 🚀 Running Live, Dry-Run & Backtester

### 1. Run Live Agent
```bash
source venv/bin/activate
python -m agent.main --live
```

### 2. Run Paper Trading / Dry-Run Mode
```bash
source venv/bin/activate
python -m agent.main --dry-run
```

### 3. Run Historical Backtest ($100 Starting Balance)
```bash
source venv/bin/activate
python run_backtest.py 1H
```

### 4. Run Web Dashboard
```bash
uvicorn dashboard.backend.app:app --host 0.0.0.0 --port 8000
```
Access dashboard at `http://localhost:8000`.

---

## 🌉 Remote MetaTrader 5 (MT5) Bridge Integration

When running on Linux, PAXIS connects to a lightweight MT5 bridge running on Windows/Wine:

1. On the MT5 host machine: `python -m mt5_bridge.server --port 8001`
2. Enable **AutoTrading** in the MetaTrader 5 top toolbar.
3. Configure `MT5_REMOTE_IP` and `MT5_REMOTE_PORT` in `.env`.

---

## 🧪 Unit Testing

Execute the complete pytest suite (214 tests):
```bash
source venv/bin/activate
pytest -q
```
