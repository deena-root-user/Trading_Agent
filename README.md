# 🪐 PAXIS Agent — Autonomous Quant & Multimodal Forex Trading System

> An autonomous LLM-driven Forex trading agent built on a **Deterministic-First, LLM-Last** architecture: Smart Money Concepts (SMC) quantitative engine, Market Regime Detector, 18-point Safety Gate, DeepSeek R1 qualitative validation, Adversarial Critic, Causal Backtesting Engine, and real-time control dashboard.

---

## 📖 Table of Contents
1. [Core Features](#-core-features)
2. [System Architecture (Pro Trader v2)](#-system-architecture-pro-trader-v2)
3. [8-Stage Pro Trader Pipeline](#-8-stage-pro-trader-pipeline)
4. [Backtesting & Historical Data](#-backtesting--historical-data)
5. [Tech Stack](#-tech-stack)
6. [Installation & Setup](#-installation--setup)
7. [Configuration (.env)](#-configuration-env)
8. [Running the Agent & Backtester](#-running-the-agent--backtester)
9. [MT5 Bridge Commands for Windows](#-metatrader-5-mt5-bridge-commands-for-windows)
10. [Unit Testing](#-unit-testing)

---

## ⚡ Core Features

- **⚡ Deterministic-First, LLM-Last Architecture**: 80%+ of low-quality setups are rejected in microseconds by mathematical rules before calling the LLM, saving inference time and preventing price level hallucinations.
- **📊 Quantitative SMC Engine**: Calculates Bos, CHoCH, Order Blocks, Fair Value Gaps (FVG), Equal Highs/Lows (EQH/EQL), Liquidity Sweeps, Inducement, Displacement strength, and 50% Equilibrium Premium/Discount zones.
- **🎯 Market Regime Classification**: Classifies market state into 7 regime types using ADX, Bollinger Band squeezes, and multi-timeframe structure alignment. Automatically outputs `NO_TRADE` in ranging or compressing markets.
- **🧠 DeepSeek R1 Qualitative Validation**: DeepSeek R1 32B validates pre-computed structured JSON payloads for qualitative institutional logic.
- **🥊 Adversarial Critic Mode**: A second-turn "devil's advocate" review for borderline setups (confluence scores 0.65–0.84) to eliminate false positives.
- **📜 Causal Backtesting Engine**: Replays historical MT5 CSV data (4H, 1H, 15M, 1M) bar-by-bar with zero look-ahead bias and rolling walk-forward validation.
- **🛡️ 18-Point Risk Gate**: Ironclad safety gateway evaluating daily drawdowns, spread limits, reward-to-risk ratio (min 1.5:1), session blackout windows, and position caps.
- **💾 Full Feature Logger**: Logs 11 data layers per decision into SQLite (`paxis_features.db`) for regime-filtered performance analytics and self-evolution learning.
- **📱 Telegram Remote Control**: Real-time Telegram bot broadcasting trading activity (entry, exit, risk blocks) and receiving remote commands (`/status`, `/kill`, `/pause`, `/resume`, `/pnl`, `/lot`).
- **🎛️ Premium Control Dashboard**: Modern React-based Web App with dark glassmorphic UI, real-time WebSockets, config editing, and emergency liquidation switches.

---

## 🏗️ System Architecture (Pro Trader v2)

```
MT5 Multi-Timeframe Data Feed (4H, 1H, 15M, 1M)
  │
  ├── 1. Feature Layer: SMC Engine (Causal) + Session Engine + Extended Indicators
  │
  ├── 2. Stage 1: Market Regime Detector (7 regime types; 3 trigger instant NO_TRADE)
  │
  ├── 3. Stage 2: Strategy Engine (Selects exact strategy for active regime)
  │
  ├── 4. Stage 3: Candidate Trade Generator (Calculates exact Entry/SL/TP1/TP2/TP3 & R:R)
  │
  ├── 5. Stage 4: 18-Point Pre-Trade Validator (6 mandatory rules must pass)
  │
  ├── 6. Stage 5: Multi-Factor Confluence Engine (Weighted 0.0–1.0 score; threshold ≥ 0.65)
  │
  ├── 7. Stage 6: Structured LLM Validation (DeepSeek R1 validates JSON payload)
  │
  ├── 8. Stage 7: Adversarial Critic Mode (Fires on borderline [0.65–0.84] setups)
  │
  └── 9. Stage 8: Risk Gate & Execution (Final hard parameter checks & MT5 order placement)
```

---

## 🔄 8-Stage Pro Trader Pipeline

| Stage | Component | Output / Action |
|---|---|---|
| **1. Feature Engine** | `smc_engine.py`, `session_engine.py`, `indicators.py` | Computes SMC swing breaks, OBs, FVGs, Premium/Discount, session levels (PDH/PDL), BB squeeze, RSI divergence, and realized vol. |
| **2. Regime Detector** | `regime_detector.py` | Classifies 7 regimes. Blocks `RANGING`, `COMPRESSING`, `UNCERTAIN` instantly as `NO_TRADE`. |
| **3. Strategy Engine** | `strategy_engine.py` | Selects valid SMC strategy (e.g. *FVG Retracement*, *OB Reaction*, *BOS Continuation*). |
| **4. Trade Generator** | `trade_generator.py` | Computes exact mathematical Entry, SL, TP1, TP2, TP3, and R:R ratios from SMC POIs. |
| **5. Validator** | `validator.py` | Enforces 18-point checklist (6 mandatory checks exit instantly if failed). |
| **6. Confluence Engine** | `confluence_engine.py` | Multi-factor weighted scoring across 8 categories (must reach threshold $\ge 0.65$). |
| **7. LLM Reasoning** | `prompt_builder_v2.py`, `ollama_client.py` | DeepSeek R1 32B validates JSON context and flags contradictions. |
| **8. Adversarial Critic** | `pro_trader_pipeline.py` | Second-turn review probing potential failure modes for borderline setups. |

---

## 📈 Backtesting & Historical Data

PAXIS includes a standalone backtesting engine (`agent/backtest/engine.py`) and execution runner (`run_backtest.py`) that processes MetaTrader 5 exported CSV data (e.g. `XAUUSD_H1_202301030200_202607311000.csv`).

### Performance Metrics Computed:
- **Win Rate & Loss Rate %**
- **Profit Factor & Recovery Factor**
- **Expectancy ($/trade and R-multiples per trade)**
- **Sharpe Ratio (Annualized) & Calmar Ratio**
- **Max Drawdown ($ and %)**
- **Regime & Strategy Win Rate Breakdown**

To execute a full historical backtest over 3.5 years of data:
```bash
python run_backtest.py
```

---

## 💻 Tech Stack

### Backend
- **Core Engine**: Python 3.11 / 3.13
- **Primary LLM**: DeepSeek R1 32B (via local Ollama)
- **API Framework**: FastAPI + WebSockets
- **Database**: SQLite + SQLAlchemy Async + WAL Mode (`paxis_trades.db`, `paxis_features.db`)
- **Task Orchestration**: APScheduler
- **Quantitative Libraries**: `pandas`, `numpy`, `pandas-ta-classic`
- **Logging**: Loguru

### Frontend
- **Framework**: React 18 + Vite
- **Styling**: Dark glassmorphic design system
- **Visual Charts**: Recharts
- **Icons**: Lucide React

---

## 🛠️ Installation & Setup

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/your-username/paxis-agent.git
cd paxis-agent
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies
```bash
cd dashboard/frontend
npm install
npm run build
cd ../..
```

---

## ⚙️ Configuration (.env)

Copy `.env.example` to `.env` and configure:
```env
DRY_RUN=true
OLLAMA_MODEL=deepseek-r1:32b
PRO_TRADER_MODE=true
CONFLUENCE_THRESHOLD=0.65
USE_ADVERSARIAL_CRITIC=true
TRADING_PAIRS=XAUUSD,EURUSD,GBPUSD
RISK_PERCENT=1.0
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## 🚀 Running the Agent & Backtester

### 1. Run Historical Backtest
```bash
python run_backtest.py
```

### 2. Run Dashboard Backend
```bash
uvicorn dashboard.backend.app:app --host 0.0.0.0 --port 8000
```
- Dashboard URL: `http://localhost:8000`

### 3. Run Live Agent / Paper Trading Mode
```bash
python -m agent.main --dry-run
```

---

## 🌉 MetaTrader 5 (MT5) Bridge Commands for Windows

### Remote MT5 Bridge Server on Windows:
1. On Windows: `pip install MetaTrader5 mt5-bridge`
2. Start MT5 software and log into your account.
3. Start bridge server: `mt5-bridge server --host 0.0.0.0 --port 8001`
4. On Agent machine (`.env`):
   ```env
   MT5_REMOTE_IP=192.168.1.50
   MT5_REMOTE_PORT=8001
   ```

---

## 🧪 Unit Testing

Run the full pytest suite:
```bash
pytest
```
