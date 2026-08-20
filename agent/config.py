"""
PAXIS Agent — Configuration
Loads all settings from .env via pydantic-settings.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Agent ─────────────────────────────────────────────────────────────────
    dry_run: bool = Field(True, description="If True, no real orders are placed")
    agent_name: str = "PAXIS Agent"

    # ── MT5 ───────────────────────────────────────────────────────────────────
    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = "JustMarkets-Demo"
    mt5_path: str = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    mt5_remote_ip: str = ""
    mt5_remote_port: int = 8000

    # ── Pairs ─────────────────────────────────────────────────────────────────
    trading_pairs: str = "XAUUSD"

    @property
    def pairs_list(self) -> List[str]:
        return [p.strip() for p in self.trading_pairs.split(",") if p.strip()]

    # ── Ollama / LLM / Remote API Options ──────────────────────────────────────
    use_local_ollama: bool = Field(False, description="true = use local Ollama GPU, false = use remote LLM API")
    llm_provider: str = Field("api", description="'ollama' or 'api'")
    llm_api_key: str = Field("sk-b56ecc128d7cca90-e880a8-a1f43d23", description="API Key for OmniRoute AI Gateway")
    llm_api_base_url: str = Field("http://34.93.80.53:20128/v1", description="Base URL for OmniRoute AI Gateway")
    llm_api_model: str = Field("zm/deepseek/deepseek-chat", description="Model name for OmniRoute AI Gateway")

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = Field("qwen2.5:14b", description="Primary decision model — Qwen 2.5 14B precision trading model")
    ollama_fallback_model: str = Field("qwen2.5:3b", description="Fallback model if primary times out")
    ollama_temperature: float = Field(0.1, description="Low temperature for consistent deterministic reasoning")
    ollama_top_p: float = Field(0.85, description="Top-p sampling — slightly below 1.0 to reduce hallucination")
    inference_timeout_seconds: int = Field(120, description="Timeout for text inference requests before fallback")
    deepseek_thinking_mode: bool = Field(False, description="Enable DeepSeek R1 extended thinking tokens")

    # ── Vision (disabled in Pro Trader text mode) ──────────────────────────────
    enable_vision: bool = Field(False, description="Vision disabled in Pro Trader mode — using structured JSON data")
    vision_timeout_seconds: int = Field(60, description="Timeout for vision LLM requests before text fallback")

    # ── Analysis Engine ────────────────────────────────────────────────────────
    confluence_llm_threshold: float = Field(0.50, description="Minimum confluence score to call LLM")
    confluence_threshold: float = 0.60
    confluence_critic_threshold: float = Field(0.85, description="Confluence score above which critic is bypassed")
    use_adversarial_critic: bool = Field(True, description="Enable adversarial critic on borderline setups")
    max_num_predict_tokens: int = Field(1024, description="Max tokens LLM generates per response")
    num_ctx_tokens: int = Field(4096, description="Context window size for LLM (fits 100% in GPU VRAM)")

    # ── Risk ──────────────────────────────────────────────────────────────────
    lot_size: float = Field(0.01, description="Fixed lot size — editable via dashboard")
    use_dynamic_risk: bool = Field(True, description="Risk a percentage of account balance rather than fixed lot size")
    risk_percent: float = Field(1.0, description="Percentage of account balance to risk per trade (e.g. 1.0 = 1%)")
    auto_breakeven_ratio: float = Field(1.0, description="Move Stop Loss to entry price when trade profit reaches X * Risk (ratio)")
    trailing_stop_atr_multiplier: float = Field(2.0, description="Trail Stop Loss by X * ATR. Set to 0.0 to disable.")
    enforce_trend_alignment: bool = Field(True, description="Require H1 and H4 EMA trend alignment before trade entry")
    disable_risk_gate: bool = Field(False, description="Completely bypass and disable all risk gate checks")
    min_confidence: float = 0.70
    max_open_trades: int = 2
    max_spread_pips: float = 3.0
    max_daily_loss_usd: float = 50.0
    min_rr_ratio: float = Field(2.0, description="Minimum R:R ratio — Pro Trader mode requires 2.0")
    news_blackout_minutes: int = 30

    # ── Scalping ──────────────────────────────────────────────────────────────
    scalping_mode: bool = Field(True, description="Enable specialized scalping mode for tight short-term trades")
    scalping_target_profit_usd: float = Field(1.0, description="Take profit target in USD for the base lot size (0.01 lots)")
    scalping_sl_usd: float = Field(4.5, description="Stop loss in USD for the base lot size (0.01 lots) — allows buffer beyond OB")

    # ── Auto-Execute Scalping Mode ────────────────────────────────────────────
    auto_scalp_mode: bool = Field(False, description="Enable fully autonomous scalp execution — LLM opens/closes trades every cycle")
    auto_scalp_cycle_minutes: int = Field(3, description="Cycle interval in minutes for auto-scalp mode (default: 3)")
    auto_scalp_max_trades: int = Field(2, description="Hard cap on concurrent open positions in auto-scalp mode (cannot exceed 2)")
    auto_scalp_sl_usd: float = Field(4.5, description="Fixed stop loss in USD per 0.01 lot — allows buffer beyond OB")
    auto_scalp_tp_usd: float = Field(1.0, description="Fixed take profit in USD per 0.01 lot — always overrides LLM output")
    auto_scalp_use_vision: bool = Field(False, description="Enable vision screenshots during auto-scalp cycles (default False for maximum execution speed)")

    # ── Pro Trader Mode (4-Timeframe SMC) ──────────────────────────────────────
    pro_trader_mode: bool = Field(True, description="Enable 4-Timeframe SMC Pro Trader Mode (4H, 1H, 15M, 1M)")
    pro_trader_use_tradingview_scrape: bool = Field(True, description="Use Playwright to scrape live TradingView charts with smc_core_model.pine indicator")
    tradingview_chart_url: str = Field("https://www.tradingview.com/chart/eTq2RTXP/", description="TradingView chart layout URL")
    tradingview_session_id: Optional[str] = Field(None, description="TradingView sessionid cookie for loading authenticated private layouts")
    pro_trader_min_rr: float = Field(2.0, description="Minimum Risk-to-Reward ratio required in Pro Trader mode")
    max_slippage_points: float = Field(1.5, description="Max allowed price drift in points between chart capture and live MT5 execution")
    max_vision_failures: int = Field(2, description="Max consecutive vision failures before pausing vision")
    disable_vision_fallback: bool = Field(False, description="If True, vision analysis is strictly preserved and never falls back to text-only mode")

    # ── Scheduler ─────────────────────────────────────────────────────────────
    trade_cycle_minutes: int = 5
    position_poll_seconds: int = 30

    # ── Sessions (UTC, "HH:MM") ───────────────────────────────────────────────
    london_session_start: str = "07:00"
    london_session_end: str = "12:00"
    ny_session_start: str = "13:00"
    ny_session_end: str = "17:00"

    # ── Telegram ──────────────────────────────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_silent_holds: bool = True

    # ── Dashboard ─────────────────────────────────────────────────────────────
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8000
    dashboard_secret_key: str = "change-this-secret-key"

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./paxis_trades.db"

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_dir: str = "logs"

    @field_validator("lot_size")
    @classmethod
    def lot_size_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("LOT_SIZE must be > 0")
        return round(v, 2)

    @field_validator("min_confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("MIN_CONFIDENCE must be between 0.0 and 1.0")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()


# Convenience alias
settings = get_settings()
