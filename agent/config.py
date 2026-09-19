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
    llm_api_model: str = Field("kr/claude-sonnet-4.5", description="Model name for OmniRoute AI Gateway (Kiro AI Claude Sonnet 4.5)")

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

    # ── Strategy Mode (Pure Deterministic — No LLM/API) ──────────────────────
    strategy_mode: bool = Field(False, description="When True, skip ALL LLM/Ollama/API calls — use only deterministic strategy conditions (regime + strategy + validator + confluence). Best for backtesting and pure mechanical execution.")
    trading_strategy_mode: str = Field("SMC,ICT,SCALP", description="Comma-separated trading strategy modes. Options: SMC, ICT, SCALP, BREAKOUT. Enables parallel execution of all listed modes.")

    # ── Trading Environment ────────────────────────────────────────────────────
    trading_environment: str = Field("PAPER", description="DEVELOPMENT | PAPER | LIVE. Controls risk bypass behavior — LIVE environment disables all risk gate bypasses.")

    # ── Parallel Strategy Execution ────────────────────────────────────────────
    enable_parallel_strategies: bool = Field(True, description="Run SMC, ICT, BREAKOUT and SCALP strategies simultaneously rather than serially.")
    parallel_strategy_timeout_seconds: float = Field(8.0, description="Max seconds to wait for each parallel strategy before skipping.")
    signal_dedup_window_seconds: float = Field(300.0, description="Suppress duplicate signals on same symbol+direction within this window.")

    # ── Dual Entry Mode: Structural ────────────────────────────────────────────
    structural_mode_enabled: bool = Field(True, description="Enable structural (retracement-into-POI) entry mode.")
    structural_poi_tolerance_enabled: bool = Field(True, description="Enable configurable tolerance so price need not hit the exact center of the POI.")
    structural_poi_tolerance_atr: float = Field(1.5, description="POI tolerance in ATR multiples. Price within ±(ATR × this) of POI center is considered 'in zone'.")
    structural_require_retracement: bool = Field(True, description="Require price to retrace into POI before entry.")
    structural_require_candle_confirmation: bool = Field(True, description="Require a candle close inside zone for confirmation.")

    # ── Dual Entry Mode: Breakout ──────────────────────────────────────────────
    breakout_mode_enabled: bool = Field(True, description="Enable breakout (BOS + displacement) entry mode. Complements structural mode.")
    breakout_entry_type: str = Field("confirmation", description="Breakout entry type: 'close' | 'confirmation' | 'micro_retest'. Default: confirmation.")
    breakout_require_close_confirmation: bool = Field(True, description="Require candle close beyond structure level. Wick-only breaks rejected.")
    breakout_require_displacement: bool = Field(True, description="Require measurable displacement candle after BOS.")
    breakout_displacement_atr_min: float = Field(0.8, description="Minimum breakout candle range / ATR to qualify as displacement.")
    breakout_min_rr: float = Field(2.0, description="Minimum R:R for breakout trades. Do not lower to increase trade count.")
    breakout_max_spread_pips: float = Field(3.0, description="Max spread during breakout entry.")
    breakout_cooldown_seconds: float = Field(300.0, description="Cooldown after a breakout trade before allowing another on same BOS level.")
    breakout_sl_method: str = Field("protected_swing", description="SL reference for breakout: 'breakout_candle' | 'protected_swing' | 'displacement_origin'.")
    breakout_regime_filter: bool = Field(True, description="Apply regime filter: EXHAUSTION regime rejects breakout, others evaluated by config.")
    breakout_regime_allowed: str = Field("TRENDING,COMPRESSION,RANGE", description="Comma-separated regimes that allow breakout trades.")

    # ── HTF Cache Fix (Fixes zone-wait paralysis) ──────────────────────────────
    htf_cache_expiry_minutes: float = Field(15.0, description="HTF fingerprint cache expiry. 0 = no cache. Fixes the issue where HOLD is returned indefinitely when HTF data unchanged.")

    # ── Kill Zone Priority ─────────────────────────────────────────────────────
    kill_zone_priority_bonus: float = Field(0.10, description="Confluence bonus for setups that occur during London/NY kill zones.")
    london_kill_zone_start: str = Field("07:00", description="London kill zone start UTC.")
    london_kill_zone_end: str = Field("09:00", description="London kill zone end UTC.")
    ny_kill_zone_start: str = Field("12:00", description="NY kill zone start UTC.")
    ny_kill_zone_end: str = Field("14:00", description="NY kill zone end UTC.")
    asian_kill_zone_start: str = Field("00:00", description="Asian kill zone start UTC.")
    asian_kill_zone_end: str = Field("03:00", description="Asian kill zone end UTC.")

    # ── Analysis Engine ────────────────────────────────────────────────────────
    confluence_llm_threshold: float = Field(0.60, description="Minimum confluence score to call LLM analysis")
    confluence_api_threshold: float = Field(0.65, description="Minimum confluence score to route to Remote API (60%-65% uses local Ollama)")
    confluence_threshold: float = 0.60
    confluence_critic_threshold: float = Field(0.85, description="Confluence score above which critic is bypassed")
    use_adversarial_critic: bool = Field(True, description="Enable adversarial critic on borderline setups")
    max_num_predict_tokens: int = Field(2048, description="Max tokens LLM generates per response")
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
    enable_focus_mode: bool = Field(True, description="Enable High Focus Mode elevation on consecutive losses")
    max_open_trades: int = 2
    max_spread_pips: float = 3.0
    max_daily_loss_usd: float = 50.0
    min_rr_ratio: float = Field(2.0, description="Minimum R:R ratio — Pro Trader mode requires 2.0")
    news_blackout_minutes: int = 30
    require_candle_close_confirmation: bool = Field(True, description="Only enter trades when the 1M candle has just closed (within first 25s of new candle). Prevents mid-candle wick entries.")
    candle_close_window_seconds: int = Field(25, description="Max allowed seconds into new 1M candle for entry execution (default 25s)")
    progressive_breakeven: bool = Field(True, description="Enable progressive profit-locking SL ratchet instead of single-step breakeven")
    breakeven_trigger_r: float = Field(0.25, description="Trigger first breakeven at this R-multiple (e.g., 0.25 = +0.25R)")
    profit_lock_steps: str = Field("0.25:0.0,0.5:0.1,1.0:0.25,1.5:0.5,2.0:1.0,2.5:1.5", description="Progressive SL steps as 'trigger_R:lock_R' pairs")
    target_open_pnl_cutoff: float = Field(0.0, description="Target total open PnL cutoff USD to close all positions to lock profit")
    protect_trade1_on_trade2: bool = Field(True, description="Move Trade 1 to breakeven when Trade 2 is opened")
    require_candle_close_confirm: bool = Field(True, description="Only enter trade on candle close confirmation")
    basket_target_profit_usd: float = Field(0.0, description="Alias for basket hard target profit USD")
    basket_soft_target_usd: float = Field(15.0, description="When total floating PnL >= this, tighten all SLs to lock current profit (don't close). 0.0=disabled")
    basket_hard_target_usd: float = Field(25.0, description="When total floating PnL >= this, close ALL positions immediately. 0.0=disabled")
    basket_soft_loss_cutoff_usd: float = Field(10.0, description="Soft loss cutoff USD ($10-$12). Evaluates pullback probability before cutting loss. Closes if low prob, holds if high prob.")
    basket_sl_loss_usd: float = Field(20.0, description="When total floating loss <= -this, close ALL positions immediately. 0.0=disabled")
    second_trade_confluence_boost: float = Field(0.10, description="Extra confluence score required for 2nd trade when 1st trade is in profit")
    second_trade_lock_first_profit: bool = Field(True, description="When opening 2nd trade, auto-tighten SL on 1st profitable trade to lock profit")
    max_trade_risk_percent: float = Field(2.0, description="Max allowed risk percent of account balance per trade")
    max_micro_account_loss_usd: float = Field(1.50, description="Hard USD risk cap per trade for accounts under $50 USD")
    counter_trend_min_confluence: float = Field(0.75, description="Min confluence required for counter-trend trades when 4H opposes")

    # ── Scalping ──────────────────────────────────────────────────────────────
    scalping_mode: bool = Field(True, description="Enable specialized scalping mode for tight short-term trades")
    scalping_target_profit_usd: float = Field(1.0, description="Take profit target in USD for the base lot size (0.01 lots)")
    scalping_sl_usd: float = Field(4.5, description="Stop loss in USD for the base lot size (0.01 lots) — allows buffer beyond OB")

    # ── Auto-Execute Scalping Mode (R-multiple based — NOT dollar based) ────────
    auto_scalp_mode: bool = Field(False, description="Enable fully autonomous scalp execution — LLM opens/closes trades every cycle")
    auto_scalp_cycle_minutes: int = Field(3, description="Cycle interval in minutes for auto-scalp mode (default: 3)")
    auto_scalp_max_trades: int = Field(2, description="Hard cap on concurrent open positions in auto-scalp mode (cannot exceed 2)")
    # R-multiple based SL/TP (replaces broken dollar-based values)
    auto_scalp_sl_atr_multiplier: float = Field(0.5, description="Auto-scalp SL = entry ± (ATR × this). Default 0.5. Backtested value.")
    auto_scalp_tp1_r: float = Field(0.8, description="Auto-scalp TP1 = entry + (sl_dist × this R multiple). Default 0.8R.")
    auto_scalp_tp2_r: float = Field(1.5, description="Auto-scalp TP2 = entry + (sl_dist × this R multiple). Default 1.5R.")
    auto_scalp_min_rr: float = Field(1.2, description="Auto-scalp minimum acceptable R:R ratio.")
    auto_scalp_use_vision: bool = Field(False, description="Enable vision screenshots during auto-scalp cycles (default False for maximum execution speed)")
    # Legacy dollar-based fields kept for backward compat — NOT used by the new pipeline
    auto_scalp_sl_usd: float = Field(4.5, description="DEPRECATED: legacy dollar-based SL. Not used by new auto_scalp_strategy.")
    auto_scalp_tp_usd: float = Field(1.0, description="DEPRECATED: legacy dollar-based TP. Not used by new auto_scalp_strategy.")

    # ── Pro Trader Mode (4-Timeframe SMC) ──────────────────────────────────────
    pro_trader_mode: bool = Field(True, description="Enable 4-Timeframe SMC Pro Trader Mode (4H, 1H, 15M, 1M)")
    pro_trader_use_tradingview_scrape: bool = Field(True, description="Use Playwright to scrape live TradingView charts with smc_core_model.pine indicator")
    tradingview_chart_url: str = Field("https://www.tradingview.com/chart/eTq2RTXP/", description="TradingView chart layout URL")
    tradingview_session_id: Optional[str] = Field(None, description="TradingView sessionid cookie for loading authenticated private layouts")
    pro_trader_min_rr: float = Field(2.0, description="Minimum Risk-to-Reward ratio required in Pro Trader mode")
    max_slippage_points: float = Field(1.5, description="Max allowed price drift in points between chart capture and live MT5 execution")
    max_vision_failures: int = Field(2, description="Max consecutive vision failures before pausing vision")
    disable_vision_fallback: bool = Field(False, description="If True, vision analysis is strictly preserved and never falls back to text-only mode")

    # ── Breakout / Retest Detection ─────────────────────────────────────────
    breakout_retest_enabled: bool = Field(True, description="Enable breakout→displacement→fallback→retest→continuation detection")
    breakout_min_displacement_atr: float = Field(0.8, description="Min candle body / ATR ratio to qualify as displacement")
    breakout_retest_tolerance_atr: float = Field(0.3, description="Retest zone tolerance: level ± (ATR * this value)")
    breakout_max_age_bars_15m: int = Field(40, description="Max 15M bars before a breakout setup expires")
    breakout_max_age_bars_1h: int = Field(12, description="Max 1H bars before a breakout setup expires")
    breakout_max_distance_atr: float = Field(3.0, description="Max distance from broken level (in ATR) to still be 'waiting for retest'")
    breakout_min_rejection_body_ratio: float = Field(0.4, description="Min body/range ratio for a rejection candle")
    breakout_max_retest_depth_atr: float = Field(1.0, description="Max penetration past broken level before setup is invalidated")
    breakout_require_m1_confirmation: bool = Field(True, description="Require M1 MSS/BOS/rejection candle before ENTRY_READY")
    breakout_max_active_setups: int = Field(3, description="Max concurrent breakout setups tracked per symbol")

    # ── Scheduler ─────────────────────────────────────────────────────────────
    trade_cycle_minutes: int = 5
    position_poll_seconds: int = 1

    # ── Sessions (UTC, "HH:MM") ───────────────────────────────────────────────
    enforce_session_hours: bool = Field(False, description="When False (default for 24/5 XAUUSD), trades 24 hours Mon-Fri including Asian session. Set True to strictly enforce London/NY hours.")
    asian_session_start: str = "22:00"
    asian_session_end: str = "07:00"
    london_session_start: str = "07:00"
    london_session_end: str = "16:00"
    ny_session_start: str = "12:00"
    ny_session_end: str = "21:00"

    # ── Breakout & Retest Engine ─────────────────────────────────────────────
    breakout_retest_enabled: bool = True
    breakout_min_displacement_atr: float = 0.8
    breakout_retest_tolerance_atr: float = 0.3
    breakout_max_age_bars_15m: int = 40
    breakout_max_distance_atr: float = 3.0
    breakout_min_rejection_body_ratio: float = 0.4

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

    @field_validator("trading_strategy_mode")
    @classmethod
    def validate_strategy_mode(cls, v: str) -> str:
        """Validates comma-separated strategy modes. Supports SMC, ICT, SCALP, BREAKOUT."""
        valid_modes = {"SMC", "ICT", "SCALP", "BREAKOUT"}
        parts = [p.strip().upper() for p in v.split(",") if p.strip()]
        if not parts:
            raise ValueError("TRADING_STRATEGY_MODE must contain at least one mode")
        invalid = [p for p in parts if p not in valid_modes]
        if invalid:
            raise ValueError(
                f"TRADING_STRATEGY_MODE: unknown modes {invalid}. Valid: {sorted(valid_modes)}"
            )
        return ",".join(parts)

    @property
    def active_strategy_modes(self) -> list[str]:
        """Return list of active strategy modes from comma-separated config."""
        return [p.strip().upper() for p in self.trading_strategy_mode.split(",") if p.strip()]

    @property
    def is_live(self) -> bool:
        return self.trading_environment.upper() == "LIVE"

    @property
    def breakout_regime_list(self) -> list[str]:
        return [r.strip().upper() for r in self.breakout_regime_allowed.split(",") if r.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()


# Convenience alias
settings = get_settings()
