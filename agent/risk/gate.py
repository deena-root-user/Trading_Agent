"""
PAXIS Agent — Risk Gate
Hard rules that MUST all pass before any order is placed.
The LLM cannot override these checks under any circumstance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from loguru import logger

from agent.config import settings


@dataclass
class RiskCheckResult:
    passed: bool
    calculated_lot: float = 0.01
    failed_checks: List[str] = field(default_factory=list)
    blocked_reason: str = ""

    def __bool__(self) -> bool:
        return self.passed


class RiskGate:
    """
    Validates a trade decision against 10 hard risk rules.
    All checks must pass — one failure forces HOLD.
    """

    def __init__(self):
        # Runtime state (updated by main loop)
        self._daily_pnl: float = 0.0
        self._today_date: Optional[str] = None
        self._agent_paused: bool = False
        self._loss_history: dict = {}  # key -> {"time": datetime, "entry_price": float}

    def record_loss(self, symbol: str, action: str, entry_price: float) -> None:
        """Record a losing trade to enforce directional cooldown and prevent repeated losses."""
        key = f"{symbol.upper()}_{action.upper()}"
        self._loss_history[key] = {
            "time": datetime.now(timezone.utc),
            "entry_price": entry_price,
        }
        logger.info(f"🛡️ Directional Loss Shield activated for {action} {symbol} at {entry_price:.2f} — 15m Cooldown Active")

    # ── State Setters ─────────────────────────────────────────────────────────

    def update_daily_pnl(self, pnl: float) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._today_date:
            self._daily_pnl = 0.0
            self._today_date = today
        self._daily_pnl = pnl

    def set_paused(self, paused: bool) -> None:
        self._agent_paused = paused
        logger.info(f"Agent {'PAUSED' if paused else 'RESUMED'} by risk gate")

    @property
    def is_paused(self) -> bool:
        return self._agent_paused

    # ── Main Check ────────────────────────────────────────────────────────────

    def check(
        self,
        *,
        symbol: str,
        action: str,
        confidence: float,
        entry: float,
        sl: float,
        tp: float,
        rr_ratio: float,
        spread_pips: float,
        open_positions: List[dict],
        news_blocked: bool = False,
        news_reason: str = "",
        h1_trend: Optional[str] = None,
        h4_trend: Optional[str] = None,
    ) -> RiskCheckResult:
        """
        Run all risk checks. Returns RiskCheckResult.
        """
        failures = []

        # ── Check 1: Agent not paused ─────────────────────────────────────────
        if self._agent_paused:
            return RiskCheckResult(
                passed=False,
                failed_checks=["AGENT_PAUSED"],
                blocked_reason="Agent is paused — use /resume to restart",
            )

        # Calculate lot size (Dynamic vs. Static)
        calculated_lot = settings.lot_size
        if settings.use_dynamic_risk and action in ("BUY", "SELL") and entry > 0 and sl > 0:
            try:
                from agent.data.mt5_feed import mt5_feed
                balance = mt5_feed.get_account_balance() or 10000.0
                
                # Determine pip size (3 digits for JPY e.g. USDJPY, 5 digits for standard pairs e.g. EURUSD)
                sym_upper = symbol.upper()
                pip_size = 0.01 if "JPY" in sym_upper or any(x in sym_upper for x in ["XAU", "GOLD"]) else 0.0001
                sl_distance_pips = abs(entry - sl) / pip_size
                
                if sl_distance_pips > 0:
                    risk_usd = balance * (settings.risk_percent / 100.0)
                    # Multi-Asset Pip Value Calibration:
                    # Forex & Gold (0.01 / 0.0001 pip): $10.0 USD per pip per 1.0 lot
                    # US30 / Indices (1.0 pt): $1.0 USD per point per 1.0 lot
                    # Crypto / BTCUSD: $1.0 USD per point per 1.0 lot
                    if any(x in sym_upper for x in ["US30", "DE30", "NDX", "SPX", "BTC", "ETH"]):
                        pip_value_per_lot = 1.0
                    else:
                        pip_value_per_lot = 10.0

                    raw_lot = risk_usd / (sl_distance_pips * pip_value_per_lot)
                    # Clamp between 0.01 and 10.0 lots for safety
                    calculated_lot = max(0.01, min(10.0, round(raw_lot, 2)))
                    logger.info(f"Dynamic Risk Sizing: balance={balance:.2f} | risk_usd={risk_usd:.2f} | sl_pips={sl_distance_pips:.1f} | calculated_lot={calculated_lot}")
                else:
                    logger.warning("SL distance is 0, using fallback static lot size.")
            except Exception as exc:
                logger.error(f"Error calculating dynamic risk: {exc}")

        # ── Check 2: Confidence threshold ─────────────────────────────────────
        min_conf = getattr(settings, "min_confidence", 0.70)
        try:
            if getattr(settings, "enable_focus_mode", True) and not getattr(settings, "strategy_mode", False):
                from agent.evolution.self_evolution import self_evolution_engine
                evo_metrics = self_evolution_engine.get_metrics()
                if evo_metrics.focus_mode:
                    min_conf = max(min_conf, evo_metrics.focus_min_confidence)
                    logger.info(f"High Focus Mode active: requiring min confidence >= {min_conf:.0%}")
        except Exception:
            pass

        if confidence < min_conf:
            failures.append(
                f"LOW_CONFIDENCE: {confidence:.0%} < {min_conf:.0%} required"
            )

        # ── Check 3: News blackout ─────────────────────────────────────────────
        if news_blocked:
            failures.append(f"NEWS_BLACKOUT: {news_reason}")

        # ── Check 4: Daily drawdown limit ─────────────────────────────────────
        if self._daily_pnl <= -abs(settings.max_daily_loss_usd):
            failures.append(
                f"DAILY_LOSS_LIMIT: daily P&L {self._daily_pnl:.2f} "
                f"<= -{settings.max_daily_loss_usd:.2f} USD"
            )
            self._agent_paused = True
            logger.critical(
                f"Daily loss limit hit ({self._daily_pnl:.2f} USD) — agent AUTO-PAUSED"
            )

        # ── Check 5: Max open trades ───────────────────────────────────────────
        if len(open_positions) >= settings.max_open_trades:
            failures.append(
                f"MAX_OPEN_TRADES: {len(open_positions)} open "
                f"(max={settings.max_open_trades})"
            )

        # ── Check 6: No duplicate position on same pair ────────────────────────
        sym_clean = symbol.upper().replace("/", "")
        existing_symbols = [p.get("symbol", "").upper().replace("/", "") for p in open_positions]
        if sym_clean in existing_symbols:
            failures.append(f"DUPLICATE_POSITION: already have open trade on {symbol}")

        # ── Check 7: Spread check ─────────────────────────────────────────────
        if spread_pips > settings.max_spread_pips:
            failures.append(
                f"HIGH_SPREAD: {spread_pips} pips > {settings.max_spread_pips} pips max"
            )

        # ── Check 8: RR ratio ─────────────────────────────────────────────────
        if action in ("BUY", "SELL"):
            if getattr(settings, "pro_trader_mode", False):
                required_rr = getattr(settings, "pro_trader_min_rr", 2.0)
            elif settings.scalping_mode:
                required_rr = 0.3
            else:
                required_rr = settings.min_rr_ratio

            if rr_ratio < required_rr:
                failures.append(
                    f"LOW_RR: {rr_ratio:.2f} < {required_rr:.2f} required"
                )

        # ── Check 9: SL/TP directional sanity ────────────────────────────────
        if action == "BUY" and entry > 0:
            if sl >= entry:
                failures.append(f"INVALID_SL: BUY sl={sl} must be < entry={entry}")
            if tp <= entry:
                failures.append(f"INVALID_TP: BUY tp={tp} must be > entry={entry}")
        elif action == "SELL" and entry > 0:
            if sl <= entry:
                failures.append(f"INVALID_SL: SELL sl={sl} must be > entry={entry}")
            if tp >= entry:
                failures.append(f"INVALID_TP: SELL tp={tp} must be < entry={entry}")

        # ── Check 10: Multi-Timeframe Trend Alignment ────────────────────────
        if settings.enforce_trend_alignment and action in ("BUY", "SELL"):
            if not settings.scalping_mode:
                # In standard swing/intraday mode, enforce strict H1/H4 alignment
                if action == "BUY":
                    if h1_trend == "BEARISH" or h4_trend == "BEARISH":
                        failures.append(f"TREND_MISALIGNMENT: H1={h1_trend}, H4={h4_trend} trend is bearish")
                elif action == "SELL":
                    if h1_trend == "BULLISH" or h4_trend == "BULLISH":
                        failures.append(f"TREND_MISALIGNMENT: H1={h1_trend}, H4={h4_trend} trend is bullish")
            else:
                # In scalping mode, micro-timeframe momentum (M1/M5) is primary.
                # Only block if BOTH medium (M5) and macro (M15) trends strongly oppose the trade AND confidence is low (< 0.75).
                if action == "BUY" and h1_trend == "BEARISH" and h4_trend == "BEARISH" and confidence < 0.75:
                    failures.append(f"TREND_MISALIGNMENT: M5={h1_trend}, M15={h4_trend} trends are strongly bearish")
                elif action == "SELL" and h1_trend == "BULLISH" and h4_trend == "BULLISH" and confidence < 0.75:
                    failures.append(f"TREND_MISALIGNMENT: M5={h1_trend}, M15={h4_trend} trends are strongly bullish")

        # ── Check 11: Directional Loss Cooldown Shield ────────────────────────
        key = f"{symbol.upper()}_{action.upper()}"
        if hasattr(self, "_loss_history") and key in self._loss_history:
            loss_info = self._loss_history[key]
            elapsed_sec = (datetime.now(timezone.utc) - loss_info["time"]).total_seconds()
            cooldown_period = 900  # 15 minutes (900 seconds)

            if elapsed_sec < cooldown_period:
                prev_entry = loss_info["entry_price"]
                mins_remaining = (cooldown_period - elapsed_sec) / 60.0
                if confidence < 0.85:
                    failures.append(
                        f"DIRECTIONAL_LOSS_COOLDOWN: Previous {action} {symbol} lost {elapsed_sec/60:.1f}m ago — "
                        f"cooldown strictly active for {mins_remaining:.1f}m (prev entry: {prev_entry:.2f}, requires conf >=85%)"
                    )

        # ── Check 12: Capital Protection USD Risk Guard ──────────────────────
        if action in ("BUY", "SELL") and entry > 0 and sl > 0:
            try:
                from agent.data.mt5_feed import mt5_feed
                raw_bal = mt5_feed.get_account_balance() or 10000.0
                raw_eq = mt5_feed.get_account_equity() or raw_bal
                balance = min(raw_bal, raw_eq)

                max_micro_cap = getattr(settings, "max_micro_account_loss_usd", 1.50)
                max_risk_pct = getattr(settings, "max_trade_risk_percent", 2.0)

                if balance < 50.0:
                    allowed_risk_usd = min(max_micro_cap, max(0.50, balance * 0.15))
                else:
                    allowed_risk_usd = balance * (max_risk_pct / 100.0)

                sl_dist_pts = abs(entry - sl)
                sym_up = symbol.upper()

                if any(x in sym_up for x in ["US30", "DE30", "NDX", "SPX", "BTC", "ETH"]):
                    usd_per_point = 1.0 * calculated_lot
                elif any(x in sym_up for x in ["XAU", "GOLD"]):
                    usd_per_point = 100.0 * calculated_lot  # 0.01 lot = $1.0 USD / pt
                elif "JPY" in sym_up:
                    usd_per_point = 100.0 * calculated_lot
                else:
                    usd_per_point = 100000.0 * calculated_lot

                actual_usd_risk = sl_dist_pts * usd_per_point

                if actual_usd_risk > allowed_risk_usd * 1.05:
                    failures.append(
                        f"CAPITAL_PROTECTION_BLOCKED: Trade SL risk (${actual_usd_risk:.2f} USD) "
                        f"exceeds max allowed capital risk (${allowed_risk_usd:.2f} USD for ${balance:.2f} balance, max 15% of micro balance or $1.50 cap)"
                    )
            except Exception as exc:
                logger.error(f"Error evaluating capital protection risk guard: {exc}")

        if failures:
            reason = " | ".join(failures)
            logger.warning(f"Risk gate BLOCKED {action} {symbol}: {reason}")
            return RiskCheckResult(
                passed=False,
                calculated_lot=calculated_lot,
                failed_checks=failures,
                blocked_reason=reason,
            )

        logger.info(
            f"Risk gate PASSED ✓ | {action} {symbol} | "
            f"conf={confidence:.0%} | RR={rr_ratio:.2f} | spread={spread_pips}pips | lot={calculated_lot}"
        )
        return RiskCheckResult(passed=True, calculated_lot=calculated_lot)

    def check_session(self) -> Tuple[bool, str]:
        """
        Check if current UTC time is within active market trading sessions.
        Returns (is_active, session_name).
        """
        now = datetime.now(timezone.utc)
        weekday = now.weekday()  # 0=Monday ... 4=Friday, 5=Saturday, 6=Sunday

        # Market close check on weekends (Saturday & Sunday before 22:00 UTC)
        if weekday == 5 or (weekday == 6 and now.hour < 22):
            logger.info(f"Market closed on weekend ({now.strftime('%A %H:%M')} UTC)")
            return False, "Weekend (Market Closed)"

        current_time = now.strftime("%H:%M")

        def in_window(start: str, end: str) -> bool:
            if start <= end:
                return start <= current_time <= end
            else:
                # Session wraps across midnight UTC (e.g. 22:00 to 07:00)
                return current_time >= start or current_time <= end

        # If strict session hours enforcement is explicitly enabled:
        if getattr(settings, "enforce_session_hours", False):
            if in_window(settings.london_session_start, settings.london_session_end):
                return True, "London"
            if in_window(settings.ny_session_start, settings.ny_session_end):
                return True, "New York"
            if in_window(getattr(settings, "asian_session_start", "22:00"), getattr(settings, "asian_session_end", "07:00")):
                return True, "Asian"
            logger.info(f"Outside enforced trading session hours at {current_time} UTC")
            return False, "Outside Enforced Hours"

        # Default 24/5 trading mode for XAUUSD & Forex: Active all weekday hours.
        # BUG-10 FIX: Check Overlap first (most specific), then individual sessions.
        # Note: This block intentionally returns True for all hours — it's a
        # session NAMING mechanism, not a session FILTER. Filtering only happens
        # when enforce_session_hours=True (above).
        if in_window("12:00", "16:00"):
            return True, "London/NY Overlap"
        elif in_window("07:00", "12:00"):
            return True, "London"
        elif in_window("16:00", "21:00"):
            return True, "New York"
        elif in_window("22:00", "07:00") or in_window("00:00", "08:00"):
            return True, "Asian"

        return True, "24H Market Open"


# Singleton
risk_gate = RiskGate()
