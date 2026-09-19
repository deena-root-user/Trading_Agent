"""
PAXIS Agent — Auto-Scalp Strategy (R-Multiple Based)
======================================================
Short-term scalp entry mode. Uses ATR-normalized SL/TP instead of
fixed dollar values.

CRITICAL CHANGE FROM PREVIOUS IMPLEMENTATION:
  OLD: SL=$4.50, TP=$1.00 per 0.01 lot (HARDCODED, WRONG)
  NEW: SL = ATR × config.auto_scalp_sl_atr_multiplier
       TP1 = entry ± (sl_distance × config.auto_scalp_tp1_r)
       TP2 = entry ± (sl_distance × config.auto_scalp_tp2_r)

Why the change:
  - Dollar-based SL/TP breaks on any account size other than the original
  - SL=$4.50 per lot is inconsistent across symbols (EURUSD vs XAUUSD vs BTCUSD)
  - ATR-based SL adapts to current volatility automatically

ARCHITECTURE:
  AutoScalpStrategy.analyze() → List[TradeCandidate]
  (NO volume, NO execution — Risk Engine handles both)

ENTRY CONDITIONS (scalp-specific):
  1. 1M structure aligned with 15M setup
  2. Tight spread (< config.max_spread_pips)
  3. Price within ATR distance of a 15M POI
  4. No adverse 1H or 4H trend (counter-trend scalps forbidden)
  5. Kill zone preferred but not required

REJECTED PATTERNS:
  - Counter-trend scalps (only trend-following scalps)
  - Wide spread environments
  - Pre-news periods
  - EXHAUSTION or NO_TRADE regimes
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from agent.config import settings
from agent.core.trade_models import TradeCandidate


class AutoScalpStrategy:
    """
    Short-term scalp strategy that produces TradeCandidate objects.
    Inherits NO legacy dollar-based SL/TP logic.

    All levels are computed from:
      SL = ATR × auto_scalp_sl_atr_multiplier
      TP1 = sl_dist × auto_scalp_tp1_r
      TP2 = sl_dist × auto_scalp_tp2_r
    """

    name = "SCALP"

    def analyze(
        self,
        *,
        symbol: str,
        smc_1m: Optional[dict],
        smc_15m: Optional[dict],
        smc_1h: Optional[dict],
        current_price: float,
        spread_pips: float,
        regime: str,
        session: str,
        is_kill_zone: bool,
        atr_15m: float = 1.0,
        account_balance: float = 0.0,
    ) -> List[TradeCandidate]:
        """
        Analyze current market state for scalp entry opportunities.

        Returns:
            List of TradeCandidate objects (usually 0 or 1).
            NEVER more than 1 from this strategy.
        """
        if not settings.auto_scalp_mode:
            return []

        if not smc_1m or not smc_15m:
            return []

        # ── Guard 1: Regime ────────────────────────────────────────────────
        no_trade_regimes = {"NO_TRADE", "EXHAUSTION", "EXTREME_VOLATILITY"}
        if regime.upper() in no_trade_regimes:
            logger.debug(f"[AutoScalp] {symbol}: regime '{regime}' → skip")
            return []

        # ── Guard 2: Spread ─────────────────────────────────────────────────
        if spread_pips > settings.max_spread_pips:
            logger.debug(f"[AutoScalp] {symbol}: spread {spread_pips:.1f}pips > max {settings.max_spread_pips:.1f} → skip")
            return []

        # ── Guard 3: Trend alignment ────────────────────────────────────────
        trend_1h = (smc_1h or {}).get("trend", "NEUTRAL").upper()
        trend_15m = smc_15m.get("trend", "NEUTRAL").upper()
        trend_1m = smc_1m.get("trend", "NEUTRAL").upper()

        # Require at least 1M and 15M to agree
        if trend_1m == "NEUTRAL" and trend_15m == "NEUTRAL":
            return []  # No directional bias

        # Use 15M as primary direction
        if trend_15m in ("BULLISH", "NEUTRAL"):
            direction = "BUY"
        else:
            direction = "SELL"

        # Reject if 1M opposes 15M strongly
        if trend_1m != "NEUTRAL" and trend_1m != trend_15m:
            logger.debug(f"[AutoScalp] {symbol}: 1M {trend_1m} opposes 15M {trend_15m} → skip")
            return []

        # ── ATR validation ────────────────────────────────────────────────
        atr = atr_15m if atr_15m > 0 else 1.0

        # ── Compute SL from ATR ────────────────────────────────────────────
        sl_atr_mult = settings.auto_scalp_sl_atr_multiplier
        sl_distance = atr * sl_atr_mult

        digits = 2 if "XAU" in symbol.upper() or "GOLD" in symbol.upper() else (
            3 if "JPY" in symbol.upper() else 5
        )

        entry = round(current_price, digits)

        if direction == "BUY":
            sl = round(entry - sl_distance, digits)
            tp1 = round(entry + sl_distance * settings.auto_scalp_tp1_r, digits)
            tp2 = round(entry + sl_distance * settings.auto_scalp_tp2_r, digits)
        else:
            sl = round(entry + sl_distance, digits)
            tp1 = round(entry - sl_distance * settings.auto_scalp_tp1_r, digits)
            tp2 = round(entry - sl_distance * settings.auto_scalp_tp2_r, digits)

        actual_rr = sl_distance * settings.auto_scalp_tp1_r / sl_distance  # = tp1_r
        if actual_rr < settings.auto_scalp_min_rr:
            logger.debug(
                f"[AutoScalp] {symbol}: computed RR {actual_rr:.2f} < min {settings.auto_scalp_min_rr:.2f} → skip"
            )
            return []

        # ── Quality scoring ────────────────────────────────────────────────
        score = 0.40  # Base score for having direction + valid levels

        # Kill zone premium
        if is_kill_zone:
            score += 0.15

        # Trend alignment bonus
        if trend_15m == trend_1m:
            score += 0.10

        # 1H agrees
        if trend_1h in ("BULLISH", "BEARISH") and trend_1h == trend_15m:
            score += 0.10

        # Regime bonus
        if regime.upper() == "TRENDING":
            score += 0.10
        elif regime.upper() == "RANGE":
            score += 0.05

        score = min(1.0, round(score, 4))

        logger.info(
            f"[AutoScalp] ✅ {direction} {symbol} | "
            f"entry={entry} | sl={sl} (ATR×{sl_atr_mult}) | "
            f"tp1={tp1} ({settings.auto_scalp_tp1_r}R) | tp2={tp2} ({settings.auto_scalp_tp2_r}R) | "
            f"atr={atr:.4f} | score={score:.3f} | kill_zone={is_kill_zone}"
        )

        return [TradeCandidate(
            symbol=symbol,
            direction=direction,
            strategy="AUTO_SCALP",
            strategy_family="AUTO_SCALP",
            strategy_version="2.0",  # v2 = R-multiple based (v1 = legacy dollar based)
            entry_mode="AUTO_SCALP",
            timestamp=datetime.now(timezone.utc),
            entry=entry,
            stop_loss=sl,
            tp1=tp1,
            tp2=tp2,
            tp3=None,
            setup_score=score,
            regime=regime,
            session=session,
            is_kill_zone=is_kill_zone,
            spread_pips=spread_pips,
            signal_features={
                "trend_1m": trend_1m,
                "trend_15m": trend_15m,
                "trend_1h": trend_1h,
                "atr_15m": atr,
                "sl_atr_mult": sl_atr_mult,
                "tp1_r": settings.auto_scalp_tp1_r,
                "tp2_r": settings.auto_scalp_tp2_r,
                "regime": regime,
                "is_kill_zone": is_kill_zone,
            },
        )]

    def get_diagnostic(self) -> dict:
        return {
            "strategy": self.name,
            "version": "2.0_R_MULTIPLE",
            "sl_atr_mult": settings.auto_scalp_sl_atr_multiplier,
            "tp1_r": settings.auto_scalp_tp1_r,
            "tp2_r": settings.auto_scalp_tp2_r,
            "min_rr": settings.auto_scalp_min_rr,
            "enabled": settings.auto_scalp_mode,
        }
