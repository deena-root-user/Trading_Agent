"""
PAXIS Agent — Technical Indicator Calculator
Computes RSI, MACD, Bollinger Bands, EMA, ATR using pandas-ta.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from loguru import logger

try:
    import pandas_ta as ta
    TA_AVAILABLE = True
except ImportError:
    try:
        import pandas_ta_classic as ta
        TA_AVAILABLE = True
    except ImportError:
        TA_AVAILABLE = False
        logger.warning("pandas-ta / pandas-ta-classic not available")


@dataclass
class IndicatorSnapshot:
    """Clean snapshot of all indicator values for a single symbol / timeframe."""
    symbol: str
    timeframe: str
    timestamp: Optional[pd.Timestamp] = None

    # Price
    close: float = 0.0
    open_: float = 0.0
    high: float = 0.0
    low: float = 0.0

    # RSI
    rsi: float = 50.0

    # MACD
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_hist: float = 0.0
    macd_cross: str = "NONE"          # "BULLISH" | "BEARISH" | "NONE"

    # Bollinger Bands
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_position: str = "MIDDLE"       # "ABOVE_UPPER" | "NEAR_UPPER" | "MIDDLE" | "NEAR_LOWER" | "BELOW_LOWER"
    bb_width: float = 0.0
    bb_squeeze: bool = False            # True when BB width is in bottom 20th percentile (low vol)

    # EMA
    ema5: float = 0.0
    ema9: float = 0.0
    ema20: float = 0.0
    ema21: float = 0.0
    ema50: float = 0.0
    ema200: float = 0.0
    ema_trend: str = "NEUTRAL"        # "BULLISH" | "BEARISH" | "NEUTRAL"
    ema5_20_cross: str = "NONE"       # "GOLDEN" | "DEATH" | "NONE"
    ema9_21_cross: str = "NONE"       # "GOLDEN" | "DEATH" | "NONE"

    # ATR
    atr: float = 0.0
    atr_pips: float = 0.0

    # Realized Volatility
    realized_vol_20: float = 0.0       # 20-period realized volatility (annualized)

    # RSI Divergence
    rsi_divergence: str = "NONE"       # "BULLISH" | "BEARISH" | "HIDDEN_BULLISH" | "HIDDEN_BEARISH" | "NONE"

    # Volume
    volume: float = 0.0
    volume_avg: float = 0.0
    volume_ratio: float = 1.0         # current / avg

    # Stochastic RSI
    stoch_rsi_k: float = 50.0
    stoch_rsi_d: float = 50.0
    stoch_rsi_status: str = "NEUTRAL" # "OVERSOLD" | "OVERBOUGHT" | "BULLISH_CROSS" | "BEARISH_CROSS" | "NEUTRAL"

    # ADX (Trend Strength)
    adx: float = 0.0
    dmp: float = 0.0                   # +DI
    dmn: float = 0.0                   # -DI
    adx_trend_strength: str = "WEAK"   # "STRONG_TREND" | "MODERATE_TREND" | "WEAK_RANGING"

    # Pivot Points (Floor Trader)
    pivot: float = 0.0
    r1: float = 0.0
    s1: float = 0.0
    r2: float = 0.0
    s2: float = 0.0

    # Smart Money Concepts (SMC)
    smc_fvg_detected: str = "NONE"     # "BULLISH_FVG" | "BEARISH_FVG" | "NONE"
    smc_fvg_gap_size: float = 0.0
    smc_fvg_top: float = 0.0
    smc_fvg_bottom: float = 0.0
    smc_order_block: str = "NONE"       # "BULLISH_OB" | "BEARISH_OB" | "NONE"
    smc_ob_top: float = 0.0
    smc_ob_bottom: float = 0.0
    smc_market_structure: str = "NEUTRAL" # "BOS_BULLISH" | "BOS_BEARISH" | "CHOCH_BULLISH" | "CHOCH_BEARISH" | "NEUTRAL"

    def to_prompt_dict(self) -> dict:
        """Return a clean dict suitable for injection into LLM prompt."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": str(self.timestamp),
            "price": {
                "open": round(self.open_, 5),
                "high": round(self.high, 5),
                "low": round(self.low, 5),
                "close": round(self.close, 5),
            },
            "rsi_14": round(self.rsi, 2),
            "stochastic_rsi": {
                "k": round(self.stoch_rsi_k, 2),
                "d": round(self.stoch_rsi_d, 2),
                "status": self.stoch_rsi_status,
            },
            "macd": {
                "line": round(self.macd, 6),
                "signal": round(self.macd_signal, 6),
                "histogram": round(self.macd_hist, 6),
                "cross": self.macd_cross,
            },
            "adx_14": {
                "adx": round(self.adx, 2),
                "plus_di": round(self.dmp, 2),
                "minus_di": round(self.dmn, 2),
                "strength": self.adx_trend_strength,
            },
            "bollinger_bands": {
                "upper": round(self.bb_upper, 5),
                "middle": round(self.bb_middle, 5),
                "lower": round(self.bb_lower, 5),
                "position": self.bb_position,
                "width_pct": round(self.bb_width, 4),
                "squeeze": self.bb_squeeze,
            },
            "ema": {
                "ema5": round(self.ema5, 5),
                "ema9": round(self.ema9, 5),
                "ema20": round(self.ema20, 5),
                "ema21": round(self.ema21, 5),
                "ema50": round(self.ema50, 5),
                "ema200": round(self.ema200, 5),
                "trend": self.ema_trend,
                "ema5_20_cross": self.ema5_20_cross,
                "ema9_21_cross": self.ema9_21_cross,
            },
            "pivot_points": {
                "pivot": round(self.pivot, 5),
                "r1": round(self.r1, 5),
                "s1": round(self.s1, 5),
                "r2": round(self.r2, 5),
                "s2": round(self.s2, 5),
            },
            "smart_money_concepts": {
                "fvg": {
                    "type": self.smc_fvg_detected,
                    "top": round(self.smc_fvg_top, 5),
                    "bottom": round(self.smc_fvg_bottom, 5),
                    "gap_size": round(self.smc_fvg_gap_size, 5),
                },
                "order_block": {
                    "type": self.smc_order_block,
                    "top": round(self.smc_ob_top, 5),
                    "bottom": round(self.smc_ob_bottom, 5),
                },
                "structure": self.smc_market_structure,
            },
            "atr_14": {
                "raw": round(self.atr, 6),
                "pips": round(self.atr_pips, 1),
            },
            "realized_vol_20": round(self.realized_vol_20, 6),
            "rsi_divergence": self.rsi_divergence,
            "volume": {
                "current": self.volume,
                "avg_20": round(self.volume_avg, 1),
                "ratio": round(self.volume_ratio, 2),
            },
        }

    def to_tradingview_dict(self) -> dict:
        """Return clean dict structured for TradingView chart overlays & technical analysis HUD."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "price": {
                "close": round(self.close, 2 if "XAU" in self.symbol else 5),
                "high": round(self.high, 2 if "XAU" in self.symbol else 5),
                "low": round(self.low, 2 if "XAU" in self.symbol else 5),
            },
            "support_resistance": {
                "resistance_2": round(self.r2, 2 if "XAU" in self.symbol else 5),
                "resistance_1": round(self.r1, 2 if "XAU" in self.symbol else 5),
                "pivot": round(self.pivot, 2 if "XAU" in self.symbol else 5),
                "support_1": round(self.s1, 2 if "XAU" in self.symbol else 5),
                "support_2": round(self.s2, 2 if "XAU" in self.symbol else 5),
            },
            "rsi": {
                "value": round(self.rsi, 1),
                "status": "OVERBOUGHT" if self.rsi >= 70 else ("OVERSOLD" if self.rsi <= 30 else "NEUTRAL")
            },
            "ema_trend": {
                "ema50": round(self.ema50, 2 if "XAU" in self.symbol else 5),
                "ema200": round(self.ema200, 2 if "XAU" in self.symbol else 5),
                "trend": self.ema_trend,
                "cross": self.ema9_21_cross,
            },
            "smart_money_concepts": {
                "fvg_type": self.smc_fvg_detected,
                "fvg_top": round(self.smc_fvg_top, 2 if "XAU" in self.symbol else 5),
                "fvg_bottom": round(self.smc_fvg_bottom, 2 if "XAU" in self.symbol else 5),
                "order_block_type": self.smc_order_block,
                "ob_top": round(self.smc_ob_top, 2 if "XAU" in self.symbol else 5),
                "ob_bottom": round(self.smc_ob_bottom, 2 if "XAU" in self.symbol else 5),
                "market_structure": self.smc_market_structure,
            }
        }


class IndicatorCalculator:
    """Calculates all technical indicators from an OHLCV DataFrame."""

    # Pip sizes for common instruments
    _JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY"}

    def _pip_size(self, symbol: str) -> float:
        sym_upper = symbol.upper()
        if sym_upper in self._JPY_PAIRS or any(x in sym_upper for x in ["XAU", "GOLD"]):
            return 0.01
        return 0.0001

    def calculate(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> Optional[IndicatorSnapshot]:
        """
        Calculate all indicators from a candle DataFrame.
        Requires columns: time, open, high, low, close, volume
        Returns an IndicatorSnapshot or None on error.
        """
        if df is None or len(df) < 15:
            logger.warning(f"Insufficient candle data for indicators ({symbol}): {len(df) if df is not None else 0} rows")
            return None

        try:
            snap = IndicatorSnapshot(symbol=symbol, timeframe=timeframe)
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            snap.timestamp = latest["time"]
            snap.close = float(latest["close"])
            snap.open_ = float(latest["open"])
            snap.high = float(latest["high"])
            snap.low = float(latest["low"])
            snap.volume = float(latest.get("volume", 0))

            pip = self._pip_size(symbol)

            # ── RSI ───────────────────────────────────────────────────────────
            rsi_series = ta.rsi(df["close"], length=14)
            if rsi_series is not None and not rsi_series.empty:
                snap.rsi = float(rsi_series.iloc[-1])

            # ── MACD ──────────────────────────────────────────────────────────
            macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty:
                snap.macd        = float(macd_df.iloc[-1, 0])   # MACD_12_26_9
                snap.macd_hist   = float(macd_df.iloc[-1, 1])   # MACDh_12_26_9
                snap.macd_signal = float(macd_df.iloc[-1, 2])   # MACDs_12_26_9

                prev_macd   = float(macd_df.iloc[-2, 0])
                prev_signal = float(macd_df.iloc[-2, 2])
                if prev_macd < prev_signal and snap.macd > snap.macd_signal:
                    snap.macd_cross = "BULLISH"
                elif prev_macd > prev_signal and snap.macd < snap.macd_signal:
                    snap.macd_cross = "BEARISH"

            # ── Bollinger Bands ───────────────────────────────────────────────
            bb_df = ta.bbands(df["close"], length=20, std=2)
            if bb_df is not None and not bb_df.empty:
                snap.bb_upper  = float(bb_df.iloc[-1, 0])   # BBU_20_2.0
                snap.bb_middle = float(bb_df.iloc[-1, 1])   # BBM_20_2.0
                snap.bb_lower  = float(bb_df.iloc[-1, 2])   # BBL_20_2.0
                band_width = snap.bb_upper - snap.bb_lower
                snap.bb_width = band_width / snap.bb_middle if snap.bb_middle else 0

                # BB Squeeze detection — width in bottom 20th percentile of last 120 bars
                try:
                    bb_width_series = (bb_df.iloc[:, 0] - bb_df.iloc[:, 2]) / bb_df.iloc[:, 1]
                    bb_width_series = bb_width_series.dropna()
                    if len(bb_width_series) >= 20:
                        pct_20 = bb_width_series.tail(120).quantile(0.20)
                        snap.bb_squeeze = bool(snap.bb_width <= pct_20)
                except Exception:
                    pass

                c = snap.close
                if c > snap.bb_upper:
                    snap.bb_position = "ABOVE_UPPER"
                elif c > snap.bb_middle + 0.3 * (snap.bb_upper - snap.bb_middle):
                    snap.bb_position = "NEAR_UPPER"
                elif c < snap.bb_lower:
                    snap.bb_position = "BELOW_LOWER"
                elif c < snap.bb_middle - 0.3 * (snap.bb_middle - snap.bb_lower):
                    snap.bb_position = "NEAR_LOWER"
                else:
                    snap.bb_position = "MIDDLE"

            # ── EMA ───────────────────────────────────────────────────────────
            for length, attr in [(5, "ema5"), (9, "ema9"), (20, "ema20"), (21, "ema21"), (50, "ema50"), (200, "ema200")]:
                if len(df) >= length:
                    s = ta.ema(df["close"], length=length)
                    if s is not None and not s.empty:
                        setattr(snap, attr, float(s.iloc[-1]))

            # EMA trend (evaluated across EMA 5, 20, 50)
            if snap.ema5 > snap.ema20 > snap.ema50 or snap.ema9 > snap.ema21 > snap.ema50:
                snap.ema_trend = "BULLISH"
            elif snap.ema5 < snap.ema20 < snap.ema50 or snap.ema9 < snap.ema21 < snap.ema50:
                snap.ema_trend = "BEARISH"
            else:
                snap.ema_trend = "NEUTRAL"

            # EMA 5/20 cross (Fast Momentum Cross)
            if len(df) >= 21:
                ema5_series  = ta.ema(df["close"], length=5)
                ema20_series = ta.ema(df["close"], length=20)
                if ema5_series is not None and ema20_series is not None and len(ema5_series) >= 2:
                    prev_ema5  = float(ema5_series.iloc[-2])
                    prev_ema20 = float(ema20_series.iloc[-2])
                    if prev_ema5 < prev_ema20 and snap.ema5 > snap.ema20:
                        snap.ema5_20_cross = "GOLDEN"
                    elif prev_ema5 > prev_ema20 and snap.ema5 < snap.ema20:
                        snap.ema5_20_cross = "DEATH"

            # EMA 9/21 cross
            if len(df) >= 22:
                prev_ema9  = float(ta.ema(df["close"], length=9).iloc[-2])
                prev_ema21 = float(ta.ema(df["close"], length=21).iloc[-2])
                if prev_ema9 < prev_ema21 and snap.ema9 > snap.ema21:
                    snap.ema9_21_cross = "GOLDEN"
                elif prev_ema9 > prev_ema21 and snap.ema9 < snap.ema21:
                    snap.ema9_21_cross = "DEATH"

            # ── ATR ───────────────────────────────────────────────────────────
            atr_series = ta.atr(df["high"], df["low"], df["close"], length=14)
            if atr_series is not None and not atr_series.empty:
                snap.atr = float(atr_series.iloc[-1])
                snap.atr_pips = round(snap.atr / pip, 1)

            # ── Realized Volatility (20-period) ──────────────────────────────
            try:
                import numpy as np
                if len(df) >= 21:
                    log_returns = np.log(df["close"].tail(21) / df["close"].tail(21).shift(1)).dropna()
                    if len(log_returns) >= 10:
                        snap.realized_vol_20 = float(log_returns.std() * np.sqrt(252))
            except Exception:
                pass

            # ── RSI Divergence Detection ──────────────────────────────────────
            try:
                if rsi_series is not None and len(rsi_series) >= 30 and len(df) >= 30:
                    clean_rsi = rsi_series.dropna()
                    if len(clean_rsi) >= 20:
                        rsi_vals = clean_rsi.tail(20).values
                        k = len(rsi_vals)
                        snap.rsi_divergence = self._detect_rsi_divergence(
                            df["close"].tail(k).values,
                            df["high"].tail(k).values,
                            df["low"].tail(k).values,
                            rsi_vals,
                        )
            except Exception:
                pass

            # ── Volume ────────────────────────────────────────────────────────
            if "volume" in df.columns:
                snap.volume_avg = float(df["volume"].tail(20).mean())
                snap.volume_ratio = (
                    snap.volume / snap.volume_avg if snap.volume_avg > 0 else 1.0
                )

            # ── Stochastic RSI ────────────────────────────────────────────────
            try:
                stoch_df = ta.stochrsi(df["close"], length=14, rsi_length=14, k=3, d=3)
                if stoch_df is not None and not stoch_df.empty:
                    snap.stoch_rsi_k = float(stoch_df.iloc[-1, 0])
                    snap.stoch_rsi_d = float(stoch_df.iloc[-1, 1])
                    if snap.stoch_rsi_k < 20 and snap.stoch_rsi_d < 20:
                        snap.stoch_rsi_status = "OVERSOLD"
                    elif snap.stoch_rsi_k > 80 and snap.stoch_rsi_d > 80:
                        snap.stoch_rsi_status = "OVERBOUGHT"
                    elif len(stoch_df) >= 2:
                        prev_k = float(stoch_df.iloc[-2, 0])
                        prev_d = float(stoch_df.iloc[-2, 1])
                        if prev_k < prev_d and snap.stoch_rsi_k > snap.stoch_rsi_d:
                            snap.stoch_rsi_status = "BULLISH_CROSS"
                        elif prev_k > prev_d and snap.stoch_rsi_k < snap.stoch_rsi_d:
                            snap.stoch_rsi_status = "BEARISH_CROSS"
            except Exception:
                pass

            # ── ADX (Trend Strength) ──────────────────────────────────────────
            try:
                adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
                if adx_df is not None and not adx_df.empty:
                    snap.adx = float(adx_df.iloc[-1, 0])
                    snap.dmp = float(adx_df.iloc[-1, 1])
                    snap.dmn = float(adx_df.iloc[-1, 2])
                    if snap.adx >= 25:
                        snap.adx_trend_strength = "STRONG_TREND"
                    elif snap.adx >= 20:
                        snap.adx_trend_strength = "MODERATE_TREND"
                    else:
                        snap.adx_trend_strength = "WEAK_RANGING"
            except Exception:
                pass

            # ── Pivot Points (Floor Trader) ──────────────────────────────────
            if len(df) >= 2:
                last_high  = float(df["high"].iloc[-2])
                last_low   = float(df["low"].iloc[-2])
                last_close = float(df["close"].iloc[-2])
                snap.pivot = (last_high + last_low + last_close) / 3.0
                snap.r1    = 2 * snap.pivot - last_low
                snap.s1    = 2 * snap.pivot - last_high
                snap.r2    = snap.pivot + (last_high - last_low)
                snap.s2    = snap.pivot - (last_high - last_low)

            # ── Smart Money Concepts (SMC: FVG, Order Block, BOS/CHoCH) ──────
            # 1. Fair Value Gap (FVG)
            for idx in range(len(df) - 1, max(len(df) - 10, 2), -1):
                c1 = df.iloc[idx - 2]
                c3 = df.iloc[idx]
                if c3["low"] > c1["high"]:
                    snap.smc_fvg_detected = "BULLISH_FVG"
                    snap.smc_fvg_bottom = float(c1["high"])
                    snap.smc_fvg_top = float(c3["low"])
                    snap.smc_fvg_gap_size = round((snap.smc_fvg_top - snap.smc_fvg_bottom) / pip, 1)
                    break
                elif c3["high"] < c1["low"]:
                    snap.smc_fvg_detected = "BEARISH_FVG"
                    snap.smc_fvg_bottom = float(c3["high"])
                    snap.smc_fvg_top = float(c1["low"])
                    snap.smc_fvg_gap_size = round((snap.smc_fvg_top - snap.smc_fvg_bottom) / pip, 1)
                    break

            # 2. Order Block (OB)
            for idx in range(len(df) - 2, max(len(df) - 15, 2), -1):
                candle = df.iloc[idx]
                next_candle = df.iloc[idx + 1]
                if candle["close"] < candle["open"] and next_candle["close"] > candle["high"]:
                    snap.smc_order_block = "BULLISH_OB"
                    snap.smc_ob_bottom = float(candle["low"])
                    snap.smc_ob_top = float(candle["high"])
                    break
                elif candle["close"] > candle["open"] and next_candle["close"] < candle["low"]:
                    snap.smc_order_block = "BEARISH_OB"
                    snap.smc_ob_bottom = float(candle["low"])
                    snap.smc_ob_top = float(candle["high"])
                    break

            # 3. Market Structure (BOS / CHoCH)
            if len(df) >= 20:
                recent_high = float(df["high"].tail(20).iloc[:-2].max())
                recent_low  = float(df["low"].tail(20).iloc[:-2].min())
                if snap.close > recent_high:
                    snap.smc_market_structure = "BOS_BULLISH"
                elif snap.close < recent_low:
                    snap.smc_market_structure = "BOS_BEARISH"

            logger.debug(
                f"Indicators ✓ {symbol} {timeframe} | "
                f"RSI={snap.rsi:.1f} | StochRSI={snap.stoch_rsi_status} | "
                f"SMC_FVG={snap.smc_fvg_detected} | SMC_OB={snap.smc_order_block} | "
                f"EMA_trend={snap.ema_trend} | ATR_pips={snap.atr_pips}"
            )
            return snap

        except Exception as exc:
            logger.error(f"Indicator calculation error ({symbol}): {exc}")
            return None

    @staticmethod
    def _detect_rsi_divergence(
        close: "np.ndarray",
        high: "np.ndarray",
        low: "np.ndarray",
        rsi: "np.ndarray",
    ) -> str:
        """
        Detect RSI divergence by comparing last two swing extremes.
        Returns: "BULLISH" | "BEARISH" | "HIDDEN_BULLISH" | "HIDDEN_BEARISH" | "NONE"
        """
        import numpy as np
        n = len(close)
        if n < 10 or len(rsi) != n or len(high) != n or len(low) != n:
            return "NONE"

        # Find local lows (for bullish divergence)
        local_lows = []
        for i in range(2, n - 2):
            if low[i] <= low[i - 1] and low[i] <= low[i - 2] and low[i] <= low[i + 1] and low[i] <= low[i + 2]:
                local_lows.append(i)

        # Find local highs (for bearish divergence)
        local_highs = []
        for i in range(2, n - 2):
            if high[i] >= high[i - 1] and high[i] >= high[i - 2] and high[i] >= high[i + 1] and high[i] >= high[i + 2]:
                local_highs.append(i)

        # Bullish divergence: price makes lower low, RSI makes higher low
        if len(local_lows) >= 2:
            i1, i2 = local_lows[-2], local_lows[-1]
            if low[i2] < low[i1] and rsi[i2] > rsi[i1]:
                return "BULLISH"
            # Hidden bullish: price makes higher low, RSI makes lower low
            if low[i2] > low[i1] and rsi[i2] < rsi[i1]:
                return "HIDDEN_BULLISH"

        # Bearish divergence: price makes higher high, RSI makes lower high
        if len(local_highs) >= 2:
            i1, i2 = local_highs[-2], local_highs[-1]
            if high[i2] > high[i1] and rsi[i2] < rsi[i1]:
                return "BEARISH"
            # Hidden bearish: price makes lower high, RSI makes higher high
            if high[i2] < high[i1] and rsi[i2] > rsi[i1]:
                return "HIDDEN_BEARISH"

        return "NONE"


# Singleton
indicator_calculator = IndicatorCalculator()
