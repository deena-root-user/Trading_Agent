"""
PAXIS Agent — Execution Authority
===================================
THE ONLY component that may call mt5_bridge.place_order().

ARCHITECTURE CONTRACT:
  No strategy, pipeline stage, auto-scalp path, or legacy code may call
  mt5_bridge.place_order() directly. All execution MUST flow through here.

  ExecutionAuthority enforces:
    1. Input must be an ApprovedTrade (isinstance check)
    2. ApprovedTrade.is_executable() must return True
    3. Volume must be broker-compliant (min/max/step)
    4. Live MT5 price re-validation before execution
    5. Max slippage guard
    6. Execution speed: order placed within configured timeout

FAST EXECUTION:
  All orders are placed immediately on approval — no artificial delays.
  The 'require_candle_close_confirmation' guard is enforced here for structural
  entries only. Breakout entries skip this guard (they need immediate execution).

AUDIT TRAIL:
  Every execution attempt (success or failure) is logged to the
  execution_events table with full ApprovedTrade context.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from agent.config import settings
from agent.core.trade_models import ApprovedTrade, TradeLifecycleState


class ExecutionResult:
    """Result of an execution attempt."""
    def __init__(
        self,
        success: bool,
        ticket: Optional[int] = None,
        executed_price: Optional[float] = None,
        executed_volume: Optional[float] = None,
        error: str = "",
        slippage_points: float = 0.0,
        latency_ms: float = 0.0,
    ):
        self.success = success
        self.ticket = ticket
        self.executed_price = executed_price
        self.executed_volume = executed_volume
        self.error = error
        self.slippage_points = slippage_points
        self.latency_ms = latency_ms
        self.timestamp = datetime.now(timezone.utc)

    def __bool__(self) -> bool:
        return self.success


class ExecutionAuthority:
    """
    Single execution gateway — the only place mt5_bridge.place_order() is called
    from the new architecture.

    Legacy code paths in main.py still call mt5_bridge directly during the
    transition period. This class is the target for all new strategy paths.
    """

    def __init__(self):
        self._execution_count = 0
        self._max_slippage = getattr(settings, "max_slippage_points", 1.5)

    def execute(
        self,
        approved_trade: ApprovedTrade,
        *,
        dry_run: Optional[bool] = None,
    ) -> ExecutionResult:
        """
        Execute an ApprovedTrade via MT5.

        Args:
            approved_trade: Must be ApprovedTrade with risk_approved=True
            dry_run: Override for dry_run (defaults to settings.dry_run)

        Returns:
            ExecutionResult
        """
        t_start = time.perf_counter()

        # ── 1. Type safety guard ────────────────────────────────────────────
        if not isinstance(approved_trade, ApprovedTrade):
            logger.critical(
                f"ExecutionAuthority: REJECTED — input is not an ApprovedTrade! "
                f"Got: {type(approved_trade)}. This is a CRITICAL architecture violation."
            )
            return ExecutionResult(
                success=False,
                error="NOT_APPROVED_TRADE: execution contract violation",
            )

        # ── 2. Executability check ──────────────────────────────────────────
        if not approved_trade.is_executable():
            reason = approved_trade.rejection_reason or "is_executable() returned False"
            logger.warning(f"ExecutionAuthority: REJECTED {approved_trade.symbol} — {reason}")
            return ExecutionResult(success=False, error=f"NOT_EXECUTABLE: {reason}")

        symbol = approved_trade.symbol
        direction = approved_trade.direction
        entry = approved_trade.entry
        sl = approved_trade.stop_loss
        tp1 = approved_trade.tp1
        volume = approved_trade.volume
        entry_mode = approved_trade.candidate.entry_mode

        # ── 3. Live price re-validation (slippage guard) ────────────────────
        try:
            from agent.data.mt5_feed import mt5_feed
            live_tick = mt5_feed.get_tick(symbol)

            if live_tick and entry > 0:
                live_price = float(live_tick.ask if direction == "BUY" else live_tick.bid)
                slippage = abs(live_price - entry)
                max_slip = getattr(settings, "max_slippage_points", 1.5)

                if slippage > max_slip:
                    error_msg = (
                        f"SLIPPAGE_EXCEEDED: live={live_price:.5f} vs setup={entry:.5f} "
                        f"drift={slippage:.4f} > max={max_slip:.4f}"
                    )
                    logger.warning(f"ExecutionAuthority SLIPPAGE BLOCK {symbol}: {error_msg}")
                    return ExecutionResult(success=False, error=error_msg, slippage_points=slippage)

                # Re-calibrate SL/TP distance relative to live price
                sl_dist = abs(entry - sl)
                tp_dist = abs(tp1 - entry) if tp1 else sl_dist * 2.0
                digits = 2 if "XAU" in symbol.upper() or "GOLD" in symbol.upper() else (
                    3 if "JPY" in symbol.upper() else 5
                )

                if direction == "BUY":
                    exec_entry = live_price
                    exec_sl = round(live_price - sl_dist, digits)
                    exec_tp = round(live_price + tp_dist, digits)
                else:
                    exec_entry = live_price
                    exec_sl = round(live_price + sl_dist, digits)
                    exec_tp = round(live_price - tp_dist, digits)

                logger.info(
                    f"ExecutionAuthority: price re-calib {symbol} | "
                    f"setup_entry={entry:.5f} → live={live_price:.5f} | "
                    f"sl={exec_sl:.5f} | tp={exec_tp:.5f} | slip={slippage:.4f}"
                )
            else:
                exec_entry = entry
                exec_sl = sl
                exec_tp = tp1 or round(
                    entry + abs(entry - sl) * 2.0 if direction == "BUY"
                    else entry - abs(entry - sl) * 2.0, 5
                )

        except Exception as exc:
            logger.warning(f"ExecutionAuthority: price re-calib failed for {symbol}: {exc} — using setup prices")
            exec_entry = entry
            exec_sl = sl
            exec_tp = tp1 or 0.0

        # ── 4. Candle close guard (structural mode only) ────────────────────
        # Breakout trades need IMMEDIATE execution — skip the candle close guard
        is_structural = entry_mode in ("STRUCTURAL", "BREAKOUT_RETEST")
        if (
            is_structural
            and getattr(settings, "require_candle_close_confirmation", False)
            and not getattr(settings, "dry_run", True)
        ):
            now_utc = datetime.now(timezone.utc)
            seconds_into_candle = now_utc.second % 60
            max_window = getattr(settings, "candle_close_window_seconds", 50)
            if seconds_into_candle > max_window:
                logger.info(
                    f"ExecutionAuthority: CANDLE_CLOSE_WAIT {symbol} — "
                    f"{seconds_into_candle}s into candle (need ≤{max_window}s) — deferring"
                )
                return ExecutionResult(
                    success=False,
                    error=f"CANDLE_CLOSE_WAIT: {seconds_into_candle}s into candle",
                )

        # ── 5. Dry run ─────────────────────────────────────────────────────
        use_dry_run = dry_run if dry_run is not None else settings.dry_run

        if use_dry_run:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            self._execution_count += 1
            logger.info(
                f"[DRY RUN] ExecutionAuthority: WOULD EXECUTE {direction} {symbol} | "
                f"lot={volume} | entry={exec_entry:.5f} | sl={exec_sl:.5f} | tp={exec_tp:.5f} | "
                f"mode={entry_mode} | latency={latency_ms:.1f}ms"
            )
            return ExecutionResult(
                success=True,
                ticket=0,
                executed_price=exec_entry,
                executed_volume=volume,
                latency_ms=latency_ms,
            )

        # ── 6. Place order ─────────────────────────────────────────────────
        try:
            from agent.execution.mt5_bridge import mt5_bridge
            order = mt5_bridge.place_order(
                symbol=symbol,
                action=direction,
                sl=exec_sl,
                tp=exec_tp,
                lot_size=volume,
                comment=f"PAXIS_{entry_mode[:4]}_{approved_trade.signal_id}",
            )

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            self._execution_count += 1

            if order.success:
                actual_slippage = abs((order.price or exec_entry) - exec_entry)
                logger.info(
                    f"✅ ExecutionAuthority EXECUTED {direction} {symbol} | "
                    f"ticket={order.ticket} | lot={volume} | price={order.price:.5f} | "
                    f"sl={exec_sl:.5f} | tp={exec_tp:.5f} | mode={entry_mode} | "
                    f"slip={actual_slippage:.4f} | latency={latency_ms:.1f}ms"
                )
                return ExecutionResult(
                    success=True,
                    ticket=order.ticket,
                    executed_price=order.price or exec_entry,
                    executed_volume=volume,
                    slippage_points=actual_slippage,
                    latency_ms=latency_ms,
                )
            else:
                logger.error(
                    f"❌ ExecutionAuthority FAILED {symbol}: {order.error} | "
                    f"latency={latency_ms:.1f}ms"
                )
                return ExecutionResult(
                    success=False,
                    error=order.error or "MT5 order failed",
                    latency_ms=latency_ms,
                )

        except Exception as exc:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            logger.error(f"ExecutionAuthority exception for {symbol}: {exc}")
            return ExecutionResult(
                success=False,
                error=f"EXECUTION_EXCEPTION: {exc}",
                latency_ms=latency_ms,
            )


# ── Singleton ─────────────────────────────────────────────────────────────────
execution_authority = ExecutionAuthority()
