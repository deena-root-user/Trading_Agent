"""
PAXIS Agent — Parallel Strategy Analyzer
==========================================
Runs SMC, ICT, BREAKOUT, and SCALP strategy analyses concurrently
using ThreadPoolExecutor, then deduplicates and ranks candidates.

DESIGN:
  • Each strategy runs in its own thread with a timeout
  • All candidates are returned as TradeCandidate objects
  • Deduplication: same symbol + direction within signal_dedup_window_seconds → one trade
  • Ranking: setup_score + is_kill_zone bonus → highest quality selected
  • All strategies share the SAME Risk Engine → no per-strategy risk bypass

ARCHITECTURE:
  StrategyAnalyzer.analyze_all(context: PipelineContext)
        ↓ ThreadPoolExecutor
    ┌───┬───┬───┬───┐
    SMC ICT BRK SCA  (parallel threads)
    └───┴───┴───┴───┘
        ↓ merge
  List[TradeCandidate]  (deduplicated, ranked)
        ↓
  Risk Engine → ApprovedTrade
        ↓
  ExecutionAuthority → MT5

KILL ZONE PRIORITY:
  Setups that occur during London/NY/Asian kill zones receive a confluence bonus.
  This is applied BEFORE deduplication so kill zone setups are preferred.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from agent.config import settings
from agent.core.trade_models import PipelineContext, TradeCandidate


# ══════════════════════════════════════════════════════════════════════════════
# KILL ZONE HELPER
# ══════════════════════════════════════════════════════════════════════════════


def _parse_time(hhmm: str) -> Tuple[int, int]:
    """Parse 'HH:MM' string into (hour, minute) tuple."""
    try:
        h, m = hhmm.split(":")
        return int(h), int(m)
    except Exception:
        return 0, 0


def is_in_kill_zone(dt: Optional[datetime] = None) -> bool:
    """
    Return True if current UTC time is within any configured kill zone.

    Kill zones (default):
      London:   07:00–09:00 UTC
      NY:       12:00–14:00 UTC
      Asian:    00:00–03:00 UTC
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    hour = dt.hour
    minute = dt.minute
    current_minutes = hour * 60 + minute

    def _in_zone(start_hhmm: str, end_hhmm: str) -> bool:
        sh, sm = _parse_time(start_hhmm)
        eh, em = _parse_time(end_hhmm)
        start_mins = sh * 60 + sm
        end_mins = eh * 60 + em
        if start_mins <= end_mins:
            return start_mins <= current_minutes < end_mins
        else:  # wraps midnight
            return current_minutes >= start_mins or current_minutes < end_mins

    return (
        _in_zone(settings.london_kill_zone_start, settings.london_kill_zone_end)
        or _in_zone(settings.ny_kill_zone_start, settings.ny_kill_zone_end)
        or _in_zone(settings.asian_kill_zone_start, settings.asian_kill_zone_end)
    )


def get_kill_zone_name(dt: Optional[datetime] = None) -> str:
    """Return the name of the current kill zone, or 'NONE'."""
    if dt is None:
        dt = datetime.now(timezone.utc)

    hour = dt.hour
    minute = dt.minute
    current_minutes = hour * 60 + minute

    def _in_zone(start_hhmm: str, end_hhmm: str) -> bool:
        sh, sm = _parse_time(start_hhmm)
        eh, em = _parse_time(end_hhmm)
        start_mins = sh * 60 + sm
        end_mins = eh * 60 + em
        if start_mins <= end_mins:
            return start_mins <= current_minutes < end_mins
        return current_minutes >= start_mins or current_minutes < end_mins

    if _in_zone(settings.london_kill_zone_start, settings.london_kill_zone_end):
        return "LONDON_KILL_ZONE"
    if _in_zone(settings.ny_kill_zone_start, settings.ny_kill_zone_end):
        return "NY_KILL_ZONE"
    if _in_zone(settings.asian_kill_zone_start, settings.asian_kill_zone_end):
        return "ASIAN_KILL_ZONE"
    return "NONE"


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL DEDUPLICATION
# ══════════════════════════════════════════════════════════════════════════════


class SignalDeduplicator:
    """
    Prevents sending the same trade signal multiple times within a time window.

    Deduplication key: symbol + direction (not strategy, since two strategies may
    legitimately identify the same structure independently).

    When two strategies both signal BUY XAUUSD simultaneously:
    → Only the higher-quality candidate (higher setup_score) is kept.
    """

    def __init__(self):
        # key → (signal_id, timestamp)
        self._seen: Dict[str, Tuple[str, float]] = {}

    def _key(self, c: TradeCandidate) -> str:
        # Breakout mode adds structure_id to prevent cross-level conflicts
        if c.entry_mode == "BREAKOUT" and c.structure_id:
            return f"{c.symbol}_{c.direction}_{c.structure_id}"
        return f"{c.symbol}_{c.direction}"

    def deduplicate(self, candidates: List[TradeCandidate]) -> List[TradeCandidate]:
        """
        Remove duplicate candidates. For identical keys, keep the highest-scoring one.

        BUG-4 FIX: Signals are NOT added to the dedup set here.
        Only record_executed() adds to the set — so rejected/blocked signals
        never lock out valid re-evaluation in the next cycle.
        """
        window = settings.signal_dedup_window_seconds
        now = time.time()

        # Evict expired entries
        expired = [k for k, (_, ts) in self._seen.items() if now - ts > window]
        for k in expired:
            del self._seen[k]

        # Group by dedup key, keep best score per key
        best_by_key: Dict[str, TradeCandidate] = {}
        for c in candidates:
            key = self._key(c)
            existing = best_by_key.get(key)
            if existing is None or c.setup_score > existing.setup_score:
                best_by_key[key] = c

        # Filter out only keys that were ACTUALLY EXECUTED recently
        result = []
        for key, c in best_by_key.items():
            if key in self._seen:
                sig_id, ts = self._seen[key]
                if now - ts < window:
                    logger.debug(
                        f"[Dedup] Suppressed duplicate {c.symbol} {c.direction} "
                        f"(seen {now - ts:.0f}s ago, window={window:.0f}s)"
                    )
                    continue

            # Accept candidate — do NOT record in _seen here.
            # Only record_executed() adds to _seen, so rejected/blocked
            # candidates never lock out the next cycle's evaluation.
            result.append(c)

        return result

    def record_executed(self, candidate: TradeCandidate) -> None:
        """Call after a trade is executed to refresh the dedup window."""
        key = self._key(candidate)
        self._seen[key] = (candidate.signal_id, time.time())


# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL ANALYZER
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class StrategyResult:
    strategy: str
    candidates: List[TradeCandidate]
    elapsed_ms: float
    error: Optional[str] = None


class ParallelAnalyzer:
    """
    Runs all configured strategies in parallel and returns deduplicated,
    kill-zone-prioritized TradeCandidate list.

    Usage:
        analyzer = ParallelAnalyzer()
        candidates = analyzer.analyze_all(context)
        # All candidates ready for Risk Engine
    """

    def __init__(self):
        self._deduplicator = SignalDeduplicator()
        # Strategy instances (lazy loaded)
        self._strategies: Dict[str, Any] = {}

    def _load_strategies(self, active_modes: list[str]) -> None:
        """Lazy-load strategy instances."""
        from agent.strategy.smc_strategy import SMCStrategy
        from agent.strategy.ict_strategy import ICTStrategy
        from agent.strategy.breakout_strategy import BreakoutStrategy
        from agent.strategy.auto_scalp_strategy import AutoScalpStrategy

        _cls_map = {
            "SMC":     SMCStrategy,
            "ICT":     ICTStrategy,
            "BREAKOUT": BreakoutStrategy,
            "SCALP":   AutoScalpStrategy,
        }

        for mode in active_modes:
            if mode not in self._strategies:
                cls = _cls_map.get(mode)
                if cls:
                    try:
                        self._strategies[mode] = cls()
                        logger.info(f"[ParallelAnalyzer] Loaded strategy: {mode}")
                    except Exception as exc:
                        logger.error(f"[ParallelAnalyzer] Failed to load {mode}: {exc}")

    def _run_strategy(
        self,
        mode: str,
        strategy: Any,
        context: PipelineContext,
        is_kill_zone: bool,
    ) -> StrategyResult:
        """
        Run a single strategy within a thread.
        All exceptions are caught and returned as StrategyResult.error.
        """
        t_start = time.perf_counter()
        try:
            candidates: List[TradeCandidate] = []

            if mode in ("SMC", "ICT"):
                # Existing strategies use the StrategyBase interface
                from agent.strategy.base import StrategyContext
                ctx = StrategyContext(
                    symbol=context.symbol,
                    current_price=context.mid_price,
                    smc_4h=context.smc_4h,
                    smc_1h=context.smc_1h,
                    smc_15m=context.smc_15m,
                    smc_1m=context.smc_1m,
                    session_data=context.session_dict,
                    current_session=context.session,
                )
                decision = strategy.analyze(ctx)
                if decision and decision.is_actionable and decision.candidates:
                    for c in decision.candidates:
                        tc = _convert_legacy_candidate(c, context, mode)
                        if tc:
                            candidates.append(tc)

            elif mode == "BREAKOUT":
                candidates = strategy.analyze(
                    symbol=context.symbol,
                    df_15m=None,   # Pipeline context doesn't carry raw DFs — strategies use SMC
                    df_1h=None,
                    df_4h=None,
                    df_1m=None,
                    smc_15m=context.smc_15m,
                    smc_1h=context.smc_1h,
                    smc_4h=context.smc_4h,
                    current_price=context.mid_price,
                    spread_pips=context.spread_pips,
                    regime=context.regime,
                    session=context.session,
                    is_kill_zone=is_kill_zone,
                    atr_15m=context.atr_15m,
                    atr_1h=context.atr_1h,
                )

            elif mode == "SCALP":
                candidates = strategy.analyze(
                    symbol=context.symbol,
                    smc_1m=context.smc_1m,
                    smc_15m=context.smc_15m,
                    smc_1h=context.smc_1h,
                    current_price=context.mid_price,
                    spread_pips=context.spread_pips,
                    regime=context.regime,
                    session=context.session,
                    is_kill_zone=is_kill_zone,
                    atr_15m=context.atr_15m,
                    account_balance=context.account_balance,
                )

            # Apply kill zone bonus to all candidates
            kz_bonus = settings.kill_zone_priority_bonus if is_kill_zone else 0.0
            for c in candidates:
                c.is_kill_zone = is_kill_zone
                if is_kill_zone and kz_bonus > 0:
                    c.setup_score = min(1.0, c.setup_score + kz_bonus)

            elapsed = (time.perf_counter() - t_start) * 1000.0
            if candidates:
                logger.info(
                    f"[ParallelAnalyzer] {mode}: {len(candidates)} candidate(s) | {elapsed:.1f}ms | "
                    f"kill_zone={is_kill_zone}"
                )
            return StrategyResult(strategy=mode, candidates=candidates, elapsed_ms=elapsed)

        except Exception as exc:
            elapsed = (time.perf_counter() - t_start) * 1000.0
            logger.error(f"[ParallelAnalyzer] {mode} error: {exc}")
            return StrategyResult(
                strategy=mode, candidates=[], elapsed_ms=elapsed, error=str(exc)
            )

    def analyze_all(self, context: PipelineContext) -> List[TradeCandidate]:
        """
        Run all configured strategies in parallel.
        Returns deduplicated, kill-zone-prioritized candidates.

        Args:
            context: Full PipelineContext snapshot (immutable)

        Returns:
            List[TradeCandidate] sorted by setup_score descending.
            Empty list if no valid setups found.
        """
        active_modes = settings.active_strategy_modes
        if not active_modes:
            return []

        self._load_strategies(active_modes)
        is_kill_zone = is_in_kill_zone()
        kz_name = get_kill_zone_name()

        if is_kill_zone:
            logger.info(f"[ParallelAnalyzer] 🎯 Kill Zone active: {kz_name}")

        timeout = settings.parallel_strategy_timeout_seconds
        all_candidates: List[TradeCandidate] = []
        results: List[StrategyResult] = []

        if not settings.enable_parallel_strategies or len(active_modes) == 1:
            # Sequential fallback (for single mode or if parallel disabled)
            for mode in active_modes:
                strategy = self._strategies.get(mode)
                if strategy:
                    result = self._run_strategy(mode, strategy, context, is_kill_zone)
                    results.append(result)
        else:
            # Parallel execution
            futures = {}
            with ThreadPoolExecutor(
                max_workers=len(active_modes),
                thread_name_prefix="paxis_strategy",
            ) as executor:
                for mode in active_modes:
                    strategy = self._strategies.get(mode)
                    if strategy:
                        future = executor.submit(
                            self._run_strategy, mode, strategy, context, is_kill_zone
                        )
                        futures[future] = mode

                for future in as_completed(futures, timeout=timeout):
                    mode = futures[future]
                    try:
                        result = future.result(timeout=0.1)
                        results.append(result)
                    except FutureTimeoutError:
                        logger.warning(f"[ParallelAnalyzer] {mode} timed out after {timeout}s")
                    except Exception as exc:
                        logger.error(f"[ParallelAnalyzer] {mode} future error: {exc}")

        # Collect all candidates
        for result in results:
            all_candidates.extend(result.candidates)

        if not all_candidates:
            return []

        # Deduplicate
        deduped = self._deduplicator.deduplicate(all_candidates)

        # Sort by setup_score descending (kill zone already factored into score)
        deduped.sort(key=lambda c: c.setup_score, reverse=True)

        logger.info(
            f"[ParallelAnalyzer] {context.symbol} | "
            f"strategies={active_modes} | "
            f"raw={len(all_candidates)} | deduped={len(deduped)} | "
            f"kill_zone={kz_name}"
        )

        return deduped

    def record_executed(self, candidate: TradeCandidate) -> None:
        """Must be called after each successful execution to update dedup state."""
        self._deduplicator.record_executed(candidate)

    def get_diagnostics(self) -> dict:
        """Return diagnostic information for /debug command."""
        return {
            "active_strategies": list(self._strategies.keys()),
            "configured_modes": settings.active_strategy_modes,
            "parallel_enabled": settings.enable_parallel_strategies,
            "kill_zone_now": is_in_kill_zone(),
            "kill_zone_name": get_kill_zone_name(),
            "dedup_window_seconds": settings.signal_dedup_window_seconds,
        }


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY CANDIDATE CONVERTER
# ══════════════════════════════════════════════════════════════════════════════


def _convert_legacy_candidate(
    c: Any, context: PipelineContext, mode: str
) -> Optional[TradeCandidate]:
    """
    Convert a legacy StrategyCandidiate (from ICT/SMC strategy decision)
    to the canonical TradeCandidate.

    Legacy candidates have different field names — this converter normalizes them.
    """
    try:
        direction = getattr(c, "direction", "")
        if direction in ("LONG", "long"):
            direction = "BUY"
        elif direction in ("SHORT", "short"):
            direction = "SELL"

        entry = float(getattr(c, "entry", 0.0) or 0.0)
        sl = float(getattr(c, "stop_loss", 0.0) or 0.0)
        tp1 = float(getattr(c, "take_profit_1", 0.0) or 0.0)
        tp2 = getattr(c, "take_profit_2", None)
        tp3 = getattr(c, "take_profit_3", None)

        if entry <= 0 or sl <= 0:
            return None

        return TradeCandidate(
            symbol=context.symbol,
            direction=direction,
            strategy=getattr(c, "setup_type", mode) or mode,
            strategy_family=mode,
            strategy_version="1.0",
            entry_mode="STRUCTURAL",
            timestamp=datetime.now(timezone.utc),
            entry=entry,
            stop_loss=sl,
            tp1=tp1 if tp1 > 0 else None,
            tp2=float(tp2) if tp2 else None,
            tp3=float(tp3) if tp3 else None,
            setup_score=float(getattr(c, "score", 0.0) or 0.0),
            confluence_score=float(getattr(c, "risk_reward", 0.0) or 0.0),
            regime=context.regime,
            session=context.session,
            spread_pips=context.spread_pips,
        )
    except Exception as exc:
        logger.debug(f"_convert_legacy_candidate error: {exc}")
        return None


# ── Singleton ─────────────────────────────────────────────────────────────────
parallel_analyzer = ParallelAnalyzer()
