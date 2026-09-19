"""
PAXIS Agent — Broker-Aware Risk Calculator
==========================================
The SINGLE authoritative position-sizing authority for the entire system.

CONTRACT:
  • Never uses "XAU" in symbol, "JPY" in symbol, or any other heuristic
  • Always queries MT5 symbol_info() when available
  • Falls back to a pre-loaded BrokerContractSpec when MT5 unavailable (e.g. backtest)
  • After computing raw lot, rounds to broker volume step and re-checks actual risk
  • Rejects if actual risk > allowed risk by more than tolerance
  • Returns an ApprovedTrade — the only object ExecutionAuthority accepts

Usage:
    candidate = TradeCandidate(...)
    approved = broker_risk_calculator.approve(
        candidate, account_balance=1000.0, risk_pct=1.0
    )
    if approved.is_executable():
        execution_authority.execute(approved)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from agent.core.trade_models import ApprovedTrade, TradeCandidate


# ══════════════════════════════════════════════════════════════════════════════
# BROKER CONTRACT SPEC
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BrokerContractSpec:
    """
    Full broker contract specification for a single symbol.
    Populated from MT5 symbol_info() — never hardcoded.
    """
    symbol: str = ""
    contract_size: float = 100000.0     # Units per lot (e.g. 100 for XAUUSD, 100000 for FX)
    tick_size: float = 0.00001          # Minimum price increment
    tick_value: float = 1.0             # Value of one tick in account currency (per lot)
    volume_step: float = 0.01           # Minimum volume increment
    volume_min: float = 0.01            # Minimum allowed lot size
    volume_max: float = 100.0           # Maximum allowed lot size
    stops_level: int = 0                # Minimum distance for SL/TP in points
    freeze_level: int = 0               # Freeze level for pending orders
    currency_profit: str = "USD"        # Profit currency
    currency_margin: str = "USD"        # Margin currency
    digits: int = 5                     # Price decimal places
    point: float = 0.00001             # Minimum price change

    # Derived: value per price point per lot (computed after loading)
    value_per_point_per_lot: float = 0.0

    def is_valid(self) -> bool:
        return (
            self.contract_size > 0
            and self.tick_size > 0
            and self.tick_value > 0
            and self.volume_step > 0
            and self.volume_min > 0
            and self.volume_min <= self.volume_max
        )

    def normalize_volume(self, raw_volume: float) -> float:
        """
        Round raw_volume to the broker's volume step and clamp to [volume_min, volume_max].
        Example: raw=0.037, step=0.01 → 0.03
        """
        if self.volume_step <= 0:
            return max(self.volume_min, min(self.volume_max, raw_volume))
        normalized = math.floor(raw_volume / self.volume_step) * self.volume_step
        normalized = round(normalized, 8)  # eliminate floating point residue
        return max(self.volume_min, min(self.volume_max, normalized))

    def compute_lot_size(self, risk_usd: float, stop_distance: float) -> float:
        """
        Compute required lot size to risk exactly risk_usd given stop_distance in price.
        Uses tick_value / tick_size to get value_per_point_per_lot.
        Returns normalized lot size.
        """
        vpp = self.value_per_point_per_lot
        if vpp <= 0:
            if self.tick_size > 0 and self.tick_value > 0:
                vpp = self.tick_value / self.tick_size
            else:
                logger.error(f"BrokerContractSpec for {self.symbol}: cannot compute vpp (tick_size={self.tick_size}, tick_value={self.tick_value})")
                return self.volume_min

        if stop_distance <= 0:
            return self.volume_min

        raw_lot = risk_usd / (stop_distance * vpp)
        return self.normalize_volume(raw_lot)

    def compute_risk_usd(self, lot_size: float, stop_distance: float) -> float:
        """
        Compute ACTUAL risk in USD for a given lot size and stop distance.
        Used after normalization to verify actual risk ≤ allowed risk.
        """
        vpp = self.value_per_point_per_lot
        if vpp <= 0:
            if self.tick_size > 0 and self.tick_value > 0:
                vpp = self.tick_value / self.tick_size
            else:
                return 0.0
        return lot_size * stop_distance * vpp

    def min_stop_distance(self) -> float:
        """Minimum allowed SL/TP distance in price (broker stops_level in points)."""
        return self.stops_level * self.point


# ══════════════════════════════════════════════════════════════════════════════
# RISK REJECTION REASONS
# ══════════════════════════════════════════════════════════════════════════════


REJECTION_NO_CONTRACT = "NO_BROKER_CONTRACT"
REJECTION_INVALID_CONTRACT = "INVALID_BROKER_CONTRACT"
REJECTION_INVALID_STOP = "INVALID_STOP_DISTANCE"
REJECTION_BELOW_MIN_STOP = "BELOW_BROKER_MIN_STOP"
REJECTION_ZERO_VOLUME = "COMPUTED_ZERO_VOLUME"
REJECTION_BELOW_MIN_VOL = "BELOW_BROKER_MIN_VOLUME"
REJECTION_ACTUAL_RISK_EXCEEDS = "ACTUAL_RISK_EXCEEDS_ALLOWED"
REJECTION_INVALID_CANDIDATE = "INVALID_TRADE_CANDIDATE"
REJECTION_DAILY_LOSS = "DAILY_LOSS_LIMIT"
REJECTION_MAX_DRAWDOWN = "MAX_DRAWDOWN"
REJECTION_MAX_OPEN_RISK = "MAX_OPEN_RISK"
REJECTION_CONSECUTIVE_LOSSES = "MAX_CONSECUTIVE_LOSSES"
REJECTION_AGENT_PAUSED = "AGENT_PAUSED"
REJECTION_LOW_RR = "LOW_RR_RATIO"
REJECTION_SPREAD_TOO_HIGH = "SPREAD_TOO_HIGH"
REJECTION_NEWS_BLACKOUT = "NEWS_BLACKOUT"
REJECTION_MT5_CALC_FAILED = "MT5_CALC_FAILED"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════


class BrokerRiskCalculator:
    """
    Authoritative position-sizing authority.

    Flow:
      1. Load broker contract from MT5 (if available) or use cached/pre-loaded spec
      2. Validate trade candidate geometry
      3. Compute raw lot → normalize to broker step
      4. Re-compute ACTUAL risk after normalization
      5. Verify actual risk ≤ allowed risk (+ tolerance)
      6. Return ApprovedTrade or rejection

    All callers receive either an ApprovedTrade (executable) or a rejected
    ApprovedTrade (risk_approved=False). No exceptions are thrown to callers.
    """

    # Per-trade tolerance: actual risk may exceed allowed by up to this factor
    # e.g. 1.05 means actual risk ≤ allowed × 1.05 (5% tolerance due to vol step)
    ACTUAL_RISK_TOLERANCE = 1.05

    def __init__(self):
        # symbol → BrokerContractSpec (loaded from MT5 or pre-set for backtest)
        self._contracts: dict[str, BrokerContractSpec] = {}
        self._environment: str = "PAPER"  # Set from settings at startup

    def set_environment(self, env: str) -> None:
        """DEVELOPMENT | PAPER | LIVE"""
        self._environment = env.upper()

    def set_contract(self, symbol: str, spec: BrokerContractSpec) -> None:
        """Pre-load a contract specification (for backtest or testing)."""
        self._contracts[symbol] = spec
        logger.info(f"BrokerRiskCalculator: pre-loaded contract for {symbol}")

    def load_from_mt5(self, symbol: str) -> Optional[BrokerContractSpec]:
        """
        Query MT5 for broker contract specification.
        Caches result. Returns None if MT5 unavailable.
        Never uses hardcoded asset-class assumptions.
        """
        try:
            import MetaTrader5 as mt5
            info = mt5.symbol_info(symbol)
            if info is None:
                logger.warning(f"MT5 symbol_info returned None for {symbol}")
                return self._contracts.get(symbol)

            spec = BrokerContractSpec(
                symbol=symbol,
                contract_size=float(info.trade_contract_size),
                tick_size=float(info.trade_tick_size),
                tick_value=float(info.trade_tick_value),
                volume_step=float(info.volume_step),
                volume_min=float(info.volume_min),
                volume_max=float(info.volume_max),
                stops_level=int(info.trade_stops_level),
                freeze_level=int(info.trade_freeze_level),
                currency_profit=str(info.currency_profit),
                currency_margin=str(info.currency_margin),
                digits=int(info.digits),
                point=float(info.point),
            )
            if spec.tick_size > 0 and spec.tick_value > 0:
                spec.value_per_point_per_lot = spec.tick_value / spec.tick_size

            self._contracts[symbol] = spec
            logger.info(
                f"BrokerRiskCalculator: loaded MT5 contract for {symbol} | "
                f"contract_size={spec.contract_size} | tick={spec.tick_size} | "
                f"tick_value={spec.tick_value} | vpp={spec.value_per_point_per_lot:.4f} | "
                f"vol_step={spec.volume_step} | min={spec.volume_min} | max={spec.volume_max} | "
                f"stops_level={spec.stops_level}"
            )
            return spec

        except ImportError:
            logger.debug("MT5 not available — using pre-loaded contract spec")
            return self._contracts.get(symbol)
        except Exception as exc:
            logger.error(f"Failed to load MT5 contract for {symbol}: {exc}")
            return self._contracts.get(symbol)

    def _try_mt5_order_calc(
        self, symbol: str, direction: str, volume: float, entry: float, sl: float
    ) -> Optional[float]:
        """
        Use mt5.order_calc_profit() to verify actual risk calculation.
        Returns risk in account currency or None if unavailable.
        """
        try:
            import MetaTrader5 as mt5
            order_type = mt5.ORDER_TYPE_BUY if direction.upper() == "BUY" else mt5.ORDER_TYPE_SELL
            # Profit when SL is hit = negative value = actual risk
            profit = mt5.order_calc_profit(order_type, symbol, volume, entry, sl)
            if profit is not None:
                return abs(float(profit))
        except Exception:
            pass
        return None

    def get_contract(self, symbol: str) -> Optional[BrokerContractSpec]:
        """Return cached contract or load from MT5."""
        if symbol in self._contracts:
            return self._contracts[symbol]
        return self.load_from_mt5(symbol)

    def approve(
        self,
        candidate: TradeCandidate,
        account_balance: float,
        risk_pct: float = 1.0,
        *,
        # Optional account state for portfolio-level checks
        daily_pnl_usd: float = 0.0,
        max_daily_loss_usd: float = 50.0,
        open_positions_count: int = 0,
        max_open_trades: int = 2,
        total_open_risk_pct: float = 0.0,
        max_total_open_risk_pct: float = 4.0,
        consecutive_losses: int = 0,
        max_consecutive_losses: int = 3,
        current_drawdown_pct: float = 0.0,
        max_drawdown_pct: float = 5.0,
        agent_paused: bool = False,
        min_rr_ratio: float = 1.5,
        # Risk multiplier (self-evolution can reduce, never increase)
        risk_multiplier: float = 1.0,
    ) -> ApprovedTrade:
        """
        The single entry point for all trade sizing decisions.

        Returns ApprovedTrade with risk_approved=True if all checks pass.
        Returns ApprovedTrade with risk_approved=False on any rejection.
        NEVER raises exceptions to callers.
        """
        def _reject(reason: str, codes: list[str] | None = None) -> ApprovedTrade:
            at = ApprovedTrade(
                candidate=candidate,
                risk_approved=False,
                rejection_reason=reason,
                rejection_codes=codes or [reason],
                environment=self._environment,
            )
            logger.warning(f"[RiskCalc REJECTED] {candidate.symbol} {candidate.direction} | {reason}")
            return at

        # ── 0. Candidate validity ────────────────────────────────────────────
        if not candidate.is_valid():
            return _reject(
                REJECTION_INVALID_CANDIDATE,
                [REJECTION_INVALID_CANDIDATE],
            )

        symbol = candidate.symbol
        direction = candidate.direction

        # ── 1. Portfolio-level gating (fast checks first) ────────────────────
        if agent_paused:
            return _reject(REJECTION_AGENT_PAUSED)

        if daily_pnl_usd < -abs(max_daily_loss_usd):
            return _reject(REJECTION_DAILY_LOSS)

        if current_drawdown_pct > max_drawdown_pct:
            return _reject(REJECTION_MAX_DRAWDOWN)

        if consecutive_losses >= max_consecutive_losses:
            return _reject(REJECTION_CONSECUTIVE_LOSSES)

        if open_positions_count >= max_open_trades:
            return _reject(REJECTION_MAX_OPEN_RISK)

        if total_open_risk_pct >= max_total_open_risk_pct:
            return _reject(REJECTION_MAX_OPEN_RISK)

        # ── 2. RR check ──────────────────────────────────────────────────────
        if candidate.rr_ratio < min_rr_ratio:
            return _reject(
                f"{REJECTION_LOW_RR}: {candidate.rr_ratio:.2f} < {min_rr_ratio:.2f}",
                [REJECTION_LOW_RR],
            )

        # ── 3. Get broker contract ───────────────────────────────────────────
        contract = self.get_contract(symbol)
        if contract is None:
            return _reject(REJECTION_NO_CONTRACT, [REJECTION_NO_CONTRACT])

        if not contract.is_valid():
            return _reject(REJECTION_INVALID_CONTRACT, [REJECTION_INVALID_CONTRACT])

        # ── 4. Stop distance validation ──────────────────────────────────────
        stop_distance = abs(candidate.entry - candidate.stop_loss)
        if stop_distance <= 0:
            return _reject(REJECTION_INVALID_STOP, [REJECTION_INVALID_STOP])

        min_stop = contract.min_stop_distance()
        if stop_distance < min_stop:
            return _reject(
                f"{REJECTION_BELOW_MIN_STOP}: stop {stop_distance:.5f} < broker min {min_stop:.5f}",
                [REJECTION_BELOW_MIN_STOP],
            )

        # ── 5. Compute position size ─────────────────────────────────────────
        effective_multiplier = max(0.0, min(1.0, risk_multiplier))
        allowed_risk_usd = account_balance * (risk_pct / 100.0) * effective_multiplier

        if effective_multiplier < 1.0:
            logger.info(
                f"BrokerRiskCalc: supervisory multiplier={effective_multiplier:.2f} → "
                f"effective risk=${allowed_risk_usd:.2f} (base=${account_balance * (risk_pct / 100.0):.2f})"
            )

        raw_lot = contract.compute_lot_size(allowed_risk_usd, stop_distance)
        normalized_lot = contract.normalize_volume(raw_lot)

        if normalized_lot <= 0:
            return _reject(REJECTION_ZERO_VOLUME, [REJECTION_ZERO_VOLUME])

        if normalized_lot < contract.volume_min:
            return _reject(
                f"{REJECTION_BELOW_MIN_VOL}: {normalized_lot} < broker min {contract.volume_min}",
                [REJECTION_BELOW_MIN_VOL],
            )

        # ── 6. Verify actual risk after normalization ────────────────────────
        # First try MT5 order_calc_profit (most accurate)
        actual_risk_usd = self._try_mt5_order_calc(symbol, direction, normalized_lot, candidate.entry, candidate.stop_loss)
        mt5_calc_used = actual_risk_usd is not None

        if actual_risk_usd is None:
            # Fall back to contract spec calculation
            actual_risk_usd = contract.compute_risk_usd(normalized_lot, stop_distance)

        # Reject if actual risk significantly exceeds allowed (due to volume rounding up)
        if actual_risk_usd > allowed_risk_usd * self.ACTUAL_RISK_TOLERANCE:
            # Try one step smaller lot
            smaller_lot = contract.normalize_volume(normalized_lot - contract.volume_step)
            if smaller_lot >= contract.volume_min:
                actual_risk_smaller = contract.compute_risk_usd(smaller_lot, stop_distance)
                if actual_risk_smaller <= allowed_risk_usd * self.ACTUAL_RISK_TOLERANCE:
                    logger.info(
                        f"BrokerRiskCalc: reduced lot {normalized_lot} → {smaller_lot} "
                        f"to keep actual risk ${actual_risk_smaller:.2f} ≤ allowed ${allowed_risk_usd:.2f}"
                    )
                    normalized_lot = smaller_lot
                    actual_risk_usd = actual_risk_smaller
                else:
                    return _reject(
                        f"{REJECTION_ACTUAL_RISK_EXCEEDS}: actual ${actual_risk_smaller:.2f} > allowed ${allowed_risk_usd:.2f}",
                        [REJECTION_ACTUAL_RISK_EXCEEDS],
                    )
            else:
                return _reject(
                    f"{REJECTION_BELOW_MIN_VOL}: reduced lot {smaller_lot} < broker min {contract.volume_min}",
                    [REJECTION_BELOW_MIN_VOL],
                )

        risk_pct_actual = (actual_risk_usd / account_balance * 100.0) if account_balance > 0 else 0.0

        logger.info(
            f"BrokerRiskCalc APPROVED: {symbol} {direction} | "
            f"lot={normalized_lot} | stop_dist={stop_distance:.5f} | "
            f"allowed=${allowed_risk_usd:.2f} | actual=${actual_risk_usd:.2f} ({risk_pct_actual:.2f}%) | "
            f"mt5_calc={'YES' if mt5_calc_used else 'CONTRACT_SPEC'} | "
            f"rr={candidate.rr_ratio:.2f}"
        )

        return ApprovedTrade(
            candidate=candidate,
            volume=normalized_lot,
            allowed_risk_usd=allowed_risk_usd,
            actual_risk_usd=actual_risk_usd,
            risk_pct=risk_pct_actual,
            sl_distance_points=stop_distance,
            tick_value=contract.tick_value,
            contract_size=contract.contract_size,
            broker_volume_step=contract.volume_step,
            broker_min_volume=contract.volume_min,
            broker_max_volume=contract.volume_max,
            broker_stops_level=contract.stops_level,
            risk_approved=True,
            environment=self._environment,
        )

    def calculate_lot_for_backtest(
        self,
        symbol: str,
        entry: float,
        sl: float,
        risk_usd: float,
    ) -> float:
        """
        Simplified sizing for backtest — uses pre-loaded contract spec only.
        Returns normalized lot or volume_min as fallback.
        """
        contract = self._contracts.get(symbol)
        if contract is None or not contract.is_valid():
            logger.warning(f"BrokerRiskCalc: no contract for {symbol} in backtest — using min lot")
            return 0.01

        stop_distance = abs(entry - sl)
        if stop_distance <= 0:
            return contract.volume_min

        raw_lot = contract.compute_lot_size(risk_usd, stop_distance)
        return contract.normalize_volume(raw_lot)


# ── Singleton ─────────────────────────────────────────────────────────────────
broker_risk_calculator = BrokerRiskCalculator()
