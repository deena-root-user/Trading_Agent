"""
PAXIS Agent — v4 Risk Engine
==============================
Computes position sizing and validates risk budget.

Engine Responsibility: "How much can we risk, and does this trade fit within our risk budget?"

CRITICAL: Uses broker-specific contract validation via MT5 symbol_info.
NEVER assumes fixed contract specifications. Always queries the broker.
"""
from __future__ import annotations

from typing import List, Optional

from loguru import logger

from agent.v4.v4_types import (
    BrokerContractSpec,
    RejectionCode,
    RiskResult,
    V4Config,
)


class RiskEngine:
    """
    Position sizing and risk budget validation.
    Queries MT5 for actual broker contract specifications.
    """

    def __init__(self, config: Optional[V4Config] = None):
        self.config = config or V4Config()
        self._cached_contract: Optional[BrokerContractSpec] = None

    def evaluate(
        self,
        *,
        symbol: str,
        direction: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        risk_pct: float = 1.0,             # % of balance to risk
        # Account state
        account_balance: float = 0.0,
        account_equity: float = 0.0,
        daily_pnl_usd: float = 0.0,
        open_positions_count: int = 0,
        total_open_risk_pct: float = 0.0,
        current_drawdown_pct: float = 0.0,
        consecutive_losses: int = 0,
        # Broker contract (pre-loaded or will use cached)
        broker_contract: Optional[BrokerContractSpec] = None,
        # Supervisory risk multiplier (0.0-1.0, default 1.0 = no reduction)
        risk_multiplier: float = 1.0,
    ) -> RiskResult:
        """
        Compute position size and validate risk budget.

        Returns:
            RiskResult with lot size, risk values, and any rejections.
        """
        result = RiskResult()
        result.account_balance = account_balance
        result.account_equity = account_equity
        result.daily_pnl_usd = daily_pnl_usd
        result.open_positions_count = open_positions_count
        result.total_open_risk_pct = total_open_risk_pct
        result.current_drawdown_pct = current_drawdown_pct
        result.consecutive_losses = consecutive_losses

        rejections: List[RejectionCode] = []

        # ── 1. Get broker contract ───────────────────────────────────────────
        contract = broker_contract or self._cached_contract
        result.broker_contract = contract

        if contract is None or not contract.is_valid():
            rejections.append(RejectionCode.BROKER_VALIDATION_FAILED)
            result.rejection_codes = rejections
            result.reasoning = "No valid broker contract specification loaded"
            return result

        # ── 2. Stop distance ────────────────────────────────────────────────
        stop_distance = abs(entry_price - sl_price)
        result.stop_distance_price_points = stop_distance

        if stop_distance <= 0:
            rejections.append(RejectionCode.INVALID_STOP)
            result.rejection_codes = rejections
            result.reasoning = "Invalid stop distance (<= 0)"
            return result

        # Check stops_level (broker minimum distance for SL/TP)
        min_stop_points = contract.stops_level * contract.point
        if stop_distance < min_stop_points:
            rejections.append(RejectionCode.INVALID_STOP)
            result.rejection_codes = rejections
            result.reasoning = (
                f"Stop distance {stop_distance:.5f} < broker minimum "
                f"{min_stop_points:.5f} ({contract.stops_level} points)"
            )
            return result

        # ── 3. Account-level risk checks ─────────────────────────────────────
        # Daily loss limit
        max_daily_loss = account_balance * (self.config.max_equity_drawdown_pct / 100.0)
        if abs(daily_pnl_usd) > max_daily_loss and daily_pnl_usd < 0:
            rejections.append(RejectionCode.DAILY_LOSS_LIMIT)

        # Max drawdown
        if current_drawdown_pct > self.config.max_equity_drawdown_pct:
            rejections.append(RejectionCode.MAX_DRAWDOWN)

        # Max consecutive losses
        if consecutive_losses >= self.config.max_consecutive_losses:
            rejections.append(RejectionCode.MAX_CONSECUTIVE_LOSSES)

        # Max open risk
        if total_open_risk_pct >= self.config.max_total_open_risk_pct:
            rejections.append(RejectionCode.MAX_OPEN_RISK)

        # Max positions
        if open_positions_count >= self.config.max_open_trades:
            rejections.append(RejectionCode.MAX_OPEN_RISK)

        # ── 4. Position sizing ───────────────────────────────────────────────
        if account_balance > 0 and len(rejections) == 0:
            # Apply supervisory risk multiplier (can only REDUCE, never increase)
            effective_multiplier = max(0.0, min(1.0, risk_multiplier))
            risk_usd = account_balance * (risk_pct / 100.0) * effective_multiplier
            if effective_multiplier < 1.0:
                logger.info(
                    f"Risk engine: supervisory multiplier={effective_multiplier:.2f} → "
                    f"effective risk=${risk_usd:.2f} (base=${account_balance * (risk_pct / 100.0):.2f})"
                )
            lot_size = contract.compute_lot_size(risk_usd, stop_distance)
            actual_risk = contract.compute_risk_usd(lot_size, stop_distance)

            result.lot_size = lot_size
            result.risk_usd = actual_risk
            result.risk_pct = (actual_risk / account_balance * 100.0) if account_balance > 0 else 0.0

            if lot_size < contract.volume_min:
                rejections.append(RejectionCode.BROKER_VALIDATION_FAILED)
                result.reasoning = (
                    f"Computed lot size {lot_size} < broker minimum {contract.volume_min}"
                )
        else:
            result.lot_size = 0.0
            result.risk_usd = 0.0
            result.risk_pct = 0.0

        # ── 5. Result ────────────────────────────────────────────────────────
        result.rejection_codes = rejections
        result.is_approved = len(rejections) == 0 and result.lot_size > 0

        if not result.reasoning:
            result.reasoning = (
                f"Risk: lot={result.lot_size} | risk=${result.risk_usd:.2f} "
                f"({result.risk_pct:.2f}%) | stop_dist={stop_distance:.5f} | "
                f"approved={result.is_approved} | "
                f"rejections={[r.value for r in rejections]}"
            )

        logger.debug(f"Risk engine: {result.reasoning}")

        return result

    def load_contract_from_mt5(self, symbol: str) -> Optional[BrokerContractSpec]:
        """
        Load broker contract specifications from MT5.
        Returns None if MT5 is unavailable.

        This queries the ACTUAL broker values — never hardcoded.
        """
        try:
            import MetaTrader5 as mt5

            info = mt5.symbol_info(symbol)
            if info is None:
                logger.warning(f"MT5 symbol_info returned None for {symbol}")
                return None

            contract = BrokerContractSpec(
                symbol=symbol,
                contract_size=info.trade_contract_size,
                tick_size=info.trade_tick_size,
                tick_value=info.trade_tick_value,
                volume_step=info.volume_step,
                volume_min=info.volume_min,
                volume_max=info.volume_max,
                stops_level=info.trade_stops_level,
                freeze_level=info.trade_freeze_level,
                currency_profit=info.currency_profit,
                currency_margin=info.currency_margin,
                digits=info.digits,
                point=info.point,
            )

            # Compute value per point per lot
            if contract.tick_size > 0 and contract.tick_value > 0:
                contract.value_per_point_per_lot = (
                    contract.tick_value / contract.tick_size
                )

            self._cached_contract = contract
            logger.info(
                f"Loaded broker contract for {symbol}: "
                f"contract_size={contract.contract_size} | "
                f"tick_size={contract.tick_size} | tick_value={contract.tick_value} | "
                f"vol_step={contract.volume_step} | "
                f"stops_level={contract.stops_level}"
            )
            return contract

        except ImportError:
            logger.warning("MetaTrader5 not available — using cached contract")
            return self._cached_contract
        except Exception as e:
            logger.error(f"Failed to load MT5 contract for {symbol}: {e}")
            return self._cached_contract

    def set_contract(self, contract: BrokerContractSpec) -> None:
        """Manually set a broker contract (for testing or non-MT5 environments)."""
        self._cached_contract = contract


# ── Singleton ─────────────────────────────────────────────────────────────────
risk_engine = RiskEngine()
