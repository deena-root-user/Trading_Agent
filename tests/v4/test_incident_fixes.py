"""
PAXIS v4 — Incident Architecture Fixes Test Suite
===================================================
Tests for:
1. Telegram trade close alert ticket deduplication.
2. Direct MT5 profit extraction without adding swap/commission.
3. Directional Loss Shield strict 15m cooldown (no adverse price bypass).
4. Mandatory NO_ACTIVE_OPPOSING_IMPULSE check in TradeValidator.
"""
from datetime import datetime, timezone
import pytest

from agent.notify.telegram_bot import PaxisBot
from agent.risk.gate import RiskGate
from agent.analysis.validator import TradeValidator


class TestTelegramAlertDeduplication:
    def test_close_alert_deduplication(self):
        bot = PaxisBot()
        # Mock _send method to count sent alerts
        sent_messages = []
        bot._send = lambda msg: sent_messages.append(msg)

        # First alert for ticket #2387454733
        bot.send_trade_close(
            symbol="XAUUSD", action="SELL", pnl=0.78, outcome="WIN ✅", ticket=2387454733
        )
        assert len(sent_messages) == 1

        # Duplicate alert for ticket #2387454733
        bot.send_trade_close(
            symbol="XAUUSD", action="SELL", pnl=0.78, outcome="WIN ✅", ticket=2387454733
        )
        # Should NOT send a second message!
        assert len(sent_messages) == 1

        # Alert for a different ticket #2387454734
        bot.send_trade_close(
            symbol="XAUUSD", action="SELL", pnl=1.50, outcome="WIN ✅", ticket=2387454734
        )
        assert len(sent_messages) == 2


class TestDirectionalLossShieldStrictCooldown:
    def test_rally_does_not_bypass_loss_shield(self):
        gate = RiskGate()
        # Record loss at entry 4414.01
        gate.record_loss("XAUUSD", "SELL", 4414.01)

        # Attempt to sell again at a HIGHER price (4419.46) during a rally with conf=0.74
        res = gate.check(
            symbol="XAUUSD",
            action="SELL",
            confidence=0.74,
            entry=4419.46,
            sl=4429.14,
            tp=4390.93,
            rr_ratio=2.5,
            spread_pips=2.0,
            open_positions=[],
        )

        assert not res.passed
        assert any("DIRECTIONAL_LOSS_COOLDOWN" in fail for fail in res.failed_checks)


class TestValidatorMandatoryImpulseBlocker:
    def test_active_bullish_displacement_blocks_short(self):
        validator = TradeValidator()

        res = validator.validate(
            symbol="XAUUSD",
            direction="SHORT",
            smc_1m={
                "displacement_detected": True,
                "displacement_direction": "BULLISH",
            },
            smc_15m={"displacement_detected": True, "displacement_direction": "BULLISH"},
            proposed_entry=4420.0,
            proposed_sl=4429.0,
            proposed_tp=4390.0,
            spread_pips=2.0,
            open_positions_count=0,
            max_open_trades=3,
            daily_pnl_usd=0.0,
            max_daily_loss_usd=20.0,
        )

        assert not res.is_valid
        assert "NO_ACTIVE_OPPOSING_IMPULSE" in res.mandatory_failures
