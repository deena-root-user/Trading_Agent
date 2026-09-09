"""
Critical validator tests — verifies the fixes for:
  C-01: Check #3 (ACTIVE_POI_EXISTS) no longer has `or True` bypass
  C-02: Check #2 (STRUCTURE_BREAK_PRESENT) no longer accepts wrong-direction breaks
  H-02: Check #17 (MIN_RR_RATIO) no longer has 0.02 tolerance
  Mandatory checks (13, 14, 15, 16, 17) must block trades when violated
"""
import pytest
from unittest.mock import patch
from agent.analysis.validator import TradeValidator, ValidatorResult


# ── Fixtures ──────────────────────────────────────────────────────────────────
# NOTE: Validator uses direction="LONG"/"SHORT" (not BUY/SELL)
# is_bull = direction == "LONG"
# expected break direction = "BULLISH" for LONG, "BEARISH" for SHORT

def _make_smc(trend="BULLISH", bull_obs=None, bear_obs=None,
              bull_fvgs=None, bear_fvgs=None, breaks=None, **kwargs):
    """Create a minimal SMC dict for testing."""
    d = {
        "trend": trend,
        "active_bullish_obs": bull_obs or [],
        "active_bearish_obs": bear_obs or [],
        "active_bullish_fvgs": bull_fvgs or [],
        "active_bearish_fvgs": bear_fvgs or [],
        "recent_breaks": breaks or [],
        "recent_sweeps": [],
        "premium_discount": "DISCOUNT",
        "displacement_detected": True,
        "displacement_direction": "BULLISH",
        "inducement_swept": True,
    }
    d.update(kwargs)
    return d


def _make_smc_1m(direction="BULLISH"):
    return {
        "trend": direction,
        "recent_breaks": [{"direction": direction, "bars_ago": 2}],
    }


def _run_validator(validator, direction="LONG", smc_4h=None, smc_1h=None,
                   smc_15m=None, smc_1m=None, **kwargs):
    """Run validator with sensible defaults for non-tested params."""
    defaults = dict(
        direction=direction,
        smc_4h=smc_4h or _make_smc(),
        smc_1h=smc_1h or _make_smc(),
        smc_15m=smc_15m or _make_smc(),
        smc_1m=smc_1m or _make_smc_1m(),
        spread_pips=2.0,
        max_spread_pips=30.0,
        current_price=2000.0,
        proposed_entry=2000.0,
        proposed_sl=1995.0,
        proposed_tp=2010.0,
        current_session="LONDON",
        is_trading_session=True,
        fib_score=0.6,
    )
    defaults.update(kwargs)
    return validator.validate(**defaults)


# ── C-01: POI Check Must Actually Validate ────────────────────────────────────

class TestC01_POICheckNotBypassed:
    """Check #3 (ACTIVE_POI_EXISTS) must FAIL when no OBs or FVGs exist."""

    def test_no_poi_fails_check_3(self):
        validator = TradeValidator()
        # SMC data with ZERO order blocks and ZERO FVGs
        empty_smc = _make_smc(bull_obs=[], bull_fvgs=[], bear_obs=[], bear_fvgs=[])
        result = _run_validator(
            validator,
            direction="LONG",
            smc_4h=empty_smc,
            smc_1h=empty_smc,
            smc_15m=empty_smc,
        )
        check_3 = next((c for c in result.checks if c.check_id == 3), None)
        assert check_3 is not None, "Check #3 not found in validator output"
        assert check_3.passed is False, (
            "CRITICAL: Check #3 (ACTIVE_POI_EXISTS) passed with NO POIs — "
            "the 'or True' bypass was NOT removed"
        )

    def test_with_ob_passes_check_3(self):
        validator = TradeValidator()
        # LONG direction → validator looks for active_bullish_obs
        smc_with_ob = _make_smc(bull_obs=[{"price_is_inside": True}])
        result = _run_validator(
            validator, direction="LONG", smc_1h=smc_with_ob
        )
        check_3 = next((c for c in result.checks if c.check_id == 3), None)
        assert check_3 is not None
        assert check_3.passed is True, "Check #3 should pass when bullish OB exists for LONG"

    def test_with_fvg_passes_check_3(self):
        validator = TradeValidator()
        # LONG direction → validator looks for active_bullish_fvgs
        smc_with_fvg = _make_smc(bull_fvgs=[{"price_is_inside": True}])
        result = _run_validator(
            validator, direction="LONG", smc_1h=smc_with_fvg
        )
        check_3 = next((c for c in result.checks if c.check_id == 3), None)
        assert check_3 is not None
        assert check_3.passed is True, "Check #3 should pass when bullish FVG exists for LONG"


# ── C-02: Structure Break Direction Must Match ────────────────────────────────

class TestC02_StructureBreakDirection:
    """Check #2 must reject breaks that don't match trade direction."""

    def test_wrong_direction_break_fails_check_2(self):
        validator = TradeValidator()
        # LONG trade expects BULLISH breaks, but only BEARISH breaks exist
        bearish_break_smc = _make_smc(
            breaks=[{"direction": "BEARISH", "bars_ago": 5}]
        )
        result = _run_validator(
            validator,
            direction="LONG",
            smc_4h=bearish_break_smc,
            smc_1h=bearish_break_smc,
            smc_15m=bearish_break_smc,
        )
        check_2 = next((c for c in result.checks if c.check_id == 2), None)
        assert check_2 is not None
        assert check_2.passed is False, (
            "CRITICAL: Check #2 passed with BEARISH breaks for a LONG trade — "
            "the direction-agnostic fallback was NOT removed"
        )

    def test_correct_direction_break_passes_check_2(self):
        validator = TradeValidator()
        # LONG trade expects BULLISH breaks
        bullish_break_smc = _make_smc(
            breaks=[{"direction": "BULLISH", "bars_ago": 5}]
        )
        result = _run_validator(
            validator,
            direction="LONG",
            smc_4h=bullish_break_smc,
        )
        check_2 = next((c for c in result.checks if c.check_id == 2), None)
        assert check_2 is not None
        assert check_2.passed is True, "Check #2 should pass with BULLISH break for LONG trade"

    def test_no_breaks_fails_check_2(self):
        validator = TradeValidator()
        no_break_smc = _make_smc(breaks=[])
        result = _run_validator(
            validator,
            smc_4h=no_break_smc,
            smc_1h=no_break_smc,
            smc_15m=no_break_smc,
        )
        check_2 = next((c for c in result.checks if c.check_id == 2), None)
        assert check_2 is not None
        assert check_2.passed is False


# ── H-02: R:R Tolerance Removed ───────────────────────────────────────────────

class TestH02_RRToleranceRemoved:
    """Check #17 must NOT pass with R:R below min_rr_ratio."""

    def test_rr_1_98_fails_when_min_is_2_0(self):
        validator = TradeValidator()
        # Entry=2000, SL=1995 → risk=5. TP=2009.90 → reward=9.90 → R:R=1.98
        result = _run_validator(
            validator,
            proposed_entry=2000.0,
            proposed_sl=1995.0,
            proposed_tp=2009.90,
        )
        check_17 = next((c for c in result.checks if c.check_id == 17), None)
        assert check_17 is not None
        assert check_17.passed is False, (
            "Check #17 should FAIL for R:R 1.98 when min is 2.0 — "
            "the 0.02 tolerance was NOT removed"
        )

    def test_rr_2_0_exactly_passes(self):
        validator = TradeValidator()
        # Entry=2000, SL=1995 → risk=5. TP=2010 → reward=10 → R:R=2.0
        result = _run_validator(
            validator,
            proposed_entry=2000.0,
            proposed_sl=1995.0,
            proposed_tp=2010.0,
        )
        check_17 = next((c for c in result.checks if c.check_id == 17), None)
        assert check_17 is not None
        assert check_17.passed is True


# ── Mandatory Checks Block Trades ─────────────────────────────────────────────

class TestMandatoryChecksBlock:
    """All mandatory checks (13, 14, 15, 16, 17) must prevent trade execution."""

    def test_news_blackout_blocks(self):
        validator = TradeValidator()
        result = _run_validator(validator, news_blocked=True, news_reason="NFP")
        assert result.is_valid is False
        assert "NO_NEWS_BLACKOUT" in result.mandatory_failures

    def test_spread_too_wide_blocks(self):
        validator = TradeValidator()
        result = _run_validator(validator, spread_pips=100.0, max_spread_pips=5.0)
        assert result.is_valid is False
        assert "SPREAD_WITHIN_LIMIT" in result.mandatory_failures

    def test_max_positions_blocks(self):
        validator = TradeValidator()
        result = _run_validator(
            validator, open_positions_count=5, max_open_trades=2
        )
        assert result.is_valid is False
        assert "MAX_POSITIONS_NOT_EXCEEDED" in result.mandatory_failures

    def test_daily_loss_blocks(self):
        validator = TradeValidator()
        result = _run_validator(
            validator, daily_pnl_usd=-200.0, max_daily_loss_usd=50.0
        )
        assert result.is_valid is False
        assert "DAILY_LOSS_LIMIT_OK" in result.mandatory_failures

    def test_rr_too_low_blocks(self):
        validator = TradeValidator()
        # R:R = 0.5 (risk=5, reward=2.5)
        result = _run_validator(
            validator,
            proposed_entry=2000.0,
            proposed_sl=1995.0,
            proposed_tp=2002.5,
        )
        assert result.is_valid is False
        assert "MIN_RR_RATIO" in result.mandatory_failures
