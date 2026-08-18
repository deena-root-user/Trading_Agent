import pytest
from agent.config import settings

@pytest.fixture(autouse=True)
def reset_settings():
    """Autouse fixture to reset settings to standard defaults before each test."""
    settings.scalping_mode = False
    settings.enforce_trend_alignment = True
    settings.use_dynamic_risk = False
    settings.min_rr_ratio = 1.5
    settings.min_confidence = 0.70
    settings.max_spread_pips = 3.0
    settings.max_daily_loss_usd = 50.0
    settings.max_open_trades = 2
    settings.lot_size = 0.01
    settings.auto_breakeven_ratio = 1.0
    settings.trailing_stop_atr_multiplier = 2.0
    yield
    # Clean up after test as well
    settings.scalping_mode = False
    settings.enforce_trend_alignment = True
    settings.use_dynamic_risk = False
    settings.min_rr_ratio = 1.5
