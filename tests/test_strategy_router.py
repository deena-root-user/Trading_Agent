"""
Tests for StrategyRouter and strategy isolation.
"""
import pytest
from agent.strategy.base import StrategyBase, StrategyContext, StrategyDecision
from agent.strategy.router import StrategyRouter


class MockStrategy(StrategyBase):
    """Mock strategy for testing."""
    def __init__(self, strategy_name: str):
        self._name = strategy_name
        self._call_count = 0

    @property
    def name(self) -> str:
        return self._name

    def analyze(self, context: StrategyContext) -> StrategyDecision:
        self._call_count += 1
        return StrategyDecision(
            strategy_mode=self._name,
            action="BUY" if self._name == "SMC" else "SELL",
            direction="LONG" if self._name == "SMC" else "SHORT",
            is_actionable=True,
            confidence=0.80,
            reasoning=f"{self._name} strategy analyzed",
        )

    def get_diagnostic(self) -> dict:
        return {"strategy": self._name, "calls": self._call_count}


@pytest.fixture
def context():
    return StrategyContext(symbol="XAUUSD", current_price=4300.0)


class TestStrategyRouter:
    def test_routes_to_smc_by_default(self, context):
        """Router should use SMC strategy by default."""
        router = StrategyRouter("SMC")
        smc = MockStrategy("SMC")
        ict = MockStrategy("ICT")
        router.register(smc)
        router.register(ict)

        decision = router.analyze(context)
        assert decision.strategy_mode == "SMC"
        assert decision.action == "BUY"
        assert smc._call_count == 1
        assert ict._call_count == 0  # ICT should NOT be called

    def test_routes_to_ict_when_configured(self, context):
        """Router should use ICT strategy when TRADING_STRATEGY_MODE=ICT."""
        router = StrategyRouter("ICT")
        smc = MockStrategy("SMC")
        ict = MockStrategy("ICT")
        router.register(smc)
        router.register(ict)

        decision = router.analyze(context)
        assert decision.strategy_mode == "ICT"
        assert decision.action == "SELL"
        assert smc._call_count == 0  # SMC should NOT be called
        assert ict._call_count == 1

    def test_inactive_strategy_cannot_generate_trades(self, context):
        """Only the active strategy should be called."""
        router = StrategyRouter("SMC")
        smc = MockStrategy("SMC")
        ict = MockStrategy("ICT")
        router.register(smc)
        router.register(ict)

        # Make 10 analysis calls
        for _ in range(10):
            router.analyze(context)

        assert smc._call_count == 10
        assert ict._call_count == 0

    def test_hold_when_no_strategy_registered(self, context):
        """Router should return HOLD if the configured strategy isn't registered."""
        router = StrategyRouter("ICT")
        # Don't register ICT

        decision = router.analyze(context)
        assert decision.action == "HOLD"
        assert "not registered" in decision.reasoning.lower() or "STRATEGY_NOT_REGISTERED" in decision.block_reason

    def test_strategy_mode_tagged_on_decision(self, context):
        """Strategy mode should be tagged on all decisions and candidates."""
        router = StrategyRouter("SMC")
        router.register(MockStrategy("SMC"))

        decision = router.analyze(context)
        assert decision.strategy_mode == "SMC"

    def test_mode_property(self):
        """Router should expose the active mode."""
        router = StrategyRouter("ICT")
        assert router.mode == "ICT"

    def test_diagnostic_from_active_strategy(self, context):
        """Diagnostic should come from the active strategy."""
        router = StrategyRouter("SMC")
        smc = MockStrategy("SMC")
        router.register(smc)

        # Trigger analysis first
        router.analyze(context)

        diag = router.get_diagnostic()
        assert diag["strategy"] == "SMC"
        assert diag["strategy_mode"] == "SMC"

    def test_get_all_strategies(self):
        """Should return info about all registered strategies."""
        router = StrategyRouter("SMC")
        router.register(MockStrategy("SMC"))
        router.register(MockStrategy("ICT"))

        info = router.get_all_strategies()
        assert info["active_mode"] == "SMC"
        assert "SMC" in info["registered"]
        assert "ICT" in info["registered"]
        assert info["active"] == "SMC"


class TestConfigValidation:
    def test_valid_smc_config(self):
        """SMC should be accepted."""
        from agent.config import Settings
        s = Settings(trading_strategy_mode="SMC")
        assert s.trading_strategy_mode == "SMC"

    def test_valid_ict_config(self):
        """ICT should be accepted."""
        from agent.config import Settings
        s = Settings(trading_strategy_mode="ICT")
        assert s.trading_strategy_mode == "ICT"

    def test_case_insensitive_config(self):
        """Config should be case-insensitive."""
        from agent.config import Settings
        s = Settings(trading_strategy_mode="smc")
        assert s.trading_strategy_mode == "SMC"

    def test_invalid_config_rejected(self):
        """Invalid strategy mode should raise an error."""
        from agent.config import Settings
        with pytest.raises(Exception):  # Pydantic validation error
            Settings(trading_strategy_mode="INVALID")


class TestBacktestEngineStrategyRouting:
    def test_backtest_config_trading_strategy_mode(self):
        """BacktestConfig should accept trading_strategy_mode and pass it to BacktestEngine."""
        from agent.backtest.engine import BacktestConfig, BacktestEngine
        cfg_smc = BacktestConfig(trading_strategy_mode="SMC")
        cfg_ict = BacktestConfig(trading_strategy_mode="ICT")
        assert cfg_smc.trading_strategy_mode == "SMC"
        assert cfg_ict.trading_strategy_mode == "ICT"
        eng_smc = BacktestEngine(cfg_smc)
        eng_ict = BacktestEngine(cfg_ict)
        assert eng_smc.config.trading_strategy_mode == "SMC"
        assert eng_ict.config.trading_strategy_mode == "ICT"

