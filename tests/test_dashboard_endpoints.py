import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
import dashboard.backend.database as db_mod
from dashboard.backend.models import Base
from dashboard.backend.app import app

@pytest.fixture(autouse=True)
def use_test_db(tmp_path):
    """Use an isolated temp SQLite database for all endpoint tests to avoid polluting paxis_trades.db."""
    test_db_file = tmp_path / "test_paxis_dash.db"
    test_db_url = f"sqlite+aiosqlite:///{test_db_file}"
    test_engine = create_async_engine(test_db_url, echo=False, connect_args={"check_same_thread": False})
    test_sessionmaker = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    orig_engine = db_mod.engine
    orig_sessionmaker = db_mod.AsyncSessionLocal

    db_mod.engine = test_engine
    db_mod.AsyncSessionLocal = test_sessionmaker

    # Run init_db synchronously via async event loop
    import asyncio
    async def init_tables():
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    loop = asyncio.get_event_loop()
    if loop.is_running():
        loop.create_task(init_tables())
    else:
        loop.run_until_complete(init_tables())

    yield

    db_mod.engine = orig_engine
    db_mod.AsyncSessionLocal = orig_sessionmaker


def test_status_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert "agent_running" in data
        assert "paused" in data
        assert "lot_size" in data

def test_internal_decision_logging():
    with TestClient(app) as client:
        payload = {
            "symbol": "XAUUSD",
            "action": "BUY",
            "confidence": 0.85,
            "entry": 2350.50,
            "sl": 2345.00,
            "tp": 2360.00,
            "rr_ratio": 1.7,
            "pattern": "Double Bottom",
            "session": "London",
            "reasoning": "Strong support bounce",
            "risk_passed": True,
            "block_reason": "",
            "executed": True,
            "ticket": 123456
        }
        # Post decision log
        response = client.post("/api/internal/decision", json=payload)
        assert response.status_code == 200
        assert response.json() == {"success": True}

        # Verify it was logged in decisions endpoint
        response = client.get("/api/decisions?limit=5")
        assert response.status_code == 200
        decisions = response.json().get("decisions", [])
        assert len(decisions) > 0
        latest = decisions[0]
        assert latest["symbol"] == "XAUUSD"
        assert latest["action"] == "BUY"
        assert latest["confidence"] == 0.85
        assert latest["ticket"] == 123456

def test_internal_trade_flow():
    import random
    ticket = random.randint(1000000, 9999999)
    with TestClient(app) as client:
        open_payload = {
            "ticket": ticket,
            "symbol": "EURUSD",
            "action": "BUY",
            "lot_size": 0.1,
            "entry_price": 1.08500,
            "sl": 1.08300,
            "tp": 1.08900,
            "pattern": "Bullish Engulfing",
            "confidence": 0.80,
            "reasoning": "Moving average crossover",
            "dry_run": True
        }
        # Open trade
        response = client.post("/api/internal/trade/open", json=open_payload)
        assert response.status_code == 200
        assert response.json() == {"success": True}

        # Close trade
        close_payload = {
            "ticket": ticket,
            "close_price": 1.08850,
            "pnl": 35.00,
            "outcome": "WIN"
        }
        response = client.post("/api/internal/trade/close", json=close_payload)
        assert response.status_code == 200
        assert response.json() == {"success": True}

        # Verify it's listed under closed trades
        response = client.get("/api/trades?status=CLOSED")
        assert response.status_code == 200
        trades = response.json().get("trades", [])
        assert len(trades) > 0
        matched = [t for t in trades if t["ticket"] == ticket]
        assert len(matched) == 1
        trade = matched[0]
        assert trade["status"] == "CLOSED"
        assert trade["pnl"] == 35.00
        assert trade["outcome"] == "WIN"

def test_open_trades_endpoint():
    import random
    ticket = random.randint(1000000, 9999999)
    with TestClient(app) as client:
        # Create an open trade first via the internal API
        open_payload = {
            "ticket": ticket,
            "symbol": "XAUUSD",
            "action": "BUY",
            "lot_size": 0.05,
            "entry_price": 2350.00,
            "sl": 2340.00,
            "tp": 2370.00,
            "pattern": "Test Pattern",
            "confidence": 0.90,
            "reasoning": "Support test",
            "dry_run": True
        }
        response = client.post("/api/internal/trade/open", json=open_payload)
        assert response.status_code == 200

        # Query open trades
        response = client.get("/api/trades/open")
        assert response.status_code == 200
        data = response.json()
        assert "positions" in data
        assert "count" in data
        
        positions = data["positions"]
        matched = [p for p in positions if p["ticket"] == ticket]
        assert len(matched) == 1
        pos = matched[0]
        assert pos["symbol"] == "XAUUSD"
        assert pos["type"] == "BUY"
        assert pos["volume"] == 0.05
        assert pos["price_open"] == 2350.00
        assert "price_current" in pos
        assert "profit" in pos
        
        # Clean up by closing it
        close_payload = {
            "ticket": ticket,
            "close_price": 2355.00,
            "pnl": 25.00,
            "outcome": "WIN"
        }
        client.post("/api/internal/trade/close", json=close_payload)


def test_internal_trade_open_duplicate_handling():
    import random
    ticket = random.randint(10000000, 99999999)
    with TestClient(app) as client:
        open_payload = {
            "ticket": ticket,
            "symbol": "EURUSD",
            "action": "BUY",
            "lot_size": 0.01,
            "entry_price": 1.08500,
            "sl": 1.08300,
            "tp": 1.08900,
            "pattern": "Test",
            "confidence": 0.80,
            "reasoning": "Duplicate test",
            "dry_run": True
        }
        # First insertion should succeed
        response = client.post("/api/internal/trade/open", json=open_payload)
        assert response.status_code == 200
        assert response.json() == {"success": True}

        # Second insertion with the same ticket should fail gracefully and return success: False with error msg
        response2 = client.post("/api/internal/trade/open", json=open_payload)
        assert response2.status_code == 200
        assert response2.json()["success"] is False
        assert "error" in response2.json()

        # Clean up
        close_payload = {
            "ticket": ticket,
            "close_price": 1.08500,
            "pnl": 0.0,
            "outcome": "LOSS"
        }
        client.post("/api/internal/trade/close", json=close_payload)


def test_tradingview_ai_suggestion_endpoint():
    with patch("agent.llm.ollama_client.ollama_client.chat") as mock_chat:
        mock_chat.return_value = '{"action": "BUY", "confidence": 0.85, "entry": 2400.0, "sl": 2390.0, "tp": 2420.0, "reasoning": "Strong momentum", "pattern": "Bullish Flag"}'
        with TestClient(app) as client:
            response = client.post("/api/tradingview/ai-suggestion?symbol=XAUUSD")
            assert response.status_code == 200
            data = response.json()
            assert "move_suggestion" in data
            assert "ai_decision" in data
            assert data["ai_decision"]["action"] == "BUY"
            assert data["ai_decision"]["confidence"] == 0.85
            assert "technical_analysis" in data
