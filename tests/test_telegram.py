import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from agent.config import settings
from agent.notify.telegram_bot import PaxisBot

@pytest.fixture
def mock_update():
    update = MagicMock()
    update.effective_chat = MagicMock()
    update.effective_chat.id = 123456
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    return update

@pytest.fixture
def mock_context():
    ctx = MagicMock()
    ctx.args = []
    return ctx

@pytest.fixture
def bot():
    bot_inst = PaxisBot()
    bot_inst._agent_ref = MagicMock()
    return bot_inst

@pytest.mark.asyncio
async def test_authenticated_decorator_success(bot, mock_update, mock_context):
    # Set matching chat ID
    settings.telegram_chat_id = "123456"
    
    # Define a dummy handler wrapped with the authenticated decorator
    @PaxisBot.authenticated
    async def dummy_handler(self, update, ctx):
        return "success"
        
    res = await dummy_handler(bot, mock_update, mock_context)
    assert res == "success"
    mock_update.message.reply_text.assert_not_called()

@pytest.mark.asyncio
async def test_authenticated_decorator_unauthorized(bot, mock_update, mock_context):
    # Mismatching chat ID
    settings.telegram_chat_id = "999999"
    
    @PaxisBot.authenticated
    async def dummy_handler(self, update, ctx):
        return "success"
        
    res = await dummy_handler(bot, mock_update, mock_context)
    assert res is None
    mock_update.message.reply_text.assert_called_once()
    assert "Unauthorized" in mock_update.message.reply_text.call_args[0][0]

@pytest.mark.asyncio
async def test_authenticated_decorator_missing_config(bot, mock_update, mock_context):
    # Unconfigured/empty chat ID
    settings.telegram_chat_id = ""
    
    @PaxisBot.authenticated
    async def dummy_handler(self, update, ctx):
        return "success"
        
    res = await dummy_handler(bot, mock_update, mock_context)
    assert res is None
    mock_update.message.reply_text.assert_called_once()
    assert "Unauthorized" in mock_update.message.reply_text.call_args[0][0]

@pytest.mark.asyncio
async def test_cmd_start(bot, mock_update, mock_context):
    settings.telegram_chat_id = "123456"
    await bot._cmd_start(mock_update, mock_context)
    mock_update.message.reply_text.assert_called_once()
    assert "remote control online" in mock_update.message.reply_text.call_args[0][0]

@pytest.mark.asyncio
async def test_cmd_help(bot, mock_update, mock_context):
    settings.telegram_chat_id = "123456"
    await bot._cmd_help(mock_update, mock_context)
    mock_update.message.reply_text.assert_called_once()
    assert "/buy" in mock_update.message.reply_text.call_args[0][0]
    assert "/sell" in mock_update.message.reply_text.call_args[0][0]
    assert "/close" in mock_update.message.reply_text.call_args[0][0]
    assert "/modify" in mock_update.message.reply_text.call_args[0][0]

@pytest.mark.asyncio
async def test_cmd_summary(bot, mock_update, mock_context):
    settings.telegram_chat_id = "123456"
    bot._agent_ref.get_detailed_summary.return_value = "System Summary Report"
    
    await bot._cmd_summary(mock_update, mock_context)
    bot._agent_ref.get_detailed_summary.assert_called_once()
    mock_update.message.reply_text.assert_called_once_with("System Summary Report", parse_mode="HTML")

@pytest.mark.asyncio
async def test_cmd_buy_default_params(bot, mock_update, mock_context):
    settings.telegram_chat_id = "123456"
    settings.lot_size = 0.05
    mock_context.args = ["xauusd"]
    
    # Setup agent mock return value
    bot._agent_ref.place_manual_order.return_value = {
        "success": True,
        "ticket": 98765,
        "price": 2350.50,
        "sl": 0.0,
        "tp": 0.0,
        "volume": 0.05,
    }
    
    await bot._cmd_buy(mock_update, mock_context)
    
    bot._agent_ref.place_manual_order.assert_called_once_with(
        symbol="XAUUSD",
        action="BUY",
        lot_size=0.05,
        sl_pips=0.0,
        tp_pips=0.0,
    )
    
    assert any("Placed Successfully" in call.args[0] for call in mock_update.message.reply_text.call_args_list)

@pytest.mark.asyncio
async def test_cmd_buy_with_params(bot, mock_update, mock_context):
    settings.telegram_chat_id = "123456"
    mock_context.args = ["eurusd", "0.02", "150", "300"]
    
    bot._agent_ref.place_manual_order.return_value = {
        "success": True,
        "ticket": 98766,
        "price": 1.08500,
        "sl": 1.07000,
        "tp": 1.11500,
        "volume": 0.02,
    }
    
    await bot._cmd_buy(mock_update, mock_context)
    
    bot._agent_ref.place_manual_order.assert_called_once_with(
        symbol="EURUSD",
        action="BUY",
        lot_size=0.02,
        sl_pips=150.0,
        tp_pips=300.0,
    )
    assert any("Placed Successfully" in call.args[0] for call in mock_update.message.reply_text.call_args_list)

@pytest.mark.asyncio
async def test_cmd_buy_invalid_args(bot, mock_update, mock_context):
    settings.telegram_chat_id = "123456"
    
    # 1. No symbol
    mock_context.args = []
    await bot._cmd_buy(mock_update, mock_context)
    mock_update.message.reply_text.assert_called_with(
        "❌ <b>Missing Symbol.</b>\nUsage: <code>/buy &lt;symbol&gt; [volume] [sl_pips] [tp_pips]</code>\nExample: <code>/buy XAUUSD 0.01 100 200</code>",
        parse_mode="HTML"
    )
    
    # 2. Invalid volume
    mock_context.args = ["xauusd", "invalid_vol"]
    await bot._cmd_buy(mock_update, mock_context)
    assert "Invalid volume" in mock_update.message.reply_text.call_args[0][0]
    
    # 3. Invalid SL
    mock_context.args = ["xauusd", "0.02", "-50"]
    await bot._cmd_buy(mock_update, mock_context)
    assert "Invalid Stop Loss" in mock_update.message.reply_text.call_args[0][0]

@pytest.mark.asyncio
async def test_cmd_close_success(bot, mock_update, mock_context):
    settings.telegram_chat_id = "123456"
    mock_context.args = ["12345"]
    bot._agent_ref.close_manual_position.return_value = True
    
    await bot._cmd_close(mock_update, mock_context)
    bot._agent_ref.close_manual_position.assert_called_once_with(12345)
    assert any("closed successfully" in call.args[0] for call in mock_update.message.reply_text.call_args_list)

@pytest.mark.asyncio
async def test_cmd_close_invalid_ticket(bot, mock_update, mock_context):
    settings.telegram_chat_id = "123456"
    mock_context.args = ["abc"]
    
    await bot._cmd_close(mock_update, mock_context)
    bot._agent_ref.close_manual_position.assert_not_called()
    assert "Invalid ticket ID" in mock_update.message.reply_text.call_args[0][0]

@pytest.mark.asyncio
async def test_cmd_modify_success(bot, mock_update, mock_context):
    settings.telegram_chat_id = "123456"
    mock_context.args = ["12345", "50", "100"]
    bot._agent_ref.modify_manual_position_stops.return_value = {
        "success": True,
        "sl": 1.0950,
        "tp": 1.1100,
    }
    
    await bot._cmd_modify(mock_update, mock_context)
    bot._agent_ref.modify_manual_position_stops.assert_called_once_with(12345, 50.0, 100.0)
    assert any("Stops Modified" in call.args[0] for call in mock_update.message.reply_text.call_args_list)

@pytest.mark.asyncio
async def test_cmd_modify_invalid_args(bot, mock_update, mock_context):
    settings.telegram_chat_id = "123456"
    mock_context.args = ["12345", "abc", "100"]
    
    await bot._cmd_modify(mock_update, mock_context)
    bot._agent_ref.modify_manual_position_stops.assert_not_called()
    assert "Invalid arguments" in mock_update.message.reply_text.call_args[0][0]
