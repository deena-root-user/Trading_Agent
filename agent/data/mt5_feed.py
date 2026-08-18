"""
PAXIS Agent — MT5 Data Feed
Connects to MetaTrader 5 terminal and pulls OHLCV candles + tick data.
NOTE: MetaTrader5 Python lib only works on Windows with MT5 terminal running.
"""
from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import requests
from loguru import logger

from agent.config import settings

# Graceful import — MT5 lib only available on Windows with terminal
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.info("MetaTrader5 library not available — running in simulated mode")


# ─── Timeframe constants ──────────────────────────────────────────────────────
TIMEFRAMES: Dict[str, int] = {}
if MT5_AVAILABLE:
    TIMEFRAMES = {
        "M1":  mt5.TIMEFRAME_M1,
        "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }


@dataclass
class TickData:
    symbol: str
    bid: float
    ask: float
    spread_pips: float
    time: float


class MT5Feed:
    """Manages MT5 connection and data retrieval, with full high-fidelity simulation on Linux."""

    _connected: bool = False
    _remote_active: bool = False
    _active_symbols: List[str] = []
    _symbol_map: Dict[str, str] = {}

    # ── Simulated Data States for Linux ───────────────────────────────────────
    _simulated_positions: Dict[int, dict] = {}
    _closed_simulated_records: Dict[int, dict] = {}
    _simulated_balance: float = 10000.0
    _ticket_counter: int = 10000000
    _simulated_prices: Dict[str, float] = {}
    _last_tick_fetch_time: Dict[str, float] = {}
    _last_remote_check_time: float = 0.0

    def __init__(self):
        self._init_ticket_counter()

    def _init_ticket_counter(self) -> None:
        """Initialize the simulated ticket counter based on the highest existing ticket in DB."""
        import sqlite3
        import os
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_path = os.path.join(base_dir, "paxis_trades.db")
        
        max_ticket = 10000000
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
                if cursor.fetchone():
                    cursor.execute("SELECT MAX(ticket) FROM trades")
                    row = cursor.fetchone()
                    if row and row[0] is not None:
                        max_ticket = max(max_ticket, int(row[0]))
                conn.close()
            except Exception as exc:
                logger.error(f"Failed to query max ticket from DB: {exc}")
                
        self._ticket_counter = max_ticket + 1
        logger.info(f"Initialized MT5Feed simulated ticket counter to {self._ticket_counter}")

    def _fetch_active_symbols(self) -> bool:
        """Fetch the list of active symbols from the remote bridge."""
        if not settings.mt5_remote_ip:
            return False
        try:
            url = f"http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}/symbols"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                self._active_symbols = response.json()
                self._symbol_map = {s.upper().replace("/", ""): s for s in self._active_symbols}
                logger.info(f"Loaded {len(self._active_symbols)} active symbols from remote bridge.")
                return True
        except Exception as exc:
            logger.error(f"Failed to fetch active symbols from remote bridge: {exc}")
        return False

    def resolve_symbol(self, symbol: str) -> str:
        """Resolve standard base symbol (like XAUUSD) to broker active symbol (like XAUUSD.m)."""
        if not symbol:
            return symbol
        sym_upper = symbol.upper().replace("/", "")
        if sym_upper in self._symbol_map:
            return self._symbol_map[sym_upper]
        
        # Auto-fetch active symbols from remote bridge if empty
        if not self._active_symbols and settings.mt5_remote_ip:
            self._fetch_active_symbols()
            if sym_upper in self._symbol_map:
                return self._symbol_map[sym_upper]

        for active in self._active_symbols:
            active_upper = active.upper().replace("/", "")
            if active_upper == sym_upper or active_upper.startswith(sym_upper) or sym_upper.startswith(active_upper):
                self._symbol_map[sym_upper] = active
                return active
        return symbol

    def _get_yfinance_candles(
        self,
        symbol: str,
        timeframe: str = "M5",
        count: int = 200,
    ) -> Optional[pd.DataFrame]:
        """Fetch candles for symbol on timeframe from Yahoo Finance."""
        tf_map = {
            "M1": ("1m", "1d"),
            "M5": ("5m", "5d"),
            "M15": ("15m", "5d"),
            "H1": ("1h", "30d"),
            "H4": ("1h", "60d"),  # aggregate 1h candles
            "D1": ("1d", "365d"),
        }
        
        interval, range_val = tf_map.get(timeframe.upper(), ("5m", "5d"))
        
        # Map broker symbol to Yahoo Finance symbol
        sym = symbol.upper().replace("/", "").strip()
        if "." in sym:
            sym = sym.split(".")[0]
            
        if sym in ("XAUUSD", "GOLD", "XAU"):
            yahoo_sym = "GC=F"
        elif sym in ("XAGUSD", "SILVER", "XAG"):
            yahoo_sym = "SI=F"
        elif sym in ("USOUSD", "WTI", "CL"):
            yahoo_sym = "CL=F"
        elif len(sym) == 6:
            yahoo_sym = f"{sym}=X"
        else:
            yahoo_sym = sym

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_sym}?interval={interval}&range={range_val}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=2)
            if response.status_code != 200:
                logger.warning(f"Yahoo Finance returned status {response.status_code} for {yahoo_sym}")
                return None
            
            data = response.json()
            result = data.get("chart", {}).get("result", [])
            if not result:
                return None
                
            res = result[0]
            timestamps = res.get("timestamp", [])
            quote = res.get("indicators", {}).get("quote", [{}])[0]
            
            opens = quote.get("open", [])
            highs = quote.get("high", [])
            lows = quote.get("low", [])
            closes = quote.get("close", [])
            volumes = quote.get("volume", [])
            
            if not timestamps or not closes:
                return None
                
            df = pd.DataFrame({
                "time": pd.to_datetime(timestamps, unit="s", utc=True),
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes
            })
            
            # Fill missing values
            df = df.ffill().bfill()
            df["spread"] = 15.0  # default mock spread
            
            # Type casting
            df["open"] = df["open"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)
            df["spread"] = df["spread"].astype(float)
            
            # Resample for H4
            if timeframe.upper() == "H4":
                df.set_index("time", inplace=True)
                resampled = df.resample("4h").agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                    "spread": "mean"
                })
                resampled = resampled.dropna().reset_index()
                df = resampled
                
            df = df.sort_values("time").reset_index(drop=True)
            df = df.tail(count).reset_index(drop=True)
            return df
            
        except Exception as e:
            logger.warning(f"Error fetching Yahoo Finance candles for {yahoo_sym}: {e}")
            return None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Initialize connection to MT5 terminal (or start simulator on Linux, or use remote bridge)."""
        if settings.mt5_remote_ip:
            logger.info(f"Connecting to remote MT5 bridge at http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}...")
            try:
                # Try standard health endpoint first
                url = f"http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}/health"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    try:
                        response.json()
                        logger.info("Remote MT5 bridge connected successfully ✓ (health)")
                        self._connected = True
                        self._remote_active = True
                        self._fetch_active_symbols()
                        return True
                    except Exception:
                        logger.warning("Remote /health returned status 200 but not valid JSON (e.g. port conflict/HTML dashboard).")
                # Try status endpoint if health is 404/error
                url_status = f"http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}/status"
                response_status = requests.get(url_status, timeout=5)
                if response_status.status_code == 200:
                    try:
                        response_status.json()
                        logger.info("Remote MT5 bridge connected successfully ✓ (status)")
                        self._connected = True
                        self._remote_active = True
                        self._fetch_active_symbols()
                        return True
                    except Exception:
                        logger.warning("Remote /status returned status 200 but not valid JSON.")
                logger.error(f"Remote MT5 bridge returned status code {response.status_code}")
            except Exception as exc:
                logger.error(
                    f"\n"
                    f"========================================================================\n"
                    f"⚠️  WARNING: Could not connect to remote MT5 bridge at http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}\n"
                    f"Error: {exc}\n"
                    f"------------------------------------------------------------------------\n"
                    f"To fix this error, ensure that:\n"
                    f"  1. mt5-bridge is running on your Windows PC. Start it using command:\n"
                    f"     pip install mt5-bridge\n"
                    f"     mt5-bridge server --host 0.0.0.0 --port {settings.mt5_remote_port}\n"
                    f"  2. Your Windows firewall allows incoming connections on port {settings.mt5_remote_port}.\n"
                    f"  3. IP address {settings.mt5_remote_ip} is correct and reachable from this machine.\n"
                    f"========================================================================\n"
                    f"👉 FALLING BACK to high-fidelity simulated mode so the agent can run!\n"
                )
            
            # Safe simulated fallback
            self._connected = True
            self._remote_active = False
            return True

        if not MT5_AVAILABLE:
            logger.info("MT5 library not available — running in simulated mode")
            self._connected = True
            return True

        try:
            if not mt5.initialize(
                path=settings.mt5_path,
                login=settings.mt5_login,
                password=settings.mt5_password,
                server=settings.mt5_server,
                timeout=10_000,
            ):
                err = mt5.last_error()
                logger.error(f"MT5 init failed: {err}")
                return False

            account = mt5.account_info()
            if account is None:
                logger.error("MT5 connected but account_info() returned None")
                return False

            self._connected = True
            logger.info(
                f"MT5 connected ✓ | Account: {account.login} | "
                f"Balance: {account.balance:.2f} {account.currency} | "
                f"Server: {account.server}"
            )
            return True

        except Exception as exc:
            logger.error(f"MT5 connect exception: {exc}")
            return False

    def disconnect(self) -> None:
        """Shut down MT5 connection."""
        if settings.mt5_remote_ip:
            self._connected = False
            self._remote_active = False
            logger.info("MT5 remote bridge disconnected")
            return

        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")
        elif not MT5_AVAILABLE:
            self._connected = False
            logger.info("MT5 simulator disconnected")

    def is_connected(self) -> bool:
        """Check if MT5 terminal is still connected."""
        if settings.mt5_remote_ip:
            current_time = time.time()
            if not self._remote_active:
                # If we are in simulated fallback, periodically check if remote came back online
                if current_time - self._last_remote_check_time > 10.0:
                    self._last_remote_check_time = current_time
                    logger.debug("Checking if remote MT5 bridge came back online...")
                    try:
                        url = f"http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}/health"
                        response = requests.get(url, timeout=1) # ultra short timeout to avoid UI lag
                        if response.status_code == 200:
                            response.json()
                            logger.info("Remote MT5 bridge detected online! Swapping from simulated to remote mode.")
                            self._remote_active = True
                            self._connected = True
                            self._fetch_active_symbols()
                            return True
                    except Exception:
                        pass
                    
                    try:
                        url_status = f"http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}/status"
                        response_status = requests.get(url_status, timeout=1)
                        if response_status.status_code == 200:
                            response_status.json()
                            logger.info("Remote MT5 bridge detected online! Swapping from simulated to remote mode.")
                            self._remote_active = True
                            self._connected = True
                            self._fetch_active_symbols()
                            return True
                    except Exception:
                        pass
                
                # Remain in simulated fallback
                return True

            try:
                url = f"http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}/health"
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    try:
                        response.json()
                        self._connected = True
                        if not self._active_symbols:
                            self._fetch_active_symbols()
                        return True
                    except Exception:
                        pass
                url_status = f"http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}/status"
                response_status = requests.get(url_status, timeout=2)
                if response_status.status_code == 200:
                    try:
                        response_status.json()
                        self._connected = True
                        if not self._active_symbols:
                            self._fetch_active_symbols()
                        return True
                    except Exception:
                        pass
            except Exception:
                pass
            logger.warning("Remote MT5 bridge connection lost. Falling back to simulated mode.")
            self._remote_active = False
            self._connected = True
            self._last_remote_check_time = current_time
            return True

        if not MT5_AVAILABLE:
            return self._connected
        if not self._connected:
            return False
        return mt5.terminal_info() is not None

    def reconnect(self, retries: int = 3, delay: float = 5.0) -> bool:
        """Attempt to reconnect to MT5 with retries."""
        for attempt in range(1, retries + 1):
            logger.info(f"MT5 reconnect attempt {attempt}/{retries}")
            self.disconnect()
            if self.connect():
                return True
            time.sleep(delay)
        logger.error("MT5 reconnect failed after all retries")
        return False

    # ── Simulator Helper Methods ──────────────────────────────────────────────

    def create_simulated_position(self, symbol: str, action: str, sl: float, tp: float, volume: float, comment: str = "PAXIS") -> dict:
        """Create a new mock open position."""
        tick = self.get_tick(symbol)
        price = tick.ask if action == "BUY" else tick.bid
        ticket = self._ticket_counter
        self._ticket_counter += 1

        pos = {
            "ticket": ticket,
            "symbol": symbol.upper().replace("/", ""),
            "type": action.upper(),
            "volume": float(volume),
            "price_open": price,
            "price_current": price,
            "sl": float(sl) if sl else 0.0,
            "tp": float(tp) if tp else 0.0,
            "profit": 0.0,
            "time_open": pd.Timestamp.now(tz="UTC"),
            "comment": comment
        }
        self._simulated_positions[ticket] = pos
        logger.info(f"[SIMULATOR] Created simulated {action} position for {symbol} | price={price} | ticket={ticket}")
        return pos

    def close_simulated_position(self, ticket: int, close_price: Optional[float] = None, reason: str = "MANUAL") -> bool:
        """Close a mock position, compute exact realized PnL, and update simulated balance."""
        if ticket in self._simulated_positions:
            pos = self._simulated_positions.pop(ticket)
            sym = pos["symbol"]
            contract_size = 100.0 if "XAU" in sym or "GOLD" in sym else 100000.0

            if close_price is not None and close_price > 0:
                final_exit = float(close_price)
            else:
                tick = self.get_tick(sym)
                if tick:
                    final_exit = tick.bid if pos["type"] == "BUY" else tick.ask
                else:
                    final_exit = pos["price_current"]

            if pos["type"] == "BUY":
                pnl = round((final_exit - pos["price_open"]) * contract_size * pos["volume"], 2)
            else:
                pnl = round((pos["price_open"] - final_exit) * contract_size * pos["volume"], 2)

            self._simulated_balance = round(self._simulated_balance + pnl, 2)
            closed_record = {
                **pos,
                "close_price": final_exit,
                "price_current": final_exit,
                "profit": pnl,
                "pnl": pnl,
                "status": "CLOSED",
                "outcome": "WIN" if pnl > 0 else "LOSS",
                "time_close": pd.Timestamp.now(tz="UTC"),
                "reason": reason,
            }
            self._closed_simulated_records[ticket] = closed_record
            logger.info(
                f"[SIMULATOR] Closed simulated position {ticket} | exit={final_exit:.5f} | "
                f"final P&L={pnl} USD | new balance={self._simulated_balance}"
            )
            return True
        return False

    def modify_simulated_position(self, ticket: int, sl: float, tp: float) -> bool:
        """Modify simulated position SL/TP parameters."""
        if ticket in self._simulated_positions:
            self._simulated_positions[ticket]["sl"] = float(sl)
            self._simulated_positions[ticket]["tp"] = float(tp)
            logger.info(f"[SIMULATOR] Modified simulated position {ticket} | new SL={sl} | new TP={tp}")
            return True
        return False

    # ── Candle Data ───────────────────────────────────────────────────────────

    def get_candles(
        self,
        symbol: str,
        timeframe: str = "M5",
        count: int = 200,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch last `count` OHLCV candles for `symbol` on `timeframe`.
        """
        if not self.is_connected():
            logger.warning(f"MT5 not connected — cannot fetch candles for {symbol}")
            return None

        if settings.mt5_remote_ip and self._remote_active:
            resolved = self.resolve_symbol(symbol)
            try:
                url = f"http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}/rates/{resolved}?timeframe={timeframe}&count={count}"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        df = pd.DataFrame(data)
                        if "time" in df.columns:
                            df["time"] = pd.to_datetime(df["time"])
                        if "tick_volume" in df.columns:
                            df = df.rename(columns={"tick_volume": "volume"})
                        required = ["time", "open", "high", "low", "close", "volume"]
                        for col in required:
                            if col not in df.columns:
                                raise ValueError(f"Missing column {col} in remote candles")
                        if "spread" not in df.columns:
                            df["spread"] = 15.0
                        df = df[["time", "open", "high", "low", "close", "volume", "spread"]]
                        df = df.sort_values("time").reset_index(drop=True)
                        logger.debug(f"Fetched {len(df)} remote candles for {symbol} (resolved: {resolved}) {timeframe}")
                        return df
            except Exception as exc:
                logger.debug(f"get_candles({symbol}) remote query failed (falling back to simulation): {exc}")

        # If running in simulated mode or remote bridge failed/offline
        if not MT5_AVAILABLE or (settings.mt5_remote_ip and not self._remote_active):
            # First try Yahoo Finance fallback
            df_yf = self._get_yfinance_candles(symbol, timeframe, count)
            if df_yf is not None and not df_yf.empty:
                logger.info(f"Successfully fetched {len(df_yf)} candles from Yahoo Finance fallback for {symbol} {timeframe}")
                return df_yf
            
            logger.warning(f"Yahoo Finance fallback failed or returned empty for {symbol} {timeframe}. Using synthetic generator.")
            # Generate simulated candle data
            tick = self.get_tick(symbol)
            base_price = tick.bid if tick else 4530.0

            freq_map = {
                "M1": "1min",
                "M5": "5min",
                "M15": "15min",
                "H1": "1h",
                "H4": "4h",
                "D1": "1D"
            }
            freq = freq_map.get(timeframe, "5min")
            times = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=count, freq=freq)

            pip_size = 0.01 if "JPY" in symbol.upper() or "XAU" in symbol.upper() or "GOLD" in symbol.upper() else 0.0001
            
            # Generate a nice random walk
            prices = [base_price]
            for _ in range(count - 1):
                change = random.uniform(-4, 4) * pip_size
                prices.append(round(prices[-1] + change, 5))
            
            prices = prices[::-1] # End with the latest base price

            data = []
            for i, t in enumerate(times):
                close_p = prices[i]
                open_p = prices[i - 1] if i > 0 else round(close_p - random.uniform(-1, 1) * pip_size, 5)
                high_p = max(open_p, close_p) + round(random.uniform(0.1, 2) * pip_size, 5)
                low_p = min(open_p, close_p) - round(random.uniform(0.1, 2) * pip_size, 5)
                vol = random.randint(100, 1000)
                spread = random.randint(10, 20)

                data.append({
                    "time": t,
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "volume": float(vol),
                    "spread": float(spread)
                })

            df = pd.DataFrame(data)
            return df

        tf = TIMEFRAMES.get(timeframe)
        if tf is None:
            logger.error(f"Unknown timeframe: {timeframe}")
            return None

        try:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
            if rates is None or len(rates) == 0:
                logger.warning(f"No candle data returned for {symbol} {timeframe}")
                return None

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.rename(columns={"tick_volume": "volume"})
            df = df[["time", "open", "high", "low", "close", "volume", "spread"]]
            df = df.sort_values("time").reset_index(drop=True)

            logger.debug(f"Fetched {len(df)} candles for {symbol} {timeframe}")
            return df

        except Exception as exc:
            logger.error(f"get_candles({symbol}, {timeframe}) error: {exc}")
            return None

    # ── Tick / Spread ─────────────────────────────────────────────────────────

    def get_tick(self, symbol: str) -> Optional[TickData]:
        """Get current bid/ask tick for a symbol."""
        if not self.is_connected():
            return None

        if settings.mt5_remote_ip and self._remote_active:
            resolved = self.resolve_symbol(symbol)
            try:
                url = f"http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}/tick/{resolved}"
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict):
                        bid = float(data.get("bid", 0.0))
                        ask = float(data.get("ask", 0.0))
                        pip_size = 0.01 if "JPY" in symbol.upper() or "XAU" in symbol.upper() or "GOLD" in symbol.upper() else 0.0001
                        spread_pips = float(data.get("spread_pips", round((ask - bid) / pip_size, 1) if bid > 0 else 1.5))
                        return TickData(
                            symbol=symbol,
                            bid=bid,
                            ask=ask,
                            spread_pips=spread_pips,
                            time=float(data.get("time", time.time())),
                        )
            except Exception as exc:
                logger.error(f"get_tick({symbol}) remote error: {exc}")

        # Check if we should run simulated/fallback tick retrieval
        if not MT5_AVAILABLE or (settings.mt5_remote_ip and not self._remote_active):
            sym = symbol.upper().replace("/", "").strip()
            if "." in sym:
                sym = sym.split(".")[0]
            
            pip_size = 0.01 if "JPY" in sym or "XAU" in sym or "GOLD" in sym else 0.0001
            
            current_time = time.time()
            cached_price = self._simulated_prices.get(sym)
            last_fetch = self._last_tick_fetch_time.get(sym, 0)
            
            fetched_new = False
            if cached_price is None or (current_time - last_fetch > 60.0):
                # Try fetching latest close price from Yahoo Finance using M1 timeframe
                try:
                    df_yf = self._get_yfinance_candles(symbol, timeframe="M1", count=1)
                    if df_yf is not None and not df_yf.empty:
                        price = float(df_yf.iloc[-1]["close"])
                        self._simulated_prices[sym] = price
                        self._last_tick_fetch_time[sym] = current_time
                        fetched_new = True
                        logger.info(f"Updated live tick price for {symbol} ({sym}) from Yahoo Finance: {price}")
                except Exception as exc:
                    logger.debug(f"Failed to fetch live tick from Yahoo Finance for {symbol}: {exc}")
            
            if not fetched_new:
                price = self._simulated_prices.get(sym)
                if price is None:
                    # Fallback default values
                    if "XAU" in sym or "GOLD" in sym:
                        price = 4530.00
                    elif "EURUSD" in sym:
                        price = 1.08500
                    elif "GBPUSD" in sym:
                        price = 1.27000
                    elif "AUDJPY" in sym:
                        price = 98.000
                    else:
                        price = 100.000 if "JPY" in sym else 1.00000
                
                # Apply small random fluctuation between API calls
                price = round(price + random.uniform(-2, 2) * pip_size, 5)
                self._simulated_prices[sym] = price

            spread_pips = round(random.uniform(1.0, 2.2), 1)
            spread_raw = spread_pips * pip_size
            bid = round(price - spread_raw / 2, 5)
            ask = round(price + spread_raw / 2, 5)

            return TickData(
                symbol=symbol,
                bid=bid,
                ask=ask,
                spread_pips=spread_pips,
                time=time.time(),
            )

        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.warning(f"No tick data for {symbol}")
                return None

            info = mt5.symbol_info(symbol)
            digits = info.digits if info else 5
            pip_size = 0.01 if digits == 3 else 0.0001
            spread_pips = round((tick.ask - tick.bid) / pip_size, 1)

            return TickData(
                symbol=symbol,
                bid=tick.bid,
                ask=tick.ask,
                spread_pips=spread_pips,
                time=tick.time,
            )

        except Exception as exc:
            logger.error(f"get_tick({symbol}) error: {exc}")
            return None

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account_balance(self) -> Optional[float]:
        """Return current account balance."""
        if not self.is_connected():
            return None
        if settings.mt5_remote_ip and self._remote_active:
            try:
                url = f"http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}/account"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict):
                        return float(data.get("balance") or data.get("account_balance") or 0.0)
            except Exception as exc:
                logger.debug(f"Remote balance fetch failed, checking status: {exc}")
            try:
                url = f"http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}/status"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict) and "balance" in data:
                        return float(data["balance"])
            except Exception:
                pass
            return self._simulated_balance

        if not MT5_AVAILABLE:
            return self._simulated_balance
        info = mt5.account_info()
        return info.balance if info else None

    def get_account_equity(self) -> Optional[float]:
        """Return current account equity."""
        if not self.is_connected():
            return None
        if settings.mt5_remote_ip and self._remote_active:
            try:
                url = f"http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}/account"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict):
                        return float(data.get("equity") or data.get("account_equity") or 0.0)
            except Exception as exc:
                logger.debug(f"Remote equity fetch failed, checking status: {exc}")
            try:
                url = f"http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}/status"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict) and "equity" in data:
                        return float(data["equity"])
            except Exception:
                pass
            positions = self.get_open_positions()
            floating_pnl = sum(p.get("profit", 0.0) for p in positions)
            return round(self._simulated_balance + floating_pnl, 2)

        if not MT5_AVAILABLE:
            positions = self.get_open_positions()
            floating_pnl = sum(p.get("profit", 0.0) for p in positions)
            return round(self._simulated_balance + floating_pnl, 2)
        info = mt5.account_info()
        return info.equity if info else None

    def get_open_positions(self, symbol: Optional[str] = None) -> List[dict]:
        """
        Return list of all open positions (or filtered by symbol).
        """
        if not self.is_connected():
            return []

        if settings.mt5_remote_ip and self._remote_active:
            try:
                url = f"http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}/positions"
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        result = []
                        for pos in data:
                            if not isinstance(pos, dict):
                                continue
                            pos_sym = pos.get("symbol", "").upper().replace("/", "")
                            target_sym = self.resolve_symbol(symbol).upper().replace("/", "") if symbol else None
                            if target_sym and pos_sym != target_sym and pos_sym != symbol.upper().replace("/", ""):
                                continue
                            
                            # Normalize type
                            pos_type_raw = str(pos.get("type", pos.get("action", ""))).upper()
                            pos_type = "BUY" if "BUY" in pos_type_raw or pos_type_raw == "0" else "SELL"
                            
                            # Parse open time safely
                            time_open_raw = pos.get("time_open", pos.get("time", time.time()))
                            try:
                                time_open = pd.Timestamp(time_open_raw)
                            except Exception:
                                time_open = pd.Timestamp.now(tz="UTC")
                                
                            result.append({
                                "ticket": int(pos.get("ticket", 0)),
                                "symbol": pos_sym,
                                "type": pos_type,
                                "volume": float(pos.get("volume", pos.get("qty", pos.get("lots", 0.01)))),
                                "price_open": float(pos.get("price_open", pos.get("open_price", 0.0))),
                                "price_current": float(pos.get("price_current", pos.get("current_price", pos.get("price", 0.0)))),
                                "sl": float(pos.get("sl", 0.0)),
                                "tp": float(pos.get("tp", 0.0)),
                                "profit": float(pos.get("profit", pos.get("pnl", 0.0))),
                                "time_open": time_open,
                            })
                        return result
            except Exception as exc:
                logger.error(f"get_open_positions remote error: {exc}")

        if not MT5_AVAILABLE or (settings.mt5_remote_ip and not self._remote_active):
            # Process simulated positions, check SL/TP hits in background
            res = []
            for ticket, pos in list(self._simulated_positions.items()):
                if symbol and pos["symbol"].upper() != symbol.upper():
                    continue

                sym = pos["symbol"]
                tick = self.get_tick(sym)
                if tick:
                    pos["price_current"] = tick.ask if pos["type"] == "SELL" else tick.bid
                    contract_size = 100.0 if "XAU" in sym or "GOLD" in sym else 100000.0
                    
                    # Calculate floating profit
                    if pos["type"] == "BUY":
                        pos["profit"] = round((pos["price_current"] - pos["price_open"]) * contract_size * pos["volume"], 2)
                    else:
                        pos["profit"] = round((pos["price_open"] - pos["price_current"]) * contract_size * pos["volume"], 2)

                    # Simulating SL/TP hits
                    if pos["sl"] > 0:
                        if pos["type"] == "BUY" and pos["price_current"] <= pos["sl"]:
                            logger.info(f"[SIMULATOR] Position {ticket} hit STOP LOSS ({pos['sl']})")
                            self.close_simulated_position(ticket, close_price=pos["sl"], reason="STOP_LOSS")
                            continue
                        elif pos["type"] == "SELL" and pos["price_current"] >= pos["sl"]:
                            logger.info(f"[SIMULATOR] Position {ticket} hit STOP LOSS ({pos['sl']})")
                            self.close_simulated_position(ticket, close_price=pos["sl"], reason="STOP_LOSS")
                            continue

                    if pos["tp"] > 0:
                        if pos["type"] == "BUY" and pos["price_current"] >= pos["tp"]:
                            logger.info(f"[SIMULATOR] Position {ticket} hit TAKE PROFIT ({pos['tp']})")
                            self.close_simulated_position(ticket, close_price=pos["tp"], reason="TAKE_PROFIT")
                            continue
                        elif pos["type"] == "SELL" and pos["price_current"] <= pos["tp"]:
                            logger.info(f"[SIMULATOR] Position {ticket} hit TAKE PROFIT ({pos['tp']})")
                            self.close_simulated_position(ticket, close_price=pos["tp"], reason="TAKE_PROFIT")
                            continue

                res.append(pos)
            return res

        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()

            if positions is None:
                return []

            result = []
            for pos in positions:
                result.append({
                    "ticket":     pos.ticket,
                    "symbol":     pos.symbol,
                    "type":       "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                    "volume":     pos.volume,
                    "price_open": pos.price_open,
                    "price_current": pos.price_current,
                    "sl":         pos.sl,
                    "tp":         pos.tp,
                    "profit":     pos.profit,
                    "time_open":  pd.Timestamp(pos.time, unit="s", tz="UTC"),
                })
            return result

        except Exception as exc:
            logger.error(f"get_open_positions error: {exc}")
            return []


    def get_closed_trade_details(self, ticket: int, last_known_pos: Optional[dict] = None) -> dict:
        """
        Retrieve true realized closed trade details (close_price, pnl, outcome)
        from MT5 deal history, remote bridge, or simulated close records.
        """
        base = dict(last_known_pos) if last_known_pos else {"ticket": ticket}

        # 1. Check simulated closed records
        if ticket in self._closed_simulated_records:
            sim_closed = self._closed_simulated_records[ticket]
            base["close_price"] = sim_closed.get("close_price", base.get("price_current", 0.0))
            base["profit"] = sim_closed.get("profit", 0.0)
            base["pnl"] = sim_closed.get("profit", 0.0)
            base["outcome"] = sim_closed.get("outcome", "WIN" if base["profit"] > 0 else "LOSS")
            return base

        # 2. Remote MT5 Bridge
        if settings.mt5_remote_ip and self._remote_active:
            try:
                import requests
                url = f"http://{settings.mt5_remote_ip}:{settings.mt5_remote_port}/history?ticket={ticket}"
                resp = requests.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    deals = data if isinstance(data, list) else data.get("deals", [data])
                    if deals:
                        total_pnl = sum(
                            float(d.get("profit", 0)) + float(d.get("swap", 0)) + float(d.get("commission", 0)) + float(d.get("fee", 0))
                            for d in deals
                        )
                        out_deals = [d for d in deals if str(d.get("entry", "")).upper() in ("1", "OUT", "ENTRY_OUT")]
                        close_price = float(out_deals[-1]["price"]) if out_deals and "price" in out_deals[-1] else float(deals[-1].get("price", base.get("price_current", 0.0)))
                        base["close_price"] = close_price
                        base["profit"] = round(total_pnl, 2)
                        base["pnl"] = round(total_pnl, 2)
                        base["outcome"] = "WIN" if total_pnl > 0 else "LOSS"
                        return base
            except Exception as exc:
                logger.error(f"Remote get_closed_trade_details exception for ticket {ticket}: {exc}")

        # 3. Native Local MT5 (Windows)
        if MT5_AVAILABLE and not (settings.mt5_remote_ip and not self._remote_active):
            try:
                deals = mt5.history_deals_get(position=ticket)
                if deals:
                    total_pnl = sum(d.profit + d.swap + d.commission + getattr(d, "fee", 0.0) for d in deals)
                    out_deals = [d for d in deals if d.entry == getattr(mt5, "DEAL_ENTRY_OUT", 1)]
                    close_price = out_deals[-1].price if out_deals else deals[-1].price
                    base["close_price"] = close_price
                    base["profit"] = round(total_pnl, 2)
                    base["pnl"] = round(total_pnl, 2)
                    base["outcome"] = "WIN" if total_pnl > 0 else "LOSS"
                    return base
            except Exception as exc:
                logger.error(f"Local MT5 history_deals_get error for ticket {ticket}: {exc}")

        # Fallback: calculate using current tick if available
        if base.get("symbol"):
            tick = self.get_tick(base["symbol"])
            if tick:
                close_p = tick.bid if base.get("type") == "BUY" else tick.ask
                contract_size = 100.0 if "XAU" in base["symbol"] or "GOLD" in base["symbol"] else 100000.0
                vol = float(base.get("volume", 0.01))
                open_p = float(base.get("price_open", close_p))
                if base.get("type") == "BUY":
                    pnl = round((close_p - open_p) * contract_size * vol, 2)
                else:
                    pnl = round((open_p - close_p) * contract_size * vol, 2)
                base["close_price"] = close_p
                base["profit"] = pnl
                base["pnl"] = pnl
                base["outcome"] = "WIN" if pnl > 0 else "LOSS"
                return base

        base["pnl"] = base.get("profit", 0.0)
        base["outcome"] = "WIN" if base["pnl"] > 0 else "LOSS"
        return base


# ── Singleton ─────────────────────────────────────────────────────────────────
mt5_feed = MT5Feed()
