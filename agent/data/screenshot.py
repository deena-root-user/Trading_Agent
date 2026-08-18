"""
PAXIS Agent — Chart Screenshot Engine
Captures MT5 chart window and encodes as base64 PNG for Ollama vision API.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class ChartCapture:
    def __init__(self, save_dir: str = "logs/screenshots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def capture(
        self,
        symbol: str,
        monitor_index: int = 1,
        region: Optional[dict] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> Optional[str]:
        """Capture chart screenshot → base64 PNG. Falls back to synthetic chart in headless environments."""
        import sys
        import os
        
        # If headless Linux is detected, use synthetic chart generator fallback
        if sys.platform != "win32" and "DISPLAY" not in os.environ:
            logger.debug(f"Headless environment detected (no $DISPLAY) — using high-fidelity synthetic chart fallback for {symbol}.")
            return self._generate_synthetic_chart(symbol, df)

        if not MSS_AVAILABLE:
            logger.warning(f"mss not installed — falling back to synthetic chart for {symbol}")
            return self._generate_synthetic_chart(symbol, df)
        try:
            with mss.mss() as sct:
                monitor = region or sct.monitors[monitor_index]
                shot = sct.grab(monitor)

                if PIL_AVAILABLE:
                    img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
                    if img.width > 1280:
                        ratio = 1280 / img.width
                        img = img.resize((1280, int(img.height * ratio)), Image.LANCZOS)
                    debug_path = self.save_dir / f"paxis_{symbol}_latest.png"
                    img.save(str(debug_path), "PNG")
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
                else:
                    raw_path = self.save_dir / f"paxis_{symbol}_latest.png"
                    mss.tools.to_png(shot.rgb, shot.size, output=str(raw_path))
                    with open(raw_path, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode("utf-8")

                logger.debug(f"Screenshot captured for {symbol} ({len(encoded)//1024}KB)")
                return encoded
        except Exception as exc:
            logger.warning(f"Screenshot failed for {symbol} ({exc}) — falling back to high-fidelity synthetic chart.")
            return self._generate_synthetic_chart(symbol, df)

    def _generate_synthetic_chart(self, symbol: str, df: Optional[pd.DataFrame] = None) -> Optional[str]:
        """Generates a premium, dark-themed synthetic candlestick chart using PIL and encodes it as base64."""
        if not PIL_AVAILABLE:
            logger.warning(f"Pillow (PIL) is not installed — cannot generate synthetic chart for {symbol}")
            return None

        # Fetch candles dynamically if not provided
        if df is None or df.empty:
            try:
                from agent.data.mt5_feed import mt5_feed
                df = mt5_feed.get_candles(symbol, "M5", 100)
            except Exception as exc:
                logger.error(f"Failed to fetch candles for synthetic chart: {exc}")
                return None

        if df is None or df.empty:
            logger.warning(f"No candle data available to generate synthetic chart for {symbol}")
            return None

        try:
            from PIL import ImageDraw, ImageFont
            
            # Dimensions
            width, height = 1280, 720
            img = Image.new("RGB", (width, height), color="#0F172A")  # Slate-900 Dark Theme
            draw = ImageDraw.Draw(img)

            # Margins
            left_margin = 80
            right_margin = 100
            top_margin = 80
            bottom_margin = 80

            plot_width = width - left_margin - right_margin
            plot_height = height - top_margin - bottom_margin

            # Use last 80 candles for perfect visual spacing
            df_plot = df.tail(80).copy().reset_index(drop=True)
            N = len(df_plot)
            if N == 0:
                return None

            # Compute EMAs for overlay
            # Compute EMAs for overlay (EMA 5, EMA 20, EMA 50)
            df_plot["ema5"] = df_plot["close"].ewm(span=5, adjust=False).mean()
            df_plot["ema20"] = df_plot["close"].ewm(span=20, adjust=False).mean()
            df_plot["ema50"] = df_plot["close"].ewm(span=50, adjust=False).mean()

            price_min = min(df_plot["low"].min(), df_plot["ema50"].min(), df_plot["ema20"].min(), df_plot["ema5"].min())
            price_max = max(df_plot["high"].max(), df_plot["ema50"].max(), df_plot["ema20"].max(), df_plot["ema5"].max())
            price_range = price_max - price_min
            if price_range == 0:
                price_range = 1.0

            # Dynamic padding to prevent clipping
            price_min -= price_range * 0.08
            price_max += price_range * 0.08
            price_range = price_max - price_min

            # Load modern system font if possible
            font_title = font_text = font_price = font_badge = None
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
            ]
            font_regular_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
            ]

            for path in font_paths:
                try:
                    font_title = ImageFont.truetype(path, 22)
                    font_badge = ImageFont.truetype(path, 16)
                    font_price = ImageFont.truetype(path, 15)
                    break
                except IOError:
                    continue

            for path in font_regular_paths:
                try:
                    font_text = ImageFont.truetype(path, 12)
                    break
                except IOError:
                    continue

            if font_title is None:
                font_title = font_badge = font_price = font_text = ImageFont.load_default()

            # 1. Draw Grid Lines and Price Labels
            num_grid_lines = 6
            for i in range(num_grid_lines):
                y = top_margin + int(i * (plot_height / (num_grid_lines - 1)))
                price = price_max - i * (price_range / (num_grid_lines - 1))
                # Subtly styled grid line
                draw.line([(left_margin, y), (width - right_margin, y)], fill="#1E293B", width=1)
                # Price text on the right axis
                draw.text((width - right_margin + 12, y - 7), f"{price:.2f}", fill="#64748B", font=font_text)

            # 2. Draw Candlesticks (Wicks and Bodies)
            for i in range(N):
                candle = df_plot.iloc[i]
                x = left_margin + int((i + 0.5) * (plot_width / N))
                candle_width = max(3, int(0.6 * (plot_width / N)))

                # Transform price → Y coordinates
                y_open = top_margin + plot_height - int((candle["open"] - price_min) / price_range * plot_height)
                y_close = top_margin + plot_height - int((candle["close"] - price_min) / price_range * plot_height)
                y_high = top_margin + plot_height - int((candle["low"] - price_min) / price_range * plot_height) if candle["high"] < candle["low"] else top_margin + plot_height - int((candle["high"] - price_min) / price_range * plot_height)
                y_low = top_margin + plot_height - int((candle["low"] - price_min) / price_range * plot_height)

                is_bullish = candle["close"] >= candle["open"]
                color = "#10B981" if is_bullish else "#EF4444"  # Vibrant Emerald Green / Scarlet Red

                # Wick (thin vertical line)
                draw.line([(x, y_high), (x, y_low)], fill=color, width=1)

                # Body (filled rectangle)
                y_top = min(y_open, y_close)
                y_bottom = max(y_open, y_close)
                if y_top == y_bottom:
                    y_bottom += 1  # Handle doji candles nicely

                draw.rectangle(
                    [x - candle_width // 2, y_top, x + candle_width // 2, y_bottom],
                    fill=color,
                    outline=color
                )

            # 3. Draw EMA Overlays (EMA 5, EMA 20, EMA 50)
            ema5_points = []
            ema20_points = []
            ema50_points = []
            for i in range(N):
                x = left_margin + int((i + 0.5) * (plot_width / N))
                y_ema5 = top_margin + plot_height - int((df_plot.loc[i, "ema5"] - price_min) / price_range * plot_height)
                y_ema20 = top_margin + plot_height - int((df_plot.loc[i, "ema20"] - price_min) / price_range * plot_height)
                y_ema50 = top_margin + plot_height - int((df_plot.loc[i, "ema50"] - price_min) / price_range * plot_height)
                ema5_points.append((x, y_ema5))
                ema20_points.append((x, y_ema20))
                ema50_points.append((x, y_ema50))

            if len(ema5_points) > 1:
                draw.line(ema5_points, fill="#06B6D4", width=2)   # Vibrant Cyan for EMA 5
            if len(ema20_points) > 1:
                draw.line(ema20_points, fill="#F59E0B", width=2)  # Amber Gold for EMA 20
            if len(ema50_points) > 1:
                draw.line(ema50_points, fill="#3B82F6", width=2)  # Royal Blue for EMA 50

            # 3.5 Draw Smart Money Concepts (SMC) Overlays (Order Blocks & Fair Value Gaps)
            # Scan last 15 candles for Order Blocks
            for i in range(N - 2, max(0, N - 20), -1):
                c_curr = df_plot.iloc[i]
                c_next = df_plot.iloc[i + 1]
                # Bullish OB (Bearish candle prior to strong bullish move)
                if c_curr["close"] < c_curr["open"] and c_next["close"] > c_curr["high"]:
                    ob_top_y = top_margin + plot_height - int((c_curr["high"] - price_min) / price_range * plot_height)
                    ob_bot_y = top_margin + plot_height - int((c_curr["low"] - price_min) / price_range * plot_height)
                    start_x = left_margin + int((i + 0.5) * (plot_width / N))
                    draw.rectangle([start_x, ob_top_y, width - right_margin, ob_bot_y], outline="#10B981", width=1)
                    draw.text((start_x + 8, ob_top_y + 2), "BULLISH OB", fill="#10B981", font=font_text)
                    break
                # Bearish OB (Bullish candle prior to strong bearish move)
                elif c_curr["close"] > c_curr["open"] and c_next["close"] < c_curr["low"]:
                    ob_top_y = top_margin + plot_height - int((c_curr["high"] - price_min) / price_range * plot_height)
                    ob_bot_y = top_margin + plot_height - int((c_curr["low"] - price_min) / price_range * plot_height)
                    start_x = left_margin + int((i + 0.5) * (plot_width / N))
                    draw.rectangle([start_x, ob_top_y, width - right_margin, ob_bot_y], outline="#EF4444", width=1)
                    draw.text((start_x + 8, ob_top_y + 2), "BEARISH OB", fill="#EF4444", font=font_text)
                    break

            # 4. Header metadata and Title Block
            draw.text((left_margin, 25), f"PAXIS INSTITUTIONAL AGENT (SMC + EMA 5/20)", fill="#38BDF8", font=font_text)
            draw.text((left_margin, 42), f"{symbol} — Smart Money Concepts (FVG / OB)", fill="#F8FAFC", font=font_title)

            # Current price badge with bullish/bearish color encoding
            last_candle = df_plot.iloc[-1]
            last_price = last_candle["close"]
            prev_price = df_plot.iloc[-2]["close"] if N > 1 else last_price
            change_pct = ((last_price - prev_price) / prev_price) * 100
            price_color = "#10B981" if last_price >= prev_price else "#EF4444"

            # Draw current price badge background
            draw.rectangle(
                [width - right_margin - 240, 25, width - right_margin, 65],
                fill="#1E293B",
                outline="#334155"
            )
            draw.text(
                (width - right_margin - 225, 33),
                f"{last_price:.2f} ({change_pct:+.2f}%)",
                fill=price_color,
                font=font_badge
            )

            # Draw dynamic legend block at the bottom
            draw.rectangle([left_margin, height - 55, width - right_margin, height - 25], fill="#1E293B", outline="#334155")
            draw.text((left_margin + 15, height - 47), "LEGEND:", fill="#94A3B8", font=font_text)
            
            # EMA 5 Legend (Cyan)
            draw.line([(left_margin + 75, height - 40), (left_margin + 105, height - 40)], fill="#06B6D4", width=2)
            draw.text((left_margin + 112, height - 47), "EMA(5)", fill="#F8FAFC", font=font_text)
            
            # EMA 20 Legend (Amber Gold)
            draw.line([(left_margin + 175, height - 40), (left_margin + 205, height - 40)], fill="#F59E0B", width=2)
            draw.text((left_margin + 212, height - 47), "EMA(20)", fill="#F8FAFC", font=font_text)

            # SMC Order Block Legend
            draw.text((left_margin + 280, height - 47), "⚡ SMC OB", fill="#818CF8", font=font_text)
            draw.text((left_margin + 360, height - 47), "🟢 Bullish", fill="#10B981", font=font_text)
            draw.text((left_margin + 440, height - 47), "🔴 Bearish", fill="#EF4444", font=font_text)
            draw.text((width - right_margin - 220, height - 47), "SMC EVOLUTION CHART", fill="#64748B", font=font_text)

            # Save latest PNG preview for debugging and logs
            debug_path = self.save_dir / f"paxis_{symbol}_latest.png"
            img.save(str(debug_path), "PNG")

            # Encode to Base64
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

            logger.info(f"High-fidelity synthetic chart generated for {symbol} ({len(encoded)//1024}KB)")
            return encoded

        except Exception as exc:
            logger.error(f"Failed to generate synthetic chart for {symbol}: {exc}")
            return None

    def capture_pro_trader_grid(
        self,
        symbol: str,
        df_4h: Optional[pd.DataFrame] = None,
        df_1h: Optional[pd.DataFrame] = None,
        df_15m: Optional[pd.DataFrame] = None,
        df_1m: Optional[pd.DataFrame] = None,
        use_tv_scrape: bool = True,
    ) -> Optional[str]:
        """
        Captures or renders the Pro Trader 4-Timeframe (4H, 1H, 15M, 1M) visual chart grid.
        First attempts TradingView Playwright web scraping (Option B); falls back to high-fidelity Python SMC 4-grid renderer.
        """
        if use_tv_scrape:
            try:
                import asyncio
                from agent.data.tradingview_scraper import tradingview_scraper
                logger.info(f"🌐 Attempting TradingView Playwright capture for Pro Trader {symbol}...")
                
                # Check if running inside existing event loop
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    import nest_asyncio
                    nest_asyncio.apply()
                    tv_b64 = loop.run_until_complete(tradingview_scraper.capture_tradingview_charts(symbol))
                else:
                    tv_b64 = asyncio.run(tradingview_scraper.capture_tradingview_charts(symbol))

                if tv_b64:
                    return tv_b64
            except Exception as exc:
                logger.warning(f"TradingView Playwright capture error ({exc}) — using text-based prompt fallback.")
            return None

        return self._generate_pro_trader_synthetic_grid(symbol, df_4h, df_1h, df_15m, df_1m)

    def capture_pro_trader_multi_images(
        self,
        symbol: str = "XAUUSD",
        use_tv_scrape: bool = True,
    ) -> List[str]:
        """
        Captures 4 individual full-screen (1920x1080) TradingView chart images (4H, 1H, 15M, 1M).
        Returns list of 4 base64 encoded JPEG strings.
        """
        if use_tv_scrape:
            try:
                import asyncio
                from agent.data.tradingview_scraper import tradingview_scraper
                logger.info(f"🌐 Capturing 4 full-screen (1920x1080) TradingView charts for Pro Trader {symbol}...")
                
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    import nest_asyncio
                    nest_asyncio.apply()
                    imgs = loop.run_until_complete(tradingview_scraper.capture_tradingview_charts_multi(symbol))
                else:
                    imgs = asyncio.run(tradingview_scraper.capture_tradingview_charts_multi(symbol))

                if imgs:
                    return imgs
            except Exception as exc:
                logger.warning(f"TradingView Playwright multi capture error ({exc})")
        return []

    def _generate_pro_trader_synthetic_grid(
        self,
        symbol: str,
        df_4h: Optional[pd.DataFrame],
        df_1h: Optional[pd.DataFrame],
        df_15m: Optional[pd.DataFrame],
        df_1m: Optional[pd.DataFrame],
    ) -> Optional[str]:
        """Generates a 4-panel visual grid image for 4H, 1H, 15M, and 1M with SMC overlays."""
        if not PIL_AVAILABLE:
            return None

        from PIL import ImageDraw, ImageFont
        from agent.data.smc_engine import smc_engine

        total_width, total_height = 1920, 1080
        grid_img = Image.new("RGB", (total_width, total_height), color="#0F172A")

        sub_w, sub_h = total_width // 2, total_height // 2
        panels = [
            ("4H Macro Chart", df_4h, "4H", (0, 0)),
            ("1H Intermediate Chart", df_1h, "1H", (sub_w, 0)),
            ("15M Setup Chart (POI)", df_15m, "15M", (0, sub_h)),
            ("1M Micro Entry Chart", df_1m, "1M", (sub_w, sub_h)),
        ]

        for title, df, tf, (pos_x, pos_y) in panels:
            if df is None or df.empty:
                continue

            sub_img = Image.new("RGB", (sub_w, sub_h), color="#0F172A")
            draw = ImageDraw.Draw(sub_img)

            # Analyze SMC
            smc_data = smc_engine.analyze(df, symbol, tf)
            df_plot = df.tail(60).copy().reset_index(drop=True)
            N = len(df_plot)
            if N == 0:
                continue

            left_margin, right_margin = 60, 70
            top_margin, bottom_margin = 50, 40
            pw = sub_w - left_margin - right_margin
            ph = sub_h - top_margin - bottom_margin

            p_min = df_plot["low"].min()
            p_max = df_plot["high"].max()
            pr = p_max - p_min if p_max > p_min else 1.0
            p_min -= pr * 0.05
            p_max += pr * 0.05
            pr = p_max - p_min

            # Draw Candles
            for i in range(N):
                c = df_plot.iloc[i]
                x = left_margin + int((i + 0.5) * (pw / N))
                cw = max(2, int(0.5 * (pw / N)))

                y_open = top_margin + ph - int((c["open"] - p_min) / pr * ph)
                y_close = top_margin + ph - int((c["close"] - p_min) / pr * ph)
                y_high = top_margin + ph - int((c["high"] - p_min) / pr * ph)
                y_low = top_margin + ph - int((c["low"] - p_min) / pr * ph)

                is_bull = c["close"] >= c["open"]
                col = "#10B981" if is_bull else "#EF4444"
                draw.line([(x, y_high), (x, y_low)], fill=col, width=1)
                draw.rectangle([x - cw // 2, min(y_open, y_close), x + cw // 2, max(y_open, y_close)], fill=col)

            # Draw active OBs
            for ob in smc_data.active_order_blocks():
                ob_col = "#10B981" if ob.is_bullish else "#EF4444"
                y_t = top_margin + ph - int((ob.top - p_min) / pr * ph)
                y_b = top_margin + ph - int((ob.bottom - p_min) / pr * ph)
                y_t, y_b = min(y_t, y_b), max(y_t, y_b)
                if 0 <= y_t <= sub_h and 0 <= y_b <= sub_h:
                    draw.rectangle([left_margin, y_t, sub_w - right_margin, y_b], outline=ob_col, width=1)

            # Draw Title & Badge
            draw.text((left_margin, 15), f"PAXIS PRO TRADER — {symbol} {tf} ({title})", fill="#38BDF8")
            draw.rectangle([sub_w - right_margin - 100, 10, sub_w - right_margin, 35], fill="#1E293B", outline="#334155")
            draw.text((sub_w - right_margin - 90, 15), f"TREND: {smc_data.trend}", fill="#FACC15")

            grid_img.paste(sub_img, (pos_x, pos_y))

        debug_path = self.save_dir / f"paxis_{symbol}_pro_trader_grid.png"
        grid_img.save(str(debug_path), "PNG")

        buf = io.BytesIO()
        grid_img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        logger.info(f"Generated Pro Trader synthetic 4-timeframe grid image ({len(encoded)//1024}KB)")
        return encoded


chart_capture = ChartCapture()


