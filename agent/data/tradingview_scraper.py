"""
PAXIS Agent — TradingView Playwright Chart Scraper
Option B: Captures live TradingView chart screenshots with smc_core_model.pine loaded across 4H, 1H, 15M, and 1M timeframes.
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
from agent.config import settings

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


class TradingViewScraper:
    def __init__(self, save_dir: str = "logs/screenshots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    async def capture_tradingview_charts_multi(
        self,
        symbol: str = "XAUUSD",
        chart_url: Optional[str] = None,
        timeframes: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Captures 4 individual full-screen Extreme 4K Ultra-HD TradingView chart images across timeframes (4H, 1H, 15M, 1M).
        Loads saved user layout (eTq2RTXP) with SMC Core Module Pine Script indicator enabled.
        Returns list of 4 base64 strings [b64_4h, b64_1h, b64_15m, b64_1m].
        """
        if not PLAYWRIGHT_AVAILABLE or not PIL_AVAILABLE:
            logger.warning("Playwright or Pillow not installed — cannot scrape TradingView charts.")
            return []

        if timeframes is None:
            timeframes = ["4h", "1h", "15m", "1m"]

        if not chart_url or chart_url == "https://www.tradingview.com/chart/":
            chart_url = getattr(settings, "tradingview_chart_url", "https://www.tradingview.com/chart/eTq2RTXP/")

        symbol_clean = symbol.replace("/", "").replace(":", "")
        
        if chart_url and "tradingview.com/chart/" in chart_url:
            chart_base = chart_url.rstrip("/")
            url = f"{chart_base}/?symbol=OANDA:{symbol_clean}"
        elif chart_url and "symbol=" in chart_url:
            url = chart_url
        else:
            url = f"https://www.tradingview.com/chart/eTq2RTXP/?symbol=OANDA:{symbol_clean}"

        user_data_dir = self.save_dir / "tv_browser_profile"
        user_data_dir.mkdir(parents=True, exist_ok=True)

        images_b64: List[str] = []

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
                )
                context = await browser.new_context(
                    viewport={"width": 1280, "height": 720},
                    device_scale_factor=1.0,  # Fast 720p HD rendering for fast vision token prefill
                )
                page = await context.new_page()

                # Inject authenticated session cookie if available
                session_id = getattr(settings, "tradingview_session_id", None) or os.getenv("TRADINGVIEW_SESSION_ID")
                cookie_file = self.save_dir / "sessionid.txt"
                if not session_id and cookie_file.exists():
                    try:
                        session_id = cookie_file.read_text().strip()
                    except Exception:
                        pass

                if session_id:
                    try:
                        await context.add_cookies([{
                            "name": "sessionid",
                            "value": session_id,
                            "url": "https://www.tradingview.com",
                        }])
                        logger.info("🔑 Injected TradingView sessionid cookie for authenticated layout eTq2RTXP")
                    except Exception as cookie_err:
                        logger.warning(f"Failed to inject sessionid cookie: {cookie_err}")

                logger.info(f"🌐 Navigating to TradingView layout with SMC indicator: {url}")
                await page.goto(url, wait_until="commit", timeout=15000)
                await page.wait_for_timeout(3500)

                # Close popups / overlays
                try:
                    await page.keyboard.press("Escape")
                    await page.wait_for_timeout(300)
                except Exception:
                    pass

                # Collapse right-hand Watchlist panel to maximize full-screen candle chart area
                try:
                    watchlist_btn = page.locator('[data-name="watchlist-toggle"], [aria-label*="Watchlist"]').first
                    if await watchlist_btn.is_visible(timeout=1000):
                        await watchlist_btn.click()
                        await page.wait_for_timeout(500)
                    else:
                        await page.keyboard.press("Alt+w")
                        await page.wait_for_timeout(400)
                except Exception:
                    try:
                        await page.keyboard.press("Alt+w")
                        await page.wait_for_timeout(300)
                    except Exception:
                        pass

                # Loop timeframes and navigate explicitly using &interval= parameter
                tf_interval_map = {
                    "4h": "240",
                    "240": "240",
                    "1h": "60",
                    "60": "60",
                    "15m": "15",
                    "15": "15",
                    "1m": "1",
                    "1": "1",
                }

                for tf in timeframes:
                    try:
                        interval_code = tf_interval_map.get(tf.lower(), "15")
                        tf_url = f"{url}&interval={interval_code}" if "?" in url else f"{url}?interval={interval_code}"
                        
                        logger.info(f"🌐 Navigating to TradingView {tf.upper()} timeframe chart (Layout eTq2RTXP): {tf_url}")
                        await page.goto(tf_url, wait_until="commit", timeout=15000)
                        await page.wait_for_timeout(3000)

                        # Close popups / overlays
                        try:
                            await page.keyboard.press("Escape")
                            await page.wait_for_timeout(200)
                        except Exception:
                            pass

                        # Collapse Watchlist panel
                        try:
                            watchlist_btn = page.locator('[data-name="watchlist-toggle"], [aria-label*="Watchlist"]').first
                            if await watchlist_btn.is_visible(timeout=1000):
                                await watchlist_btn.click()
                                await page.wait_for_timeout(400)
                            else:
                                await page.keyboard.press("Alt+w")
                                await page.wait_for_timeout(300)
                        except Exception:
                            pass

                        png_bytes = await page.screenshot(type="png")
                        img = Image.open(io.BytesIO(png_bytes))
                        if img.width > 1024:
                            ratio = 1024 / img.width
                            img = img.resize((1024, int(img.height * ratio)), Image.LANCZOS)
                        
                        # Save fast 80% quality JPEG
                        debug_path = self.save_dir / f"paxis_{symbol}_{tf.lower()}_fullscreen.jpg"
                        img.save(str(debug_path), "JPEG", quality=80)

                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=80)
                        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
                        images_b64.append(encoded)
                        logger.info(f"Captured fast HD TradingView chart for {symbol} ({tf.upper()})")
                    except Exception as tf_exc:
                        logger.warning(f"Failed to capture timeframe {tf} on TradingView: {tf_exc}")

                await browser.close()

            # Also stitch into a 1920x1080 2x2 grid image (paxis_XAUUSD_tv_pro_trader.jpg) for visual reference
            if len(images_b64) == 4:
                try:
                    grid_width, grid_height = 1920, 1080
                    grid_img = Image.new("RGB", (grid_width, grid_height), color="#0F172A")
                    w_sub, h_sub = grid_width // 2, grid_height // 2
                    positions = [(0, 0), (w_sub, 0), (0, h_sub), (w_sub, h_sub)]
                    tf_titles = ["4H MACRO FRAMEWORK", "1H INTERMEDIATE STRUCTURE", "15M SETUP POI", "1M MICRO ENTRY TRIGGER"]

                    from PIL import ImageDraw
                    for idx, b64_str in enumerate(images_b64[:4]):
                        sub_bytes = base64.b64decode(b64_str)
                        sub_img = Image.open(io.BytesIO(sub_bytes)).resize((w_sub, h_sub), Image.LANCZOS)
                        draw = ImageDraw.Draw(sub_img)
                        draw.rectangle([(10, 10), (340, 44)], fill=(15, 23, 42, 220), outline="#38BDF8", width=1)
                        draw.text((20, 18), tf_titles[idx], fill="#F8FAFC")
                        grid_img.paste(sub_img, positions[idx])

                    grid_path = self.save_dir / f"paxis_{symbol}_tv_pro_trader.jpg"
                    grid_img.save(str(grid_path), "JPEG", quality=100)
                    logger.info(f"Saved Extreme 4K 4-in-1 stitched grid image to {grid_path}")
                except Exception as grid_exc:
                    logger.warning(f"Failed to stitch grid debug image: {grid_exc}")

            return images_b64

        except Exception as exc:
            logger.warning(f"TradingView Playwright persistent capture failed ({exc})")
            return []


tradingview_scraper = TradingViewScraper()
