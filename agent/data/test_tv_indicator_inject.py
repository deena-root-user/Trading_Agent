import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        user_data_dir = "logs/screenshots/tv_browser_profile"
        context = await p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=True,
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=2.0,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
        )
        page = context.pages[0] if context.pages else await context.new_page()
        
        print("Navigating to TradingView eTq2RTXP...")
        await page.goto("https://www.tradingview.com/chart/eTq2RTXP/?symbol=OANDA:XAUUSD", wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(3000)
        
        # Click Indicators toolbar button
        print("Clicking Indicators toolbar button...")
        try:
            ind_btn = page.locator('#header-toolbar-indicators, [data-name="open-indicators-dialog"], button[aria-label*="Indicators"]').first
            if await ind_btn.is_visible(timeout=3000):
                await ind_btn.click()
                print("Clicked Indicators button!")
                await page.wait_for_timeout(1500)
            else:
                print("Indicators button not visible, pressing 'Insert' key...")
                await page.keyboard.press("Insert")
                await page.wait_for_timeout(1500)
        except Exception as e:
            print(f"Error clicking indicators: {e}")

        # Check if search dialog opened
        dialog = page.locator('[data-name="indicators-dialog"], [aria-label*="Indicators"], [data-role="dialog"]')
        if await dialog.is_visible(timeout=3000):
            print("Indicators dialog is visible!")
            search_input = page.locator('input[data-role="search"], input[type="text"]').first
            await search_input.fill("SMC Core Module")
            await page.wait_for_timeout(1500)
            await page.screenshot(path="logs/screenshots/test_indicator_search.png")
            print("Saved debug screenshot to logs/screenshots/test_indicator_search.png")
        else:
            print("Dialog still not visible!")

        await page.screenshot(path="logs/screenshots/test_indicator_inject.png")
        print("Saved debug screenshot to logs/screenshots/test_indicator_inject.png")
        await context.close()

if __name__ == "__main__":
    asyncio.run(run())
