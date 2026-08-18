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
        
        # Click Pine Editor tab at bottom toolbar
        print("Looking for Pine Editor tab...")
        try:
            pine_tab = page.locator('button:has-text("Pine Editor"), [data-name="pine_editor"], [aria-label*="Pine"]').first
            if await pine_tab.is_visible(timeout=3000):
                await pine_tab.click()
                print("Clicked Pine Editor tab!")
                await page.wait_for_timeout(2000)
                await page.screenshot(path="logs/screenshots/test_pine_editor.png")
            else:
                print("Pine Editor tab not found directly.")
        except Exception as e:
            print(f"Error opening Pine Editor: {e}")

        await context.close()

if __name__ == "__main__":
    asyncio.run(run())
