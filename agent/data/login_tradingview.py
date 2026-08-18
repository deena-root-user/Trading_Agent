"""
PAXIS Agent — TradingView Login Helper
Launches headful Playwright browser to log into TradingView once and persist session cookies.
Or accepts sessionid cookie input to save directly.
"""
import sys
import asyncio
from pathlib import Path

async def main():
    profile_dir = Path("logs/screenshots/tv_browser_profile")
    profile_dir.mkdir(parents=True, exist_ok=True)
    cookie_file = Path("logs/screenshots/sessionid.txt")

    print("======================================================================")
    print("🔑 PAXIS TradingView Session Setup")
    print("======================================================================")
    print("Option 1: Paste your TradingView 'sessionid' cookie value.")
    print("Option 2: Launch browser window to log in manually.\n")

    val = input("Enter 'sessionid' cookie OR press Enter for browser login: ").strip()

    if val:
        cookie_file.write_text(val)
        print(f"✅ Saved TradingView sessionid to {cookie_file}")
        print("Your layout eTq2RTXP and SMC Core Module indicator will now load on all runs!")
        return

    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            print("\n🌐 Launching Chrome browser... Please log into TradingView in the window.")
            context = await p.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                headless=False,
                viewport={"width": 1920, "height": 1080},
                args=["--no-sandbox"]
            )
            page = context.pages[0] if context.pages else await context.new_page()
            await page.goto("https://www.tradingview.com/chart/eTq2RTXP/", wait_until="networkidle")
            
            input("\n👉 Log into your TradingView account in the browser window, then press ENTER here to save session...")
            
            cookies = await context.cookies()
            for c in cookies:
                if c["name"] == "sessionid":
                    cookie_file.write_text(c["value"])
                    print(f"✅ Extracted and saved sessionid cookie: {c['value'][:10]}...")
            
            await context.close()
            print("✅ TradingView profile session saved successfully!")
    except Exception as exc:
        print(f"⚠️ Browser launch error (running on headless server): {exc}")
        print("\n💡 TIP: On a headless server, paste your 'sessionid' cookie value directly:")
        print("1. Open Chrome DevTools (F12) on TradingView.")
        print("2. Go to Application -> Cookies -> https://www.tradingview.com.")
        print("3. Copy the 'sessionid' value and paste it into TRADINGVIEW_SESSION_ID in .env or logs/screenshots/sessionid.txt!")

if __name__ == "__main__":
    asyncio.run(main())
