"""
PAXIS AGENT — Full Real-Time LLM Benchmark Suite (Unlimited Timeout)
Compares qwen2.5:3b, qwen2.5:14b, deepseek-r1:32b, and qwen2.5vl:32b.
Measures exact real-time execution speed (seconds), token counts, and accuracy.
"""
import json
import time
import httpx

MODELS_TO_TEST = ["qwen2.5:3b", "qwen2.5:14b", "deepseek-r1:32b", "qwen2.5vl:32b"]

SCENARIOS = [
    {
        "name": "Scenario A: Strong Bullish SMC Alignment (High Prob BUY)",
        "prompt": """[INST] System Role: You are PAXIS Pro Trader AI, an elite SMC Gold (XAUUSD) Scalper.
Evaluate this setup and return JSON ONLY:

Symbol: XAUUSD
Current Price: 4350.50
Spread: 1.2 pips
Session: London Open
4H Bias: Bullish (BOS at 4320.00, active Bullish OB 4310-4325)
1H Structure: Bullish (Higher Highs & Higher Lows)
15M Setup POI: Price retesting 15M Bullish Order Block (4348.00-4352.00) + Bullish FVG
1M Micro Trigger: Bullish CHoCH breakout above 4351.20, RSI(14)=32 (Oversold), StochRSI=OVERSOLD
Reward:Risk: 2.8 (SL: 4344.00, TP: 4368.70)

Return valid JSON:
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0 - 1.0,
  "entry": float,
  "sl": float,
  "tp": float,
  "trade_thesis": "string detailed SMC reasoning"
}
[/INST]"""
    },
    {
        "name": "Scenario B: Conflict / High Risk Trap (High Prob HOLD)",
        "prompt": """[INST] System Role: You are PAXIS Pro Trader AI, an elite SMC Gold (XAUUSD) Scalper.
Evaluate this setup and return JSON ONLY:

Symbol: XAUUSD
Current Price: 4350.50
Spread: 4.8 pips (HIGH SPREAD)
Session: Asian Session (Low Volatility)
4H Bias: Bearish (Downtrend)
1H Structure: Ranging / Indecisive
15M Setup POI: Counter-trend bullish bounce
1M Micro Trigger: No CHoCH, RSI(14)=52 (Neutral)
News Event: High Impact US CPI Release in 8 minutes!
Reward:Risk: 1.1 (Low R:R)

Return valid JSON:
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0 - 1.0,
  "entry": float,
  "sl": float,
  "tp": float,
  "trade_thesis": "string detailed SMC reasoning"
}
[/INST]"""
    },
    {
        "name": "Scenario C: Bearish Liquidity Sweep & Reversal (High Prob SELL)",
        "prompt": """[INST] System Role: You are PAXIS Pro Trader AI, an elite SMC Gold (XAUUSD) Scalper.
Evaluate this setup and return JSON ONLY:

Symbol: XAUUSD
Current Price: 4385.00
Spread: 1.0 pips
Session: NY Session
4H Bias: Bearish
1H Structure: Bearish CHoCH confirmed at 4400.00
15M Setup POI: Liquidity Sweep of Asian Highs (4388.50) into 15M Bearish OB (4384-4390) + FVG
1M Micro Trigger: Bearish CHoCH break below 4383.50, RSI(14)=76 (Overbought), StochRSI=OVERBOUGHT
Reward:Risk: 3.2 (SL: 4391.00, TP: 4365.80)

Return valid JSON:
{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0 - 1.0,
  "entry": float,
  "sl": float,
  "tp": float,
  "trade_thesis": "string detailed SMC reasoning"
}
[/INST]"""
    }
]

def run_benchmark():
    url = "http://localhost:11434/api/chat"
    results = {}

    for model in MODELS_TO_TEST:
        print(f"\n==========================================")
        print(f" Testing Model: {model}")
        print(f"==========================================")
        results[model] = []

        for sc in SCENARIOS:
            print(f"\n[START] {model} on {sc['name']}...")
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": sc["prompt"]}],
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1024,
                }
            }

            t0 = time.time()
            try:
                # 600 second (10 minute) timeout to capture full 32B generation times accurately
                resp = httpx.post(url, json=payload, timeout=600)
                elapsed = time.time() - t0
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                tokens = data.get("eval_count", "?")

                # Parse JSON
                parsed = None
                try:
                    # Strip <think> if reasoning model
                    text = content.strip()
                    if "<think>" in text:
                        if "</think>" in text:
                            text = text.split("</think>")[-1].strip()
                    parsed = json.loads(text)
                except Exception:
                    pass

                action = parsed.get("action") if parsed else "FAILED_PARSE"
                conf = parsed.get("confidence") if parsed else 0.0
                thesis = parsed.get("trade_thesis") or parsed.get("reasoning") if parsed else content[:100]

                results[model].append({
                    "scenario": sc["name"],
                    "elapsed_s": round(elapsed, 2),
                    "elapsed_min": round(elapsed / 60, 2),
                    "tokens": tokens,
                    "valid_json": parsed is not None,
                    "action": action,
                    "confidence": conf,
                    "thesis": str(thesis)[:150]
                })

                print(f"✓ {model} | Time: {elapsed:.1f}s ({elapsed/60:.2f}m) | Action: {action} | Conf: {conf} | Tokens: {tokens}")
                print(f"  Thesis: {str(thesis)[:120]}...")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"❌ {model} FAILED after {elapsed:.1f}s: {e}")
                results[model].append({
                    "scenario": sc["name"],
                    "elapsed_s": round(elapsed, 2),
                    "elapsed_min": round(elapsed / 60, 2),
                    "tokens": 0,
                    "valid_json": False,
                    "action": "ERROR",
                    "confidence": 0.0,
                    "thesis": str(e)
                })

    print("\n\n" + "="*70)
    print(" SUMMARY BENCHMARK REPORT (REAL-TIME TIMINGS) ")
    print("="*70)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_benchmark()
