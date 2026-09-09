"""
OmniRoute API Bridge Server — General External Use
Routes all Claude Desktop / Anthropic API calls directly to OmniRoute (kr/claude-sonnet-4.5).
"""
import time
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="OmniRoute API Bridge")

OMNIROUTE_BASE_URL = "http://34.93.80.53:20128/v1"
OMNIROUTE_API_KEY = "sk-b56ecc128d7cca90-e880a8-a1f43d23"
OMNIROUTE_MODEL = "kr/claude-sonnet-4.5"


@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": "OmniRoute AI Gateway",
        "model": OMNIROUTE_MODEL,
        "endpoint": OMNIROUTE_BASE_URL,
    }


@app.post("/v1/messages")
async def handle_messages(request: Request):
    try:
        body = await request.json()
        anth_messages = body.get("messages", [])
        system_prompt = body.get("system", "")

        openai_messages = []
        if system_prompt:
            if isinstance(system_prompt, list):
                system_text = "\n".join(
                    s.get("text", "") if isinstance(s, dict) else str(s)
                    for s in system_prompt
                )
            else:
                system_text = str(system_prompt)
            openai_messages.append({"role": "system", "content": system_text})

        for msg in anth_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                ]
                content_str = "\n".join(text_parts)
            else:
                content_str = str(content)
            openai_messages.append({"role": role, "content": content_str})

        payload = {
            "model": OMNIROUTE_MODEL,
            "messages": openai_messages,
            "temperature": body.get("temperature", 0.7),
            "max_tokens": body.get("max_tokens", 2048),
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {OMNIROUTE_API_KEY}",
            "Content-Type": "application/json",
        }

        print("\n" + "=" * 70)
        print("🎯 [PROOF OF OMNIROUTE USAGE] Request Intercepted!")
        print(f"🔑 API Key:       {OMNIROUTE_API_KEY[:10]}...{OMNIROUTE_API_KEY[-8:]}")
        print(f"🌐 Target Server:  {OMNIROUTE_BASE_URL}/chat/completions")
        print(f"🤖 Target Model:   {OMNIROUTE_MODEL}")
        print(f"💬 Prompt Length:  {len(str(openai_messages))} chars")
        print("-" * 70)

        start_t = time.time()
        resp = requests.post(
            f"{OMNIROUTE_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed_sec = time.time() - start_t

        res_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        comp_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + comp_tokens)

        print(f"✅ [SUCCESS] OmniRoute Responded in {elapsed_sec:.2f}s!")
        print(f"📊 Tokens Used:    Input: {prompt_tokens} | Output: {comp_tokens} | Total: {total_tokens}")
        print("=" * 70 + "\n")

        anthropic_response = {
            "id": f"msg_{int(time.time()*1000)}",
            "type": "message",
            "role": "assistant",
            "model": body.get("model", "claude-3-5-sonnet-20241022"),
            "content": [{"type": "text", "text": res_text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": comp_tokens,
            },
        }

        return JSONResponse(content=anthropic_response)

    except Exception as exc:
        print(f"❌ [BRIDGE ERROR] {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {"type": "api_error", "message": str(exc)},
            },
        )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=4000)
