"""
PAXIS Agent — Ollama Client
Communicates with Ollama REST API to call the Plutus LLM.
Supports vision (image) + text prompts with timeout handling.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from agent.config import settings


class OllamaClient:
    """Async-ready Ollama REST API client."""

    def __init__(self):
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        self.timeout = getattr(settings, "inference_timeout_seconds", 600)
        self.vision_timeout = getattr(settings, "vision_timeout_seconds", 600)
        self._consecutive_vision_failures: int = 0
        self._max_vision_failures_before_pause: int = getattr(settings, "max_vision_failures", 2)
        self._lock = threading.Lock()

    def should_skip_vision(self) -> bool:
        """Returns True if vision is disabled in settings or paused due to consecutive timeouts."""
        if not getattr(settings, "enable_vision", True):
            return True
        if getattr(settings, "disable_vision_fallback", False):
            return False
        max_fails = getattr(settings, "max_vision_failures", 2)
        if self._consecutive_vision_failures >= max_fails:
            return True
        return False

    def reset_vision_failures(self) -> None:
        """Reset consecutive vision failure counter."""
        self._consecutive_vision_failures = 0

    def _resolve_model(self) -> str:
        """Resolve available model or fallback if configured model is missing."""
        available = self.list_models()
        if not available:
            return self.model

        # Direct match
        if any(self.model in m or m in self.model for m in available):
            return self.model

        # Try to match tag or exact name
        logger.warning(
            f"Configured Ollama model '{self.model}' not found in installed models: {available}. "
            f"Falling back to available model: '{available[0]}'."
        )
        return available[0]

    def _resolve_fallback_model(self) -> str:
        """Resolve fast text-only fallback model when vision fails or times out."""
        available = self.list_models()
        if not available:
            return getattr(settings, "ollama_fallback_model", self.model) or self.model

        fb_model = getattr(settings, "ollama_fallback_model", "qwen2.5:latest")
        if any(fb_model in m or m in fb_model for m in available):
            return fb_model
        return self._resolve_model()

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Optional[str]:
        """
        Send a chat request to Ollama. Returns raw text JSON response string.
        Automatically falls back to text-only indicators if vision processing times out.
        Thread-safe: acquires self._lock to prevent concurrent GPU/stdout collisions.
        """
        with self._lock:
            url = f"{self.base_url}/api/chat"
            active_model = self._resolve_model()
            
            has_images = any("images" in m for m in messages)

            # Automatically strip vision if vision is disabled or paused due to repeated timeouts
            if has_images and self.should_skip_vision():
                if self._consecutive_vision_failures >= self._max_vision_failures_before_pause:
                    logger.warning(
                        f"Skipping vision input due to {self._consecutive_vision_failures} consecutive vision failure(s) "
                        "— using fast text-only indicator prompt."
                    )
                text_messages = []
                for msg in messages:
                    msg_copy = dict(msg)
                    msg_copy.pop("images", None)
                    text_messages.append(msg_copy)
                messages = text_messages
                has_images = False

            import sys
            payload = {
                "model": active_model,
                "messages": messages,
                "stream": True,
                "keep_alive": "24h",  # Keep model permanently loaded in GPU VRAM for 24h
                "options": {
                    "temperature": temperature if temperature is not None else settings.ollama_temperature,
                    "top_p": top_p if top_p is not None else settings.ollama_top_p,
                    "num_predict": getattr(settings, "max_num_predict_tokens", 1024),
                    "num_ctx": getattr(settings, "num_ctx_tokens", 8192),
                    "repeat_penalty": 1.1,
                },
                "format": "json",
            }

            request_timeout = float(self.vision_timeout) if has_images else float(self.timeout)
            start = time.time()
            chunks = []
            token_count = 0
            
            try:
                with httpx.Client(timeout=request_timeout) as client:
                    with client.stream("POST", url, json=payload) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if not line:
                                continue
                            try:
                                chunk = json.loads(line)
                                content = chunk.get("message", {}).get("content", "")
                                if content:
                                    chunks.append(content)
                                    token_count += 1
                                    elapsed_so_far = time.time() - start
                                    tps = token_count / max(0.1, elapsed_so_far)
                                    sys.stdout.write(
                                        f"\r\033[K🧠 [{active_model}] Reasoning... "
                                        f"⏱️ {elapsed_so_far:.1f}s | ⚡ {token_count} tokens ({tps:.1f} t/s)"
                                    )
                                    sys.stdout.flush()
                            except Exception:
                                pass

                elapsed = time.time() - start
                if token_count > 0:
                    sys.stdout.write("\r\033[K")
                    sys.stdout.flush()

                raw_content = "".join(chunks)

                if has_images:
                    self._consecutive_vision_failures = 0

                self._log_raw(raw_content, elapsed)
                logger.info(
                    f"✅ Ollama response completed | model={active_model} | "
                    f"elapsed={elapsed:.1f}s | tokens={token_count} ({(token_count/max(0.1, elapsed)):.1f} t/s)"
                )
                return raw_content

            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                elapsed = time.time() - start
                if token_count > 0:
                    sys.stdout.write("\r\033[K")
                    sys.stdout.flush()
                err_detail = ""
                if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
                    try:
                        exc.response.read()
                        err_detail = exc.response.text
                    except Exception:
                        err_detail = str(exc)
                else:
                    err_detail = str(exc)
                
                # If 400 Bad Request occurred, attempt retry without format="json" or keep_alive adjustments
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 400:
                    logger.warning(f"Ollama returned 400 Bad Request ({err_detail.strip()}) — retrying without format constraint...")
                    try:
                        retry_payload = dict(payload)
                        retry_payload.pop("format", None)
                        retry_payload["keep_alive"] = 1800  # 30 mins as integer seconds
                        with httpx.Client(timeout=request_timeout) as client:
                            resp = client.post(url, json=retry_payload)
                            resp.raise_for_status()
                            data = resp.json()
                        raw_content = data.get("message", {}).get("content", "")
                        self._log_raw(raw_content, time.time() - start)
                        logger.info(f"✅ Ollama retry without format constraint successful | model={active_model}")
                        return raw_content
                    except Exception as retry_exc:
                        logger.error(f"Ollama retry without format constraint failed: {retry_exc}")

                # Attempt fast fallback model (e.g. qwen2.5:3b) if primary model failed or timed out
                fallback_model = self._resolve_fallback_model()
                if active_model != fallback_model:
                    logger.warning(
                        f"Ollama request to '{active_model}' failed/timed out after {elapsed:.1f}s ({err_detail.strip()}) — "
                        f"⚡ RETRYING immediately with FAST FALLBACK MODEL '{fallback_model}'..."
                    )
                    text_messages = []
                    for msg in messages:
                        msg_copy = dict(msg)
                        msg_copy.pop("images", None)
                        text_messages.append(msg_copy)

                    fallback_payload = dict(payload)
                    fallback_payload.pop("format", None)
                    fallback_payload["model"] = fallback_model
                    fallback_payload["messages"] = text_messages
                    fallback_payload["keep_alive"] = 1800
                    fallback_payload["options"] = {
                        "temperature": 0.2,
                        "top_p": 0.8,
                        "num_predict": 512,
                        "num_ctx": 4096,
                    }

                    try:
                        fb_start = time.time()
                        fb_timeout = 30.0  # Fast 30s timeout for 3b fallback model
                        with httpx.Client(timeout=fb_timeout) as client:
                            fb_resp = client.post(url, json=fallback_payload)
                            fb_resp.raise_for_status()
                            fb_data = fb_resp.json()

                        fb_elapsed = time.time() - fb_start
                        fb_content = fb_data.get("message", {}).get("content", "")
                        self._log_raw(fb_content, fb_elapsed)
                        
                        logger.info(
                            f"✅ Fast fallback successful | model={fallback_model} | "
                            f"elapsed={fb_elapsed:.1f}s | tokens={fb_data.get('eval_count', '?')}"
                        )
                        return fb_content
                    except Exception as fb_exc:
                        logger.error(f"Fallback model '{fallback_model}' also failed: {fb_exc} — forcing HOLD")
                        return None

                logger.error(f"Ollama request error after {elapsed:.1f}s ({err_detail.strip()}) — forcing HOLD")
                return None

            except httpx.ConnectError:
                if token_count > 0:
                    sys.stdout.write("\r\033[K")
                    sys.stdout.flush()
                logger.error(
                    f"Cannot connect to Ollama at {self.base_url} — "
                    "is Ollama running on localhost:11434? forcing HOLD"
                )
                return None

            except Exception as exc:
                if token_count > 0:
                    sys.stdout.write("\r\033[K")
                    sys.stdout.flush()
                logger.error(f"Ollama request error: {exc}")
                return None

    def _log_raw(self, content: str, elapsed: float) -> None:
        """Append raw LLM response to debug log file."""
        try:
            import os
            os.makedirs("logs", exist_ok=True)
            with open("logs/raw_llm.log", "a", encoding="utf-8") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
                        f"model={self.model} elapsed={elapsed:.1f}s\n")
                f.write(content)
                f.write("\n")
        except Exception:
            pass

    def is_available(self) -> bool:
        """Quick health check — returns True if Ollama is reachable."""
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Return list of available model names from Ollama."""
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []


# Singleton
ollama_client = OllamaClient()
