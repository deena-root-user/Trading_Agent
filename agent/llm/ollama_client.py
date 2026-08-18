"""
PAXIS Agent — Ollama Client
Communicates with Ollama REST API to call the Plutus LLM.
Supports vision (image) + text prompts with timeout handling.
"""
from __future__ import annotations

import json
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
        """
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

        payload = {
            "model": active_model,
            "messages": messages,
            "stream": False,
            "keep_alive": "-1",
            "options": {
                "temperature": temperature if temperature is not None else settings.ollama_temperature,
                "top_p": top_p if top_p is not None else settings.ollama_top_p,
                "num_predict": getattr(settings, "max_num_predict_tokens", 512),
                "num_ctx": getattr(settings, "num_ctx_tokens", 8192),
                "repeat_penalty": 1.1,
            },
            "format": "json",
        }

        request_timeout = float(self.vision_timeout) if has_images else float(self.timeout)
        start = time.time()
        
        try:
            with httpx.Client(timeout=request_timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

            elapsed = time.time() - start
            raw_content = data.get("message", {}).get("content", "")

            if has_images:
                self._consecutive_vision_failures = 0

            self._log_raw(raw_content, elapsed)
            logger.info(
                f"Ollama response received | model={active_model} | "
                f"elapsed={elapsed:.1f}s | tokens={data.get('eval_count', '?')}"
            )
            return raw_content

        except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            elapsed = time.time() - start
            
            # If request had vision/image data and timed out or errored, attempt text-only fallback
            if has_images:
                self._consecutive_vision_failures += 1
                fallback_model = self._resolve_fallback_model()
                logger.warning(
                    f"Ollama vision request failed/timed out after {elapsed:.1f}s ({exc}) — "
                    f"⚡ RETRYING immediately with FAST TEXT-ONLY INDICATOR FALLBACK using model '{fallback_model}'... "
                    f"(consecutive vision failures: {self._consecutive_vision_failures})"
                )
                text_messages = []
                for msg in messages:
                    msg_copy = dict(msg)
                    msg_copy.pop("images", None)
                    text_messages.append(msg_copy)

                fallback_payload = dict(payload)
                fallback_payload["model"] = fallback_model
                fallback_payload["messages"] = text_messages
                fallback_payload["keep_alive"] = "30m"
                fallback_payload["options"] = {
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "num_predict": 256,
                    "num_ctx": 2048,
                }

                try:
                    fb_start = time.time()
                    fb_timeout = float(self.timeout)
                    with httpx.Client(timeout=fb_timeout) as client:
                        fb_resp = client.post(url, json=fallback_payload)
                        fb_resp.raise_for_status()
                        fb_data = fb_resp.json()

                    fb_elapsed = time.time() - fb_start
                    fb_content = fb_data.get("message", {}).get("content", "")
                    self._log_raw(fb_content, fb_elapsed)
                    
                    logger.info(
                        f"✅ Fast text-only fallback successful | model={fallback_model} | "
                        f"elapsed={fb_elapsed:.1f}s | tokens={fb_data.get('eval_count', '?')}"
                    )
                    return fb_content
                except Exception as fb_exc:
                    logger.error(f"Text-only fallback also failed: {fb_exc} — forcing HOLD")
                    return None

            logger.error(f"Ollama request error after {elapsed:.1f}s ({exc}) — forcing HOLD")
            return None

        except httpx.ConnectError:
            logger.error(
                f"Cannot connect to Ollama at {self.base_url} — "
                "is Ollama running on localhost:11434? forcing HOLD"
            )
            return None

        except Exception as exc:
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
