"""OpenAI-compatible chat-completion client."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


Message = Mapping[str, str]


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "local"
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    timeout: float = 120.0
    max_retries: int = 4
    retry_base_seconds: float = 1.5
    json_mode: bool = False

    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip("/")
        if self.provider == "openai":
            return "https://api.openai.com/v1"
        return "http://localhost:8000/v1"

    def resolved_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            value = os.environ.get(self.api_key_env)
            if value:
                return value
        if self.provider == "openai":
            value = os.environ.get("OPENAI_API_KEY")
            if value:
                return value
            raise RuntimeError("OPENAI_API_KEY is required for provider='openai'.")
        return os.environ.get("VLLM_API_KEY", "EMPTY")


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float,
        max_tokens: int | None = None,
        response_format: Mapping[str, str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format

        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.config.resolved_base_url() + "/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.config.resolved_api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return _extract_message_content(data)
            except urllib.error.HTTPError as exc:
                last_error = RuntimeError(_format_http_error(exc))
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
            if attempt < self.config.max_retries:
                time.sleep(self.config.retry_base_seconds * (2**attempt))
        raise RuntimeError(f"LLM request failed after retries: {last_error}") from last_error


def _format_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    return f"HTTP {exc.code} from LLM API: {body[:1000]}"


def _extract_message_content(data: Mapping[str, Any]) -> str:
    try:
        choices = data["choices"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("response has no choices")
        message = choices[0].get("message", {})
        content = message.get("content")
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(f"unexpected chat completion response shape: {data}") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        if parts:
            return "\n".join(parts)
    raise ValueError(f"chat completion content is not text: {content!r}")
