from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from openai import AsyncOpenAI

from app.core import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key or settings.openai_api_key,
            base_url=base_url or settings.openai_base_url or None,
            timeout=timeout_s or settings.llm_timeout_s,
        )
        self._model = model or settings.openai_model
        self._temperature = settings.llm_temperature if temperature is None else temperature

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        max_retries: int = 2,
        backoff_s: float = 0.8,
    ) -> str:
        """Return assistant message content (expected JSON)."""

        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                logger.debug("LLM system prompt: %s", system)
                logger.debug("LLM user prompt: %s", user)

                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self._temperature,
                )
                text = resp.choices[0].message.content or ""
                logger.debug("LLM raw output: %s", text)
                return text
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("LLM call failed (attempt %s/%s): %s", attempt + 1, max_retries + 1, e)
                if attempt < max_retries:
                    await asyncio.sleep(backoff_s * (2**attempt))

        raise RuntimeError(f"LLM call failed after retries: {last_err}")
