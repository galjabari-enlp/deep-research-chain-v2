from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List

import httpx

from app.core import settings
from app.graph.models import SearchResult

logger = logging.getLogger(__name__)


class SerperError(RuntimeError):
    pass


@dataclass(frozen=True)
class SerperSearchResponse:
    results: List[SearchResult]
    raw: dict[str, Any]


class SerperClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        endpoint: str = "https://google.serper.dev/search",
        timeout_s: float | None = None,
    ) -> None:
        self._api_key = api_key or settings.serper_api_key
        self._endpoint = endpoint
        self._timeout_s = timeout_s or settings.http_timeout_s

    async def search(self, query: str, *, top_n: int | None = None) -> SerperSearchResponse:
        if not self._api_key:
            raise SerperError("SERPER_API_KEY is missing")

        payload = {"q": query}
        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            try:
                r = await client.post(self._endpoint, headers=headers, json=payload)
            except httpx.HTTPError as e:
                raise SerperError(f"Serper HTTP error: {e}") from e

        if r.status_code >= 400:
            raise SerperError(f"Serper returned {r.status_code}: {r.text}")

        data = r.json()
        organic = data.get("organic", []) or []
        n = top_n or settings.serper_top_n
        out: List[SearchResult] = []

        for item in organic[:n]:
            link = item.get("link")
            title = item.get("title") or ""
            snippet = item.get("snippet") or ""
            if not link or not title:
                continue
            try:
                out.append(SearchResult(title=title, url=link, snippet=snippet))
            except Exception as e:  # noqa: BLE001
                logger.debug("Skipping invalid serper item: %s (%s)", item, e)

        return SerperSearchResponse(results=out, raw=data)
