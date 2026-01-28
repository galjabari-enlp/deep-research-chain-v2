from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass

import httpx

from app.core import settings

logger = logging.getLogger(__name__)


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class FetchResult:
    url: str
    status_code: int
    text: str
    content_type: str | None = None


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")

# Common boilerplate / nags we don't want in evidence excerpts
_BOILERPLATE_RE = re.compile(
    r"(would you like to react to this message\?.*?continue\.|create an account.*?continue\.|log in to continue\.?|enable javascript.*?\.|cookies? policy.*?\.)",
    re.IGNORECASE,
)


def extract_text_from_html(html_text: str, *, max_chars: int = 12000) -> str:
    """Very simple HTML-to-text; enough for basic extraction/logging.

    Note: this is intentionally minimal (no heavy scraping dependencies).
    """
    if not html_text:
        return ""

    cleaned = _SCRIPT_STYLE_RE.sub(" ", html_text)
    cleaned = _TAG_RE.sub(" ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = _WS_RE.sub(" ", cleaned).strip()

    # Remove common boilerplate blocks
    cleaned = _BOILERPLATE_RE.sub(" ", cleaned)
    cleaned = _WS_RE.sub(" ", cleaned).strip()

    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "…"
    return cleaned


class PageFetcher:
    def __init__(self, *, timeout_s: float | None = None, max_bytes: int = 1_200_000) -> None:
        self._timeout_s = timeout_s or settings.http_timeout_s
        self._max_bytes = max_bytes

    async def fetch(self, url: str) -> FetchResult:
        headers = {
            "User-Agent": "deep-research-agent/0.1 (+snippets-and-fetch)"
        }

        async with httpx.AsyncClient(timeout=self._timeout_s, follow_redirects=True) as client:
            try:
                r = await client.get(url, headers=headers)
            except httpx.HTTPError as e:
                raise FetchError(f"Fetch failed: {e}") from e

        content_type = r.headers.get("content-type")
        raw = r.content[: self._max_bytes]
        text = ""
        try:
            text = raw.decode(r.encoding or "utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            text = raw.decode("utf-8", errors="ignore")

        return FetchResult(url=url, status_code=r.status_code, text=text, content_type=content_type)
