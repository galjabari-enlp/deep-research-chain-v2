from __future__ import annotations

import respx
from httpx import Response

from app.services.serper import SerperClient


@respx.mock
async def test_serper_search_parses_results() -> None:
    endpoint = "https://google.serper.dev/search"
    respx.post(endpoint).mock(
        return_value=Response(
            200,
            json={
                "organic": [
                    {
                        "title": "Example",
                        "link": "https://example.com",
                        "snippet": "Example snippet",
                    },
                    {"title": "Bad missing link"},
                ]
            },
        )
    )

    client = SerperClient(api_key="test", endpoint=endpoint)
    resp = await client.search("test query", top_n=5)
    assert len(resp.results) == 1
    assert resp.results[0].title == "Example"
    assert str(resp.results[0].url) == "https://example.com/"
