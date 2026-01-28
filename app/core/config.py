from __future__ import annotations

import logging

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # OpenAI-compatible (map from OPENAI_* env vars)
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="", validation_alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")

    # Serper
    serper_api_key: str = Field(default="", validation_alias="SERPER_API_KEY")

    # Runtime
    llm_temperature: float = 0.2
    llm_timeout_s: float = 30.0
    http_timeout_s: float = 20.0

    # Search settings
    per_iteration_search_cap: int = 4
    serper_top_n: int = 5


# This reads from .env and/or process environment.
settings = Settings()

# Debugging: confirm env/.env settings resolution at import time (no secret leakage).
_key_len = len(settings.openai_api_key or "")
_key_tail = (settings.openai_api_key or "")[-4:] if _key_len >= 4 else ""
logger.info(
    "Settings loaded openai_model=%r openai_base_url=%r openai_api_key_len=%s openai_api_key_tail=%r",
    settings.openai_model,
    settings.openai_base_url or None,
    _key_len,
    _key_tail,
)
logger.info(
    "Settings loaded serper_api_key_len=%s",
    len(settings.serper_api_key or ""),
)
