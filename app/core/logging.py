from __future__ import annotations

import logging
import os
from typing import Any


def _redact(value: str) -> str:
    if not value:
        return value
    if len(value) <= 8:
        return "***"
    return value[:3] + "***" + value[-3:]


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        api_key = os.getenv("OPENAI_API_KEY", "")
        serper_key = os.getenv("SERPER_API_KEY", "")
        for secret in [api_key, serper_key]:
            if secret and isinstance(record.msg, str):
                record.msg = record.msg.replace(secret, _redact(secret))
        return True


def setup_logging(level: str | int = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    root = logging.getLogger()
    root.addFilter(RedactingFilter())


def log_json(logger: logging.Logger, level: int, msg: str, payload: Any) -> None:
    logger.log(level, "%s | payload=%s", msg, payload)
