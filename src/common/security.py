"""Redact configured credentials from diagnostics, without logging configuration."""
from __future__ import annotations
import re
from typing import Any


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: ('[REDACTED]' if str(key).lower() in {'api_key', 'authorization', 'x-api-key', 'auth_token', 'password'} else redact(item))
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if not isinstance(value, str):
        return value
    try:
        from .config import get_config
        config = get_config()
        for role in config.provider_names():
            connection = config.provider(role)
            for secret in (connection.api_key, connection.base_url):
                if secret:
                    value = value.replace(secret, '[REDACTED]')
    except (ValueError, OSError):
        pass
    value = re.sub(r'\bsk-[A-Za-z0-9_-]{10,}\b', '[REDACTED]', value)
    value = re.sub(r'(?i)(Bearer\s+)[A-Za-z0-9_.-]{10,}', r'\1[REDACTED]', value)
    return value
