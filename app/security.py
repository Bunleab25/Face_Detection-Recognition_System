import hmac
import os
from typing import Optional

from fastapi import Header, HTTPException


_INSECURE_API_KEYS = {"my-secret-key-123", "your_secret_api_key_here"}


def validate_api_key_configuration() -> None:
    configured_key = os.getenv("FACE_API_KEY", "").strip()
    if not configured_key or configured_key in _INSECURE_API_KEYS:
        raise RuntimeError(
            "Set FACE_API_KEY to a unique secret before starting the application. "
            "See README.md for setup instructions."
        )


def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    configured_key = os.getenv("FACE_API_KEY")
    if not configured_key or not configured_key.strip():
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured",
        )

    if x_api_key is None or not hmac.compare_digest(
        x_api_key.encode("utf-8"), configured_key.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
