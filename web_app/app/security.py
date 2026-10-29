import hmac
import os
from typing import Optional

from fastapi import Header, HTTPException


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
