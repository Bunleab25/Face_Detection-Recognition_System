#!/usr/bin/env python3
"""Main entry point to launch the Face Recognition & Detection Web Application."""

import os
import sys
import uvicorn

# Ensure repository root is on PYTHONPATH
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if __name__ == "__main__":
    from app.config import settings  # noqa: F401 - loads .env before validation
    from app.security import validate_api_key_configuration

    try:
        validate_api_key_configuration()
    except RuntimeError as error:
        print(f"❌ Error: {error}", file=sys.stderr)
        sys.exit(1)

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() in ("true", "1", "yes")

    print("=" * 60)
    print("🚀 Starting Face Detection & Recognition System...")
    print(f"🌐 Server address: http://{host}:{port}")
    print(f"📖 Swagger Docs:  http://{host}:{port}/docs")
    print("=" * 60)

    uvicorn.run("app.main:app", host=host, port=port, reload=reload)
