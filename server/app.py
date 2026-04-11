"""
FastAPI application entry point for the DevOps/FinOps OpenEnv Environment.

This module creates an HTTP server exposing the cloud optimization
environment over HTTP endpoints, including the interactive playground UI.

Endpoints:
    - GET  /         Interactive playground UI
    - POST /reset    Reset the environment
    - POST /step     Execute an action
    - GET  /state    Get current environment state
    - GET  /health   Health check
    - GET  /dashboard Live fleet visualization

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860

    # Or run directly:
    python -m server.app
"""

import sys
import os

# Ensure the root directory is in the path so we can import env.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import app  # noqa: E402 — re-export the FastAPI app


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        python -m server.app
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
