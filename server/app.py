"""
FastAPI application entry point for the DevOps/FinOps OpenEnv Environment.

This module creates an HTTP server exposing the cloud optimization
environment over HTTP endpoints.

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860

    # Or run directly:
    uv run --project . server
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
        python -m devops_finops.server.app
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
