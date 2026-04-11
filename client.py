"""
client.py — Python client for the DevOps/FinOps OpenEnv environment.

Provides a clean wrapper around the HTTP API so anyone can interact
with the environment in 3 lines of code:

    from client import DevOpsEnv
    env = DevOpsEnv("https://aryan-void-openenv-devops-2.hf.space")
    obs = env.reset(task_id="easy")
    result = env.step({"action_type": "noop"})

Supports both local (http://localhost:7860) and remote HuggingFace
Spaces deployments. Thread-safe, context-manager compatible.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests


class DevOpsEnv:
    """
    Client wrapper for the DevOps/FinOps Cloud Optimizer OpenEnv API.

    Args:
        base_url: Root URL of the environment server.
                  Defaults to ENV_URL env var or http://localhost:7860.
        timeout:  Request timeout in seconds (default: 30).

    Usage:
        env = DevOpsEnv("https://aryan-void-openenv-devops-2.hf.space")
        obs = env.reset(task_id="easy")

        result = env.step({"action_type": "resize",
                           "server_id": "srv-001",
                           "instance_type": "small"})
        print(result["reward"]["score"])

        state = env.get_state()
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self.base_url = (base_url or os.environ.get("ENV_URL", "http://localhost:7860")).rstrip("/")
        self.timeout = timeout
        self.session_id: Optional[str] = None
        self._http = requests.Session()
        self._http.headers.update({"Content-Type": "application/json"})

    @classmethod
    def from_env(cls, space_id: str, timeout: int = 30) -> "DevOpsEnv":
        """
        Create a client from a HuggingFace Space ID.

        Args:
            space_id: HuggingFace Space ID, e.g. "Aryan-void/openenv-devops-2".
            timeout:  Request timeout in seconds.

        Returns:
            DevOpsEnv: Client pointing to the Space.

        Usage:
            env = DevOpsEnv.from_env("Aryan-void/openenv-devops-2")
        """
        user, repo = space_id.split("/")
        url = f"https://{user}-{repo}.hf.space"
        return cls(base_url=url, timeout=timeout)

    def reset(self, task_id: str = "easy", seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset the environment and start a new episode.

        Args:
            task_id: One of "easy", "medium", "hard", or "random".
            seed:    Optional seed for deterministic random tasks.

        Returns:
            dict: Full observation including session_id, servers, workloads, etc.
        """
        body: Dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            body["seed"] = seed
        resp = self._http.post(f"{self.base_url}/reset", json=body, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data.get("session_id")
        return data

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one action in the environment.

        Args:
            action: Action dict, e.g. {"action_type": "noop"} or
                    {"action_type": "resize", "server_id": "srv-001",
                     "instance_type": "small"}.

        Returns:
            dict: {observation, reward, done, info}.

        Raises:
            RuntimeError: If no session is active (call reset() first).
        """
        if self.session_id is None:
            raise RuntimeError("No active session. Call reset() first.")
        resp = self._http.post(
            f"{self.base_url}/step",
            json=action,
            params={"session_id": self.session_id},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def get_state(self) -> Dict[str, Any]:
        """
        Get the full current state of the environment.

        Returns:
            dict: {session_id, task_id, observation, action_history, ...}.

        Raises:
            RuntimeError: If no session is active.
        """
        if self.session_id is None:
            raise RuntimeError("No active session. Call reset() first.")
        resp = self._http.get(
            f"{self.base_url}/state",
            params={"session_id": self.session_id},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def health(self) -> Dict[str, Any]:
        """Check if the environment server is healthy."""
        resp = self._http.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        """Close the HTTP session."""
        self._http.close()

    def __enter__(self) -> "DevOpsEnv":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"DevOpsEnv(base_url={self.base_url!r}, session_id={self.session_id!r})"
