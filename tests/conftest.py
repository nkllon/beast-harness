"""Pytest fixtures for beast-harness integration tests (ephemeral Docker Redis)."""

from __future__ import annotations

import atexit
import subprocess
import time
from collections.abc import Iterator

import pytest


def _stop_docker_container(container_name: str) -> None:
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except Exception:
        pass


@pytest.fixture(scope="session")
def redis_docker() -> Iterator[tuple[str, int]]:
    """Start an isolated ephemeral Redis container on 127.0.0.1 with a free port."""
    container_name = "beast-harness-test-redis"
    _stop_docker_container(container_name)

    try:
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-p",
                "127.0.0.1:0:6379",
                "redis:latest",
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        atexit.register(_stop_docker_container, container_name)

        port_proc = subprocess.run(
            ["docker", "port", container_name, "6379/tcp"],
            capture_output=True,
            check=True,
            text=True,
        )
        port_line = port_proc.stdout.strip().splitlines()[0]
        port = int(port_line.rsplit(":", 1)[1])

        import redis

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                client = redis.Redis(host="127.0.0.1", port=port, db=0)
                client.ping()
                client.close()
                break
            except Exception:
                time.sleep(0.1)
        else:
            _stop_docker_container(container_name)
            pytest.fail("Ephemeral Docker Redis did not become ready within 10s")

        yield ("127.0.0.1", port)
    finally:
        _stop_docker_container(container_name)
