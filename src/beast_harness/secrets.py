"""Pluggable secret resolution providers without exposing secret values."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping, Sequence

from beast_harness.types import SecretProvider, SecretResolutionError


class EnvSecretProvider:
    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        self._env = dict(env) if env is not None else None

    def has_secret(self, name: str) -> bool:
        return self.get_secret(name) is not None

    def get_secret(self, name: str) -> str | None:
        source = self._env if self._env is not None else os.environ
        val = source.get(name)
        if val is None:
            return None
        stripped = val.strip()
        return stripped if stripped else None

    def __repr__(self) -> str:
        scope = f"explicit_keys={len(self._env)}" if self._env is not None else "os.environ"
        return f"EnvSecretProvider({scope})"


class CallbackSecretProvider:
    def __init__(
        self,
        resolver: Callable[[str], str | None],
        label: str = "callback",
        *,
        has_secret_fn: Callable[[str], bool] | None = None,
    ) -> None:
        self._resolver = resolver
        self._label = label
        self._has_secret_fn = has_secret_fn

    def has_secret(self, name: str) -> bool:
        if self._has_secret_fn is not None:
            return bool(self._has_secret_fn(name))
        return self.get_secret(name) is not None

    def get_secret(self, name: str) -> str | None:
        val = self._resolver(name)
        if val is None:
            return None
        stripped = val.strip()
        return stripped if stripped else None

    def __repr__(self) -> str:
        return f"CallbackSecretProvider(label={self._label!r})"


class CommandSecretProvider:
    def __init__(
        self,
        commands_by_name: Mapping[str, Sequence[str]],
        timeout: float = 10.0,
    ) -> None:
        self._commands = {k: tuple(v) for k, v in commands_by_name.items()}
        self._timeout = timeout

    def has_secret(self, name: str) -> bool:
        # Non-consuming configuration check (does not spawn CLI subprocess during discovery)
        return bool(self._commands.get(name))

    def get_secret(self, name: str) -> str | None:
        cmd = self._commands.get(name)
        if not cmd:
            return None
        try:
            res = subprocess.run(
                list(cmd),
                check=True,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise SecretResolutionError(
                f"CommandSecretProvider failed while resolving {name!r}: {type(exc).__name__}"
            ) from exc
        stripped = res.stdout.strip()
        return stripped if stripped else None

    def __repr__(self) -> str:
        return f"CommandSecretProvider(keys={sorted(self._commands.keys())!r})"


class ChainedSecretProvider:
    def __init__(self, providers: Sequence[SecretProvider]) -> None:
        self._providers = tuple(providers)

    def has_secret(self, name: str) -> bool:
        for provider in self._providers:
            try:
                if provider.has_secret(name):
                    return True
            except SecretResolutionError:
                continue
        return False

    def get_secret(self, name: str) -> str | None:
        for provider in self._providers:
            val = provider.get_secret(name)
            if val:
                return val
        return None

    def require_secret(self, name: str) -> str:
        val = self.get_secret(name)
        if not val:
            raise SecretResolutionError(
                f"Required secret {name!r} could not be resolved from any configured provider."
            )
        return val

    def __repr__(self) -> str:
        return f"ChainedSecretProvider(providers={self._providers!r})"
