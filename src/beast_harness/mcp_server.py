"""MCP tool server exposing beast-harness capabilities and session lifecycle."""

from __future__ import annotations

import json
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]
except ImportError:
    from mcp.server.mcpserver import MCPServer as FastMCP  # type: ignore[assignment]

from beast_harness.substrate import HarnessSubstrate, create_default_substrate
from beast_harness.types import (
    SCHEMA_VERSION,
    DiscoveryQuery,
    ExecutionPosture,
    TrustTier,
)


def _ok(payload: dict[str, Any]) -> str:
    return json.dumps(
        {"ok": True, "schema_version": SCHEMA_VERSION, **payload},
        sort_keys=True,
        default=str,
    )


def _err(exc: Exception) -> str:
    return json.dumps(
        {
            "ok": False,
            "schema_version": SCHEMA_VERSION,
            "error_type": type(exc).__name__,
            "message": str(exc),
        },
        sort_keys=True,
    )


def build_mcp_server(substrate: HarnessSubstrate | None = None):
    active_substrate = substrate if substrate is not None else create_default_substrate()
    mcp = FastMCP("beast-harness")

    @mcp.tool()
    async def harness_discover(
        required_capabilities: list[str] | None = None,
        preferred_capabilities: list[str] | None = None,
        exclude_model_families: list[str] | None = None,
        min_trust_tier: str | None = None,
        required_posture: str = "observe_only",
        only_alive: bool = True,
    ) -> str:
        """Discover registered SDK drivers and live Beast Network peers with full facet schemas."""
        try:
            tier = (
                TrustTier.from_shorthand(min_trust_tier)
                if min_trust_tier
                else TrustTier.RUNTIME_SELF_ANNOUNCED
            )
            query = DiscoveryQuery(
                required_capabilities=tuple(required_capabilities or ()),
                preferred_capabilities=tuple(preferred_capabilities or ()),
                exclude_model_families=tuple(exclude_model_families or ()),
                min_trust_tier=tier,
                required_posture=ExecutionPosture.from_shorthand(required_posture),
                only_alive=only_alive,
            )
            eps = await active_substrate.discover(query)
            return _ok({"endpoints": [ep.model_dump(mode="json") for ep in eps]})
        except Exception as exc:
            return _err(exc)

    @mcp.tool()
    async def harness_start(
        prompt: str,
        repo_url: str | None = None,
        ref: str = "HEAD",
        base_ref: str | None = None,
        target_endpoint_id: str | None = None,
        required_capabilities: list[str] | None = None,
        preferred_capabilities: list[str] | None = None,
        exclude_model_families: list[str] | None = None,
        posture: str = "observe_only",
        facets: dict[str, Any] | None = None,
        independent_review: bool = False,
    ) -> str:
        """Start an async agent run against a verified published git ref or standalone prompt."""
        try:
            resolved_posture = ExecutionPosture.from_shorthand(posture)
            selection = DiscoveryQuery(
                required_capabilities=tuple(required_capabilities or ()),
                preferred_capabilities=tuple(preferred_capabilities or ()),
                exclude_model_families=tuple(exclude_model_families or ()),
                required_posture=resolved_posture,
            )
            session = await active_substrate.start(
                prompt=prompt,
                repo_url=repo_url,
                ref=ref,
                base_ref=base_ref,
                target_endpoint_id=target_endpoint_id,
                selection=selection,
                posture=resolved_posture,
                facets=facets,
                independent_review=independent_review,
            )
            return _ok({"session": session.model_dump(mode="json")})
        except Exception as exc:
            return _err(exc)

    @mcp.tool()
    async def harness_send(
        session_id: str,
        prompt: str,
        facets: dict[str, Any] | None = None,
    ) -> str:
        """Send a follow-up prompt to a durable non-review session."""
        try:
            session = await active_substrate.send(
                session_id=session_id,
                prompt=prompt,
                facets=facets,
            )
            return _ok({"session": session.model_dump(mode="json")})
        except Exception as exc:
            return _err(exc)

    @mcp.tool()
    async def harness_status(session_id: str) -> str:
        """Non-blocking status check for an active or completed session."""
        try:
            session = await active_substrate.status(session_id)
            return _ok({"session": session.model_dump(mode="json")})
        except Exception as exc:
            return _err(exc)

    @mcp.tool()
    async def harness_wait(session_id: str, timeout_seconds: float = 120.0) -> str:
        """Wait up to timeout_seconds for a session's active run to reach a terminal state."""
        try:
            session = await active_substrate.wait(
                session_id,
                timeout_seconds=timeout_seconds,
            )
            return _ok({"session": session.model_dump(mode="json")})
        except Exception as exc:
            return _err(exc)

    @mcp.tool()
    async def harness_result(session_id: str) -> str:
        """Retrieve the answer, structured_output, and audit_digest for a session."""
        try:
            session = await active_substrate.result(session_id)
            return _ok({"session": session.model_dump(mode="json")})
        except Exception as exc:
            return _err(exc)

    @mcp.tool()
    async def harness_cancel(session_id: str) -> str:
        """Cancel an active session run and clean up any local read-only worktree."""
        try:
            session = await active_substrate.cancel(session_id)
            return _ok({"session": session.model_dump(mode="json")})
        except Exception as exc:
            return _err(exc)

    @mcp.tool()
    async def harness_review(
        prompt: str,
        repo_url: str,
        ref: str = "HEAD",
        base_ref: str | None = None,
        target_endpoint_id: str | None = None,
        required_capabilities: list[str] | None = None,
        preferred_capabilities: list[str] | None = None,
        exclude_model_families: list[str] | None = None,
        facets: dict[str, Any] | None = None,
        timeout_seconds: float = 300.0,
    ) -> str:
        """Run a fresh ObserveOnly independent review against a verified published git ref."""
        try:
            session = await active_substrate.review(
                prompt=prompt,
                repo_url=repo_url,
                ref=ref,
                base_ref=base_ref,
                target_endpoint_id=target_endpoint_id,
                required_capabilities=tuple(required_capabilities or ()),
                preferred_capabilities=tuple(preferred_capabilities or ()),
                exclude_model_families=tuple(exclude_model_families or ()),
                facets=facets,
                timeout_seconds=timeout_seconds,
            )
            return _ok({"session": session.model_dump(mode="json")})
        except Exception as exc:
            return _err(exc)

    return mcp


def main() -> None:
    server = build_mcp_server()
    server.run()
