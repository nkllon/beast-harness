"""Anthropic Managed Agents (v1/sessions + SSE) and local claude-agent-sdk drivers."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from beast_harness.secrets import ChainedSecretProvider
from beast_harness.types import (
    BH_NS,
    DriverRunResult,
    EndpointDescriptor,
    ExecutionPosture,
    HarnessError,
    PostureViolationError,
    RunState,
    SessionRecord,
    VerifiedRepoRef,
)

_MANAGED_AGENTS_BETA = "managed-agents-2026-04-01"
_ANTHROPIC_VERSION = "2023-06-01"

_OBSERVE_GUARDRAIL = (
    "\n\n[OBSERVE_ONLY GUARDRAIL: Inspect the repository at the pinned commit only. "
    "Do NOT modify files, create branches, commit changes, push refs, or open pull requests.]"
)

_READ_ONLY_CLAUDE_TOOLS = frozenset(
    {
        "Read",
        "Glob",
        "Grep",
        "LS",
        "StructuredOutput",
        "TodoRead",
    }
)

_MUTATING_CLAUDE_DISALLOWED_TOOLS = [
    "Bash",
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
]


def _format_repo_context_suffix(repo_ref: VerifiedRepoRef | None) -> str:
    if repo_ref is None:
        return ""
    if repo_ref.base_sha:
        return (
            f"\n\n[PINNED REPO CONTEXT: repo={repo_ref.repo_url} "
            f"base_sha={repo_ref.base_sha} head_sha={repo_ref.commit_sha} "
            f"(diff range: {repo_ref.base_sha}..{repo_ref.commit_sha})]"
        )
    return f"\n\n[PINNED REPO CONTEXT: repo={repo_ref.repo_url} commit_sha={repo_ref.commit_sha}]"


def _map_managed_session_status(raw: str) -> RunState:
    norm = raw.strip().lower()
    if norm in {"rescheduling", "queued", "pending"}:
        return RunState.QUEUED
    if norm in {"running", "in_progress"}:
        return RunState.RUNNING
    if norm in {"idle", "completed", "finished"}:
        return RunState.FINISHED
    if norm in {"terminated", "cancelled", "canceled", "archived"}:
        return RunState.CANCELLED
    if norm in {"error", "failed"}:
        return RunState.ERROR
    return RunState.RUNNING


def _extract_turn_events_and_answer(
    events: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]], str | None]:
    last_user_idx = -1
    for idx, ev in enumerate(events):
        if str(ev.get("type") or "") == "user.message":
            last_user_idx = idx

    turn_slice = events[last_user_idx + 1 :] if last_user_idx >= 0 else events
    text_chunks: list[str] = []
    audit_events: list[dict[str, Any]] = []
    error_detail: str | None = None

    for ev in turn_slice:
        ev_type = str(ev.get("type") or "")
        if ev_type == "agent.message":
            content = ev.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        txt = str(block.get("text") or "")
                        if txt:
                            text_chunks.append(txt)
            elif isinstance(ev.get("text"), str) and ev["text"]:
                text_chunks.append(ev["text"])
        elif ev_type in {
            "agent.tool_use",
            "agent.mcp_tool_use",
            "agent.custom_tool_use",
        }:
            audit_events.append(
                {
                    "type": ev_type,
                    "id": ev.get("id"),
                    "name": ev.get("name"),
                }
            )
        elif ev_type in {"session.error", "error"}:
            err_obj = ev.get("error")
            if isinstance(err_obj, dict):
                error_detail = str(err_obj.get("message") or err_obj)
            else:
                error_detail = str(err_obj or ev.get("message") or "Managed session error")

    answer = "\n".join(text_chunks).strip() if text_chunks else None
    return answer, audit_events, error_detail


class ClaudeManagedDriver:
    """Anthropic Managed Agents cloud container driver (/v1/sessions + SSE)."""

    driver_id = "claude-managed"

    def __init__(
        self,
        *,
        secrets: ChainedSecretProvider,
        base_url: str = "https://api.anthropic.com",
        http_client: httpx.AsyncClient | None = None,
        secret_key_name: str = "ANTHROPIC_API_KEY",
        github_secret_key_name: str = "GITHUB_TOKEN",
        default_agent_id: str | None = None,
        default_environment_id: str | None = None,
    ) -> None:
        self._secrets = secrets
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client
        self._secret_key_name = secret_key_name
        self._github_secret_key_name = github_secret_key_name
        self._default_agent_id = default_agent_id
        self._default_environment_id = default_environment_id

    def _resolve_default_agent_id(self) -> str | None:
        val = self._default_agent_id or os.environ.get("ANTHROPIC_MANAGED_AGENT_ID")
        return val.strip() if val and val.strip() else None

    def _resolve_default_environment_id(self) -> str | None:
        val = self._default_environment_id or os.environ.get("ANTHROPIC_MANAGED_ENVIRONMENT_ID")
        return val.strip() if val and val.strip() else None

    def is_available(self) -> bool:
        return (
            self._secrets.has_secret(self._secret_key_name)
            and self._resolve_default_agent_id() is not None
            and self._resolve_default_environment_id() is not None
        )

    def announce_capabilities_ttl(self) -> list[str]:
        extra_repos_schema = json.dumps(
            {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "minLength": 1},
                        "ref": {"type": "string", "minLength": 1},
                        "mount_path": {"type": "string", "minLength": 1},
                    },
                    "required": ["url", "ref"],
                },
            }
        )
        str_schema = json.dumps({"type": "string", "minLength": 1})
        budget_schema = json.dumps({"type": "number", "minimum": 0.01})
        vault_schema = json.dumps({"type": "array", "items": {"type": "string", "minLength": 1}})
        ttl = f"""\
@prefix bh: <{BH_NS}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<urn:beast:endpoint:claude-managed> a bh:AgentEndpoint ;
    rdfs:label "Anthropic Managed Agents Cloud Session (v1/sessions + SSE)" ;
    bh:endpointId "claude-managed" ;
    bh:driverId "claude-managed" ;
    bh:costTier "high" ;
    bh:latencyClass "cloud-container" ;
    bh:trustTier bh:RuntimeSelfAnnounced ;
    bh:defaultPosture bh:ObserveOnly ;
    bh:supportsPosture bh:ObserveOnly, bh:ProposeDiff, bh:IsolatedBranchWrite ;
    bh:hasCapability
        bh:RemoteGitShaMount,
        bh:MultiRepoMount,
        bh:IsolatedContainerExec,
        bh:DurableCloudRun,
        bh:MidRunSteering,
        bh:FollowUpRunQueue,
        bh:RepoFreePrompt,
        bh:VerifiableToolAuditStream ;
    bh:hasModelProfile [
        a bh:ModelProfile ;
        bh:modelFamily "anthropic" ;
        bh:modelIdentifier "claude-sonnet-4-6" ;
        bh:contextWindowTokens 200000
    ] ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability bh:DurableCloudRun ;
        bh:facetKey "agent_id" ;
        bh:inputJsonSchema '{str_schema}'
    ] ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability bh:IsolatedContainerExec ;
        bh:facetKey "environment_id" ;
        bh:inputJsonSchema '{str_schema}'
    ] ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability bh:MultiRepoMount ;
        bh:facetKey "extra_repos" ;
        bh:inputJsonSchema '{extra_repos_schema}'
    ] ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability bh:DurableCloudRun ;
        bh:facetKey "max_budget_usd" ;
        bh:inputJsonSchema '{budget_schema}'
    ] ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability bh:DurableCloudRun ;
        bh:facetKey "vault_ids" ;
        bh:inputJsonSchema '{vault_schema}'
    ] .
"""
        return [ttl]

    def _auth_headers(self) -> dict[str, str]:
        api_key = self._secrets.require_secret(self._secret_key_name)
        return {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "anthropic-beta": _MANAGED_AGENTS_BETA,
            "Accept": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> httpx.Response:
        headers = self._auth_headers()
        try:
            if self._http_client is not None:
                return await self._http_client.request(
                    method,
                    path,
                    headers=headers,
                    json=json_body,
                    timeout=timeout,
                )
            async with httpx.AsyncClient(base_url=self._base_url, timeout=timeout) as client:
                return await client.request(method, path, headers=headers, json=json_body)
        except httpx.HTTPError as exc:
            raise HarnessError(
                f"Anthropic Managed Agents HTTP error on {method} {path}: {exc}"
            ) from exc

    async def _fetch_session_events(self, managed_session_id: str) -> list[dict[str, Any]]:
        resp = await self._request(
            "GET",
            f"/v1/sessions/{managed_session_id}/events",
        )
        if resp.status_code >= 400:
            raise HarnessError(
                f"Anthropic Managed Agents GET /v1/sessions/{managed_session_id}/events "
                f"failed with HTTP {resp.status_code}: {resp.text}"
            )
        payload = resp.json()
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return list(payload["data"])
        if isinstance(payload, list):
            return list(payload)
        return []

    async def start_run(
        self,
        *,
        endpoint: EndpointDescriptor,
        prompt: str,
        repo_ref: VerifiedRepoRef | None,
        posture: ExecutionPosture,
        facets: dict[str, Any],
        independent_review: bool,
    ) -> DriverRunResult:
        if facets.get("vault_ids") and posture == ExecutionPosture.OBSERVE_ONLY:
            raise PostureViolationError(
                "Facet 'vault_ids' cannot be attached under ObserveOnly execution posture."
            )

        agent_id = str(facets.get("agent_id") or self._resolve_default_agent_id() or "").strip()
        env_id = str(
            facets.get("environment_id") or self._resolve_default_environment_id() or ""
        ).strip()
        if not agent_id or not env_id:
            raise HarnessError(
                "ClaudeManagedDriver requires agent_id and environment_id "
                "(set via facets or ANTHROPIC_MANAGED_AGENT_ID / ANTHROPIC_MANAGED_ENVIRONMENT_ID)."
            )

        effective_prompt = f"{prompt}{_format_repo_context_suffix(repo_ref)}"
        if posture == ExecutionPosture.OBSERVE_ONLY or independent_review:
            effective_prompt = f"{effective_prompt}{_OBSERVE_GUARDRAIL}"

        resources: list[dict[str, Any]] = []
        gh_token: str | None = None
        if (repo_ref is not None or facets.get("extra_repos")) and self._secrets.has_secret(
            self._github_secret_key_name
        ):
            gh_token = self._secrets.get_secret(self._github_secret_key_name)

        if repo_ref is not None:
            primary_res: dict[str, Any] = {
                "type": "github_repository",
                "url": repo_ref.repo_url,
                "checkout": {"type": "commit", "sha": repo_ref.commit_sha},
            }
            if gh_token:
                primary_res["authorization_token"] = gh_token
            resources.append(primary_res)

        for extra in facets.get("extra_repos", []):
            extra_res: dict[str, Any] = {
                "type": "github_repository",
                "url": str(extra["url"]),
                "checkout": {"type": "commit", "sha": str(extra["ref"])},
            }
            if extra.get("mount_path"):
                extra_res["mount_path"] = str(extra["mount_path"])
            if gh_token:
                extra_res["authorization_token"] = gh_token
            resources.append(extra_res)

        body: dict[str, Any] = {
            "agent": agent_id,
            "environment_id": env_id,
            "resources": resources,
            "initial_events": [
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": effective_prompt}],
                }
            ],
            "metadata": {
                "beast_harness_posture": posture.value,
                "independent_review": "true" if independent_review else "false",
            },
        }
        if "max_budget_usd" in facets:
            body["budget"] = {"max_usd": float(facets["max_budget_usd"])}
        if "vault_ids" in facets:
            body["vault_ids"] = [str(v) for v in facets["vault_ids"]]

        resp = await self._request("POST", "/v1/sessions", json_body=body)
        if resp.status_code >= 400:
            raise HarnessError(
                f"Anthropic Managed Agents POST /v1/sessions failed with HTTP "
                f"{resp.status_code}: {resp.text}"
            )

        data = resp.json()
        raw_sid = data.get("id")
        if not raw_sid or not str(raw_sid).strip():
            raise HarnessError("Anthropic Managed Agents response did not include session id.")
        managed_sid = str(raw_sid).strip()
        run_id = f"{managed_sid}-turn-1"
        mapped_state = _map_managed_session_status(str(data.get("status") or "running"))

        answer: str | None = None
        audit_events: list[dict[str, Any]] = []
        error_detail: str | None = None
        if mapped_state.is_terminal:
            events = await self._fetch_session_events(managed_sid)
            answer, audit_events, error_detail = _extract_turn_events_and_answer(events)
            if error_detail and mapped_state != RunState.CANCELLED:
                mapped_state = RunState.ERROR

        return DriverRunResult(
            native_agent_id=managed_sid,
            run_id=run_id,
            state=mapped_state,
            answer=answer,
            audit_events=audit_events,
            error_detail=error_detail,
        )

    async def send_followup(
        self,
        *,
        session: SessionRecord,
        prompt: str,
        facets: dict[str, Any],
    ) -> DriverRunResult:
        effective_prompt = (
            f"{prompt}{_OBSERVE_GUARDRAIL}"
            if session.posture == ExecutionPosture.OBSERVE_ONLY
            else prompt
        )
        body = {
            "events": [
                {
                    "type": "user.message",
                    "content": [{"type": "text", "text": effective_prompt}],
                }
            ]
        }
        resp = await self._request(
            "POST",
            f"/v1/sessions/{session.native_agent_id}/events",
            json_body=body,
        )
        if resp.status_code >= 400:
            raise HarnessError(
                f"Anthropic Managed Agents POST /events failed with HTTP "
                f"{resp.status_code}: {resp.text}"
            )
        run_id = f"{session.native_agent_id}-turn-{uuid.uuid4().hex[:6]}"
        return DriverRunResult(
            native_agent_id=session.native_agent_id,
            run_id=run_id,
            state=RunState.RUNNING,
        )

    async def poll_run(self, *, session: SessionRecord) -> DriverRunResult:
        resp = await self._request("GET", f"/v1/sessions/{session.native_agent_id}")
        if resp.status_code >= 400:
            raise HarnessError(
                f"Anthropic Managed Agents GET /v1/sessions/{session.native_agent_id} "
                f"failed with HTTP {resp.status_code}: {resp.text}"
            )
        data = resp.json()
        mapped_state = _map_managed_session_status(str(data.get("status") or "running"))
        events = await self._fetch_session_events(session.native_agent_id)
        answer, audit_events, error_detail = _extract_turn_events_and_answer(events)
        if error_detail and mapped_state != RunState.CANCELLED:
            mapped_state = RunState.ERROR
        return DriverRunResult(
            native_agent_id=session.native_agent_id,
            run_id=session.latest_run_id,
            state=mapped_state,
            answer=answer,
            audit_events=audit_events,
            error_detail=error_detail,
        )

    async def _try_sse_stream(
        self,
        session: SessionRecord,
        timeout_seconds: float,
    ) -> DriverRunResult | None:
        headers = self._auth_headers()
        headers["Accept"] = "text/event-stream"
        path = f"/v1/sessions/{session.native_agent_id}/events/stream"

        streamed_events: list[dict[str, Any]] = []
        final_state: RunState | None = None

        async def _consume(client: httpx.AsyncClient) -> None:
            nonlocal final_state
            async with client.stream("GET", path, headers=headers, timeout=timeout_seconds) as resp:
                if resp.status_code >= 400:
                    return
                async for line in resp.aiter_lines():
                    stripped = line.strip()
                    if not stripped.startswith("data:"):
                        continue
                    raw_payload = stripped[5:].strip()
                    if not raw_payload or raw_payload == "[DONE]":
                        continue
                    ev = json.loads(raw_payload)
                    if not isinstance(ev, dict):
                        continue
                    streamed_events.append(ev)
                    ev_type = str(ev.get("type") or "")
                    if ev_type in {"session.status_idle", "turn.completed"}:
                        final_state = RunState.FINISHED
                        break
                    if ev_type in {"session.status_terminated"}:
                        final_state = RunState.CANCELLED
                        break
                    if ev_type in {"session.error", "error"}:
                        final_state = RunState.ERROR
                        break

        try:
            if self._http_client is not None:
                await asyncio.wait_for(_consume(self._http_client), timeout=timeout_seconds)
            else:
                async with httpx.AsyncClient(base_url=self._base_url) as client:
                    await asyncio.wait_for(_consume(client), timeout=timeout_seconds)
        except Exception:
            return None

        if final_state is None:
            return None

        answer, audit_events, error_detail = _extract_turn_events_and_answer(streamed_events)
        if answer is None and final_state == RunState.FINISHED:
            full_events = await self._fetch_session_events(session.native_agent_id)
            answer, audit_events, error_detail = _extract_turn_events_and_answer(full_events)

        return DriverRunResult(
            native_agent_id=session.native_agent_id,
            run_id=session.latest_run_id,
            state=final_state,
            answer=answer,
            audit_events=audit_events,
            error_detail=error_detail,
        )

    async def wait_run(
        self,
        *,
        session: SessionRecord,
        timeout_seconds: float,
    ) -> DriverRunResult:
        deadline = time.monotonic() + max(0.1, timeout_seconds)
        streamed = await self._try_sse_stream(session, timeout_seconds=max(0.1, timeout_seconds))
        if streamed is not None and streamed.state.is_terminal:
            return streamed

        delay = 0.05
        latest: DriverRunResult | None = None
        while time.monotonic() < deadline:
            latest = await self.poll_run(session=session)
            if latest.state.is_terminal:
                return latest
            await asyncio.sleep(min(delay, max(0.01, deadline - time.monotonic())))
            delay = min(1.0, delay * 1.5)

        return latest if latest is not None else await self.poll_run(session=session)

    async def cancel_run(self, *, session: SessionRecord) -> DriverRunResult:
        await self._request(
            "POST",
            f"/v1/sessions/{session.native_agent_id}/events",
            json_body={"events": [{"type": "user.interrupt"}]},
        )
        archive_resp = await self._request(
            "POST",
            f"/v1/sessions/{session.native_agent_id}/archive",
        )
        if archive_resp.status_code >= 400 and archive_resp.status_code != 404:
            raise HarnessError(
                f"Anthropic Managed Agents archive failed with HTTP "
                f"{archive_resp.status_code}: {archive_resp.text}"
            )
        return DriverRunResult(
            native_agent_id=session.native_agent_id,
            run_id=session.latest_run_id,
            state=RunState.CANCELLED,
        )


class ClaudeSdkDriver:
    """Local claude-agent-sdk driver over verified detached read-only worktrees."""

    driver_id = "claude-sdk"

    def __init__(
        self,
        *,
        secrets: ChainedSecretProvider,
        secret_key_name: str = "ANTHROPIC_API_KEY",
        cli_binary: str = "claude",
        allow_cli_login: bool = True,
    ) -> None:
        self._secrets = secrets
        self._secret_key_name = secret_key_name
        self._cli_binary = cli_binary
        self._allow_cli_login = allow_cli_login
        self._runs: dict[str, DriverRunResult] = {}

    def is_available(self) -> bool:
        if importlib.util.find_spec("claude_agent_sdk") is None:
            return False
        if self._secrets.has_secret(self._secret_key_name):
            return True
        return self._allow_cli_login and shutil.which(self._cli_binary) is not None

    def announce_capabilities_ttl(self) -> list[str]:
        effort_schema = json.dumps(
            {
                "type": "string",
                "enum": ["low", "medium", "high", "max"],
            }
        )
        obj_schema = json.dumps({"type": "object"})
        str_schema = json.dumps({"type": "string", "minLength": 1})
        budget_schema = json.dumps({"type": "number", "minimum": 0.01})
        turns_schema = json.dumps({"type": "integer", "minimum": 1, "maximum": 100})
        ttl = f"""\
@prefix bh: <{BH_NS}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<urn:beast:endpoint:claude-sdk> a bh:AgentEndpoint ;
    rdfs:label "Claude Agent SDK Local Agent (Policy-Hooked Detached Worktree)" ;
    bh:endpointId "claude-sdk" ;
    bh:driverId "claude-sdk" ;
    bh:costTier "medium" ;
    bh:latencyClass "interactive" ;
    bh:trustTier bh:RuntimeSelfAnnounced ;
    bh:defaultPosture bh:ObserveOnly ;
    bh:supportsPosture bh:ObserveOnly, bh:ProposeDiff ;
    bh:hasCapability
        bh:LocalDetachedWorktree,
        bh:ReadOnlyPolicyHook,
        bh:AstRipgrepSearch,
        bh:StructuredOutputSchema,
        bh:ThinkingEffortControl,
        bh:FollowUpRunQueue,
        bh:RepoFreePrompt,
        bh:VerifiableToolAuditStream ;
    bh:hasModelProfile [
        a bh:ModelProfile ;
        bh:modelFamily "anthropic" ;
        bh:modelIdentifier "claude-sonnet-4-6" ;
        bh:contextWindowTokens 200000
    ] ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability bh:ThinkingEffortControl ;
        bh:facetKey "thinking_effort" ;
        bh:inputJsonSchema '{effort_schema}'
    ] ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability bh:StructuredOutputSchema ;
        bh:facetKey "structured_output_schema" ;
        bh:inputJsonSchema '{obj_schema}'
    ] ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability bh:RepoFreePrompt ;
        bh:facetKey "model" ;
        bh:inputJsonSchema '{str_schema}'
    ] ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability bh:RepoFreePrompt ;
        bh:facetKey "max_budget_usd" ;
        bh:inputJsonSchema '{budget_schema}'
    ] ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability bh:RepoFreePrompt ;
        bh:facetKey "max_turns" ;
        bh:inputJsonSchema '{turns_schema}'
    ] .
"""
        return [ttl]

    def build_agent_options(
        self,
        *,
        workspace_path: str | None,
        posture: ExecutionPosture,
        facets: dict[str, Any],
        resume_session_id: str | None = None,
        api_key: str | None = None,
    ) -> Any:
        try:
            import claude_agent_sdk as cas
        except ImportError as exc:
            raise HarnessError("claude extra (claude-agent-sdk) is not installed.") from exc

        async def _observe_only_policy(
            tool_name: str,
            _input_data: dict[str, Any],
            _context: Any,
        ) -> Any:
            if tool_name in _READ_ONLY_CLAUDE_TOOLS:
                return cas.PermissionResultAllow()
            return cas.PermissionResultDeny(
                message=(f"Tool {tool_name!r} is denied under beast-harness ObserveOnly posture."),
                interrupt=False,
            )

        async def _propose_diff_policy(
            tool_name: str,
            _input_data: dict[str, Any],
            _context: Any,
        ) -> Any:
            if tool_name == "Bash":
                return cas.PermissionResultDeny(
                    message="Arbitrary shell execution is denied under ProposeDiff posture.",
                    interrupt=False,
                )
            return cas.PermissionResultAllow()

        env_overrides: dict[str, str] = {}
        if api_key:
            env_overrides["ANTHROPIC_API_KEY"] = api_key

        if posture == ExecutionPosture.OBSERVE_ONLY:
            disallowed = list(_MUTATING_CLAUDE_DISALLOWED_TOOLS)
            can_use = _observe_only_policy
        else:
            disallowed = ["Bash"]
            can_use = _propose_diff_policy

        output_fmt: dict[str, Any] | None = None
        if isinstance(facets.get("structured_output_schema"), dict):
            output_fmt = {
                "type": "json_schema",
                "schema": facets["structured_output_schema"],
            }

        cli_resolved = shutil.which(self._cli_binary)
        opts_kwargs: dict[str, Any] = {
            "cwd": workspace_path,
            "disallowed_tools": disallowed,
            "can_use_tool": can_use,
            "env": env_overrides,
            "setting_sources": [],
        }
        if cli_resolved:
            opts_kwargs["cli_path"] = Path(cli_resolved)
        if facets.get("model"):
            opts_kwargs["model"] = str(facets["model"])
        if facets.get("thinking_effort"):
            opts_kwargs["effort"] = str(facets["thinking_effort"])
        if facets.get("max_budget_usd") is not None:
            opts_kwargs["max_budget_usd"] = float(facets["max_budget_usd"])
        if facets.get("max_turns") is not None:
            opts_kwargs["max_turns"] = int(facets["max_turns"])
        if output_fmt is not None:
            opts_kwargs["output_format"] = output_fmt
        if resume_session_id is not None:
            opts_kwargs["resume"] = resume_session_id

        return cas.ClaudeAgentOptions(**opts_kwargs)

    async def _execute_query(
        self,
        *,
        workspace_path: str | None,
        posture: ExecutionPosture,
        facets: dict[str, Any],
        prompt: str,
        resume_session_id: str | None = None,
    ) -> DriverRunResult:
        try:
            import claude_agent_sdk as cas
        except ImportError as exc:
            raise HarnessError("claude extra (claude-agent-sdk) is not installed.") from exc

        api_key: str | None = None
        if self._secrets.has_secret(self._secret_key_name):
            api_key = self._secrets.require_secret(self._secret_key_name)
        elif not (self._allow_cli_login and shutil.which(self._cli_binary)):
            raise HarnessError(
                f"ClaudeSdkDriver requires {self._secret_key_name} or an installed 'claude' CLI."
            )

        opts = self.build_agent_options(
            workspace_path=workspace_path,
            posture=posture,
            facets=facets,
            resume_session_id=resume_session_id,
            api_key=api_key,
        )

        text_parts: list[str] = []
        audit_events: list[dict[str, Any]] = []
        structured_output: dict[str, Any] | None = None
        resolved_session_id = resume_session_id or f"claude-{uuid.uuid4().hex[:12]}"
        final_result_text: str | None = None
        error_detail: str | None = None
        state = RunState.FINISHED

        try:
            async for msg in cas.query(prompt=prompt, options=opts):
                if isinstance(msg, cas.AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, cas.TextBlock) and block.text:
                            text_parts.append(block.text)
                        elif isinstance(block, cas.ToolUseBlock):
                            audit_events.append(
                                {
                                    "type": "tool_use",
                                    "id": block.id,
                                    "name": block.name,
                                }
                            )
                elif isinstance(msg, cas.ResultMessage):
                    if msg.session_id:
                        resolved_session_id = str(msg.session_id)
                    if msg.result:
                        final_result_text = str(msg.result)
                    if isinstance(msg.structured_output, dict):
                        structured_output = msg.structured_output
                    if msg.is_error:
                        state = RunState.ERROR
                        errs = getattr(msg, "errors", None)
                        error_detail = (
                            "; ".join(str(e) for e in errs)
                            if errs
                            else (final_result_text or "Claude SDK query reported is_error=True")
                        )
        except Exception as exc:
            if isinstance(exc, HarnessError):
                raise
            raise HarnessError(f"ClaudeSdkDriver query failed: {exc}") from exc

        answer = final_result_text or ("\n".join(text_parts).strip() if text_parts else None)
        if structured_output is None and isinstance(facets.get("structured_output_schema"), dict):
            candidates = [
                *([final_result_text] if final_result_text else []),
                *reversed(text_parts),
            ]
            for cand in candidates:
                try:
                    parsed = json.loads(cand.strip())
                    if isinstance(parsed, dict):
                        structured_output = parsed
                        break
                except json.JSONDecodeError:
                    continue

        run_id = f"{resolved_session_id}-run-{uuid.uuid4().hex[:6]}"
        result = DriverRunResult(
            native_agent_id=resolved_session_id,
            run_id=run_id,
            state=state,
            answer=answer,
            structured_output=structured_output,
            audit_events=audit_events,
            error_detail=error_detail,
        )
        self._runs[run_id] = result
        return result

    async def start_run(
        self,
        *,
        endpoint: EndpointDescriptor,
        prompt: str,
        repo_ref: VerifiedRepoRef | None,
        posture: ExecutionPosture,
        facets: dict[str, Any],
        independent_review: bool,
    ) -> DriverRunResult:
        ws_path = repo_ref.local_worktree_path if repo_ref is not None else None
        effective_prompt = f"{prompt}{_format_repo_context_suffix(repo_ref)}"
        if posture == ExecutionPosture.OBSERVE_ONLY or independent_review:
            effective_prompt = f"{effective_prompt}{_OBSERVE_GUARDRAIL}"
        return await self._execute_query(
            workspace_path=ws_path,
            posture=posture,
            facets=facets,
            prompt=effective_prompt,
            resume_session_id=None,
        )

    async def send_followup(
        self,
        *,
        session: SessionRecord,
        prompt: str,
        facets: dict[str, Any],
    ) -> DriverRunResult:
        merged_facets = {**session.facets, **facets}
        effective_prompt = (
            f"{prompt}{_OBSERVE_GUARDRAIL}"
            if session.posture == ExecutionPosture.OBSERVE_ONLY
            else prompt
        )
        return await self._execute_query(
            workspace_path=session.worktree_path,
            posture=session.posture,
            facets=merged_facets,
            prompt=effective_prompt,
            resume_session_id=session.native_agent_id,
        )

    async def poll_run(self, *, session: SessionRecord) -> DriverRunResult:
        if session.latest_run_id in self._runs:
            return self._runs[session.latest_run_id]
        return DriverRunResult(
            native_agent_id=session.native_agent_id,
            run_id=session.latest_run_id,
            state=session.status,
            answer=session.answer,
            structured_output=session.structured_output,
            error_detail=session.error_detail,
        )

    async def wait_run(
        self,
        *,
        session: SessionRecord,
        timeout_seconds: float,
    ) -> DriverRunResult:
        return await self.poll_run(session=session)

    async def cancel_run(self, *, session: SessionRecord) -> DriverRunResult:
        cancelled = DriverRunResult(
            native_agent_id=session.native_agent_id,
            run_id=session.latest_run_id,
            state=RunState.CANCELLED,
        )
        self._runs[session.latest_run_id] = cancelled
        return cancelled
