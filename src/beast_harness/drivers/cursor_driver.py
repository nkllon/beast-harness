"""Cursor Cloud Agents v1 and Local cursor-sdk drivers."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
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

_OBSERVE_GUARDRAIL = (
    "\n\n[OBSERVE_ONLY GUARDRAIL: Inspect the repository at the pinned commit only. "
    "Do NOT create branches, commit changes, push refs, or open pull requests.]"
)


def _map_cursor_status(raw: str) -> RunState:
    norm = raw.strip().lower()
    if norm in {"queued", "pending", "creating"}:
        return RunState.QUEUED
    if norm in {"running", "in_progress", "starting"}:
        return RunState.RUNNING
    if norm in {"busy"}:
        return RunState.BUSY
    if norm in {"completed", "finished", "succeeded", "done"}:
        return RunState.FINISHED
    if norm in {"cancelled", "canceled", "stopped"}:
        return RunState.CANCELLED
    if norm in {"failed", "error"}:
        return RunState.ERROR
    return RunState.RUNNING


class CursorCloudDriver:
    driver_id = "cursor-cloud"

    def __init__(
        self,
        *,
        secrets: ChainedSecretProvider,
        base_url: str = "https://api.cursor.com",
        http_client: httpx.AsyncClient | None = None,
        secret_key_name: str = "CURSOR_API_KEY",
    ) -> None:
        self._secrets = secrets
        self._base_url = base_url.rstrip("/")
        self._http_client = http_client
        self._secret_key_name = secret_key_name

    def announce_capabilities_ttl(self) -> list[str]:
        extra_repos_schema = json.dumps(
            {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "minLength": 1},
                        "ref": {"type": "string", "minLength": 1},
                    },
                    "required": ["url", "ref"],
                },
            }
        )
        str_schema = json.dumps({"type": "string", "minLength": 1})
        bool_schema = json.dumps({"type": "boolean"})
        ttl = f"""\
@prefix bh: <{BH_NS}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<urn:beast:endpoint:cursor-cloud> a bh:AgentEndpoint ;
    rdfs:label "Cursor Cloud Agent (v1 REST + SSE)" ;
    bh:endpointId "cursor-cloud" ;
    bh:driverId "cursor-cloud" ;
    bh:trustTier bh:RuntimeSelfAnnounced ;
    bh:defaultPosture bh:ObserveOnly ;
    bh:supportsPosture bh:ObserveOnly, bh:ProposeDiff, bh:IsolatedBranchWrite ;
    bh:hasCapability
        bh:RemoteGitShaMount,
        bh:MultiRepoMount,
        bh:SemanticCodebaseIndex,
        bh:DurableCloudRun,
        bh:FollowUpRunQueue,
        bh:RepoFreePrompt,
        bh:VerifiableToolAuditStream ;
    bh:hasModelProfile [
        a bh:ModelProfile ;
        bh:modelFamily "cursor-composer" ;
        bh:modelIdentifier "composer-2.5" ;
        bh:contextWindowTokens 200000
    ] ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability bh:DurableCloudRun ;
        bh:facetKey "model" ;
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
        bh:facetKey "environment_id" ;
        bh:inputJsonSchema '{str_schema}'
    ] ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability bh:DurableCloudRun ;
        bh:facetKey "allow_pr" ;
        bh:inputJsonSchema '{bool_schema}'
    ] .
"""
        return [ttl]

    def _auth_headers(self) -> dict[str, str]:
        token = self._secrets.require_secret(self._secret_key_name)
        return {
            "Authorization": f"Bearer {token}",
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
        if facets.get("allow_pr") and posture != ExecutionPosture.ISOLATED_BRANCH_WRITE:
            raise PostureViolationError(
                "Facet 'allow_pr=True' requires posture IsolatedBranchWrite."
            )

        effective_prompt = (
            f"{prompt}{_OBSERVE_GUARDRAIL}"
            if (posture == ExecutionPosture.OBSERVE_ONLY or independent_review)
            else prompt
        )

        repos: list[dict[str, str]] = []
        if repo_ref is not None:
            repos.append({"url": repo_ref.repo_url, "ref": repo_ref.commit_sha})
        for extra in facets.get("extra_repos", []):
            repos.append({"url": str(extra["url"]), "ref": str(extra["ref"])})

        auto_create_pr = (
            bool(facets.get("allow_pr", False))
            if posture == ExecutionPosture.ISOLATED_BRANCH_WRITE
            else False
        )
        body: dict[str, Any] = {
            "prompt": {"text": effective_prompt},
            "repos": repos,
            "autoCreatePr": auto_create_pr,
        }
        if "model" in facets:
            body["model"] = facets["model"]
        if "environment_id" in facets:
            body["environmentId"] = facets["environment_id"]

        resp = await self._request("POST", "/v1/agents", json_body=body)
        if resp.status_code >= 400:
            raise HarnessError(
                f"Cursor Cloud POST /v1/agents failed with HTTP {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        agent_obj = data.get("agent", data)
        run_obj = data.get("run", data)
        agent_id = str(agent_obj.get("id") or data.get("id"))
        run_id = str(run_obj.get("runId") or agent_obj.get("latestRunId") or f"{agent_id}-run-1")
        raw_status = str(run_obj.get("status") or agent_obj.get("status") or "running")
        steps = list(run_obj.get("steps") or [])
        return DriverRunResult(
            native_agent_id=agent_id,
            run_id=run_id,
            state=_map_cursor_status(raw_status),
            answer=run_obj.get("output") or data.get("output"),
            audit_events=steps,
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
        body: dict[str, Any] = {"prompt": {"text": effective_prompt}}
        if "model" in facets:
            body["model"] = facets["model"]

        resp = await self._request(
            "POST",
            f"/v1/agents/{session.native_agent_id}/runs",
            json_body=body,
        )
        if resp.status_code == 409:
            return DriverRunResult(
                native_agent_id=session.native_agent_id,
                run_id=session.latest_run_id,
                state=RunState.BUSY,
                error_detail="Cursor Cloud agent already has an active run.",
            )
        if resp.status_code >= 400:
            raise HarnessError(
                f"Cursor Cloud follow-up failed with HTTP {resp.status_code}: {resp.text}"
            )
        data = resp.json()
        run_id = str(data.get("runId") or data.get("id") or session.latest_run_id)
        return DriverRunResult(
            native_agent_id=session.native_agent_id,
            run_id=run_id,
            state=_map_cursor_status(str(data.get("status") or "running")),
            answer=data.get("output"),
            audit_events=list(data.get("steps") or []),
        )

    async def poll_run(self, *, session: SessionRecord) -> DriverRunResult:
        resp = await self._request(
            "GET",
            f"/v1/agents/{session.native_agent_id}/runs/{session.latest_run_id}",
        )
        if resp.status_code >= 400:
            raise HarnessError(
                f"Cursor Cloud poll_run failed with HTTP {resp.status_code}: {resp.text}"
            )
        data = resp.json()
        return DriverRunResult(
            native_agent_id=session.native_agent_id,
            run_id=str(data.get("runId") or session.latest_run_id),
            state=_map_cursor_status(str(data.get("status") or "running")),
            answer=data.get("output"),
            audit_events=list(data.get("steps") or []),
            error_detail=data.get("error"),
        )

    async def _try_sse_stream(
        self,
        session: SessionRecord,
        timeout_seconds: float,
    ) -> DriverRunResult | None:
        headers = self._auth_headers()
        headers["Accept"] = "text/event-stream"
        path = f"/v1/agents/{session.native_agent_id}/runs/{session.latest_run_id}/stream"

        audit_events: list[dict[str, Any]] = []
        answer: str | None = None
        final_state: RunState | None = None
        error_detail: str | None = None

        async def _consume(client: httpx.AsyncClient) -> None:
            nonlocal answer, final_state, error_detail
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
                    event = json.loads(raw_payload)
                    ev_type = str(event.get("type") or "")
                    if ev_type == "step" and isinstance(event.get("step"), dict):
                        audit_events.append(event["step"])
                    if "output" in event and event["output"] is not None:
                        answer = str(event["output"])
                    if "status" in event:
                        mapped = _map_cursor_status(str(event["status"]))
                        if mapped.is_terminal:
                            final_state = mapped
                            error_detail = event.get("error")
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
        resp = await self._request(
            "POST",
            f"/v1/agents/{session.native_agent_id}/runs/{session.latest_run_id}/cancel",
        )
        if resp.status_code >= 400:
            raise HarnessError(
                f"Cursor Cloud cancel_run failed with HTTP {resp.status_code}: {resp.text}"
            )
        return DriverRunResult(
            native_agent_id=session.native_agent_id,
            run_id=session.latest_run_id,
            state=RunState.CANCELLED,
        )


class CursorLocalDriver:
    driver_id = "cursor-local"

    def __init__(
        self,
        *,
        secrets: ChainedSecretProvider,
        secret_key_name: str = "CURSOR_API_KEY",
    ) -> None:
        self._secrets = secrets
        self._secret_key_name = secret_key_name
        self._runs: dict[str, DriverRunResult] = {}

    def announce_capabilities_ttl(self) -> list[str]:
        str_schema = json.dumps({"type": "string", "minLength": 1})
        ttl = f"""\
@prefix bh: <{BH_NS}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<urn:beast:endpoint:cursor-local> a bh:AgentEndpoint ;
    rdfs:label "Cursor Local SDK Agent (Read-Only Detached Worktree)" ;
    bh:endpointId "cursor-local" ;
    bh:driverId "cursor-local" ;
    bh:trustTier bh:RuntimeSelfAnnounced ;
    bh:defaultPosture bh:ObserveOnly ;
    bh:supportsPosture bh:ObserveOnly, bh:ProposeDiff ;
    bh:hasCapability
        bh:LocalDetachedWorktree,
        bh:SemanticCodebaseIndex,
        bh:FollowUpRunQueue,
        bh:RepoFreePrompt ;
    bh:hasModelProfile [
        a bh:ModelProfile ;
        bh:modelFamily "cursor-composer" ;
        bh:modelIdentifier "composer-2.5" ;
        bh:contextWindowTokens 200000
    ] ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability bh:LocalDetachedWorktree ;
        bh:facetKey "model" ;
        bh:inputJsonSchema '{str_schema}'
    ] .
"""
        return [ttl]

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
        import cursor_sdk

        api_key = self._secrets.require_secret(self._secret_key_name)
        cwd = repo_ref.local_worktree_path if repo_ref is not None else None
        effective_prompt = (
            f"{prompt}{_OBSERVE_GUARDRAIL}"
            if (posture == ExecutionPosture.OBSERVE_ONLY or independent_review)
            else prompt
        )
        client = cursor_sdk.AsyncClient(api_key=api_key)
        try:
            agent = await cursor_sdk.AsyncAgent.create(
                client=client,
                api_key=api_key,
                model=facets.get("model"),
                local=cursor_sdk.LocalAgentOptions(cwd=cwd),
            )
            run_res = await agent.prompt(effective_prompt)
            agent_id = str(getattr(agent, "id", None) or f"local-{uuid.uuid4().hex[:10]}")
            run_id = f"{agent_id}-run-1"
            answer = str(getattr(run_res, "text", None) or run_res)
            result = DriverRunResult(
                native_agent_id=agent_id,
                run_id=run_id,
                state=RunState.FINISHED,
                answer=answer,
            )
            self._runs[run_id] = result
            return result
        finally:
            await client.close()

    async def send_followup(
        self,
        *,
        session: SessionRecord,
        prompt: str,
        facets: dict[str, Any],
    ) -> DriverRunResult:
        import cursor_sdk

        api_key = self._secrets.require_secret(self._secret_key_name)
        client = cursor_sdk.AsyncClient(api_key=api_key)
        try:
            agent = await cursor_sdk.AsyncAgent.resume(session.native_agent_id, client=client)
            run_res = await agent.prompt(prompt)
            run_id = f"{session.native_agent_id}-run-{uuid.uuid4().hex[:6]}"
            result = DriverRunResult(
                native_agent_id=session.native_agent_id,
                run_id=run_id,
                state=RunState.FINISHED,
                answer=str(getattr(run_res, "text", None) or run_res),
            )
            self._runs[run_id] = result
            return result
        finally:
            await client.close()

    async def poll_run(self, *, session: SessionRecord) -> DriverRunResult:
        if session.latest_run_id in self._runs:
            return self._runs[session.latest_run_id]
        return DriverRunResult(
            native_agent_id=session.native_agent_id,
            run_id=session.latest_run_id,
            state=session.status,
            answer=session.answer,
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
