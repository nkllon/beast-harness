"""Milestone 1 E2E acceptance tests: Cursor Cloud ASGI, Antigravity SDK, Live Beast Redis, & MCP."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from beast_harness.beast_bridge import BeastHarnessBridgeAgent
from beast_harness.drivers.antigravity_driver import AntigravitySdkDriver
from beast_harness.drivers.beast_driver import BeastPeerDriver
from beast_harness.drivers.cursor_driver import CursorCloudDriver
from beast_harness.ledger import SessionLedger
from beast_harness.mcp_server import build_mcp_server
from beast_harness.ref_verifier import PublishedRefVerifier
from beast_harness.secrets import ChainedSecretProvider, EnvSecretProvider
from beast_harness.substrate import HarnessSubstrate
from beast_harness.types import (
    BH_NS,
    DiscoveryQuery,
    ExecutionPosture,
    PostureViolationError,
    RunState,
)


def _init_published_repo(tmp_path: Path) -> tuple[str, str]:
    remote = tmp_path / "remote.git"
    remote.mkdir()
    subprocess.run(["git", "init", "--bare", "-b", "main"], cwd=remote, check=True)
    work = tmp_path / "work"
    subprocess.run(["git", "clone", str(remote), str(work)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@energration.com"], cwd=work, check=True)
    subprocess.run(["git", "config", "user.name", "Dev"], cwd=work, check=True)
    (work / "auth.py").write_text("def check_token(tok: str) -> bool:\n    return tok == 'ok'\n")
    subprocess.run(["git", "add", "auth.py"], cwd=work, check=True)
    subprocess.run(["git", "commit", "-m", "add auth"], cwd=work, check=True)
    subprocess.run(["git", "push", "-u", "origin", "main"], cwd=work, check=True)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=work, check=True, capture_output=True, text=True
    ).stdout.strip()
    return str(remote), sha


def _build_cursor_v1_asgi_app(recorded_requests: list[dict[str, Any]]) -> Starlette:
    """Real ASGI server speaking the Cursor Cloud Agents v1 JSON + SSE protocol."""
    agents: dict[str, dict[str, Any]] = {}
    runs: dict[str, dict[str, Any]] = {}
    seq = {"n": 0}

    def _check_auth(request: Request) -> bool:
        return request.headers.get("authorization") == "Bearer crsr_test_live_key"

    async def create_agent(request: Request) -> JSONResponse:
        if not _check_auth(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        body = await request.json()
        recorded_requests.append({"endpoint": "POST /v1/agents", "body": body})
        seq["n"] += 1
        agent_id = f"bc-test-{seq['n']:04d}"
        run_id = f"run-{seq['n']:04d}-1"
        prompt_text = body["prompt"]["text"]
        runs[run_id] = {
            "runId": run_id,
            "agentId": agent_id,
            "status": "running",
            "polls": 0,
            "output": f"Cursor Cloud independent review complete for prompt: {prompt_text[:60]}",
            "steps": [{"type": "tool_call", "name": "semantic_search", "readOnly": True}],
        }
        agents[agent_id] = {
            "id": agent_id,
            "status": "running",
            "createdAt": "2026-11-12T13:00:00Z",
            "repos": body.get("repos", []),
            "latestRunId": run_id,
        }
        return JSONResponse(
            {"agent": agents[agent_id], "run": {"runId": run_id, "status": "running"}}
        )

    async def get_agent(request: Request) -> JSONResponse:
        agent_id = request.path_params["agent_id"]
        return JSONResponse(agents[agent_id])

    async def get_run(request: Request) -> JSONResponse:
        run_id = request.path_params["run_id"]
        run = runs[run_id]
        run["polls"] += 1
        if run["status"] == "running" and run["polls"] >= 1:
            run["status"] = "completed"
        return JSONResponse(run)

    async def stream_run(request: Request) -> StreamingResponse:
        run_id = request.path_params["run_id"]
        run = runs[run_id]
        run["status"] = "completed"

        async def event_gen():
            step_payload = json.dumps({"type": "step", "step": run["steps"][0]})
            done_payload = json.dumps(
                {"type": "completed", "status": "completed", "output": run["output"]}
            )
            yield f"data: {step_payload}\n\n"
            yield f"data: {done_payload}\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    async def create_followup_run(request: Request) -> JSONResponse:
        agent_id = request.path_params["agent_id"]
        body = await request.json()
        recorded_requests.append({"endpoint": f"POST /v1/agents/{agent_id}/runs", "body": body})
        seq["n"] += 1
        run_id = f"run-{seq['n']:04d}-followup"
        runs[run_id] = {
            "runId": run_id,
            "agentId": agent_id,
            "status": "completed",
            "polls": 1,
            "output": f"Follow-up answer: {body['prompt']['text']}",
            "steps": [],
        }
        agents[agent_id]["latestRunId"] = run_id
        return JSONResponse(
            {"runId": run_id, "status": "completed", "output": runs[run_id]["output"]}
        )

    async def cancel_run(request: Request) -> JSONResponse:
        run_id = request.path_params["run_id"]
        runs[run_id]["status"] = "cancelled"
        return JSONResponse({"runId": run_id, "status": "cancelled"})

    return Starlette(
        routes=[
            Route("/v1/agents", create_agent, methods=["POST"]),
            Route("/v1/agents/{agent_id}", get_agent, methods=["GET"]),
            Route("/v1/agents/{agent_id}/runs", create_followup_run, methods=["POST"]),
            Route("/v1/agents/{agent_id}/runs/{run_id}", get_run, methods=["GET"]),
            Route("/v1/agents/{agent_id}/runs/{run_id}/stream", stream_run, methods=["GET"]),
            Route("/v1/agents/{agent_id}/runs/{run_id}/cancel", cancel_run, methods=["POST"]),
        ]
    )


def _extract_mcp_json(result: Any) -> dict[str, Any]:
    """Extract JSON payload across mcp 1.x (tuple) and mcp 2.x (CallToolResult)."""
    if hasattr(result, "content"):
        content = result.content
    elif isinstance(result, tuple):
        content = result[0]
    else:
        content = result
    return json.loads(content[0].text)


@pytest.mark.asyncio
async def test_cursor_cloud_driver_and_mcp_server_e2e(tmp_path: Path):
    repo_url, sha = _init_published_repo(tmp_path)
    recorded: list[dict[str, Any]] = []
    app = _build_cursor_v1_asgi_app(recorded)
    secrets = ChainedSecretProvider([EnvSecretProvider({"CURSOR_API_KEY": "crsr_test_live_key"})])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="https://api.cursor.com",
    ) as client:
        cursor_drv = CursorCloudDriver(secrets=secrets, http_client=client)
        substrate = HarnessSubstrate(
            ledger=SessionLedger(tmp_path / "ledger.sqlite3"),
            ref_verifier=PublishedRefVerifier(tmp_path / "worktrees"),
            drivers=[cursor_drv],
        )

        # 1. Independent review enforces fresh session, exact 40-char SHA, ObserveOnly, no send()
        review_rec = await substrate.review(
            prompt="Audit auth.py for timing attacks.",
            repo_url=repo_url,
            ref="main",
            required_capabilities=["SemanticCodebaseIndex", "RemoteGitShaMount"],
            exclude_model_families=["google"],
            facets={"model": "composer-2.5"},
            timeout_seconds=10.0,
        )
        assert review_rec.status == RunState.FINISHED
        assert review_rec.commit_sha == sha
        assert review_rec.native_agent_id.startswith("bc-")
        assert "Cursor Cloud independent review complete" in (review_rec.answer or "")
        assert recorded[0]["body"]["repos"][0]["ref"] == sha
        assert recorded[0]["body"].get("autoCreatePr") is False

        with pytest.raises(PostureViolationError):
            await substrate.send(
                session_id=review_rec.session_id, prompt="Try to contaminate review"
            )

        # 2. MCP Server end-to-end tool calls (discover, start, wait, send, result, cancel)
        mcp = build_mcp_server(substrate)
        tools = {t.name for t in await mcp.list_tools()}
        assert {
            "harness_discover",
            "harness_start",
            "harness_send",
            "harness_status",
            "harness_wait",
            "harness_result",
            "harness_cancel",
            "harness_review",
        } <= tools

        disc_json = _extract_mcp_json(
            await mcp.call_tool(
                "harness_discover",
                {"required_capabilities": ["SemanticCodebaseIndex"]},
            )
        )
        assert disc_json["ok"] is True
        assert disc_json["endpoints"][0]["endpoint_id"] == "cursor-cloud"

        started = _extract_mcp_json(
            await mcp.call_tool(
                "harness_start",
                {
                    "prompt": "Explain auth.py",
                    "repo_url": repo_url,
                    "ref": "main",
                    "target_endpoint_id": "cursor-cloud",
                },
            )
        )["session"]
        sid = started["session_id"]

        waited = _extract_mcp_json(
            await mcp.call_tool("harness_wait", {"session_id": sid, "timeout_seconds": 10.0})
        )["session"]
        assert waited["status"] == "finished"

        followed = _extract_mcp_json(
            await mcp.call_tool(
                "harness_send",
                {"session_id": sid, "prompt": "Now check edge cases on empty tokens."},
            )
        )["session"]
        assert "Follow-up answer: Now check edge cases" in followed["answer"]


def test_antigravity_sdk_driver_workspace_and_policy_configuration(tmp_path: Path):
    secrets = ChainedSecretProvider([EnvSecretProvider({"GEMINI_API_KEY": "test_gemini_key"})])
    drv = AntigravitySdkDriver(secrets=secrets, state_root=tmp_path / "agy_state")
    eps = drv.announce_capabilities_ttl()
    assert len(eps) == 1

    ws = tmp_path / "readonly_wt"
    ws.mkdir()
    cfg = drv.build_local_agent_config(
        workspace_path=str(ws),
        posture=ExecutionPosture.OBSERVE_ONLY,
        facets={
            "thinking_effort": "medium",
            "structured_output_schema": {
                "title": "ReviewVerdict",
                "type": "object",
                "properties": {"summary": {"type": "string"}, "blocking": {"type": "boolean"}},
                "required": ["summary", "blocking"],
            },
        },
    )
    assert str(ws) in [str(w) for w in cfg.workspaces]
    enabled_tool_values = {t.value for t in cfg.capabilities.enabled_tools}
    assert "view_file" in enabled_tool_values
    assert "edit_file" not in enabled_tool_values
    assert "create_file" not in enabled_tool_values
    assert "run_command" not in enabled_tool_values
    assert cfg.response_schema is not None


@pytest.mark.asyncio
async def test_beast_network_live_peer_and_bridge_roundtrip(
    tmp_path: Path,
    redis_docker: tuple[str, int],
):
    """Live integration against real Redis + beast_agent.BaseAgent + BeastHarnessBridgeAgent."""
    import redis.asyncio as aioredis
    from beast_agent import BaseAgent
    from beast_mailbox_core import MailboxConfig

    redis_host, redis_port = redis_docker
    probe = aioredis.Redis(host=redis_host, port=redis_port, db=14, decode_responses=True)
    await probe.ping()
    await probe.flushdb()

    domain_ttl = f"""\
@prefix bh: <{BH_NS}> .
@prefix dom: <https://nkllon.com/ontology/domain#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

dom:NetworkAudit a bh:HarnessCapability ;
    rdfs:label "Live Beast Network Audit" .

<urn:beast:endpoint:auditor-peer-1> a bh:AgentEndpoint ;
    rdfs:label "Beast Auditor Peer" ;
    bh:endpointId "auditor-peer-1" ;
    bh:driverId "beast-peer" ;
    bh:trustTier bh:RuntimeSelfAnnounced ;
    bh:defaultPosture bh:ObserveOnly ;
    bh:supportsPosture bh:ObserveOnly ;
    bh:hasCapability dom:NetworkAudit, bh:RepoFreePrompt ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability dom:NetworkAudit ;
        bh:facetKey "audit_scope" ;
        bh:inputJsonSchema '{{"type": "string"}}'
    ] .
"""

    class LiveAuditBeastPeer(BaseAgent):
        def __init__(self, mb_cfg: MailboxConfig) -> None:
            super().__init__(
                agent_id="auditor-peer-1",
                capabilities=["network_audit"],
                mailbox_url=mb_cfg,
                capability_graph_ttl=domain_ttl,
            )
            self._task_loop: asyncio.Task[None] | None = None

        async def on_startup(self) -> None:
            self._task_loop = asyncio.create_task(self._consume_stream_inbox())

        async def on_shutdown(self) -> None:
            if self._task_loop:
                self._task_loop.cancel()

        async def _consume_stream_inbox(self) -> None:
            r = aioredis.Redis(host=redis_host, port=redis_port, db=14, decode_responses=True)
            inbox = f"beast:stream:agent:{self.agent_id}:inbox"
            last_id = "0-0"
            try:
                while True:
                    entries = await r.xread({inbox: last_id}, count=5, block=200)
                    for _stream, msgs in entries:
                        for msg_id, fields in msgs:
                            last_id = msg_id
                            env = json.loads(fields["envelope"])
                            payload = env["payload"]
                            reply_to = payload["reply_stream"]
                            reply_env = {
                                "header": {
                                    "sender": self.agent_id,
                                    "timestamp": "2026-11-12T14:00:00+00:00",
                                    "id": f"reply-{payload['run_id']}",
                                },
                                "payload": {
                                    "type": "HARNESS_TASK_RESULT",
                                    "run_id": payload["run_id"],
                                    "native_agent_id": self.agent_id,
                                    "status": "finished",
                                    "answer": (
                                        f"Audited [{payload['facets'].get('audit_scope')}]: "
                                        f"{payload['prompt']}"
                                    ),
                                },
                            }
                            await r.xadd(reply_to, {"envelope": json.dumps(reply_env)})
            except asyncio.CancelledError:
                pass
            finally:
                await r.aclose()

    mb_cfg = MailboxConfig(host=redis_host, port=redis_port, db=14)
    peer = LiveAuditBeastPeer(mb_cfg)
    await peer.startup()

    recorded: list[dict[str, Any]] = []
    app = _build_cursor_v1_asgi_app(recorded)
    secrets = ChainedSecretProvider([EnvSecretProvider({"CURSOR_API_KEY": "crsr_test_live_key"})])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="https://api.cursor.com",
    ) as client:
        cursor_drv = CursorCloudDriver(secrets=secrets, http_client=client)
        beast_drv = BeastPeerDriver(
            caller_agent_id="caller-node-1",
            redis_host=redis_host,
            redis_port=redis_port,
            redis_db=14,
        )
        substrate = HarnessSubstrate(
            ledger=SessionLedger(tmp_path / "beast_ledger.sqlite3"),
            ref_verifier=PublishedRefVerifier(tmp_path / "wt"),
            drivers=[],
            beast_peer_driver=beast_drv,
        )

        # Also start a BeastHarnessBridgeAgent exposing cursor-cloud onto Beast Net!
        from beast_harness.capability_graph import parse_and_validate_capability_ttl

        cursor_ep = parse_and_validate_capability_ttl(cursor_drv.announce_capabilities_ttl()[0])[0]
        bridge = BeastHarnessBridgeAgent(
            agent_id="bridge-cursor-cloud",
            driver=cursor_drv,
            endpoint=cursor_ep,
            mailbox_config=mb_cfg,
        )
        await bridge.startup()

        try:
            await asyncio.sleep(0.15)
            discovered = await substrate.discover(
                DiscoveryQuery(
                    required_capabilities=("https://nkllon.com/ontology/domain#NetworkAudit",)
                )
            )
            assert [d.endpoint_id for d in discovered] == ["auditor-peer-1"]

            rec = await substrate.start(
                prompt="Verify VLAN isolation",
                selection=DiscoveryQuery(
                    required_capabilities=("https://nkllon.com/ontology/domain#NetworkAudit",)
                ),
                facets={"audit_scope": "vlan-40"},
            )
            finished = await substrate.wait(rec.session_id, timeout_seconds=5.0)
            assert finished.status == RunState.FINISHED
            assert finished.answer == "Audited [vlan-40]: Verify VLAN isolation"

            # Also dispatch through Beast Net to the bridged Cursor Cloud agent!
            bridged_eps = await substrate.discover(
                DiscoveryQuery(required_capabilities=("SemanticCodebaseIndex",))
            )
            assert any(ep.endpoint_id == "bridge-cursor-cloud" for ep in bridged_eps)

            bridged_run = await substrate.start(
                prompt="Standalone architectural sanity check via Beast bridge",
                target_endpoint_id="bridge-cursor-cloud",
            )
            bridged_done = await substrate.wait(bridged_run.session_id, timeout_seconds=5.0)
            assert bridged_done.status == RunState.FINISHED
            assert "Cursor Cloud independent review complete" in (bridged_done.answer or "")
        finally:
            await bridge.shutdown()
            await peer.shutdown()
            await probe.aclose()
