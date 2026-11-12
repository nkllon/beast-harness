"""Milestone 2 E2E & integration tests: Claude Managed/SDK, Codex App-Server, and Ranking."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from beast_harness.capability_graph import (
    match_and_rank_endpoints,
    parse_and_validate_capability_ttl,
)
from beast_harness.drivers.antigravity_driver import AntigravitySdkDriver
from beast_harness.drivers.claude_driver import ClaudeManagedDriver, ClaudeSdkDriver
from beast_harness.drivers.codex_driver import CodexAppServerDriver
from beast_harness.drivers.cursor_driver import CursorCloudDriver, CursorLocalDriver
from beast_harness.ledger import SessionLedger
from beast_harness.mcp_server import build_mcp_server
from beast_harness.ref_verifier import PublishedRefVerifier
from beast_harness.secrets import ChainedSecretProvider, EnvSecretProvider
from beast_harness.substrate import HarnessSubstrate
from beast_harness.types import (
    CapabilityValidationError,
    CostTier,
    DiscoveryQuery,
    ExecutionPosture,
    LatencyClass,
    PostureViolationError,
    RankingProfile,
    RunState,
)


def _extract_mcp_json(raw_res: Any) -> dict[str, Any]:
    if hasattr(raw_res, "content"):
        blocks = raw_res.content
    elif isinstance(raw_res, tuple) and len(raw_res) >= 1:
        blocks = raw_res[0]
    else:
        blocks = raw_res
    first = blocks[0]
    text = getattr(first, "text", None) or first["text"]
    return json.loads(text)


def _init_published_repo(tmp_path: Path, name: str = "m2_repo") -> tuple[str, str]:
    remote = tmp_path / f"{name}_remote.git"
    remote.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)

    work = tmp_path / f"{name}_work"
    subprocess.run(["git", "clone", str(remote), str(work)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(work), "config", "user.email", "test@nkllon.com"],
        check=True,
    )
    subprocess.run(["git", "-C", str(work), "config", "user.name", "Test"], check=True)
    (work / "module.py").write_text("def compute_delta(x: int) -> int:\n    return x + 42\n")
    subprocess.run(["git", "-C", str(work), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(work), "commit", "-m", "initial commit"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(work), "push", "origin", "HEAD"],
        check=True,
        capture_output=True,
    )
    sha = subprocess.run(
        ["git", "-C", str(work), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return str(remote), sha


class _SingleUseSecretProvider:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = dict(mapping)
        self.get_calls: dict[str, int] = {}
        self.has_calls: dict[str, int] = {}

    def has_secret(self, name: str) -> bool:
        self.has_calls[name] = self.has_calls.get(name, 0) + 1
        return name in self._mapping

    def get_secret(self, name: str) -> str | None:
        self.get_calls[name] = self.get_calls.get(name, 0) + 1
        return self._mapping.get(name)


def test_all_builtin_drivers_validate_shacl_and_support_ranking_profiles(tmp_path: Path):
    secrets = ChainedSecretProvider([EnvSecretProvider({})])
    drivers = [
        CursorCloudDriver(secrets=secrets),
        CursorLocalDriver(secrets=secrets),
        AntigravitySdkDriver(secrets=secrets, state_root=tmp_path / "agy"),
        ClaudeManagedDriver(
            secrets=secrets,
            default_agent_id="agent_1",
            default_environment_id="env_1",
        ),
        ClaudeSdkDriver(secrets=secrets),
        CodexAppServerDriver(secrets=secrets, state_root=tmp_path / "codex"),
    ]

    all_eps = []
    for drv in drivers:
        for ttl in drv.announce_capabilities_ttl():
            all_eps.extend(parse_and_validate_capability_ttl(ttl, alive=True))

    assert {ep.endpoint_id for ep in all_eps} == {
        "antigravity-sdk",
        "claude-managed",
        "claude-sdk",
        "codex-app-server",
        "cursor-cloud",
        "cursor-local",
    }

    # Verify cross-family diversity across 4 distinct model families
    families = {m.model_family for ep in all_eps for m in ep.models}
    assert families == {"anthropic", "cursor-composer", "google", "openai"}

    # LOW_COST profile ranks cursor-local (costTier=low) first, claude-managed (high) last
    low_cost = match_and_rank_endpoints(
        all_eps,
        DiscoveryQuery(ranking_profile=RankingProfile.LOW_COST),
    )
    assert low_cost[0].endpoint_id == "cursor-local"
    assert low_cost[-1].endpoint_id == "claude-managed"

    # max_cost_tier="low" filters strictly to cursor-local
    strict_low = match_and_rank_endpoints(
        all_eps,
        DiscoveryQuery(max_cost_tier=CostTier.LOW),
    )
    assert [e.endpoint_id for e in strict_low] == ["cursor-local"]

    # max_latency_class="interactive" excludes cloud-container endpoints
    interactive_only = match_and_rank_endpoints(
        all_eps,
        DiscoveryQuery(
            ranking_profile=RankingProfile.LOW_LATENCY,
            max_latency_class=LatencyClass.INTERACTIVE,
        ),
    )
    interactive_ids = {e.endpoint_id for e in interactive_only}
    assert "cursor-cloud" not in interactive_ids
    assert "claude-managed" not in interactive_ids
    assert {"claude-sdk", "codex-app-server", "antigravity-sdk", "cursor-local"} == interactive_ids

    # SHACL rejects invalid bh:costTier or bh:latencyClass values fail-closed
    bad_cost_ttl = (
        drivers[0]
        .announce_capabilities_ttl()[0]
        .replace(
            'bh:costTier "medium" ;',
            'bh:costTier "free-tier" ;',
        )
    )
    with pytest.raises(CapabilityValidationError):
        parse_and_validate_capability_ttl(bad_cost_ttl)


@pytest.mark.asyncio
async def test_claude_managed_driver_asgi_lifecycle_sse_and_posture(tmp_path: Path):
    primary_repo, primary_sha = _init_published_repo(tmp_path, "primary")
    extra_repo, extra_sha = _init_published_repo(tmp_path, "extra")

    sessions_store: dict[str, dict[str, Any]] = {}
    events_store: dict[str, list[dict[str, Any]]] = {}
    archived_ids: set[str] = set()

    def _check_headers(request: Request) -> bool:
        return (
            request.headers.get("x-api-key") == "sk-ant-test-key"
            and request.headers.get("anthropic-version") == "2023-06-01"
            and "managed-agents-2026-04-01" in request.headers.get("anthropic-beta", "")
        )

    async def create_session(request: Request):
        if not _check_headers(request):
            return JSONResponse({"error": "invalid headers"}, status_code=401)
        body = await request.json()
        sid = f"sesn_{len(sessions_store) + 1}"
        sessions_store[sid] = {
            "id": sid,
            "type": "session",
            "status": "running",
            "body": body,
        }
        init_evs = list(body.get("initial_events") or [])
        events_store[sid] = [
            *init_evs,
            {"id": "ev_tool_1", "type": "agent.tool_use", "name": "read_file"},
            {
                "id": "ev_msg_1",
                "type": "agent.message",
                "content": [{"type": "text", "text": "Claude Managed Review: LGTM (0 blockers)."}],
            },
        ]
        return JSONResponse({"id": sid, "type": "session", "status": "running"})

    async def get_session(request: Request):
        if not _check_headers(request):
            return JSONResponse({"error": "invalid headers"}, status_code=401)
        sid = request.path_params["sid"]
        sess = sessions_store[sid]
        if sid in archived_ids:
            sess["status"] = "terminated"
        else:
            sess["status"] = "idle"
        return JSONResponse({"id": sid, "type": "session", "status": sess["status"]})

    async def handle_events(request: Request):
        if not _check_headers(request):
            return JSONResponse({"error": "invalid headers"}, status_code=401)
        sid = request.path_params["sid"]
        if request.method == "GET":
            return JSONResponse({"data": events_store.get(sid, [])})
        body = await request.json()
        incoming = list(body.get("events") or [])
        events_store.setdefault(sid, []).extend(incoming)
        for ev in incoming:
            if ev.get("type") == "user.message":
                txt = ev["content"][0]["text"].splitlines()[0]
                events_store[sid].append(
                    {
                        "id": f"ev_msg_{len(events_store[sid])}",
                        "type": "agent.message",
                        "content": [{"type": "text", "text": f"Managed Turn 2 reply to: {txt}"}],
                    }
                )
            elif ev.get("type") == "user.interrupt":
                sessions_store[sid]["status"] = "terminated"
        return JSONResponse({"data": incoming})

    async def stream_events(request: Request):
        if not _check_headers(request):
            return JSONResponse({"error": "invalid headers"}, status_code=401)
        sid = request.path_params["sid"]

        async def _gen():
            for ev in events_store.get(sid, []):
                yield f"data: {json.dumps(ev)}\n\n"
            sessions_store[sid]["status"] = "idle"
            yield f"data: {json.dumps({'type': 'session.status_idle'})}\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    async def archive_session(request: Request):
        if not _check_headers(request):
            return JSONResponse({"error": "invalid headers"}, status_code=401)
        sid = request.path_params["sid"]
        archived_ids.add(sid)
        sessions_store[sid]["status"] = "terminated"
        return JSONResponse({"id": sid, "status": "terminated"})

    app = Starlette(
        routes=[
            Route("/v1/sessions", create_session, methods=["POST"]),
            Route("/v1/sessions/{sid}", get_session, methods=["GET"]),
            Route("/v1/sessions/{sid}/events", handle_events, methods=["GET", "POST"]),
            Route("/v1/sessions/{sid}/events/stream", stream_events, methods=["GET"]),
            Route("/v1/sessions/{sid}/archive", archive_session, methods=["POST"]),
        ]
    )

    broker = _SingleUseSecretProvider(
        {"ANTHROPIC_API_KEY": "sk-ant-test-key", "GITHUB_TOKEN": "ghp-test-token"}
    )
    secrets = ChainedSecretProvider([broker])

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://claude.local",
    ) as client:
        driver = ClaudeManagedDriver(
            secrets=secrets,
            base_url="http://claude.local",
            http_client=client,
            default_agent_id="agent_default_1",
            default_environment_id="env_default_1",
        )
        substrate = HarnessSubstrate(
            ledger=SessionLedger(tmp_path / "sessions.sqlite3"),
            ref_verifier=PublishedRefVerifier(tmp_path / "worktrees"),
            drivers=[driver],
        )

        # Discovery must NOT consume get_secret
        eps = await substrate.discover()
        assert [e.endpoint_id for e in eps] == ["claude-managed"]
        assert broker.get_calls == {}

        # ObserveOnly rejects vault_ids fail-closed
        with pytest.raises(PostureViolationError):
            await substrate.review(
                prompt="Review with vaults",
                repo_url=primary_repo,
                ref="HEAD",
                facets={"vault_ids": ["vlt_123"]},
            )

        # Independent review mounts both primary & extra_repos at verified 40-char commit SHAs
        rev = await substrate.review(
            prompt="Audit compute_delta implementation",
            repo_url=primary_repo,
            ref="HEAD",
            facets={
                "extra_repos": [
                    {"url": extra_repo, "ref": "HEAD", "mount_path": "/workspace/extra"}
                ],
                "max_budget_usd": 2.5,
            },
        )
        assert rev.status == RunState.FINISHED
        assert rev.answer == "Claude Managed Review: LGTM (0 blockers)."
        assert rev.audit_digest is not None
        created_body = sessions_store["sesn_1"]["body"]
        assert created_body["resources"] == [
            {
                "type": "github_repository",
                "url": primary_repo,
                "checkout": {"type": "commit", "sha": primary_sha},
                "authorization_token": "ghp-test-token",
            },
            {
                "type": "github_repository",
                "url": extra_repo,
                "checkout": {"type": "commit", "sha": extra_sha},
                "mount_path": "/workspace/extra",
                "authorization_token": "ghp-test-token",
            },
        ]

        # Multi-turn session: Turn 2 returns Turn 2's message (no Turn 1 bleed), then cancels
        s2 = await substrate.start(
            prompt="Initial question",
            target_endpoint_id="claude-managed",
        )
        s2_done = await substrate.wait(s2.session_id, timeout_seconds=5.0)
        assert s2_done.status == RunState.FINISHED

        s2_turn2 = await substrate.send(
            session_id=s2.session_id,
            prompt="Second turn question",
        )
        assert s2_turn2.status == RunState.RUNNING
        s2_turn2_done = await substrate.wait(s2.session_id, timeout_seconds=5.0)
        assert s2_turn2_done.status == RunState.FINISHED
        assert s2_turn2_done.answer == "Managed Turn 2 reply to: Second turn question"

        cancelled = await substrate.cancel(s2.session_id)
        # Note: s2_turn2_done was already FINISHED (idle); start a running one to test live archive
        s3 = await substrate.start(prompt="Cancel me", target_endpoint_id="claude-managed")
        assert s3.status == RunState.RUNNING
        s3_cancelled = await substrate.cancel(s3.session_id)
        assert s3_cancelled.status == RunState.CANCELLED
        assert s3.native_agent_id in archived_ids
        assert cancelled.status == RunState.FINISHED


@pytest.mark.asyncio
async def test_claude_sdk_driver_policy_hook_and_zero_shadow_warnings(tmp_path: Path):
    import claude_agent_sdk as cas
    from claude_agent_sdk.types import _configure_can_use_tool

    broker = _SingleUseSecretProvider({"ANTHROPIC_API_KEY": "sk-ant-local"})
    driver = ClaudeSdkDriver(secrets=ChainedSecretProvider([broker]), allow_cli_login=False)

    assert driver.is_available() is True
    assert broker.get_calls == {}

    schema = {
        "type": "object",
        "properties": {"verdict": {"type": "string"}},
        "required": ["verdict"],
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        opts = driver.build_agent_options(
            workspace_path=str(tmp_path),
            posture=ExecutionPosture.OBSERVE_ONLY,
            facets={
                "thinking_effort": "high",
                "structured_output_schema": schema,
                "max_budget_usd": 1.0,
                "max_turns": 6,
            },
            api_key="sk-ant-local",
        )
        configured = _configure_can_use_tool(opts)

    assert len(caught) == 0, f"Unexpected SDK warnings: {[str(w.message) for w in caught]}"
    assert configured.permission_prompt_tool_name == "stdio"
    assert "Bash" in configured.disallowed_tools
    assert "Write" in configured.disallowed_tools
    assert configured.output_format == {"type": "json_schema", "schema": schema}

    # Verify deterministic ObserveOnly policy callback behavior
    assert opts.can_use_tool is not None
    allow_read = await opts.can_use_tool("Read", {"file_path": "module.py"}, None)
    allow_struct = await opts.can_use_tool("StructuredOutput", {"verdict": "PASS"}, None)
    deny_write = await opts.can_use_tool("Write", {"file_path": "module.py"}, None)
    deny_bash = await opts.can_use_tool("Bash", {"command": "rm -rf /"}, None)

    assert isinstance(allow_read, cas.PermissionResultAllow)
    assert isinstance(allow_struct, cas.PermissionResultAllow)
    assert isinstance(deny_write, cas.PermissionResultDeny)
    assert isinstance(deny_bash, cas.PermissionResultDeny)


@pytest.mark.asyncio
async def test_codex_app_server_jsonrpc_subprocess_e2e_and_mcp(tmp_path: Path):
    repo_url, head_sha = _init_published_repo(tmp_path, "codex_repo")

    # Real standalone JSON-RPC v2 stdio server subprocess speaking Codex app-server protocol
    server_script = tmp_path / "fake_codex_app_server.py"
    state_file = tmp_path / "codex_threads_state.json"
    server_script.write_text(
        """\
import json
import pathlib
import sys
import time

state_path = pathlib.Path(sys.argv[1])

def load_state():
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {"threads": {}, "steered": [], "interrupted": []}

def save_state(st):
    state_path.write_text(json.dumps(st))

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()

active_thread = None
active_turn = None
hold_open = False

for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    msg = json.loads(line)
    # Enforce Codex JSON-RPC Lite framing (no unexpected fields required, handle responses)
    if "method" not in msg:
        # Response to our server approval request
        st = load_state()
        st.setdefault("approval_decisions", []).append(msg.get("result"))
        save_state(st)
        if active_thread and active_turn and not hold_open:
            verdict_payload = json.dumps({
                "verdict": "APPROVED",
                "sandbox_verified": st["threads"][active_thread]["sandbox"],
            })
            send({
                "method": "item/completed",
                "params": {
                    "threadId": active_thread,
                    "item": {
                        "id": "item_commentary",
                        "type": "agentMessage",
                        "text": "Checking module.py...",
                    },
                },
            })
            send({
                "method": "item/completed",
                "params": {
                    "threadId": active_thread,
                    "item": {"id": "item_cmd_1", "type": "commandExecution", "status": "completed"},
                },
            })
            send({
                "method": "item/completed",
                "params": {
                    "threadId": active_thread,
                    "item": {"id": "item_msg_1", "type": "agentMessage", "text": verdict_payload},
                },
            })
            send({
                "method": "turn/completed",
                "params": {
                    "threadId": active_thread,
                    "turn": {"id": active_turn, "status": "completed"},
                },
            })
        continue

    method = msg["method"]
    req_id = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        send({"id": req_id, "result": {"serverInfo": {"name": "codex-test", "version": "0.166.0"}}})
    elif method == "initialized":
        pass
    elif method == "thread/start":
        st = load_state()
        tid = f"thr_{len(st['threads']) + 1}"
        st["threads"][tid] = {
            "cwd": params.get("cwd"),
            "sandbox": params.get("sandbox"),
            "ephemeral": params.get("ephemeral"),
            "turns": 0,
        }
        save_state(st)
        active_thread = tid
        send({
            "id": req_id,
            "result": {
                "approvalPolicy": "never",
                "approvalsReviewer": "user",
                "cwd": params.get("cwd"),
                "model": params.get("model") or "gpt-5-codex",
                "modelProvider": "openai",
                "sandbox": params.get("sandbox") or "read-only",
                "thread": {"id": tid},
            },
        })
    elif method == "thread/resume":
        tid = params["threadId"]
        active_thread = tid
        send({"id": req_id, "result": {"thread": {"id": tid}}})
    elif method == "turn/start":
        st = load_state()
        tid = params["threadId"]
        st["threads"][tid]["turns"] += 1
        turn_num = st["threads"][tid]["turns"]
        save_state(st)
        active_thread = tid
        active_turn = f"{tid}_turn_{turn_num}"
        prompt_text = params["input"][0]["text"]
        hold_open = "HOLD_BACKGROUND" in prompt_text
        send({"id": req_id, "result": {"turn": {"id": active_turn, "status": "inProgress"}}})

        if hold_open:
            continue

        if turn_num == 1:
            # Probe host's ObserveOnly approval gate with a fileChange requestApproval
            send({
                "id": "srv_req_1",
                "method": "item/fileChange/requestApproval",
                "params": {"threadId": tid, "turnId": active_turn},
            })
        else:
            reply_txt = json.dumps(
                {"verdict": f"RESUMED_TURN_{turn_num}", "sandbox_verified": "read-only"}
            )
            send({
                "method": "item/completed",
                "params": {
                    "threadId": tid,
                    "item": {"id": f"msg_{turn_num}", "type": "agentMessage", "text": reply_txt},
                },
            })
            send({
                "method": "turn/completed",
                "params": {
                    "threadId": tid,
                    "turn": {"id": active_turn, "status": "completed"},
                },
            })
    elif method == "turn/steer":
        st = load_state()
        st["steered"].append(params["input"][0]["text"])
        save_state(st)
        send({"id": req_id, "result": {"turn": {"id": params["expectedTurnId"]}}})
    elif method == "turn/interrupt":
        st = load_state()
        st["interrupted"].append(params["turnId"])
        save_state(st)
        send({"id": req_id, "result": {}})
        send({
            "method": "turn/completed",
            "params": {
                "threadId": params["threadId"],
                "turn": {"id": params["turnId"], "status": "interrupted"},
            },
        })
"""
    )

    secrets = ChainedSecretProvider([EnvSecretProvider({})])
    codex_drv = CodexAppServerDriver(
        secrets=secrets,
        state_root=tmp_path / "codex_state",
        app_server_cmd=[sys.executable, "-u", str(server_script), str(state_file)],
    )
    substrate = HarnessSubstrate(
        ledger=SessionLedger(tmp_path / "sessions.sqlite3"),
        ref_verifier=PublishedRefVerifier(tmp_path / "worktrees"),
        drivers=[codex_drv],
    )
    mcp = build_mcp_server(substrate)

    # 1. MCP independent review with structured_output_schema + exclude_model_families
    schema = {
        "type": "object",
        "properties": {
            "verdict": {"type": "string"},
            "sandbox_verified": {"type": "string"},
        },
        "required": ["verdict", "sandbox_verified"],
    }
    rev_raw = await mcp.call_tool(
        "harness_review",
        {
            "prompt": "Review module.py",
            "repo_url": repo_url,
            "ref": "HEAD",
            "exclude_model_families": ["anthropic", "google"],
            "ranking_profile": "low_latency",
            "facets": {
                "reasoning_effort": "high",
                "structured_output_schema": schema,
            },
        },
    )
    rev_payload = _extract_mcp_json(rev_raw)
    assert rev_payload["ok"] is True
    rev_sess = rev_payload["session"]
    assert rev_sess["endpoint_id"] == "codex-app-server"
    assert rev_sess["commit_sha"] == head_sha
    assert rev_sess["status"] == "finished"
    assert rev_sess["structured_output"] == {
        "verdict": "APPROVED",
        "sandbox_verified": "read-only",
    }
    assert rev_sess["worktree_path"] is None

    # Verify ObserveOnly declined server fileChange requestApproval
    st_after_rev = json.loads(state_file.read_text())
    assert st_after_rev["approval_decisions"] == [{"decision": "decline"}]
    assert st_after_rev["threads"]["thr_1"]["ephemeral"] is True

    # 2. Multi-turn session resumption across separate codex app-server subprocesses
    s_start = _extract_mcp_json(
        await mcp.call_tool(
            "harness_start",
            {
                "prompt": "Turn 1 inspect repo",
                "repo_url": repo_url,
                "ref": "HEAD",
                "target_endpoint_id": "codex-app-server",
                "facets": {"structured_output_schema": schema},
            },
        )
    )["session"]
    assert s_start["status"] == "finished"
    assert s_start["worktree_path"] is not None

    s_turn2 = _extract_mcp_json(
        await mcp.call_tool(
            "harness_send",
            {
                "session_id": s_start["session_id"],
                "prompt": "Turn 2 follow-up",
            },
        )
    )["session"]
    assert s_turn2["status"] == "finished"
    assert s_turn2["structured_output"]["verdict"] == "RESUMED_TURN_2"

    # Cancel cleans up retained detached worktree
    wt_dir = Path(s_start["worktree_path"])
    assert wt_dir.exists()
    s_clean = _extract_mcp_json(
        await mcp.call_tool("harness_cancel", {"session_id": s_start["session_id"]})
    )["session"]
    assert s_clean["worktree_path"] is None
    assert not wt_dir.exists()

    # 3. Background turn -> live mid-run steering (turn/steer) -> live cancel (turn/interrupt)
    bg_start = _extract_mcp_json(
        await mcp.call_tool(
            "harness_start",
            {
                "prompt": "HOLD_BACKGROUND long analysis",
                "target_endpoint_id": "codex-app-server",
                "facets": {"background": True},
            },
        )
    )["session"]
    assert bg_start["status"] == "running"

    steered = _extract_mcp_json(
        await mcp.call_tool(
            "harness_send",
            {
                "session_id": bg_start["session_id"],
                "prompt": "Focus on edge cases",
            },
        )
    )["session"]
    assert steered["status"] == "running"

    interrupted = _extract_mcp_json(
        await mcp.call_tool(
            "harness_cancel",
            {"session_id": bg_start["session_id"]},
        )
    )["session"]
    assert interrupted["status"] == "cancelled"

    final_st = json.loads(state_file.read_text())
    assert any("Focus on edge cases" in s for s in final_st["steered"])
    assert len(final_st["interrupted"]) == 1


@pytest.mark.asyncio
async def test_m2_security_hardening_secret_protocol_base_env_stderr_and_propose_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import claude_agent_sdk as cas

    from beast_harness.substrate import create_default_substrate
    from beast_harness.types import TrustTier

    # 1. [B-2] Custom SecretProvider implementing ONLY has_secret/get_secret (no require_secret)
    raw_provider = _SingleUseSecretProvider({"OPENAI_API_KEY": "sk-brokered-single-use"})
    assert not hasattr(raw_provider, "require_secret")

    # 2. [B-3] Poison ambient os.environ with secrets and BEAST_HARNESS_DB / REDIS_URL
    monkeypatch.setenv("NEO4J_PASSWORD", "top-secret-neo4j-pw")
    monkeypatch.setenv("MAGPIE_GUARDED_RECEIPT_ID", "magpie-secret-receipt-999")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-ambient-leak")
    monkeypatch.setenv("BEAST_HARNESS_DB", str(tmp_path / "hijacked.sqlite3"))
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6399/0")

    env_dump_file = tmp_path / "child_env_dump.json"
    stderr_flood_codex = tmp_path / "stderr_flood_codex.py"
    stderr_flood_codex.write_text(
        f"""\
import json
import os
import pathlib
import sys

pathlib.Path({str(env_dump_file)!r}).write_text(json.dumps(dict(os.environ)))
# [M-3] Emit >256 KB to stderr to ensure _stderr_drain_loop prevents pipe deadlock
sys.stderr.write(("RUST_TRACE_LINE_" + ("X" * 512) + "\\n") * 600)
sys.stderr.flush()

for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    msg = json.loads(line)
    method = msg.get("method")
    req_id = msg.get("id")
    params = msg.get("params") or {{}}
    if method == "initialize":
        sys.stdout.write(json.dumps({{"id": req_id, "result": {{}}}}) + "\\n")
        sys.stdout.flush()
    elif method == "thread/start":
        sys.stdout.write(
            json.dumps({{"id": req_id, "result": {{"thread": {{"id": "thr_sec_1"}}}}}}) + "\\n"
        )
        sys.stdout.flush()
    elif method == "turn/start":
        turn_start_resp = {{
            "id": req_id,
            "result": {{"turn": {{"id": "t_sec_1", "status": "inProgress"}}}},
        }}
        sys.stdout.write(json.dumps(turn_start_resp) + "\\n")
        msg_item = {{"type": "agentMessage", "id": "m1", "text": "ISOLATED_OK"}}
        sys.stdout.write(
            json.dumps({{
                "method": "item/completed",
                "params": {{"threadId": "thr_sec_1", "item": msg_item}},
            }}) + "\\n"
        )
        done_turn = {{"id": "t_sec_1", "status": "completed"}}
        sys.stdout.write(
            json.dumps({{
                "method": "turn/completed",
                "params": {{"threadId": "thr_sec_1", "turn": done_turn}},
            }}) + "\\n"
        )
        sys.stdout.flush()
"""
    )

    clean_base_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"}
    codex_drv = CodexAppServerDriver(
        secrets=raw_provider,
        state_root=tmp_path / "codex_sec",
        app_server_cmd=[sys.executable, "-u", str(stderr_flood_codex)],
        base_env=clean_base_env,
    )
    observed_digest = parse_and_validate_capability_ttl(codex_drv.announce_capabilities_ttl()[0])[
        0
    ].capability_graph_digest

    # [M-1] Multi-digest set admission + [B-3] explicit state_dir overrides BEAST_HARNESS_DB
    explicit_state_dir = tmp_path / "explicit_state"
    substrate = create_default_substrate(
        explicit_state_dir,
        secrets=raw_provider,
        admitted_capability_digests={
            "codex-app-server": {"0" * 64, observed_digest},
        },
        drivers=[codex_drv],
        base_env=clean_base_env,
        include_beast_peers=False,
    )
    assert (explicit_state_dir / "sessions.sqlite3").exists()
    assert not (tmp_path / "hijacked.sqlite3").exists()

    eps = await substrate.discover(DiscoveryQuery(min_trust_tier=TrustTier.NEO4J_ADMITTED))
    assert [e.endpoint_id for e in eps] == ["codex-app-server"]

    rec = await substrate.start(
        prompt="Verify env scrubbing and stderr drain",
        target_endpoint_id="codex-app-server",
    )
    assert rec.status == RunState.FINISHED
    assert rec.answer == "ISOLATED_OK"
    assert raw_provider.get_calls == {"OPENAI_API_KEY": 1}

    child_env = json.loads(env_dump_file.read_text())
    assert "NEO4J_PASSWORD" not in child_env
    assert "MAGPIE_GUARDED_RECEIPT_ID" not in child_env
    assert "ANTHROPIC_API_KEY" not in child_env
    assert child_env.get("OPENAI_API_KEY") == "sk-brokered-single-use"

    # 3. [M-4] ClaudeSdkDriver ProposeDiff workspace confinement & tool allowlist
    ws_dir = tmp_path / "propose_ws"
    ws_dir.mkdir()
    outside_file = tmp_path / "outside_secret.txt"
    outside_file.write_text("do not touch")

    claude_drv = ClaudeSdkDriver(
        secrets=raw_provider,
        secret_key_name="OPENAI_API_KEY",
        allow_cli_login=False,
        base_env=clean_base_env,
    )
    propose_opts = claude_drv.build_agent_options(
        workspace_path=str(ws_dir),
        posture=ExecutionPosture.PROPOSE_DIFF,
        facets={},
        api_key="sk-ant-brokered",
    )
    assert propose_opts.env.get("ANTHROPIC_API_KEY") == "sk-ant-brokered"
    assert "NEO4J_PASSWORD" not in propose_opts.env

    in_ws = await propose_opts.can_use_tool("Edit", {"file_path": "sub/file.py"}, None)
    escape_rel = await propose_opts.can_use_tool(
        "Write", {"file_path": "../outside_secret.txt"}, None
    )
    escape_abs = await propose_opts.can_use_tool("Edit", {"file_path": str(outside_file)}, None)
    deny_task = await propose_opts.can_use_tool("Task", {"prompt": "spawn"}, None)
    deny_web = await propose_opts.can_use_tool("WebFetch", {"url": "https://evil.test"}, None)
    deny_mcp = await propose_opts.can_use_tool("mcp__ext__run", {}, None)

    assert isinstance(in_ws, cas.PermissionResultAllow)
    assert isinstance(escape_rel, cas.PermissionResultDeny)
    assert isinstance(escape_abs, cas.PermissionResultDeny)
    assert isinstance(deny_task, cas.PermissionResultDeny)
    assert isinstance(deny_web, cas.PermissionResultDeny)
    assert isinstance(deny_mcp, cas.PermissionResultDeny)

    # Without workspace_path, ProposeDiff denies file edits fail-closed
    no_ws_opts = claude_drv.build_agent_options(
        workspace_path=None,
        posture=ExecutionPosture.PROPOSE_DIFF,
        facets={},
    )
    deny_no_ws = await no_ws_opts.can_use_tool("Write", {"file_path": "file.py"}, None)
    assert isinstance(deny_no_ws, cas.PermissionResultDeny)


@pytest.mark.asyncio
async def test_claude_managed_historical_sse_idle_and_followup_vault_guard(tmp_path: Path):
    events_by_sid: dict[str, list[dict[str, Any]]] = {}

    async def create_session(request: Request):
        body = await request.json()
        sid = "sesn_replay_1"
        events_by_sid[sid] = [
            *list(body.get("initial_events") or []),
            {
                "id": "ev_1",
                "type": "agent.message",
                "content": [{"type": "text", "text": "Turn 1 answer"}],
            },
            # Historical Turn 1 idle persisted in replay stream!
            {"id": "ev_idle_1", "type": "session.status_idle"},
        ]
        return JSONResponse({"id": sid, "type": "session", "status": "running"})

    async def get_session(request: Request):
        return JSONResponse({"id": request.path_params["sid"], "type": "session", "status": "idle"})

    async def handle_events(request: Request):
        sid = request.path_params["sid"]
        if request.method == "GET":
            return JSONResponse({"data": events_by_sid[sid]})
        body = await request.json()
        for ev in body.get("events", []):
            events_by_sid[sid].append(ev)
            if ev.get("type") == "user.message":
                events_by_sid[sid].append(
                    {
                        "id": "ev_2",
                        "type": "agent.message",
                        "content": [
                            {"type": "text", "text": "Turn 2 answer after historical idle"}
                        ],
                    }
                )
                events_by_sid[sid].append({"id": "ev_idle_2", "type": "session.status_idle"})
        return JSONResponse({"data": []})

    async def stream_events(request: Request):
        sid = request.path_params["sid"]

        async def _gen():
            for ev in events_by_sid[sid]:
                yield f"data: {json.dumps(ev)}\n\n"

        return StreamingResponse(_gen(), media_type="text/event-stream")

    app = Starlette(
        routes=[
            Route("/v1/sessions", create_session, methods=["POST"]),
            Route("/v1/sessions/{sid}", get_session, methods=["GET"]),
            Route("/v1/sessions/{sid}/events", handle_events, methods=["GET", "POST"]),
            Route("/v1/sessions/{sid}/events/stream", stream_events, methods=["GET"]),
        ]
    )

    raw_provider = _SingleUseSecretProvider({"ANTHROPIC_API_KEY": "sk-ant-replay"})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://claude.local",
    ) as client:
        driver = ClaudeManagedDriver(
            secrets=raw_provider,
            base_url="http://claude.local",
            http_client=client,
            default_agent_id="ag_1",
            default_environment_id="env_1",
        )
        substrate = HarnessSubstrate(
            ledger=SessionLedger(tmp_path / "replay.sqlite3"),
            ref_verifier=PublishedRefVerifier(tmp_path / "wt"),
            drivers=[driver],
        )
        s1 = await substrate.start(prompt="Turn 1", target_endpoint_id="claude-managed")
        s1_done = await substrate.wait(s1.session_id, timeout_seconds=5.0)
        assert s1_done.answer == "Turn 1 answer"

        # [m-3] ObserveOnly follow-up rejects vault_ids fail-closed
        with pytest.raises(PostureViolationError):
            await substrate.send(
                session_id=s1.session_id,
                prompt="Try vaults on follow-up",
                facets={"vault_ids": ["vlt_999"]},
            )

        await substrate.send(session_id=s1.session_id, prompt="Turn 2")
        s2_done = await substrate.wait(s1.session_id, timeout_seconds=5.0)
        assert s2_done.status == RunState.FINISHED
        assert s2_done.answer == "Turn 2 answer after historical idle"


@pytest.mark.asyncio
async def test_cursor_local_cli_base_env_isolation_and_timeout_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from beast_harness.types import HarnessError

    monkeypatch.setenv("NEO4J_PASSWORD", "cursor-should-not-see-this")
    env_out = tmp_path / "cursor_env.json"
    fake_cursor = tmp_path / "cursor-agent"
    fake_cursor.write_text(
        f"""\
#!{sys.executable}
import json, os, pathlib, sys, time
pathlib.Path({str(env_out)!r}).write_text(json.dumps(dict(os.environ)))
if "HANG_TEST" in " ".join(sys.argv):
    time.sleep(30)
print(json.dumps({{"session_id": "cur_loc_1", "result": "Cursor isolated ok"}}))
"""
    )
    fake_cursor.chmod(0o755)

    clean_env = {"PATH": f"{tmp_path}:{os.environ.get('PATH', '/usr/bin:/bin')}"}
    raw_provider = _SingleUseSecretProvider({"CURSOR_API_KEY": "cur-brokered-key"})
    drv = CursorLocalDriver(
        secrets=raw_provider,
        cli_binary=str(fake_cursor),
        base_env=clean_env,
        cli_timeout_seconds=1.0,
    )
    ep = parse_and_validate_capability_ttl(drv.announce_capabilities_ttl()[0])[0]
    res = await drv.start_run(
        endpoint=ep,
        prompt="Quick check",
        repo_ref=None,
        posture=ExecutionPosture.OBSERVE_ONLY,
        facets={},
        independent_review=True,
    )
    assert res.state == RunState.FINISHED
    assert res.answer == "Cursor isolated ok"
    observed_env = json.loads(env_out.read_text())
    assert "NEO4J_PASSWORD" not in observed_env
    assert observed_env.get("CURSOR_API_KEY") == "cur-brokered-key"

    with pytest.raises(HarnessError, match="timed out"):
        await drv.start_run(
            endpoint=ep,
            prompt="HANG_TEST",
            repo_ref=None,
            posture=ExecutionPosture.OBSERVE_ONLY,
            facets={},
            independent_review=True,
        )
    assert drv._active_procs == {}
