"""OpenAI Codex App-Server JSON-RPC v2 driver with OS read-only sandboxing."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import tempfile
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from beast_harness.secrets import ChainedSecretProvider, ensure_chained_secret_provider
from beast_harness.types import (
    BH_NS,
    DriverRunResult,
    EndpointDescriptor,
    ExecutionPosture,
    HarnessError,
    RunState,
    SecretProvider,
    SessionRecord,
    VerifiedRepoRef,
    ensure_directory,
)

_OBSERVE_GUARDRAIL = (
    "\n\n[OBSERVE_ONLY GUARDRAIL: Inspect the repository at the pinned commit only. "
    "Do NOT modify files, create branches, commit changes, push refs, or open pull requests.]"
)


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


def _map_codex_turn_status(raw: str) -> RunState:
    norm = raw.strip().lower()
    if norm == "completed":
        return RunState.FINISHED
    if norm in {"interrupted", "cancelled", "canceled"}:
        return RunState.CANCELLED
    if norm in {"failed", "error"}:
        return RunState.ERROR
    return RunState.RUNNING


class _CodexJsonRpcConnection:
    """Async newline-delimited JSON-RPC v2 client over codex app-server stdio."""

    def __init__(
        self,
        proc: asyncio.subprocess.Process,
        *,
        posture: ExecutionPosture,
        scratch_dir: tempfile.TemporaryDirectory[str] | None = None,
    ) -> None:
        self.proc = proc
        self.posture = posture
        self._scratch_dir = scratch_dir
        self._next_id = 1
        self._write_lock = asyncio.Lock()
        self._pending: dict[int | str, asyncio.Future[dict[str, Any]]] = {}
        self.turn_done = asyncio.Event()
        self.final_turn: dict[str, Any] | None = None
        self.completed_messages: list[str] = []
        self.delta_chunks: dict[str, list[str]] = {}
        self.audit_events: list[dict[str, Any]] = []
        self.stderr_tail: deque[str] = deque(maxlen=64)
        self.protocol_error: str | None = None
        self._stderr_task = asyncio.create_task(self._stderr_drain_loop())
        self._reader_task = asyncio.create_task(self._read_loop())

    async def _stderr_drain_loop(self) -> None:
        if self.proc.stderr is None:
            return
        try:
            while True:
                chunk = await self.proc.stderr.read(4096)
                if not chunk:
                    break
                for line in chunk.decode("utf-8", errors="replace").splitlines():
                    stripped = line.strip()
                    if stripped:
                        self.stderr_tail.append(stripped[:512])
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self.proc.stdin is None:
            raise HarnessError("Codex app-server stdin is closed.")
        encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        async with self._write_lock:
            self.proc.stdin.write(encoded)
            await self.proc.stdin.drain()

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[req_id] = fut
        msg: dict[str, Any] = {"id": req_id, "method": method}
        if params is not None:
            msg["params"] = params
        try:
            await self._send_json(msg)
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(req_id, None)

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        msg: dict[str, Any] = {"method": method}
        if params is not None:
            msg["params"] = params
        await self._send_json(msg)

    async def _handle_server_request(self, msg: dict[str, Any]) -> None:
        req_id = msg["id"]
        method = str(msg.get("method") or "")
        if method in {
            "item/fileChange/requestApproval",
            "applyPatchApproval",
        }:
            decision = "accept" if self.posture == ExecutionPosture.PROPOSE_DIFF else "decline"
            self.audit_events.append(
                {
                    "type": "server_approval_request",
                    "method": method,
                    "decision": decision,
                }
            )
            await self._send_json({"id": req_id, "result": {"decision": decision}})
            return

        if method in {
            "item/commandExecution/requestApproval",
            "execCommandApproval",
            "item/permissions/requestApproval",
        }:
            self.audit_events.append(
                {
                    "type": "server_approval_request",
                    "method": method,
                    "decision": "decline",
                }
            )
            await self._send_json({"id": req_id, "result": {"decision": "decline"}})
            return

        await self._send_json(
            {
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": (
                        f"Interactive server request {method!r} unsupported in headless harness."
                    ),
                },
            }
        )

    def _handle_notification(self, msg: dict[str, Any]) -> None:
        method = str(msg.get("method") or "")
        params = msg.get("params") if isinstance(msg.get("params"), dict) else {}

        if method == "item/agentMessage/delta":
            item_id = str(params.get("itemId") or "default")
            delta = str(params.get("delta") or "")
            if delta:
                self.delta_chunks.setdefault(item_id, []).append(delta)
        elif method == "item/completed":
            item = params.get("item")
            if isinstance(item, dict):
                item_type = str(item.get("type") or "")
                if item_type == "agentMessage":
                    txt = str(item.get("text") or "")
                    if txt:
                        self.completed_messages.append(txt)
                elif item_type in {
                    "commandExecution",
                    "fileChange",
                    "mcpToolCall",
                    "dynamicToolCall",
                    "webSearch",
                }:
                    self.audit_events.append(
                        {
                            "type": item_type,
                            "id": item.get("id"),
                            "status": item.get("status"),
                        }
                    )
        elif method == "turn/completed":
            turn = params.get("turn")
            if isinstance(turn, dict):
                self.final_turn = turn
            else:
                self.final_turn = {"status": "completed"}
            self.turn_done.set()
        elif method == "error":
            err_obj = params.get("error")
            if isinstance(err_obj, dict):
                self.protocol_error = str(err_obj.get("message") or err_obj)
            else:
                self.protocol_error = str(err_obj or params or "Codex app-server error")

    async def _read_loop(self) -> None:
        assert self.proc.stdout is not None
        try:
            while True:
                raw_line = await self.proc.stdout.readline()
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue

                if "id" in msg and "method" in msg:
                    await self._handle_server_request(msg)
                elif "id" in msg:
                    req_id = msg["id"]
                    fut = self._pending.get(req_id)
                    if fut is not None and not fut.done():
                        if "error" in msg and msg["error"] is not None:
                            err = msg["error"]
                            detail = err.get("message") if isinstance(err, dict) else str(err)
                            fut.set_exception(HarnessError(f"Codex JSON-RPC error: {detail}"))
                        else:
                            res = msg.get("result")
                            fut.set_result(res if isinstance(res, dict) else {"value": res})
                elif "method" in msg:
                    self._handle_notification(msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.protocol_error = str(exc)
        finally:
            stderr_hint = "; ".join(list(self.stderr_tail)[-3:]) if self.stderr_tail else ""
            fallback_err = self.protocol_error or (
                f"Codex app-server closed stream: {stderr_hint}"
                if stderr_hint
                else "Codex app-server closed stream before replying."
            )
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(HarnessError(fallback_err))
            if not self.turn_done.is_set():
                if self.final_turn is None:
                    self.final_turn = {
                        "status": "failed",
                        "error": {"message": fallback_err},
                    }
                self.turn_done.set()

    def build_result(
        self,
        *,
        thread_id: str,
        turn_id: str,
        expect_structured: bool,
    ) -> DriverRunResult:
        if not self.turn_done.is_set():
            return DriverRunResult(
                native_agent_id=thread_id,
                run_id=turn_id,
                state=RunState.RUNNING,
                audit_events=list(self.audit_events),
            )

        turn = self.final_turn or {"status": "completed"}
        state = _map_codex_turn_status(str(turn.get("status") or "completed"))
        err_detail: str | None = None
        if isinstance(turn.get("error"), dict):
            err_detail = str(turn["error"].get("message") or turn["error"])
        elif turn.get("error"):
            err_detail = str(turn["error"])
        elif self.protocol_error and state == RunState.ERROR:
            err_detail = self.protocol_error

        if self.completed_messages:
            answer: str | None = "\n".join(self.completed_messages).strip()
        elif self.delta_chunks:
            joined = ["".join(parts) for parts in self.delta_chunks.values()]
            answer = "\n".join(joined).strip() or None
        else:
            answer = None

        structured: dict[str, Any] | None = None
        if expect_structured:
            candidates = [*reversed(self.completed_messages)]
            if answer:
                candidates.append(answer)
            for candidate in candidates:
                with contextlib.suppress(json.JSONDecodeError):
                    parsed = json.loads(candidate.strip())
                    if isinstance(parsed, dict):
                        structured = parsed
                        break

        return DriverRunResult(
            native_agent_id=thread_id,
            run_id=turn_id,
            state=state,
            answer=answer,
            structured_output=structured,
            audit_events=list(self.audit_events),
            error_detail=err_detail,
        )

    async def close(self) -> None:
        self._reader_task.cancel()
        self._stderr_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._reader_task
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._stderr_task
        if self.proc.stdin is not None:
            with contextlib.suppress(Exception):
                self.proc.stdin.close()
        if self.proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=3.0)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    self.proc.kill()
                with contextlib.suppress(Exception):
                    await self.proc.wait()
        if self._scratch_dir is not None:
            with contextlib.suppress(Exception):
                self._scratch_dir.cleanup()
            self._scratch_dir = None


@dataclass
class _ActiveCodexTurn:
    thread_id: str
    turn_id: str
    conn: _CodexJsonRpcConnection
    expect_structured: bool
    closed: bool = field(default=False)


class CodexAppServerDriver:
    """OpenAI Codex App-Server JSON-RPC v2 driver over detached read-only worktrees."""

    driver_id = "codex-app-server"

    def __init__(
        self,
        *,
        secrets: SecretProvider | ChainedSecretProvider,
        state_root: Path | None = None,
        secret_key_name: str = "OPENAI_API_KEY",
        cli_binary: str = "codex",
        allow_cli_login: bool = True,
        app_server_cmd: Sequence[str] | None = None,
        base_env: Mapping[str, str] | None = None,
    ) -> None:
        self._secrets = ensure_chained_secret_provider(secrets)
        self._state_root = (
            ensure_directory(Path(state_root))
            if state_root is not None
            else ensure_directory(Path(tempfile.gettempdir()) / "beast-harness-codex")
        )
        self._secret_key_name = secret_key_name
        self._cli_binary = cli_binary
        self._allow_cli_login = allow_cli_login
        self._app_server_cmd = tuple(app_server_cmd) if app_server_cmd is not None else None
        self._base_env = dict(base_env) if base_env is not None else None
        self._active_turns: dict[str, _ActiveCodexTurn] = {}
        self._completed_runs: dict[str, DriverRunResult] = {}

    def _which_cli(self) -> str | None:
        if self._base_env is not None and "PATH" in self._base_env:
            return shutil.which(self._cli_binary, path=self._base_env.get("PATH"))
        return shutil.which(self._cli_binary)

    def is_available(self) -> bool:
        if self._app_server_cmd is not None:
            return True
        if self._which_cli() is None:
            return False
        if self._secrets.has_secret(self._secret_key_name):
            return True
        return self._allow_cli_login

    def announce_capabilities_ttl(self) -> list[str]:
        effort_schema = json.dumps(
            {
                "type": "string",
                "enum": ["low", "medium", "high", "xhigh"],
            }
        )
        obj_schema = json.dumps({"type": "object"})
        str_schema = json.dumps({"type": "string", "minLength": 1})
        bool_schema = json.dumps({"type": "boolean"})
        timeout_schema = json.dumps({"type": "number", "minimum": 1.0, "maximum": 3600.0})
        ttl = f"""\
@prefix bh: <{BH_NS}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<urn:beast:endpoint:codex-app-server> a bh:AgentEndpoint ;
    rdfs:label "OpenAI Codex App-Server Agent (JSON-RPC v2 + OS Read-Only Sandbox)" ;
    bh:endpointId "codex-app-server" ;
    bh:driverId "codex-app-server" ;
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
        bh:MidRunSteering,
        bh:FollowUpRunQueue,
        bh:RepoFreePrompt,
        bh:VerifiableToolAuditStream ;
    bh:hasModelProfile [
        a bh:ModelProfile ;
        bh:modelFamily "openai" ;
        bh:modelIdentifier "gpt-5-codex" ;
        bh:contextWindowTokens 400000
    ] ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability bh:ThinkingEffortControl ;
        bh:facetKey "reasoning_effort" ;
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
        bh:forCapability bh:MidRunSteering ;
        bh:facetKey "background" ;
        bh:inputJsonSchema '{bool_schema}'
    ] ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability bh:RepoFreePrompt ;
        bh:facetKey "turn_timeout_seconds" ;
        bh:inputJsonSchema '{timeout_schema}'
    ] .
"""
        return [ttl]

    def _resolve_command(self) -> list[str]:
        if self._app_server_cmd is not None:
            return list(self._app_server_cmd)
        resolved = self._which_cli()
        if resolved is None:
            raise HarnessError(f"Codex binary {self._cli_binary!r} was not found on PATH.")
        return [resolved, "app-server", "--listen", "stdio://"]

    async def _spawn_connection(
        self,
        *,
        posture: ExecutionPosture,
        workspace_path: str | None,
    ) -> tuple[_CodexJsonRpcConnection, str]:
        cmd = self._resolve_command()
        env = dict(self._base_env) if self._base_env is not None else dict(os.environ)
        if self._secrets.has_secret(self._secret_key_name):
            secret_val = self._secrets.require_secret(self._secret_key_name)
            env[self._secret_key_name] = secret_val
            env["OPENAI_API_KEY"] = secret_val

        scratch: tempfile.TemporaryDirectory[str] | None = None
        if workspace_path is not None:
            effective_cwd = str(Path(workspace_path).resolve())
        else:
            scratch = tempfile.TemporaryDirectory(
                prefix="codex-repofree-",
                dir=str(self._state_root),
            )
            effective_cwd = str(Path(scratch.name).resolve())

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=effective_cwd,
                env=env,
                limit=8 * 1024 * 1024,
            )
        except Exception as exc:
            if scratch is not None:
                scratch.cleanup()
            raise HarnessError(f"Failed to launch codex app-server: {exc}") from exc

        conn = _CodexJsonRpcConnection(proc, posture=posture, scratch_dir=scratch)
        try:
            await conn.request(
                "initialize",
                {
                    "clientInfo": {
                        "name": "beast-harness",
                        "title": "Beast Harness",
                        "version": "0.2.0",
                    },
                    "capabilities": {"experimentalApi": True},
                },
                timeout=30.0,
            )
            await conn.notify("initialized")
        except Exception:
            await conn.close()
            raise
        return conn, effective_cwd

    def _sandbox_params(
        self,
        posture: ExecutionPosture,
        effective_cwd: str,
    ) -> tuple[str, dict[str, Any]]:
        if posture == ExecutionPosture.OBSERVE_ONLY:
            return "read-only", {"type": "readOnly", "networkAccess": False}
        return (
            "workspace-write",
            {
                "type": "workspaceWrite",
                "writableRoots": [effective_cwd],
                "networkAccess": False,
            },
        )

    async def _finalize_active_turn(self, active: _ActiveCodexTurn) -> DriverRunResult:
        res = active.conn.build_result(
            thread_id=active.thread_id,
            turn_id=active.turn_id,
            expect_structured=active.expect_structured,
        )
        if res.state.is_terminal and not active.closed:
            active.closed = True
            self._active_turns.pop(active.thread_id, None)
            self._completed_runs[active.turn_id] = res
            await active.conn.close()
        return res

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

        conn, effective_cwd = await self._spawn_connection(
            posture=posture,
            workspace_path=ws_path,
        )
        sandbox_mode, sandbox_policy = self._sandbox_params(posture, effective_cwd)
        expect_structured = isinstance(facets.get("structured_output_schema"), dict)

        try:
            thread_params: dict[str, Any] = {
                "cwd": effective_cwd,
                "sandbox": sandbox_mode,
                "approvalPolicy": "never",
                "ephemeral": bool(independent_review),
            }
            if facets.get("model"):
                thread_params["model"] = str(facets["model"])

            thread_resp = await conn.request("thread/start", thread_params, timeout=30.0)
            thread_obj = thread_resp.get("thread") if isinstance(thread_resp, dict) else None
            raw_tid = thread_obj.get("id") if isinstance(thread_obj, dict) else None
            if not raw_tid or not str(raw_tid).strip():
                raise HarnessError("Codex thread/start did not return a valid thread.id.")
            thread_id = str(raw_tid).strip()

            turn_params: dict[str, Any] = {
                "threadId": thread_id,
                "cwd": effective_cwd,
                "approvalPolicy": "never",
                "sandboxPolicy": sandbox_policy,
                "input": [{"type": "text", "text": effective_prompt}],
            }
            if facets.get("model"):
                turn_params["model"] = str(facets["model"])
            if facets.get("reasoning_effort"):
                turn_params["effort"] = str(facets["reasoning_effort"])
            if expect_structured:
                turn_params["outputSchema"] = facets["structured_output_schema"]

            turn_resp = await conn.request("turn/start", turn_params, timeout=30.0)
            turn_obj = turn_resp.get("turn") if isinstance(turn_resp, dict) else None
            turn_id = str(
                (turn_obj.get("id") if isinstance(turn_obj, dict) else None)
                or f"{thread_id}-turn-1"
            )
        except Exception:
            await conn.close()
            raise

        active = _ActiveCodexTurn(
            thread_id=thread_id,
            turn_id=turn_id,
            conn=conn,
            expect_structured=expect_structured,
        )
        self._active_turns[thread_id] = active

        if facets.get("background") is True:
            return DriverRunResult(
                native_agent_id=thread_id,
                run_id=turn_id,
                state=RunState.RUNNING,
            )

        timeout_s = float(facets.get("turn_timeout_seconds") or 300.0)
        try:
            await asyncio.wait_for(conn.turn_done.wait(), timeout=timeout_s)
        except TimeoutError as exc:
            await self._interrupt_and_close(active)
            raise HarnessError(f"Codex turn {turn_id!r} timed out after {timeout_s}s.") from exc

        return await self._finalize_active_turn(active)

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

        # Mid-run steering if an active turn is currently in progress
        active = self._active_turns.get(session.native_agent_id)
        if active is not None and not active.conn.turn_done.is_set():
            await active.conn.request(
                "turn/steer",
                {
                    "threadId": active.thread_id,
                    "expectedTurnId": active.turn_id,
                    "input": [{"type": "text", "text": effective_prompt}],
                },
                timeout=30.0,
            )
            return DriverRunResult(
                native_agent_id=active.thread_id,
                run_id=active.turn_id,
                state=RunState.RUNNING,
                audit_events=list(active.conn.audit_events),
            )

        # Resume completed/restarted thread via thread/resume + turn/start
        conn, effective_cwd = await self._spawn_connection(
            posture=session.posture,
            workspace_path=session.worktree_path,
        )
        sandbox_mode, sandbox_policy = self._sandbox_params(session.posture, effective_cwd)
        expect_structured = isinstance(merged_facets.get("structured_output_schema"), dict)

        try:
            resume_params: dict[str, Any] = {
                "threadId": session.native_agent_id,
                "cwd": effective_cwd,
                "sandbox": sandbox_mode,
                "approvalPolicy": "never",
            }
            if merged_facets.get("model"):
                resume_params["model"] = str(merged_facets["model"])
            await conn.request("thread/resume", resume_params, timeout=30.0)

            turn_params: dict[str, Any] = {
                "threadId": session.native_agent_id,
                "cwd": effective_cwd,
                "approvalPolicy": "never",
                "sandboxPolicy": sandbox_policy,
                "input": [{"type": "text", "text": effective_prompt}],
            }
            if merged_facets.get("model"):
                turn_params["model"] = str(merged_facets["model"])
            if merged_facets.get("reasoning_effort"):
                turn_params["effort"] = str(merged_facets["reasoning_effort"])
            if expect_structured:
                turn_params["outputSchema"] = merged_facets["structured_output_schema"]

            turn_resp = await conn.request("turn/start", turn_params, timeout=30.0)
            turn_obj = turn_resp.get("turn") if isinstance(turn_resp, dict) else None
            turn_id = str(
                (turn_obj.get("id") if isinstance(turn_obj, dict) else None)
                or f"{session.native_agent_id}-turn-{uuid.uuid4().hex[:6]}"
            )
        except Exception:
            await conn.close()
            raise

        new_active = _ActiveCodexTurn(
            thread_id=session.native_agent_id,
            turn_id=turn_id,
            conn=conn,
            expect_structured=expect_structured,
        )
        self._active_turns[session.native_agent_id] = new_active

        if facets.get("background") is True:
            return DriverRunResult(
                native_agent_id=session.native_agent_id,
                run_id=turn_id,
                state=RunState.RUNNING,
            )

        timeout_s = float(merged_facets.get("turn_timeout_seconds") or 300.0)
        try:
            await asyncio.wait_for(conn.turn_done.wait(), timeout=timeout_s)
        except TimeoutError as exc:
            await self._interrupt_and_close(new_active)
            raise HarnessError(
                f"Codex follow-up turn {turn_id!r} timed out after {timeout_s}s."
            ) from exc

        return await self._finalize_active_turn(new_active)

    async def poll_run(self, *, session: SessionRecord) -> DriverRunResult:
        active = self._active_turns.get(session.native_agent_id)
        if active is not None:
            return await self._finalize_active_turn(active)
        if session.latest_run_id in self._completed_runs:
            return self._completed_runs[session.latest_run_id]
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
        active = self._active_turns.get(session.native_agent_id)
        if active is not None:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    active.conn.turn_done.wait(),
                    timeout=max(0.1, timeout_seconds),
                )
            return await self._finalize_active_turn(active)
        return await self.poll_run(session=session)

    async def _interrupt_and_close(self, active: _ActiveCodexTurn) -> None:
        if not active.conn.turn_done.is_set():
            with contextlib.suppress(Exception):
                await active.conn.request(
                    "turn/interrupt",
                    {
                        "threadId": active.thread_id,
                        "turnId": active.turn_id,
                    },
                    timeout=5.0,
                )
        active.closed = True
        self._active_turns.pop(active.thread_id, None)
        await active.conn.close()

    async def cancel_run(self, *, session: SessionRecord) -> DriverRunResult:
        active = self._active_turns.get(session.native_agent_id)
        if active is not None:
            await self._interrupt_and_close(active)
        cancelled = DriverRunResult(
            native_agent_id=session.native_agent_id,
            run_id=session.latest_run_id,
            state=RunState.CANCELLED,
        )
        self._completed_runs[session.latest_run_id] = cancelled
        return cancelled
