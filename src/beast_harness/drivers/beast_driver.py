"""Beast Network peer discovery & Redis Streams dispatch driver."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from beast_harness.capability_graph import (
    lift_legacy_beast_capabilities_to_ttl,
    parse_and_validate_capability_ttl,
)
from beast_harness.types import (
    DriverRunResult,
    EndpointDescriptor,
    ExecutionPosture,
    RunState,
    SessionRecord,
    VerifiedRepoRef,
)


def _map_beast_status(raw: str) -> RunState:
    norm = raw.strip().lower()
    if norm in {"queued", "pending"}:
        return RunState.QUEUED
    if norm in {"busy"}:
        return RunState.BUSY
    if norm in {"finished", "completed", "done"}:
        return RunState.FINISHED
    if norm in {"cancelled", "canceled"}:
        return RunState.CANCELLED
    if norm in {"error", "failed"}:
        return RunState.ERROR
    return RunState.RUNNING


class BeastPeerDriver:
    driver_id = "beast-peer"

    def __init__(
        self,
        *,
        caller_agent_id: str = "beast-harness-caller",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: str | None = None,
    ) -> None:
        self.caller_agent_id = caller_agent_id
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_password = redis_password

    def _redis(self):
        import redis.asyncio as aioredis

        return aioredis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=self.redis_password,
            decode_responses=True,
        )

    @property
    def reply_stream(self) -> str:
        return f"beast:stream:agent:{self.caller_agent_id}:inbox"

    def announce_capabilities_ttl(self) -> list[str]:
        return []

    async def discover_live_peer_endpoints(self) -> list[EndpointDescriptor]:
        r = self._redis()
        endpoints: list[EndpointDescriptor] = []
        try:
            agent_ids = sorted(await r.smembers("beast:agents:all"))
            for aid in agent_ids:
                if aid == self.caller_agent_id:
                    continue
                raw_info = await r.get(f"beast:agents:{aid}")
                if not raw_info:
                    continue
                info = json.loads(raw_info)
                ttl = info.get("capability_graph_ttl")
                if not ttl:
                    caps = list(info.get("capabilities") or [])
                    ttl = lift_legacy_beast_capabilities_to_ttl(aid, caps)
                parsed = parse_and_validate_capability_ttl(str(ttl), alive=True)
                for ep in parsed:
                    endpoints.append(ep.model_copy(update={"driver_id": self.driver_id}))
        finally:
            await r.aclose()
        endpoints.sort(key=lambda d: d.endpoint_id)
        return endpoints

    async def _publish_task_envelope(
        self,
        *,
        target_agent_id: str,
        payload: dict[str, Any],
    ) -> None:
        r = self._redis()
        try:
            envelope = {
                "header": {
                    "sender": self.caller_agent_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "id": str(uuid.uuid4()),
                },
                "payload": payload,
            }
            await r.xadd(
                f"beast:stream:agent:{target_agent_id}:inbox",
                {"envelope": json.dumps(envelope, sort_keys=True)},
                maxlen=5000,
                approximate=True,
            )
        finally:
            await r.aclose()

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
        run_id = f"beast-run-{uuid.uuid4().hex[:12]}"
        await self._publish_task_envelope(
            target_agent_id=endpoint.endpoint_id,
            payload={
                "type": "HARNESS_TASK_REQUEST",
                "run_id": run_id,
                "prompt": prompt,
                "repo_ref": repo_ref.model_dump() if repo_ref is not None else None,
                "posture": posture.value,
                "facets": dict(facets),
                "independent_review": independent_review,
                "reply_stream": self.reply_stream,
            },
        )
        return DriverRunResult(
            native_agent_id=endpoint.endpoint_id,
            run_id=run_id,
            state=RunState.RUNNING,
        )

    async def send_followup(
        self,
        *,
        session: SessionRecord,
        prompt: str,
        facets: dict[str, Any],
    ) -> DriverRunResult:
        run_id = f"beast-run-{uuid.uuid4().hex[:12]}"
        await self._publish_task_envelope(
            target_agent_id=session.endpoint_id,
            payload={
                "type": "HARNESS_TASK_FOLLOWUP",
                "session_id": session.session_id,
                "previous_run_id": session.latest_run_id,
                "run_id": run_id,
                "prompt": prompt,
                "posture": session.posture.value,
                "facets": dict(facets),
                "reply_stream": self.reply_stream,
            },
        )
        return DriverRunResult(
            native_agent_id=session.native_agent_id,
            run_id=run_id,
            state=RunState.RUNNING,
        )

    def _parse_result_payload(
        self,
        payload: dict[str, Any],
        session: SessionRecord,
    ) -> DriverRunResult:
        return DriverRunResult(
            native_agent_id=str(payload.get("native_agent_id") or session.native_agent_id),
            run_id=str(payload.get("run_id") or session.latest_run_id),
            state=_map_beast_status(str(payload.get("status") or "running")),
            answer=payload.get("answer"),
            structured_output=payload.get("structured_output"),
            audit_events=list(payload.get("audit_events") or []),
            error_detail=payload.get("error_detail"),
        )

    async def poll_run(self, *, session: SessionRecord) -> DriverRunResult:
        r = self._redis()
        try:
            entries = await r.xrange(self.reply_stream, "-", "+")
            matched: DriverRunResult | None = None
            for _msg_id, fields in entries:
                raw_env = fields.get("envelope")
                if not raw_env:
                    continue
                env = json.loads(raw_env)
                payload = env.get("payload") or {}
                if (
                    payload.get("type") == "HARNESS_TASK_RESULT"
                    and payload.get("run_id") == session.latest_run_id
                ):
                    matched = self._parse_result_payload(payload, session)
            if matched is not None:
                return matched
        finally:
            await r.aclose()

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
        existing = await self.poll_run(session=session)
        if existing.state.is_terminal:
            return existing

        r = self._redis()
        deadline = time.monotonic() + max(0.1, timeout_seconds)
        cursor_id = "0-0"
        latest = existing
        try:
            while time.monotonic() < deadline:
                remaining_ms = max(50, int((deadline - time.monotonic()) * 1000))
                block_ms = min(250, remaining_ms)
                streams = await r.xread({self.reply_stream: cursor_id}, count=20, block=block_ms)
                for _sname, messages in streams:
                    for msg_id, fields in messages:
                        cursor_id = msg_id
                        raw_env = fields.get("envelope")
                        if not raw_env:
                            continue
                        env = json.loads(raw_env)
                        payload = env.get("payload") or {}
                        if (
                            payload.get("type") == "HARNESS_TASK_RESULT"
                            and payload.get("run_id") == session.latest_run_id
                        ):
                            latest = self._parse_result_payload(payload, session)
                            if latest.state.is_terminal:
                                return latest
                await asyncio.sleep(0.01)
        finally:
            await r.aclose()

        return latest

    async def cancel_run(self, *, session: SessionRecord) -> DriverRunResult:
        await self._publish_task_envelope(
            target_agent_id=session.endpoint_id,
            payload={
                "type": "HARNESS_TASK_CANCEL",
                "run_id": session.latest_run_id,
                "reply_stream": self.reply_stream,
            },
        )
        return DriverRunResult(
            native_agent_id=session.native_agent_id,
            run_id=session.latest_run_id,
            state=RunState.CANCELLED,
        )
