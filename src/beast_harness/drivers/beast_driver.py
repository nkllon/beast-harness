"""Beast Network peer discovery & Redis Streams dispatch driver."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from beast_harness.capability_graph import (
    lift_legacy_beast_capabilities_to_ttl,
    parse_and_validate_capability_ttl,
)
from beast_harness.types import (
    CapabilityValidationError,
    DriverRunResult,
    EndpointDescriptor,
    ExecutionPosture,
    HarnessError,
    RunState,
    SecretProvider,
    SessionRecord,
    TrustTier,
    VerifiedRepoRef,
)

logger = logging.getLogger(__name__)


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
        secrets: SecretProvider | None = None,
        redis_password_secret_name: str = "BEAST_REDIS_PASSWORD",
    ) -> None:
        self.caller_agent_id = caller_agent_id
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self._secrets = secrets
        self._redis_password_secret_name = redis_password_secret_name
        self._last_reply_cursor_by_run: dict[str, str] = {}

    def __repr__(self) -> str:
        return (
            f"BeastPeerDriver(caller_agent_id={self.caller_agent_id!r}, "
            f"redis={self.redis_host}:{self.redis_port}/{self.redis_db})"
        )

    def is_available(self) -> bool:
        return importlib.util.find_spec("redis") is not None

    def _redis(self):
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:
            raise HarnessError("redis extra is not installed for BeastPeerDriver.") from exc

        pwd = (
            self._secrets.get_secret(self._redis_password_secret_name)
            if self._secrets is not None
            else None
        )
        return aioredis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            password=pwd,
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
            # Optional fallback index of latest capability_graph_ttl from announcements stream
            announced_ttl_by_agent: dict[str, str] = {}
            stream_entries = await r.xrevrange("beast:stream:global:announcements", count=200)
            for _entry_id, fields in stream_entries:
                raw_env = fields.get("envelope")
                if not raw_env:
                    continue
                try:
                    env = json.loads(raw_env)
                except json.JSONDecodeError:
                    continue
                payload = env.get("payload") or {}
                sender = (env.get("header") or {}).get("sender")
                ttl_str = payload.get("capability_graph_ttl")
                if (
                    isinstance(sender, str)
                    and sender == payload.get("agent_id")
                    and isinstance(ttl_str, str)
                    and sender not in announced_ttl_by_agent
                ):
                    announced_ttl_by_agent[sender] = ttl_str

            for aid in agent_ids:
                if aid == self.caller_agent_id:
                    continue
                raw_info = await r.get(f"beast:agents:{aid}")
                if not raw_info:
                    continue
                try:
                    info = json.loads(raw_info)
                    if not isinstance(info, dict):
                        continue
                    ttl = info.get("capability_graph_ttl") or announced_ttl_by_agent.get(aid)
                    if not ttl:
                        caps = list(info.get("capabilities") or [])
                        ttl = lift_legacy_beast_capabilities_to_ttl(aid, caps)
                    parsed = parse_and_validate_capability_ttl(str(ttl), alive=True)
                except (
                    json.JSONDecodeError,
                    CapabilityValidationError,
                    ValueError,
                    TypeError,
                ) as exc:
                    logger.warning(
                        "Skipping Beast peer %r with invalid capability announcement: %s",
                        aid,
                        exc,
                    )
                    continue

                for ep in parsed:
                    if ep.endpoint_id != aid:
                        logger.warning(
                            "Skipping Beast peer %r with mismatched endpointId=%r",
                            aid,
                            ep.endpoint_id,
                        )
                        continue
                    # Never trust self-asserted Tier-2 Neo4jAdmitted over unauthenticated Redis;
                    # HarnessSubstrate promotes iff capability_graph_digest is admitted.
                    clamped = ep.model_copy(
                        update={
                            "driver_id": self.driver_id,
                            "trust_tier": TrustTier.RUNTIME_SELF_ANNOUNCED,
                        }
                    )
                    endpoints.append(clamped)
        except Exception as exc:
            if isinstance(exc, HarnessError):
                raise
            raise HarnessError(f"BeastPeerDriver discovery failed: {exc}") from exc
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
        except Exception as exc:
            if isinstance(exc, HarnessError):
                raise
            raise HarnessError(
                f"BeastPeerDriver failed publishing to {target_agent_id}: {exc}"
            ) from exc
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
        start_cursor = self._last_reply_cursor_by_run.get(session.latest_run_id, "-")
        min_id = f"({start_cursor}" if start_cursor not in {"-", "0-0"} else "-"
        try:
            entries = await r.xrange(self.reply_stream, min_id, "+", count=500)
            matched: DriverRunResult | None = None
            for msg_id, fields in entries:
                self._last_reply_cursor_by_run[session.latest_run_id] = msg_id
                raw_env = fields.get("envelope")
                if not raw_env:
                    continue
                try:
                    env = json.loads(raw_env)
                except json.JSONDecodeError:
                    continue
                payload = env.get("payload") or {}
                if (
                    payload.get("type") == "HARNESS_TASK_RESULT"
                    and payload.get("run_id") == session.latest_run_id
                ):
                    matched = self._parse_result_payload(payload, session)
            if matched is not None:
                return matched
        except Exception as exc:
            if isinstance(exc, HarnessError):
                raise
            raise HarnessError(f"BeastPeerDriver poll_run failed: {exc}") from exc
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
        cursor_id = self._last_reply_cursor_by_run.get(session.latest_run_id, "0-0")
        latest = existing
        try:
            while time.monotonic() < deadline:
                remaining_ms = max(50, int((deadline - time.monotonic()) * 1000))
                block_ms = min(250, remaining_ms)
                streams = await r.xread({self.reply_stream: cursor_id}, count=50, block=block_ms)
                for _sname, messages in streams:
                    for msg_id, fields in messages:
                        cursor_id = msg_id
                        self._last_reply_cursor_by_run[session.latest_run_id] = msg_id
                        raw_env = fields.get("envelope")
                        if not raw_env:
                            continue
                        try:
                            env = json.loads(raw_env)
                        except json.JSONDecodeError:
                            continue
                        payload = env.get("payload") or {}
                        if (
                            payload.get("type") == "HARNESS_TASK_RESULT"
                            and payload.get("run_id") == session.latest_run_id
                        ):
                            latest = self._parse_result_payload(payload, session)
                            if latest.state.is_terminal:
                                return latest
                await asyncio.sleep(0.01)
        except Exception as exc:
            if isinstance(exc, HarnessError):
                raise
            raise HarnessError(f"BeastPeerDriver wait_run failed: {exc}") from exc
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
