"""Bridge agent hosting any HarnessDriver on the Beast Network via BaseAgent."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import re
import uuid
from datetime import UTC, datetime
from typing import Any

from beast_agent import BaseAgent
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS

from beast_harness.types import (
    BH_NS,
    EndpointDescriptor,
    ExecutionPosture,
    HarnessDriver,
    PostureViolationError,
    RunState,
    SessionRecord,
    VerifiedRepoRef,
)

logger = logging.getLogger(__name__)
BH = Namespace(BH_NS)
_VALID_REPLY_STREAM_RE = re.compile(r"^beast:stream:agent:[A-Za-z0-9._-]+:inbox$")
_CONSUMER_GROUP = "beast-harness-bridge-cg"


def _rewrite_ttl_for_bridge_agent(
    raw_ttl: str,
    *,
    bridge_agent_id: str,
    target_endpoint_id: str,
) -> str:
    g = Graph()
    g.parse(data=raw_ttl, format="turtle")

    target_node = None
    for s in g.subjects(RDF.type, BH.AgentEndpoint):
        if str(g.value(s, BH.endpointId)) == target_endpoint_id:
            target_node = s
            break
    if target_node is None:
        raise ValueError(f"Endpoint {target_endpoint_id!r} not found in driver Turtle.")

    new_node = URIRef(f"urn:beast:endpoint:{bridge_agent_id}")
    if target_node != new_node:
        for p, o in list(g.predicate_objects(target_node)):
            g.remove((target_node, p, o))
            g.add((new_node, p, o))

    g.set((new_node, BH.endpointId, Literal(bridge_agent_id)))
    g.set((new_node, BH.driverId, Literal("beast-peer")))
    g.set((new_node, RDFS.label, Literal(f"Beast Bridge ({bridge_agent_id})")))
    return g.serialize(format="turtle")


class BeastHarnessBridgeAgent(BaseAgent):
    def __init__(
        self,
        *,
        agent_id: str,
        driver: HarnessDriver,
        endpoint: EndpointDescriptor,
        mailbox_config: Any = None,
        registration_heartbeat_interval: float = 20.0,
    ) -> None:
        matched_ttl: str | None = None
        for ttl_doc in driver.announce_capabilities_ttl():
            if endpoint.endpoint_id in ttl_doc:
                matched_ttl = ttl_doc
                break
        if matched_ttl is None:
            matched_ttl = driver.announce_capabilities_ttl()[0]

        bridged_ttl = _rewrite_ttl_for_bridge_agent(
            matched_ttl,
            bridge_agent_id=agent_id,
            target_endpoint_id=endpoint.endpoint_id,
        )
        short_caps = [
            cap.rsplit("#", 1)[-1] if "#" in cap else cap for cap in endpoint.capabilities
        ]
        super().__init__(
            agent_id=agent_id,
            capabilities=short_caps,
            mailbox_url=mailbox_config,
            capability_graph_ttl=bridged_ttl,
            registration_heartbeat_interval=registration_heartbeat_interval,
        )
        self._driver = driver
        self._wrapped_endpoint = endpoint
        self._inbox_task: asyncio.Task[None] | None = None
        self._inflight_tasks: set[asyncio.Task[None]] = set()
        self._sessions_by_run_id: dict[str, SessionRecord] = {}
        self._consumer_name = f"{agent_id}-{uuid.uuid4().hex[:8]}"

    @property
    def inbox_stream(self) -> str:
        return f"beast:stream:agent:{self.agent_id}:inbox"

    async def on_startup(self) -> None:
        # Create consumer group positioned at '$' BEFORE startup() returns so:
        # 1) Historical messages from prior runs are never replayed on restart;
        # 2) Any message published after startup() returns is guaranteed delivered.
        r = self._redis()
        try:
            with contextlib.suppress(Exception):
                await r.xgroup_create(
                    self.inbox_stream,
                    _CONSUMER_GROUP,
                    id="$",
                    mkstream=True,
                )
        finally:
            await r.aclose()
        self._inbox_task = asyncio.create_task(self._consume_stream_inbox())

    async def on_shutdown(self) -> None:
        if self._inbox_task is not None:
            self._inbox_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._inbox_task
            self._inbox_task = None
        for task in list(self._inflight_tasks):
            task.cancel()
        if self._inflight_tasks:
            await asyncio.gather(*self._inflight_tasks, return_exceptions=True)
            self._inflight_tasks.clear()

    def _redis(self):
        import redis.asyncio as aioredis

        host = getattr(self._mailbox_config, "host", "localhost")
        port = getattr(self._mailbox_config, "port", 6379)
        db = getattr(self._mailbox_config, "db", 0)
        password = getattr(self._mailbox_config, "password", None)
        return aioredis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
        )

    async def _publish_result(self, reply_to: str, result_payload: dict[str, Any]) -> None:
        if not _VALID_REPLY_STREAM_RE.match(reply_to):
            logger.warning("Refusing invalid Beast reply_stream=%r", reply_to)
            return
        r = self._redis()
        try:
            reply_envelope = {
                "header": {
                    "sender": self.agent_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "id": str(uuid.uuid4()),
                },
                "payload": result_payload,
            }
            await r.xadd(
                reply_to,
                {"envelope": json.dumps(reply_envelope, sort_keys=True)},
                maxlen=5000,
                approximate=True,
            )
        finally:
            await r.aclose()

    async def _process_envelope_payload(self, payload: dict[str, Any]) -> None:
        msg_type = str(payload.get("type") or "")
        req_run_id = payload.get("run_id")
        reply_to = payload.get("reply_stream")
        if not isinstance(req_run_id, str) or not req_run_id or not isinstance(reply_to, str):
            logger.warning("Dropping malformed Beast bridge payload missing run_id/reply_stream")
            return

        timeout_s = float(payload.get("timeout_seconds") or 300.0)
        try:
            if msg_type == "HARNESS_TASK_REQUEST":
                repo_raw = payload.get("repo_ref")
                repo_ref = VerifiedRepoRef.model_validate(repo_raw) if repo_raw else None
                posture = ExecutionPosture.from_shorthand(
                    str(payload.get("posture") or ExecutionPosture.OBSERVE_ONLY.value)
                )
                indep = bool(payload.get("independent_review", False))
                facets = dict(payload.get("facets") or {})
                started = await self._driver.start_run(
                    endpoint=self._wrapped_endpoint,
                    prompt=str(payload.get("prompt") or ""),
                    repo_ref=repo_ref,
                    posture=posture,
                    facets=facets,
                    independent_review=indep,
                )
                now = datetime.now(UTC)
                session = SessionRecord(
                    session_id=f"bridge-{req_run_id}",
                    endpoint_id=self.agent_id,
                    driver_id=self._driver.driver_id,
                    native_agent_id=started.native_agent_id,
                    latest_run_id=started.run_id,
                    posture=posture,
                    independent_review=indep,
                    facets=facets,
                    status=started.state,
                    answer=started.answer,
                    structured_output=started.structured_output,
                    audit_events=started.audit_events,
                    created_at=now,
                    updated_at=now,
                )
                final = (
                    started
                    if started.state.is_terminal
                    else await self._driver.wait_run(session=session, timeout_seconds=timeout_s)
                )
                session = session.model_copy(
                    update={
                        "latest_run_id": final.run_id,
                        "status": final.state,
                        "answer": final.answer,
                        "structured_output": final.structured_output,
                        "audit_events": final.audit_events,
                    }
                )
                self._sessions_by_run_id[req_run_id] = session

            elif msg_type == "HARNESS_TASK_FOLLOWUP":
                prev_run_id = str(payload.get("previous_run_id") or "")
                prior = self._sessions_by_run_id.get(prev_run_id)
                if prior is None:
                    raise ValueError(f"Unknown previous_run_id={prev_run_id!r} on bridge.")
                if prior.independent_review:
                    raise PostureViolationError(
                        "Independent review sessions on Beast bridge forbid follow-ups."
                    )
                followed = await self._driver.send_followup(
                    session=prior,
                    prompt=str(payload.get("prompt") or ""),
                    facets=dict(payload.get("facets") or {}),
                )
                temp_session = prior.model_copy(update={"latest_run_id": followed.run_id})
                final = (
                    followed
                    if followed.state.is_terminal
                    else await self._driver.wait_run(
                        session=temp_session,
                        timeout_seconds=timeout_s,
                    )
                )
                updated = temp_session.model_copy(
                    update={
                        "status": final.state,
                        "answer": final.answer,
                        "structured_output": final.structured_output,
                        "audit_events": final.audit_events,
                    }
                )
                self._sessions_by_run_id[req_run_id] = updated

            elif msg_type == "HARNESS_TASK_CANCEL":
                prior = self._sessions_by_run_id.get(req_run_id)
                if prior is not None:
                    final = await self._driver.cancel_run(session=prior)
                else:
                    final = None
                result_payload = {
                    "type": "HARNESS_TASK_RESULT",
                    "run_id": req_run_id,
                    "native_agent_id": final.native_agent_id if final else self.agent_id,
                    "status": RunState.CANCELLED.value,
                }
                await self._publish_result(reply_to, result_payload)
                return

            else:
                return

            result_payload = {
                "type": "HARNESS_TASK_RESULT",
                "run_id": req_run_id,
                "native_agent_id": final.native_agent_id,
                "status": final.state.value,
                "answer": final.answer,
                "structured_output": final.structured_output,
                "audit_events": final.audit_events,
                "error_detail": final.error_detail,
            }
        except Exception as exc:
            result_payload = {
                "type": "HARNESS_TASK_RESULT",
                "run_id": req_run_id,
                "native_agent_id": self.agent_id,
                "status": RunState.ERROR.value,
                "error_detail": str(exc),
            }

        await self._publish_result(reply_to, result_payload)

    async def _consume_stream_inbox(self) -> None:
        r = self._redis()
        try:
            while True:
                entries = await r.xreadgroup(
                    groupname=_CONSUMER_GROUP,
                    consumername=self._consumer_name,
                    streams={self.inbox_stream: ">"},
                    count=10,
                    block=200,
                )
                for _sname, messages in entries:
                    for msg_id, fields in messages:
                        await r.xack(self.inbox_stream, _CONSUMER_GROUP, msg_id)
                        raw_env = fields.get("envelope")
                        if not raw_env:
                            continue
                        try:
                            env = json.loads(raw_env)
                        except json.JSONDecodeError:
                            logger.warning("Dropping malformed JSON envelope %s", msg_id)
                            continue
                        payload = env.get("payload") if isinstance(env, dict) else None
                        if not isinstance(payload, dict):
                            continue
                        task = asyncio.create_task(self._process_envelope_payload(payload))
                        self._inflight_tasks.add(task)
                        task.add_done_callback(self._inflight_tasks.discard)
        except asyncio.CancelledError:
            raise
        finally:
            await r.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a BeastHarnessBridgeAgent on Beast Network.")
    parser.add_argument("--agent-id", default="bridge-cursor-cloud")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-db", type=int, default=0)
    args = parser.parse_args()

    from beast_mailbox_core import MailboxConfig

    from beast_harness.capability_graph import parse_and_validate_capability_ttl
    from beast_harness.drivers.cursor_driver import CursorCloudDriver
    from beast_harness.secrets import ChainedSecretProvider, EnvSecretProvider

    drv = CursorCloudDriver(secrets=ChainedSecretProvider([EnvSecretProvider()]))
    ep = parse_and_validate_capability_ttl(drv.announce_capabilities_ttl()[0])[0]
    mb_cfg = MailboxConfig(host=args.redis_host, port=args.redis_port, db=args.redis_db)

    async def _serve() -> None:
        agent = BeastHarnessBridgeAgent(
            agent_id=args.agent_id,
            driver=drv,
            endpoint=ep,
            mailbox_config=mb_cfg,
        )
        await agent.startup()
        try:
            await asyncio.Event().wait()
        finally:
            await agent.shutdown()

    asyncio.run(_serve())
