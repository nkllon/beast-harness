"""Bridge agent hosting any HarnessDriver on the Beast Network via BaseAgent."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
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
    RunState,
    SessionRecord,
    VerifiedRepoRef,
)

BH = Namespace(BH_NS)


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
        )
        self._driver = driver
        self._wrapped_endpoint = endpoint
        self._inbox_task: asyncio.Task[None] | None = None

    async def on_startup(self) -> None:
        self._inbox_task = asyncio.create_task(self._consume_stream_inbox())

    async def on_shutdown(self) -> None:
        if self._inbox_task is not None:
            self._inbox_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._inbox_task
            self._inbox_task = None

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

    async def _handle_request_payload(self, r: Any, payload: dict[str, Any]) -> None:
        req_run_id = str(payload["run_id"])
        reply_to = str(payload["reply_stream"])
        try:
            repo_raw = payload.get("repo_ref")
            repo_ref = VerifiedRepoRef.model_validate(repo_raw) if repo_raw else None
            posture = ExecutionPosture.from_shorthand(
                str(payload.get("posture") or ExecutionPosture.OBSERVE_ONLY.value)
            )
            started = await self._driver.start_run(
                endpoint=self._wrapped_endpoint,
                prompt=str(payload["prompt"]),
                repo_ref=repo_ref,
                posture=posture,
                facets=dict(payload.get("facets") or {}),
                independent_review=bool(payload.get("independent_review", False)),
            )
            now = datetime.now(UTC)
            temp_session = SessionRecord(
                session_id=f"bridge-{req_run_id}",
                endpoint_id=self.agent_id,
                driver_id=self._driver.driver_id,
                native_agent_id=started.native_agent_id,
                latest_run_id=started.run_id,
                posture=posture,
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
                else await self._driver.wait_run(session=temp_session, timeout_seconds=120.0)
            )
            result_payload: dict[str, Any] = {
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

        reply_envelope = {
            "header": {
                "sender": self.agent_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "id": str(uuid.uuid4()),
            },
            "payload": result_payload,
        }
        await r.xadd(reply_to, {"envelope": json.dumps(reply_envelope, sort_keys=True)})

    async def _consume_stream_inbox(self) -> None:
        r = self._redis()
        inbox = f"beast:stream:agent:{self.agent_id}:inbox"
        cursor_id = "0-0"
        try:
            while True:
                entries = await r.xread({inbox: cursor_id}, count=10, block=200)
                for _sname, messages in entries:
                    for msg_id, fields in messages:
                        cursor_id = msg_id
                        raw_env = fields.get("envelope")
                        if not raw_env:
                            continue
                        env = json.loads(raw_env)
                        payload = env.get("payload") or {}
                        if payload.get("type") == "HARNESS_TASK_REQUEST":
                            await self._handle_request_payload(r, payload)
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
