"""Core HarnessSubstrate orchestrating discovery, ref verification, sessions, and drivers."""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from beast_harness.capability_graph import (
    match_and_rank_endpoints,
    parse_and_validate_capability_ttl,
    validate_facets_for_endpoint,
)
from beast_harness.drivers.antigravity_driver import AntigravitySdkDriver
from beast_harness.drivers.beast_driver import BeastPeerDriver
from beast_harness.drivers.cursor_driver import CursorCloudDriver, CursorLocalDriver
from beast_harness.ledger import SessionLedger
from beast_harness.ref_verifier import PublishedRefVerifier
from beast_harness.secrets import ChainedSecretProvider, EnvSecretProvider
from beast_harness.types import (
    BH_NS,
    CapabilityValidationError,
    DiscoveryQuery,
    EndpointDescriptor,
    ExecutionPosture,
    HarnessDriver,
    HarnessError,
    NoMatchingEndpointError,
    RefVerificationError,
    SessionRecord,
    TrustTier,
    VerifiedRepoRef,
    ensure_directory,
)

logger = logging.getLogger(__name__)

_REPO_MOUNT_CAPS = {
    f"{BH_NS}RemoteGitShaMount",
    f"{BH_NS}LocalDetachedWorktree",
}


def _supports_repo_mount(ep: EndpointDescriptor) -> bool:
    return bool(_REPO_MOUNT_CAPS.intersection(ep.capabilities))


class HarnessSubstrate:
    def __init__(
        self,
        *,
        ledger: SessionLedger,
        ref_verifier: PublishedRefVerifier,
        drivers: Sequence[HarnessDriver],
        beast_peer_driver: BeastPeerDriver | None = None,
        admitted_capability_digests: dict[str, str] | None = None,
        caller_inbox_notifier: Any | None = None,
    ) -> None:
        self._ledger = ledger
        self._ref_verifier = ref_verifier
        self._drivers_by_id: dict[str, HarnessDriver] = {d.driver_id: d for d in drivers}
        self._beast_peer_driver = beast_peer_driver
        if beast_peer_driver is not None:
            self._drivers_by_id[beast_peer_driver.driver_id] = beast_peer_driver
        self._admitted_digests = dict(admitted_capability_digests or {})
        self._caller_inbox_notifier = caller_inbox_notifier

    async def _all_endpoints(self) -> list[EndpointDescriptor]:
        endpoints_by_id: dict[str, EndpointDescriptor] = {}
        local_endpoint_ids: set[str] = set()

        for drv in self._drivers_by_id.values():
            if drv is self._beast_peer_driver:
                continue
            drv_alive = drv.is_available()
            for raw_ttl in drv.announce_capabilities_ttl():
                for ep in parse_and_validate_capability_ttl(raw_ttl, alive=drv_alive):
                    normalized = self._normalize_trust_tier(ep)
                    endpoints_by_id[normalized.endpoint_id] = normalized
                    local_endpoint_ids.add(normalized.endpoint_id)

        if self._beast_peer_driver is not None:
            peer_eps = await self._beast_peer_driver.discover_live_peer_endpoints()
            for ep in peer_eps:
                if ep.endpoint_id in local_endpoint_ids:
                    logger.warning(
                        "Ignoring remote Beast peer attempting to shadow registered endpoint %r",
                        ep.endpoint_id,
                    )
                    continue
                normalized = self._normalize_trust_tier(ep)
                endpoints_by_id[normalized.endpoint_id] = normalized

        return sorted(endpoints_by_id.values(), key=lambda e: e.endpoint_id)

    def _normalize_trust_tier(self, ep: EndpointDescriptor) -> EndpointDescriptor:
        expected_digest = self._admitted_digests.get(ep.endpoint_id)
        tier = (
            TrustTier.NEO4J_ADMITTED
            if (expected_digest and expected_digest == ep.capability_graph_digest)
            else TrustTier.RUNTIME_SELF_ANNOUNCED
        )
        return ep.model_copy(update={"trust_tier": tier})

    def _resolve_driver(self, driver_id: str) -> HarnessDriver:
        drv = self._drivers_by_id.get(driver_id)
        if drv is None:
            raise HarnessError(f"No driver registered for driver_id={driver_id!r}.")
        return drv

    async def discover(self, query: DiscoveryQuery | None = None) -> list[EndpointDescriptor]:
        all_eps = await self._all_endpoints()
        effective_query = query if query is not None else DiscoveryQuery()
        return match_and_rank_endpoints(all_eps, effective_query)

    async def start(
        self,
        *,
        prompt: str,
        repo_url: str | None = None,
        ref: str = "HEAD",
        base_ref: str | None = None,
        target_endpoint_id: str | None = None,
        selection: DiscoveryQuery | None = None,
        posture: ExecutionPosture | str = ExecutionPosture.OBSERVE_ONLY,
        facets: dict[str, Any] | None = None,
        independent_review: bool = False,
    ) -> SessionRecord:
        if not prompt or not prompt.strip():
            raise CapabilityValidationError("prompt must be non-empty.")

        resolved_posture = (
            posture
            if isinstance(posture, ExecutionPosture)
            else ExecutionPosture.from_shorthand(posture)
        )
        if independent_review and resolved_posture != ExecutionPosture.OBSERVE_ONLY:
            raise CapabilityValidationError(
                "independent_review=True requires ObserveOnly execution posture."
            )

        facets_dict = dict(facets or {})
        effective_query = selection or DiscoveryQuery(required_posture=resolved_posture)
        if effective_query.required_posture != resolved_posture:
            effective_query = effective_query.model_copy(
                update={"required_posture": resolved_posture}
            )

        all_eps = await self._all_endpoints()
        candidate_pool = (
            [ep for ep in all_eps if _supports_repo_mount(ep)] if repo_url is not None else all_eps
        )

        if target_endpoint_id is not None:
            matched = [ep for ep in all_eps if ep.endpoint_id == target_endpoint_id]
            if not matched:
                raise NoMatchingEndpointError(
                    f"Endpoint {target_endpoint_id!r} was not found among "
                    "live/registered endpoints."
                )
            endpoint = matched[0]
            if repo_url is not None and not _supports_repo_mount(endpoint):
                raise CapabilityValidationError(
                    f"Endpoint {endpoint.endpoint_id!r} does not support repository mounting "
                    "(RemoteGitShaMount or LocalDetachedWorktree)."
                )
            if not match_and_rank_endpoints([endpoint], effective_query):
                raise NoMatchingEndpointError(
                    f"Target endpoint {target_endpoint_id!r} violates selection constraints: "
                    f"{effective_query.model_dump()}"
                )
        else:
            ranked = match_and_rank_endpoints(candidate_pool, effective_query)
            if not ranked:
                raise NoMatchingEndpointError(
                    f"No endpoint satisfied discovery constraints: {effective_query.model_dump()}"
                )
            endpoint = ranked[0]

        validate_facets_for_endpoint(endpoint, facets_dict, resolved_posture)

        if "extra_repos" in facets_dict and isinstance(facets_dict["extra_repos"], list):
            verified_extras: list[dict[str, str]] = []
            for extra in facets_dict["extra_repos"]:
                v_extra = self._ref_verifier.verify_published_ref(
                    str(extra["url"]),
                    str(extra["ref"]),
                )
                verified_extras.append({"url": v_extra.repo_url, "ref": v_extra.commit_sha})
            facets_dict["extra_repos"] = verified_extras

        session_id = f"bh-{uuid.uuid4().hex[:12]}"
        verified_ref: VerifiedRepoRef | None = None
        if repo_url is not None:
            verified_ref = self._ref_verifier.verify_published_ref(
                repo_url,
                ref,
                base_ref=base_ref,
            )
            needs_local_wt = (
                f"{BH_NS}LocalDetachedWorktree" in endpoint.capabilities
                and f"{BH_NS}RemoteGitShaMount" not in endpoint.capabilities
            )
            if needs_local_wt:
                writable_wt = (
                    resolved_posture == ExecutionPosture.PROPOSE_DIFF and not independent_review
                )
                verified_ref = self._ref_verifier.materialize_worktree(
                    verified_ref,
                    session_id=session_id,
                    writable=writable_wt,
                )
        else:
            supports_repo_free = f"{BH_NS}RepoFreePrompt" in endpoint.capabilities or bool(
                endpoint.legacy_capability_tags
            )
            if not supports_repo_free:
                raise CapabilityValidationError(
                    f"Endpoint {endpoint.endpoint_id!r} requires a published repo_url/ref."
                )

        driver = self._resolve_driver(endpoint.driver_id)
        try:
            initial = await driver.start_run(
                endpoint=endpoint,
                prompt=prompt,
                repo_ref=verified_ref,
                posture=resolved_posture,
                facets=facets_dict,
                independent_review=independent_review,
            )
        except Exception:
            if verified_ref and verified_ref.local_worktree_path:
                self._ref_verifier.cleanup_worktree(verified_ref.local_worktree_path)
            raise

        record = self._ledger.create_session(
            session_id=session_id,
            endpoint_id=endpoint.endpoint_id,
            driver_id=endpoint.driver_id,
            repo_ref=verified_ref,
            posture=resolved_posture,
            independent_review=independent_review,
            facets=facets_dict,
            initial_run=initial,
        )
        if record.status.is_terminal:
            should_release_wt = (
                independent_review or f"{BH_NS}FollowUpRunQueue" not in endpoint.capabilities
            )
            record = await self._finalize_terminal_session(
                record,
                initial,
                release_worktree=should_release_wt,
            )
        return record

    async def _should_release_worktree_on_turn_end(self, session: SessionRecord) -> bool:
        if session.independent_review:
            return True
        all_eps = await self._all_endpoints()
        matched = [ep for ep in all_eps if ep.endpoint_id == session.endpoint_id]
        if not matched:
            return True
        return f"{BH_NS}FollowUpRunQueue" not in matched[0].capabilities

    async def _finalize_terminal_session(
        self,
        session: SessionRecord,
        driver_result: Any,
        *,
        release_worktree: bool = True,
    ) -> SessionRecord:
        clear_wt = False
        if release_worktree and session.worktree_path:
            self._ref_verifier.cleanup_worktree(session.worktree_path)
            clear_wt = True
        updated = self._ledger.update_from_driver_result(
            session.session_id,
            driver_result,
            clear_worktree=clear_wt,
        )
        if self._caller_inbox_notifier is not None:
            await self._caller_inbox_notifier(updated)
        return updated

    async def send(
        self,
        *,
        session_id: str,
        prompt: str,
        facets: dict[str, Any] | None = None,
    ) -> SessionRecord:
        session = self._ledger.assert_followup_allowed(session_id)
        facets_dict = dict(facets or {})

        all_eps = await self._all_endpoints()
        matching_ep = [ep for ep in all_eps if ep.endpoint_id == session.endpoint_id]
        if matching_ep and facets_dict:
            validate_facets_for_endpoint(matching_ep[0], facets_dict, session.posture)
        if "extra_repos" in facets_dict and isinstance(facets_dict["extra_repos"], list):
            verified_extras: list[dict[str, str]] = []
            for extra in facets_dict["extra_repos"]:
                v_extra = self._ref_verifier.verify_published_ref(
                    str(extra["url"]),
                    str(extra["ref"]),
                )
                verified_extras.append({"url": v_extra.repo_url, "ref": v_extra.commit_sha})
            facets_dict["extra_repos"] = verified_extras

        driver = self._resolve_driver(session.driver_id)
        res = await driver.send_followup(session=session, prompt=prompt, facets=facets_dict)
        if res.state.is_terminal:
            release = await self._should_release_worktree_on_turn_end(session)
            return await self._finalize_terminal_session(
                session,
                res,
                release_worktree=release,
            )
        return self._ledger.update_from_driver_result(session_id, res)

    async def status(self, session_id: str) -> SessionRecord:
        session = self._ledger.get_session(session_id)
        if session.status.is_terminal:
            return session
        driver = self._resolve_driver(session.driver_id)
        polled = await driver.poll_run(session=session)
        if polled.state.is_terminal:
            release = await self._should_release_worktree_on_turn_end(session)
            return await self._finalize_terminal_session(
                session,
                polled,
                release_worktree=release,
            )
        return self._ledger.update_from_driver_result(session_id, polled)

    async def wait(self, session_id: str, *, timeout_seconds: float = 120.0) -> SessionRecord:
        session = self._ledger.get_session(session_id)
        if session.status.is_terminal:
            return session
        driver = self._resolve_driver(session.driver_id)
        waited = await driver.wait_run(session=session, timeout_seconds=timeout_seconds)
        if waited.state.is_terminal:
            release = await self._should_release_worktree_on_turn_end(session)
            return await self._finalize_terminal_session(
                session,
                waited,
                release_worktree=release,
            )
        return self._ledger.update_from_driver_result(session_id, waited)

    async def result(self, session_id: str) -> SessionRecord:
        return await self.status(session_id)

    async def cancel(self, session_id: str) -> SessionRecord:
        session = self._ledger.get_session(session_id)
        if session.status.is_terminal:
            if session.worktree_path:
                self._ref_verifier.cleanup_worktree(session.worktree_path)
            return session
        driver = self._resolve_driver(session.driver_id)
        cancelled = await driver.cancel_run(session=session)
        return await self._finalize_terminal_session(
            session,
            cancelled,
            release_worktree=True,
        )

    async def review(
        self,
        *,
        prompt: str,
        repo_url: str,
        ref: str = "HEAD",
        base_ref: str | None = None,
        target_endpoint_id: str | None = None,
        required_capabilities: Sequence[str] = (),
        preferred_capabilities: Sequence[str] = (),
        exclude_model_families: Sequence[str] = (),
        facets: dict[str, Any] | None = None,
        timeout_seconds: float = 300.0,
    ) -> SessionRecord:
        if not repo_url or not repo_url.strip():
            raise RefVerificationError(
                "Independent review requires a non-empty published repo_url."
            )

        selection = DiscoveryQuery(
            required_capabilities=tuple(required_capabilities),
            preferred_capabilities=tuple(preferred_capabilities),
            exclude_model_families=tuple(exclude_model_families),
            required_posture=ExecutionPosture.OBSERVE_ONLY,
            only_alive=True,
        )
        started = await self.start(
            prompt=prompt,
            repo_url=repo_url,
            ref=ref,
            base_ref=base_ref,
            target_endpoint_id=target_endpoint_id,
            selection=selection,
            posture=ExecutionPosture.OBSERVE_ONLY,
            facets=facets,
            independent_review=True,
        )
        return await self.wait(started.session_id, timeout_seconds=timeout_seconds)


def _default_beast_peer_driver(secrets: ChainedSecretProvider) -> BeastPeerDriver | None:
    redis_url = os.environ.get("REDIS_URL")
    redis_host = os.environ.get("BEAST_REDIS_HOST")
    if not redis_url and not redis_host:
        return None

    host = redis_host or "localhost"
    port = int(os.environ.get("BEAST_REDIS_PORT", "6379"))
    db = int(os.environ.get("BEAST_REDIS_DB", "0"))
    if redis_url and not redis_host:
        parsed = urlsplit(redis_url)
        if parsed.hostname:
            host = parsed.hostname
        if parsed.port:
            port = parsed.port
        if parsed.path and parsed.path.strip("/").isdigit():
            db = int(parsed.path.strip("/"))

    caller_id = os.environ.get("BEAST_CALLER_AGENT_ID", "beast-harness-mcp")
    candidate = BeastPeerDriver(
        caller_agent_id=caller_id,
        redis_host=host,
        redis_port=port,
        redis_db=db,
        secrets=secrets,
    )
    return candidate if candidate.is_available() else None


def create_default_substrate(state_dir: Path | None = None) -> HarnessSubstrate:
    root = ensure_directory(
        Path(state_dir) if state_dir is not None else Path.home() / ".beast-harness"
    )
    secrets = ChainedSecretProvider([EnvSecretProvider()])
    drivers: list[HarnessDriver] = [
        CursorCloudDriver(secrets=secrets),
        CursorLocalDriver(secrets=secrets),
        AntigravitySdkDriver(secrets=secrets, state_root=root / "antigravity"),
    ]
    return HarnessSubstrate(
        ledger=SessionLedger(root / "sessions.sqlite3"),
        ref_verifier=PublishedRefVerifier(root / "worktrees"),
        drivers=drivers,
        beast_peer_driver=_default_beast_peer_driver(secrets),
    )
