"""Versioned core types, errors, and driver/secret protocols for beast-harness."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.1.0"
BH_NS = "https://nkllon.com/ontology/beast-harness#"


class HarnessError(RuntimeError):
    """Base fail-closed error for beast-harness."""


class CapabilityValidationError(HarnessError):
    """Raised when an announced Turtle capability graph or facet input fails SHACL/schema checks."""


class RefVerificationError(HarnessError):
    """Raised when a repository reference is unpushed, malformed, or unreachable."""


class PostureViolationError(HarnessError):
    """Raised when a mutating posture or follow-up is requested without authorization/support."""


class SecretResolutionError(HarnessError):
    """Raised when a required provider credential cannot be resolved."""


class NoMatchingEndpointError(HarnessError):
    """Raised when no registered or live endpoint satisfies discovery constraints."""


class TrustTier(StrEnum):
    RUNTIME_SELF_ANNOUNCED = f"{BH_NS}RuntimeSelfAnnounced"
    NEO4J_ADMITTED = f"{BH_NS}Neo4jAdmitted"

    @classmethod
    def from_shorthand(cls, value: str) -> TrustTier:
        mapping = {
            "tier1": cls.RUNTIME_SELF_ANNOUNCED,
            "runtime": cls.RUNTIME_SELF_ANNOUNCED,
            "runtime_self_announced": cls.RUNTIME_SELF_ANNOUNCED,
            cls.RUNTIME_SELF_ANNOUNCED.value: cls.RUNTIME_SELF_ANNOUNCED,
            "tier2": cls.NEO4J_ADMITTED,
            "neo4j": cls.NEO4J_ADMITTED,
            "neo4j_admitted": cls.NEO4J_ADMITTED,
            cls.NEO4J_ADMITTED.value: cls.NEO4J_ADMITTED,
        }
        key = value.strip()
        if key in mapping:
            return mapping[key]
        lowered = key.lower()
        if lowered in mapping:
            return mapping[lowered]
        raise CapabilityValidationError(f"Unsupported trust tier: {value!r}")


class ExecutionPosture(StrEnum):
    OBSERVE_ONLY = f"{BH_NS}ObserveOnly"
    PROPOSE_DIFF = f"{BH_NS}ProposeDiff"
    ISOLATED_BRANCH_WRITE = f"{BH_NS}IsolatedBranchWrite"

    @classmethod
    def from_shorthand(cls, value: str) -> ExecutionPosture:
        mapping = {
            "observe": cls.OBSERVE_ONLY,
            "observe_only": cls.OBSERVE_ONLY,
            cls.OBSERVE_ONLY.value: cls.OBSERVE_ONLY,
            "propose_diff": cls.PROPOSE_DIFF,
            cls.PROPOSE_DIFF.value: cls.PROPOSE_DIFF,
            "isolated_branch_write": cls.ISOLATED_BRANCH_WRITE,
            cls.ISOLATED_BRANCH_WRITE.value: cls.ISOLATED_BRANCH_WRITE,
        }
        key = value.strip()
        if key in mapping:
            return mapping[key]
        lowered = key.lower()
        if lowered in mapping:
            return mapping[lowered]
        raise PostureViolationError(f"Unsupported execution posture: {value!r}")


class RunState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    BUSY = "busy"
    FINISHED = "finished"
    ERROR = "error"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {self.FINISHED, self.ERROR, self.CANCELLED}


class VerifiedRepoRef(BaseModel):
    """Immutable, remote-verified repository reference."""

    model_config = ConfigDict(frozen=True)

    repo_url: str
    requested_ref: str
    commit_sha: str
    base_sha: str | None = None
    local_worktree_path: str | None = None


class ModelProfileDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_family: str
    model_identifier: str
    context_window_tokens: int | None = None


class CapabilityFacetDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True)

    capability_iri: str
    facet_key: str
    input_json_schema: dict[str, Any]
    output_json_schema: dict[str, Any] | None = None


class EndpointDescriptor(BaseModel):
    """Parsed, SHACL-validated description of a driver endpoint or live Beast peer."""

    model_config = ConfigDict(frozen=True)

    endpoint_iri: str
    endpoint_id: str
    driver_id: str
    label: str
    trust_tier: TrustTier
    default_posture: ExecutionPosture
    supported_postures: tuple[ExecutionPosture, ...]
    capabilities: tuple[str, ...] = ()
    legacy_capability_tags: tuple[str, ...] = ()
    models: tuple[ModelProfileDescriptor, ...] = ()
    facets: dict[str, CapabilityFacetDescriptor] = Field(default_factory=dict)
    alive: bool = True
    capability_graph_digest: str = ""


class DiscoveryQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    required_capabilities: tuple[str, ...] = ()
    preferred_capabilities: tuple[str, ...] = ()
    exclude_model_families: tuple[str, ...] = ()
    min_trust_tier: TrustTier = TrustTier.RUNTIME_SELF_ANNOUNCED
    required_posture: ExecutionPosture = ExecutionPosture.OBSERVE_ONLY
    only_alive: bool = True


class SessionRecord(BaseModel):
    session_id: str
    endpoint_id: str
    driver_id: str
    native_agent_id: str
    latest_run_id: str
    repo_url: str | None = None
    commit_sha: str | None = None
    base_sha: str | None = None
    worktree_path: str | None = None
    posture: ExecutionPosture = ExecutionPosture.OBSERVE_ONLY
    independent_review: bool = False
    facets: dict[str, Any] = Field(default_factory=dict)
    status: RunState = RunState.QUEUED
    answer: str | None = None
    structured_output: dict[str, Any] | None = None
    audit_events: list[dict[str, Any]] = Field(default_factory=list)
    audit_digest: str | None = None
    error_detail: str | None = None
    created_at: datetime
    updated_at: datetime


class DriverRunResult(BaseModel):
    native_agent_id: str
    run_id: str
    state: RunState
    answer: str | None = None
    structured_output: dict[str, Any] | None = None
    audit_events: list[dict[str, Any]] = Field(default_factory=list)
    error_detail: str | None = None


@runtime_checkable
class SecretProvider(Protocol):
    """Pluggable secret resolution boundary (never exposes values via repr or tool args)."""

    def has_secret(self, name: str) -> bool: ...

    def get_secret(self, name: str) -> str | None: ...


@runtime_checkable
class HarnessDriver(Protocol):
    """Pluggable non-LCD execution driver protocol."""

    @property
    def driver_id(self) -> str: ...

    def is_available(self) -> bool: ...

    def announce_capabilities_ttl(self) -> list[str]: ...

    async def start_run(
        self,
        *,
        endpoint: EndpointDescriptor,
        prompt: str,
        repo_ref: VerifiedRepoRef | None,
        posture: ExecutionPosture,
        facets: dict[str, Any],
        independent_review: bool,
    ) -> DriverRunResult: ...

    async def send_followup(
        self,
        *,
        session: SessionRecord,
        prompt: str,
        facets: dict[str, Any],
    ) -> DriverRunResult: ...

    async def poll_run(self, *, session: SessionRecord) -> DriverRunResult: ...

    async def wait_run(
        self,
        *,
        session: SessionRecord,
        timeout_seconds: float,
    ) -> DriverRunResult: ...

    async def cancel_run(self, *, session: SessionRecord) -> DriverRunResult: ...


def normalize_capability_iri(token: str) -> str:
    cleaned = token.strip()
    if cleaned.startswith(("http://", "https://", "urn:")):
        return cleaned
    if cleaned.startswith("bh:"):
        return f"{BH_NS}{cleaned[3:]}"
    return f"{BH_NS}{cleaned}"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
