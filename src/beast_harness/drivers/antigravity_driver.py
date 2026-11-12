"""Workspace-mounted, policy-hooked Google Antigravity SDK driver."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, create_model

from beast_harness.secrets import ChainedSecretProvider
from beast_harness.types import (
    BH_NS,
    DriverRunResult,
    EndpointDescriptor,
    ExecutionPosture,
    RunState,
    SessionRecord,
    VerifiedRepoRef,
    ensure_directory,
)


def _json_schema_to_pydantic_model(schema: dict[str, Any]) -> type[BaseModel]:
    title = str(schema.get("title") or "StructuredHarnessResponse")
    props = schema.get("properties", {})
    required_set = set(schema.get("required", []))
    type_map: dict[str, Any] = {
        "string": str,
        "boolean": bool,
        "integer": int,
        "number": float,
        "array": list[Any],
        "object": dict[str, Any],
    }
    field_defs: dict[str, Any] = {}
    for prop_name, prop_spec in props.items():
        raw_t = prop_spec.get("type", "string") if isinstance(prop_spec, dict) else "string"
        py_t = type_map.get(str(raw_t), Any)
        if prop_name in required_set:
            field_defs[prop_name] = (py_t, ...)
        else:
            field_defs[prop_name] = (py_t | None, None)
    if not field_defs:
        field_defs["result"] = (str, ...)
    return create_model(title, **field_defs)


class AntigravitySdkDriver:
    driver_id = "antigravity-sdk"

    def __init__(
        self,
        *,
        secrets: ChainedSecretProvider,
        state_root: Path,
        secret_key_name: str = "GEMINI_API_KEY",
    ) -> None:
        self._secrets = secrets
        self._state_root = Path(state_root)
        self._secret_key_name = secret_key_name
        self._runs: dict[str, DriverRunResult] = {}

    def announce_capabilities_ttl(self) -> list[str]:
        effort_schema = json.dumps(
            {
                "type": "string",
                "enum": ["minimal", "low", "medium", "high", "extra_high"],
            }
        )
        obj_schema = json.dumps({"type": "object"})
        str_schema = json.dumps({"type": "string", "minLength": 1})
        ttl = f"""\
@prefix bh: <{BH_NS}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<urn:beast:endpoint:antigravity-sdk> a bh:AgentEndpoint ;
    rdfs:label "Google Antigravity SDK Local Agent (Policy-Hooked)" ;
    bh:endpointId "antigravity-sdk" ;
    bh:driverId "antigravity-sdk" ;
    bh:trustTier bh:RuntimeSelfAnnounced ;
    bh:defaultPosture bh:ObserveOnly ;
    bh:supportsPosture bh:ObserveOnly, bh:ProposeDiff ;
    bh:hasCapability
        bh:LocalDetachedWorktree,
        bh:ReadOnlyPolicyHook,
        bh:AstRipgrepSearch,
        bh:StructuredOutputSchema,
        bh:ThinkingEffortControl,
        bh:FollowUpRunQueue,
        bh:RepoFreePrompt ;
    bh:hasModelProfile [
        a bh:ModelProfile ;
        bh:modelFamily "google" ;
        bh:modelIdentifier "gemini-2.5-pro" ;
        bh:contextWindowTokens 1000000
    ] ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability bh:ThinkingEffortControl ;
        bh:facetKey "thinking_effort" ;
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
    ] .
"""
        return [ttl]

    def build_local_agent_config(
        self,
        *,
        workspace_path: str | None,
        posture: ExecutionPosture,
        facets: dict[str, Any],
        conversation_id: str | None = None,
        save_dir: str | None = None,
    ) -> Any:
        from google import antigravity as agy
        from google.antigravity.hooks import policy as agy_policy

        api_key = self._secrets.get_secret(self._secret_key_name)
        workspaces = [workspace_path] if workspace_path else []

        if posture == ExecutionPosture.OBSERVE_ONLY:
            enabled_tools = [
                agy.BuiltinTools.LIST_DIR,
                agy.BuiltinTools.SEARCH_DIR,
                agy.BuiltinTools.FIND_FILE,
                agy.BuiltinTools.VIEW_FILE,
                agy.BuiltinTools.FINISH,
            ]
            policies = [
                agy_policy.deny_all(),
                *(agy_policy.allow(tool) for tool in enabled_tools),
            ]
        else:
            enabled_tools = [
                agy.BuiltinTools.LIST_DIR,
                agy.BuiltinTools.SEARCH_DIR,
                agy.BuiltinTools.FIND_FILE,
                agy.BuiltinTools.VIEW_FILE,
                agy.BuiltinTools.CREATE_FILE,
                agy.BuiltinTools.EDIT_FILE,
                agy.BuiltinTools.FINISH,
            ]
            policies = (
                [agy_policy.workspace_only(workspace_path)]
                if workspace_path
                else [agy_policy.safe_defaults()]
            )

        capabilities = agy.CapabilitiesConfig(
            enable_subagents=False,
            enabled_tools=enabled_tools,
        )

        response_schema: type[BaseModel] | None = None
        if "structured_output_schema" in facets and isinstance(
            facets["structured_output_schema"], dict
        ):
            response_schema = _json_schema_to_pydantic_model(facets["structured_output_schema"])

        model_name = str(facets.get("model") or "gemini-2.5-pro")
        models_list: list[Any] | None = None
        if "thinking_effort" in facets and api_key:
            level = agy.ThinkingLevel(str(facets["thinking_effort"]))
            models_list = [
                agy.ModelTarget(
                    name=model_name,
                    types=[agy.ModelType.TEXT],
                    endpoint=agy.GeminiAPIEndpoint(
                        api_key=api_key,
                        options=agy.GeminiModelOptions(thinking_level=level),
                    ),
                )
            ]

        cfg_kwargs: dict[str, Any] = {
            "capabilities": capabilities,
            "policies": policies,
            "workspaces": workspaces,
            "response_schema": response_schema,
        }
        if conversation_id is not None:
            cfg_kwargs["conversation_id"] = conversation_id
        if save_dir is not None:
            cfg_kwargs["save_dir"] = save_dir
        if models_list is not None:
            cfg_kwargs["models"] = models_list
        else:
            cfg_kwargs["model"] = model_name
            if api_key:
                cfg_kwargs["api_key"] = api_key

        return agy.LocalAgentConfig(**cfg_kwargs)

    async def _execute_agent_turn(
        self,
        *,
        conversation_id: str,
        workspace_path: str | None,
        posture: ExecutionPosture,
        facets: dict[str, Any],
        prompt: str,
    ) -> DriverRunResult:
        from google import antigravity as agy

        self._secrets.require_secret(self._secret_key_name)
        save_dir = ensure_directory(self._state_root / conversation_id)
        cfg = self.build_local_agent_config(
            workspace_path=workspace_path,
            posture=posture,
            facets=facets,
            conversation_id=conversation_id,
            save_dir=str(save_dir),
        )

        def _sync_run() -> tuple[str, dict[str, Any] | None]:
            agent = agy.Agent(config=cfg)
            resp = agent.run(prompt) if hasattr(agent, "run") else agent.query(prompt)
            text = str(getattr(resp, "text", None) or resp)
            structured: dict[str, Any] | None = None
            parsed = getattr(resp, "parsed", None)
            if isinstance(parsed, BaseModel):
                structured = parsed.model_dump()
            elif isinstance(parsed, dict):
                structured = parsed
            return text, structured

        run_id = f"{conversation_id}-run-{uuid.uuid4().hex[:6]}"
        answer, structured = await asyncio.to_thread(_sync_run)
        result = DriverRunResult(
            native_agent_id=conversation_id,
            run_id=run_id,
            state=RunState.FINISHED,
            answer=answer,
            structured_output=structured,
        )
        self._runs[run_id] = result
        return result

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
        conv_id = f"agy-{uuid.uuid4().hex[:12]}"
        ws_path = repo_ref.local_worktree_path if repo_ref is not None else None
        return await self._execute_agent_turn(
            conversation_id=conv_id,
            workspace_path=ws_path,
            posture=posture,
            facets=facets,
            prompt=prompt,
        )

    async def send_followup(
        self,
        *,
        session: SessionRecord,
        prompt: str,
        facets: dict[str, Any],
    ) -> DriverRunResult:
        merged_facets = {**session.facets, **facets}
        return await self._execute_agent_turn(
            conversation_id=session.native_agent_id,
            workspace_path=session.worktree_path,
            posture=session.posture,
            facets=merged_facets,
            prompt=prompt,
        )

    async def poll_run(self, *, session: SessionRecord) -> DriverRunResult:
        if session.latest_run_id in self._runs:
            return self._runs[session.latest_run_id]
        return DriverRunResult(
            native_agent_id=session.native_agent_id,
            run_id=session.latest_run_id,
            state=session.status,
            answer=session.answer,
            structured_output=session.structured_output,
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
