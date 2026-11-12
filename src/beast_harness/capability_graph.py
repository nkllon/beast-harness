"""RDF/SHACL capability graph validation, legacy lifting, facet checking, and routing."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import pyshacl
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.compare import to_canonical_graph
from rdflib.namespace import RDF, RDFS

from beast_harness.ontology import read_capability_ontology_ttl, read_capability_shapes_ttl
from beast_harness.types import (
    BH_NS,
    CapabilityFacetDescriptor,
    CapabilityValidationError,
    CostTier,
    DiscoveryQuery,
    EndpointDescriptor,
    ExecutionPosture,
    LatencyClass,
    ModelProfileDescriptor,
    PostureViolationError,
    RankingProfile,
    TrustTier,
    normalize_capability_iri,
)

BH = Namespace(BH_NS)

_TRUST_RANK: dict[TrustTier, int] = {
    TrustTier.RUNTIME_SELF_ANNOUNCED: 1,
    TrustTier.NEO4J_ADMITTED: 2,
}

_COST_RANK: dict[CostTier, int] = {
    CostTier.LOW: 1,
    CostTier.MEDIUM: 2,
    CostTier.HIGH: 3,
}

_LATENCY_RANK: dict[LatencyClass, int] = {
    LatencyClass.INTERACTIVE: 1,
    LatencyClass.BATCH: 2,
    LatencyClass.CLOUD_CONTAINER: 3,
}

_VALID_JSON_TYPES = {"string", "boolean", "integer", "number", "array", "object"}
_ALLOWED_SCHEMA_KEYS = {
    "$schema",
    "title",
    "description",
    "default",
    "type",
    "enum",
    "const",
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "items",
    "properties",
    "required",
    "additionalProperties",
}


def _canonical_graph_sha256(graph: Graph) -> str:
    try:
        canonical = to_canonical_graph(graph)
        nt_lines = sorted(
            line.strip() for line in canonical.serialize(format="nt").splitlines() if line.strip()
        )
        payload = "\n".join(nt_lines).encode("utf-8")
    except Exception:
        nt_lines = sorted(
            line.strip() for line in graph.serialize(format="nt").splitlines() if line.strip()
        )
        payload = "\n".join(nt_lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_strict_int(val: Any) -> bool:
    return not isinstance(val, bool) and isinstance(val, int)


def _is_strict_num(val: Any) -> bool:
    return not isinstance(val, bool) and isinstance(val, (int, float))


def _validate_schema_definition(schema: Any, path: str) -> None:
    if not isinstance(schema, dict):
        raise CapabilityValidationError(f"{path}: JSON Schema definition must be an object.")
    unknown_keys = set(schema.keys()) - _ALLOWED_SCHEMA_KEYS
    if unknown_keys:
        raise CapabilityValidationError(
            f"{path}: unsupported JSON Schema keywords {sorted(unknown_keys)!r} "
            "(fail-closed contract validation)."
        )
    if "type" in schema and schema["type"] not in _VALID_JSON_TYPES:
        raise CapabilityValidationError(f"{path}: unsupported JSON Schema type {schema['type']!r}.")
    for int_kw in ("minLength", "maxLength", "minItems", "maxItems"):
        if int_kw in schema and (not _is_strict_int(schema[int_kw]) or schema[int_kw] < 0):
            raise CapabilityValidationError(f"{path}.{int_kw} must be a non-negative integer.")
    for num_kw in ("minimum", "maximum"):
        if num_kw in schema and not _is_strict_num(schema[num_kw]):
            raise CapabilityValidationError(f"{path}.{num_kw} must be numeric.")
    if "pattern" in schema:
        if not isinstance(schema["pattern"], str):
            raise CapabilityValidationError(f"{path}.pattern must be a string regex.")
        try:
            re.compile(schema["pattern"])
        except re.error as exc:
            raise CapabilityValidationError(f"{path}.pattern invalid regex: {exc}") from exc
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise CapabilityValidationError(f"{path}.enum must be a non-empty array.")
    if "items" in schema:
        _validate_schema_definition(schema["items"], f"{path}.items")
    if "required" in schema:
        if not isinstance(schema["required"], list) or not all(
            isinstance(k, str) for k in schema["required"]
        ):
            raise CapabilityValidationError(f"{path}.required must be an array of strings.")
    if "properties" in schema:
        if not isinstance(schema["properties"], dict):
            raise CapabilityValidationError(f"{path}.properties must be an object.")
        for prop_k, prop_spec in schema["properties"].items():
            _validate_schema_definition(prop_spec, f"{path}.properties.{prop_k}")
    if "additionalProperties" in schema and not isinstance(
        schema["additionalProperties"], (bool, dict)
    ):
        raise CapabilityValidationError(
            f"{path}.additionalProperties must be a boolean or schema object."
        )
    if isinstance(schema.get("additionalProperties"), dict):
        _validate_schema_definition(
            schema["additionalProperties"],
            f"{path}.additionalProperties",
        )


def parse_and_validate_capability_ttl(
    ttl_text: str,
    *,
    alive: bool = True,
) -> list[EndpointDescriptor]:
    if not ttl_text or not ttl_text.strip():
        raise CapabilityValidationError("Capability Turtle text must be non-empty.")

    data_graph = Graph()
    try:
        data_graph.parse(data=ttl_text, format="turtle")
    except Exception as exc:
        raise CapabilityValidationError(f"Invalid Turtle capability graph: {exc}") from exc

    ont_graph = Graph()
    ont_graph.parse(data=read_capability_ontology_ttl(), format="turtle")

    shapes_graph = Graph()
    shapes_graph.parse(data=read_capability_shapes_ttl(), format="turtle")

    conforms, _results_graph, results_text = pyshacl.validate(
        data_graph,
        shacl_graph=shapes_graph,
        ont_graph=ont_graph,
        inference="none",
        abort_on_first=False,
    )
    if not conforms:
        raise CapabilityValidationError(
            f"Capability announcement failed SHACL validation:\n{results_text}"
        )

    endpoint_nodes = sorted(data_graph.subjects(RDF.type, BH.AgentEndpoint), key=str)
    if not endpoint_nodes:
        raise CapabilityValidationError("Capability graph contains no bh:AgentEndpoint nodes.")

    digest = _canonical_graph_sha256(data_graph)
    descriptors: list[EndpointDescriptor] = []

    for ep_node in endpoint_nodes:
        ep_id_val = data_graph.value(ep_node, BH.endpointId)
        drv_id_val = data_graph.value(ep_node, BH.driverId)
        if ep_id_val is None or drv_id_val is None:
            raise CapabilityValidationError("Endpoint missing endpointId or driverId.")
        endpoint_id = str(ep_id_val)
        driver_id = str(drv_id_val)
        label_val = data_graph.value(ep_node, RDFS.label)
        label = str(label_val) if label_val is not None else endpoint_id

        trust_raw = str(data_graph.value(ep_node, BH.trustTier))
        try:
            trust_tier = TrustTier(trust_raw)
        except ValueError as exc:
            raise CapabilityValidationError(f"Invalid trustTier: {trust_raw!r}") from exc

        default_posture_raw = str(data_graph.value(ep_node, BH.defaultPosture))
        try:
            default_posture = ExecutionPosture(default_posture_raw)
        except ValueError as exc:
            raise CapabilityValidationError(
                f"Invalid defaultPosture: {default_posture_raw!r}"
            ) from exc

        supported_postures_list: list[ExecutionPosture] = []
        for p_obj in sorted(data_graph.objects(ep_node, BH.supportsPosture), key=str):
            try:
                supported_postures_list.append(ExecutionPosture(str(p_obj)))
            except ValueError as exc:
                raise CapabilityValidationError(f"Invalid supportsPosture: {p_obj!r}") from exc

        if default_posture not in supported_postures_list:
            raise CapabilityValidationError(
                f"Endpoint {endpoint_id!r} defaultPosture {default_posture} "
                "must be included in bh:supportsPosture."
            )

        capabilities = tuple(
            sorted({str(c) for c in data_graph.objects(ep_node, BH.hasCapability)})
        )
        legacy_tags = tuple(
            sorted({str(t) for t in data_graph.objects(ep_node, BH.legacyCapabilityTag)})
        )

        models: list[ModelProfileDescriptor] = []
        for mp_node in data_graph.objects(ep_node, BH.hasModelProfile):
            fam_val = data_graph.value(mp_node, BH.modelFamily)
            ident_val = data_graph.value(mp_node, BH.modelIdentifier)
            if fam_val is None or ident_val is None or not str(fam_val).strip():
                raise CapabilityValidationError(
                    f"ModelProfile on {endpoint_id!r} requires modelFamily and modelIdentifier."
                )
            ctx_val = data_graph.value(mp_node, BH.contextWindowTokens)
            try:
                ctx_int = int(str(ctx_val)) if ctx_val is not None else None
            except ValueError as exc:
                raise CapabilityValidationError(
                    f"Invalid contextWindowTokens {ctx_val!r} on {endpoint_id!r}."
                ) from exc
            models.append(
                ModelProfileDescriptor(
                    model_family=str(fam_val),
                    model_identifier=str(ident_val),
                    context_window_tokens=ctx_int,
                )
            )
        models.sort(key=lambda m: (m.model_family, m.model_identifier))

        facets: dict[str, CapabilityFacetDescriptor] = {}
        for fc_node in data_graph.objects(ep_node, BH.supportsFacet):
            cap_val = data_graph.value(fc_node, BH.forCapability)
            key_val = data_graph.value(fc_node, BH.facetKey)
            in_val = data_graph.value(fc_node, BH.inputJsonSchema)
            if cap_val is None or key_val is None or in_val is None:
                raise CapabilityValidationError(
                    f"Incomplete CapabilityFacetContract on {endpoint_id!r}."
                )
            cap_iri = str(cap_val)
            if cap_iri not in capabilities:
                raise CapabilityValidationError(
                    f"Endpoint {endpoint_id!r} declares facet contract for capability "
                    f"{cap_iri!r}, which is not in its bh:hasCapability set."
                )
            facet_key = str(key_val)
            if facet_key in facets:
                raise CapabilityValidationError(
                    f"Duplicate facetKey {facet_key!r} on endpoint {endpoint_id!r}."
                )
            try:
                parsed_in = json.loads(str(in_val))
            except json.JSONDecodeError as exc:
                raise CapabilityValidationError(
                    f"Invalid inputJsonSchema on facet {facet_key!r}: {exc}"
                ) from exc
            _validate_schema_definition(parsed_in, f"facet[{facet_key}].inputJsonSchema")

            raw_out_node = data_graph.value(fc_node, BH.outputJsonSchema)
            parsed_out: dict[str, Any] | None = None
            if raw_out_node is not None:
                try:
                    loaded_out = json.loads(str(raw_out_node))
                except json.JSONDecodeError as exc:
                    raise CapabilityValidationError(
                        f"Invalid outputJsonSchema on facet {facet_key!r}: {exc}"
                    ) from exc
                _validate_schema_definition(loaded_out, f"facet[{facet_key}].outputJsonSchema")
                parsed_out = loaded_out

            facets[facet_key] = CapabilityFacetDescriptor(
                capability_iri=cap_iri,
                facet_key=facet_key,
                input_json_schema=parsed_in,
                output_json_schema=parsed_out,
            )

        cost_raw = data_graph.value(ep_node, BH.costTier)
        cost_tier = CostTier.from_shorthand(str(cost_raw)) if cost_raw is not None else None

        lat_raw = data_graph.value(ep_node, BH.latencyClass)
        latency_class = LatencyClass.from_shorthand(str(lat_raw)) if lat_raw is not None else None

        descriptors.append(
            EndpointDescriptor(
                endpoint_iri=str(ep_node),
                endpoint_id=endpoint_id,
                driver_id=driver_id,
                label=label,
                trust_tier=trust_tier,
                default_posture=default_posture,
                supported_postures=tuple(supported_postures_list),
                capabilities=capabilities,
                legacy_capability_tags=legacy_tags,
                models=tuple(models),
                facets=facets,
                cost_tier=cost_tier,
                latency_class=latency_class,
                alive=alive,
                capability_graph_digest=digest,
            )
        )

    descriptors.sort(key=lambda d: d.endpoint_id)
    return descriptors


def lift_legacy_beast_capabilities_to_ttl(
    agent_id: str,
    capabilities: list[str],
) -> str:
    cleaned_id = agent_id.strip()
    if not cleaned_id:
        raise CapabilityValidationError("agent_id must be non-empty for legacy lifting.")
    safe_iri_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned_id)
    ep_uri = URIRef(f"urn:beast:endpoint:{safe_iri_slug}")

    g = Graph()
    g.bind("bh", BH)
    g.bind("rdfs", RDFS)
    g.add((ep_uri, RDF.type, BH.AgentEndpoint))
    g.add((ep_uri, RDFS.label, Literal(f"Beast Peer {cleaned_id}")))
    g.add((ep_uri, BH.endpointId, Literal(cleaned_id)))
    g.add((ep_uri, BH.driverId, Literal("beast-peer")))
    g.add((ep_uri, BH.trustTier, BH.RuntimeSelfAnnounced))
    g.add((ep_uri, BH.defaultPosture, BH.ObserveOnly))
    g.add((ep_uri, BH.supportsPosture, BH.ObserveOnly))
    g.add((ep_uri, BH.hasCapability, BH.RepoFreePrompt))
    for tag in capabilities:
        tag_clean = tag.strip()
        if tag_clean:
            g.add((ep_uri, BH.legacyCapabilityTag, Literal(tag_clean)))

    return g.serialize(format="turtle")


def _validate_json_schema_value(value: Any, schema: dict[str, Any], path: str) -> None:
    _validate_schema_definition(schema, path)
    expected_type = schema.get("type")
    if expected_type is not None:
        if expected_type == "string":
            if not isinstance(value, str):
                raise CapabilityValidationError(
                    f"{path}: expected string, got {type(value).__name__}"
                )
            min_len = schema.get("minLength")
            if min_len is not None and len(value) < int(min_len):
                raise CapabilityValidationError(f"{path}: string shorter than minLength={min_len}")
            max_len = schema.get("maxLength")
            if max_len is not None and len(value) > int(max_len):
                raise CapabilityValidationError(f"{path}: string exceeds maxLength={max_len}")
            pat = schema.get("pattern")
            if pat is not None and re.search(pat, value) is None:
                raise CapabilityValidationError(f"{path}: string does not match pattern {pat!r}")
        elif expected_type == "boolean":
            if not isinstance(value, bool):
                raise CapabilityValidationError(
                    f"{path}: expected boolean, got {type(value).__name__}"
                )
        elif expected_type == "integer":
            if not _is_strict_int(value):
                raise CapabilityValidationError(
                    f"{path}: expected integer, got {type(value).__name__}"
                )
            if "minimum" in schema and value < schema["minimum"]:
                raise CapabilityValidationError(
                    f"{path}: integer below minimum={schema['minimum']}"
                )
            if "maximum" in schema and value > schema["maximum"]:
                raise CapabilityValidationError(
                    f"{path}: integer above maximum={schema['maximum']}"
                )
        elif expected_type == "number":
            if not _is_strict_num(value):
                raise CapabilityValidationError(
                    f"{path}: expected number, got {type(value).__name__}"
                )
            if "minimum" in schema and value < schema["minimum"]:
                raise CapabilityValidationError(f"{path}: number below minimum={schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                raise CapabilityValidationError(f"{path}: number above maximum={schema['maximum']}")
        elif expected_type == "array":
            if not isinstance(value, list):
                raise CapabilityValidationError(
                    f"{path}: expected array, got {type(value).__name__}"
                )
            if "minItems" in schema and len(value) < int(schema["minItems"]):
                raise CapabilityValidationError(
                    f"{path}: fewer items than minItems={schema['minItems']}"
                )
            if "maxItems" in schema and len(value) > int(schema["maxItems"]):
                raise CapabilityValidationError(
                    f"{path}: more items than maxItems={schema['maxItems']}"
                )
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for idx, item in enumerate(value):
                    _validate_json_schema_value(item, item_schema, f"{path}[{idx}]")
        elif expected_type == "object":
            if not isinstance(value, dict):
                raise CapabilityValidationError(
                    f"{path}: expected object, got {type(value).__name__}"
                )
            req_keys = schema.get("required", [])
            for rk in req_keys:
                if rk not in value:
                    raise CapabilityValidationError(f"{path}: missing required key {rk!r}")
            props = schema.get("properties", {})
            addl = schema.get("additionalProperties", True)
            if addl is False:
                extra = set(value.keys()) - set(props.keys())
                if extra:
                    raise CapabilityValidationError(
                        f"{path}: unexpected properties {sorted(extra)!r}"
                    )
            for prop_k, prop_v in value.items():
                if prop_k in props and isinstance(props[prop_k], dict):
                    _validate_json_schema_value(prop_v, props[prop_k], f"{path}.{prop_k}")
                elif isinstance(addl, dict):
                    _validate_json_schema_value(prop_v, addl, f"{path}.{prop_k}")

    if "const" in schema and value != schema["const"]:
        raise CapabilityValidationError(f"{path}: value must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise CapabilityValidationError(f"{path}: value {value!r} not in enum {schema['enum']!r}")


def validate_facets_for_endpoint(
    endpoint: EndpointDescriptor,
    facets: dict[str, Any],
    posture: ExecutionPosture,
) -> None:
    if posture not in endpoint.supported_postures:
        raise PostureViolationError(
            f"Endpoint {endpoint.endpoint_id!r} does not support "
            f"execution posture {posture.value!r}."
        )

    for key, value in facets.items():
        contract = endpoint.facets.get(key)
        if contract is None:
            raise CapabilityValidationError(
                f"Endpoint {endpoint.endpoint_id!r} does not declare facet {key!r} "
                f"(supported facets: {sorted(endpoint.facets.keys())})."
            )
        _validate_json_schema_value(value, contract.input_json_schema, f"facets.{key}")


def _endpoint_matches_capability(endpoint: EndpointDescriptor, token: str) -> bool:
    normalized = normalize_capability_iri(token)
    if normalized in endpoint.capabilities:
        return True
    return token.strip() in endpoint.legacy_capability_tags


def match_and_rank_endpoints(
    endpoints: list[EndpointDescriptor],
    query: DiscoveryQuery,
) -> list[EndpointDescriptor]:
    excluded_families = {fam.strip().lower() for fam in query.exclude_model_families if fam.strip()}
    min_rank = _TRUST_RANK[query.min_trust_tier]

    candidates: list[tuple[tuple[int, ...], str, EndpointDescriptor]] = []
    for ep in endpoints:
        if query.only_alive and not ep.alive:
            continue
        if _TRUST_RANK[ep.trust_tier] < min_rank:
            continue
        if query.required_posture not in ep.supported_postures:
            continue
        if query.max_cost_tier is not None:
            if ep.cost_tier is None or _COST_RANK[ep.cost_tier] > _COST_RANK[query.max_cost_tier]:
                continue
        if query.max_latency_class is not None:
            if (
                ep.latency_class is None
                or _LATENCY_RANK[ep.latency_class] > _LATENCY_RANK[query.max_latency_class]
            ):
                continue
        if excluded_families and any(
            m.model_family.strip().lower() in excluded_families for m in ep.models
        ):
            continue
        if not all(_endpoint_matches_capability(ep, req) for req in query.required_capabilities):
            continue

        pref_score = sum(
            1 for pref in query.preferred_capabilities if _endpoint_matches_capability(ep, pref)
        )
        trust_score = _TRUST_RANK[ep.trust_tier]
        default_missing = (
            4
            if query.ranking_profile in {RankingProfile.LOW_COST, RankingProfile.LOW_LATENCY}
            else 2
        )
        cost_rank = _COST_RANK[ep.cost_tier] if ep.cost_tier is not None else default_missing
        latency_rank = (
            _LATENCY_RANK[ep.latency_class] if ep.latency_class is not None else default_missing
        )

        if query.ranking_profile == RankingProfile.LOW_COST:
            rank_tuple: tuple[int, ...] = (-trust_score, cost_rank, -pref_score, latency_rank)
        elif query.ranking_profile == RankingProfile.LOW_LATENCY:
            rank_tuple = (-trust_score, latency_rank, -pref_score, cost_rank)
        elif query.ranking_profile == RankingProfile.CAPABILITY_FIRST:
            rank_tuple = (
                -pref_score,
                -len(ep.capabilities),
                -trust_score,
                cost_rank,
                latency_rank,
            )
        else:
            rank_tuple = (-pref_score, -trust_score, cost_rank, latency_rank)

        candidates.append((rank_tuple, ep.endpoint_id, ep))

    candidates.sort(key=lambda item: (item[0], item[1]))
    return [ep for _rank, _eid, ep in candidates]
