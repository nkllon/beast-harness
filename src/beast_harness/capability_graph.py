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
    DiscoveryQuery,
    EndpointDescriptor,
    ExecutionPosture,
    ModelProfileDescriptor,
    PostureViolationError,
    TrustTier,
    normalize_capability_iri,
)

BH = Namespace(BH_NS)

_TRUST_RANK: dict[TrustTier, int] = {
    TrustTier.RUNTIME_SELF_ANNOUNCED: 1,
    TrustTier.NEO4J_ADMITTED: 2,
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
        endpoint_id = str(data_graph.value(ep_node, BH.endpointId))
        driver_id = str(data_graph.value(ep_node, BH.driverId))
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
            fam = str(data_graph.value(mp_node, BH.modelFamily))
            ident = str(data_graph.value(mp_node, BH.modelIdentifier))
            ctx_val = data_graph.value(mp_node, BH.contextWindowTokens)
            ctx_int = int(str(ctx_val)) if ctx_val is not None else None
            models.append(
                ModelProfileDescriptor(
                    model_family=fam,
                    model_identifier=ident,
                    context_window_tokens=ctx_int,
                )
            )
        models.sort(key=lambda m: (m.model_family, m.model_identifier))

        facets: dict[str, CapabilityFacetDescriptor] = {}
        for fc_node in data_graph.objects(ep_node, BH.supportsFacet):
            cap_iri = str(data_graph.value(fc_node, BH.forCapability))
            if cap_iri not in capabilities:
                raise CapabilityValidationError(
                    f"Endpoint {endpoint_id!r} declares facet contract for capability "
                    f"{cap_iri!r}, which is not in its bh:hasCapability set."
                )
            facet_key = str(data_graph.value(fc_node, BH.facetKey))
            if facet_key in facets:
                raise CapabilityValidationError(
                    f"Duplicate facetKey {facet_key!r} on endpoint {endpoint_id!r}."
                )
            raw_in = str(data_graph.value(fc_node, BH.inputJsonSchema))
            try:
                parsed_in = json.loads(raw_in)
            except json.JSONDecodeError as exc:
                raise CapabilityValidationError(
                    f"Invalid inputJsonSchema on facet {facet_key!r}: {exc}"
                ) from exc
            if not isinstance(parsed_in, dict):
                raise CapabilityValidationError(
                    f"Facet {facet_key!r} inputJsonSchema must be a JSON object."
                )

            raw_out_node = data_graph.value(fc_node, BH.outputJsonSchema)
            parsed_out: dict[str, Any] | None = None
            if raw_out_node is not None:
                try:
                    loaded_out = json.loads(str(raw_out_node))
                except json.JSONDecodeError as exc:
                    raise CapabilityValidationError(
                        f"Invalid outputJsonSchema on facet {facet_key!r}: {exc}"
                    ) from exc
                if not isinstance(loaded_out, dict):
                    raise CapabilityValidationError(
                        f"Facet {facet_key!r} outputJsonSchema must be a JSON object."
                    )
                parsed_out = loaded_out

            facets[facet_key] = CapabilityFacetDescriptor(
                capability_iri=cap_iri,
                facet_key=facet_key,
                input_json_schema=parsed_in,
                output_json_schema=parsed_out,
            )

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
        elif expected_type == "boolean":
            if not isinstance(value, bool):
                raise CapabilityValidationError(
                    f"{path}: expected boolean, got {type(value).__name__}"
                )
        elif expected_type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
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
            if isinstance(value, bool) or not isinstance(value, (int, float)):
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
            if schema.get("additionalProperties") is False:
                extra = set(value.keys()) - set(props.keys())
                if extra:
                    raise CapabilityValidationError(
                        f"{path}: unexpected properties {sorted(extra)!r}"
                    )
            for prop_k, prop_v in value.items():
                if prop_k in props and isinstance(props[prop_k], dict):
                    _validate_json_schema_value(prop_v, props[prop_k], f"{path}.{prop_k}")
        else:
            raise CapabilityValidationError(f"{path}: unsupported schema type {expected_type!r}")

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

    candidates: list[tuple[tuple[int, int, str], EndpointDescriptor]] = []
    for ep in endpoints:
        if query.only_alive and not ep.alive:
            continue
        if _TRUST_RANK[ep.trust_tier] < min_rank:
            continue
        if query.required_posture not in ep.supported_postures:
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
        sort_key = (-pref_score, -_TRUST_RANK[ep.trust_tier], ep.endpoint_id)
        candidates.append((sort_key, ep))

    candidates.sort(key=lambda item: item[0])
    return [ep for _key, ep in candidates]
