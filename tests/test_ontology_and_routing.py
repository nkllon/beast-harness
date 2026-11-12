"""Milestone 0 & 1 acceptance tests: Ontology, SHACL validation, facets, and routing."""

from __future__ import annotations

import json

import pytest
from rdflib import Graph, Namespace

from beast_harness.capability_graph import (
    lift_legacy_beast_capabilities_to_ttl,
    match_and_rank_endpoints,
    parse_and_validate_capability_ttl,
    validate_facets_for_endpoint,
)
from beast_harness.ontology import read_capability_ontology_ttl, read_capability_shapes_ttl
from beast_harness.types import (
    BH_NS,
    CapabilityValidationError,
    DiscoveryQuery,
    ExecutionPosture,
    PostureViolationError,
    TrustTier,
)

BH = Namespace(BH_NS)

CURSOR_SAMPLE_TTL = f"""\
@prefix bh: <{BH_NS}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<urn:beast:endpoint:cursor-cloud> a bh:AgentEndpoint ;
    rdfs:label "Cursor Cloud Agent" ;
    bh:endpointId "cursor-cloud" ;
    bh:driverId "cursor-cloud" ;
    bh:trustTier bh:RuntimeSelfAnnounced ;
    bh:defaultPosture bh:ObserveOnly ;
    bh:supportsPosture bh:ObserveOnly, bh:ProposeDiff, bh:IsolatedBranchWrite ;
    bh:hasCapability
        bh:RemoteGitShaMount,
        bh:MultiRepoMount,
        bh:SemanticCodebaseIndex,
        bh:DurableCloudRun,
        bh:FollowUpRunQueue,
        bh:RepoFreePrompt ;
    bh:hasModelProfile [
        a bh:ModelProfile ;
        bh:modelFamily "cursor-composer" ;
        bh:modelIdentifier "composer-2.5" ;
        bh:contextWindowTokens 200000
    ] ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability bh:MultiRepoMount ;
        bh:facetKey "extra_repos" ;
        bh:inputJsonSchema '{{"type": "array", "items": {{"type": "string"}}}}'
    ] .
"""

DOMAIN_PEER_TTL = f"""\
@prefix bh: <{BH_NS}> .
@prefix dom: <https://nkllon.com/ontology/domain#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

dom:LeanFormalProof a bh:HarnessCapability ;
    rdfs:label "Lean 4 Formal Theorem Verification" .

<urn:beast:endpoint:lean-prover-1> a bh:AgentEndpoint ;
    rdfs:label "Beast Lean Prover Node" ;
    bh:endpointId "lean-prover-1" ;
    bh:driverId "beast-peer" ;
    bh:trustTier bh:Neo4jAdmitted ;
    bh:defaultPosture bh:ObserveOnly ;
    bh:supportsPosture bh:ObserveOnly ;
    bh:hasCapability dom:LeanFormalProof, bh:RepoFreePrompt ;
    bh:hasModelProfile [
        a bh:ModelProfile ;
        bh:modelFamily "google" ;
        bh:modelIdentifier "gemini-3.8-flash"
    ] ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability dom:LeanFormalProof ;
        bh:facetKey "theorem_module" ;
        bh:inputJsonSchema '{{"type": "string", "minLength": 1}}'
    ] .
"""


def test_ontology_declares_intended_audience_inside_artifacts():
    """Both ontology and SHACL shapes declare bh:intendedAudience inside the artifacts."""
    for raw in (read_capability_ontology_ttl(), read_capability_shapes_ttl()):
        g = Graph()
        g.parse(data=raw, format="turtle")
        audiences = [str(o) for o in g.objects(None, BH.intendedAudience)]
        assert any("LLM" in a for a in audiences)


def test_shacl_accepts_valid_cursor_and_open_world_domain_peer_graphs():
    cursor_eps = parse_and_validate_capability_ttl(CURSOR_SAMPLE_TTL)
    assert len(cursor_eps) == 1
    ep = cursor_eps[0]
    assert ep.endpoint_id == "cursor-cloud"
    assert f"{BH_NS}SemanticCodebaseIndex" in ep.capabilities
    assert "extra_repos" in ep.facets
    assert ep.capability_graph_digest

    peer_eps = parse_and_validate_capability_ttl(DOMAIN_PEER_TTL)
    assert len(peer_eps) == 1
    assert "https://nkllon.com/ontology/domain#LeanFormalProof" in peer_eps[0].capabilities
    assert peer_eps[0].trust_tier == TrustTier.NEO4J_ADMITTED


def test_shacl_rejects_mutating_default_posture_and_undeclared_facet_capability():
    bad_posture_ttl = CURSOR_SAMPLE_TTL.replace(
        "bh:defaultPosture bh:ObserveOnly ;",
        "bh:defaultPosture bh:IsolatedBranchWrite ;",
    )
    with pytest.raises(CapabilityValidationError):
        parse_and_validate_capability_ttl(bad_posture_ttl)

    undeclared_facet_cap_ttl = f"""\
@prefix bh: <{BH_NS}> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

<urn:beast:endpoint:bad-facet> a bh:AgentEndpoint ;
    rdfs:label "Bad Facet Endpoint" ;
    bh:endpointId "bad-facet" ;
    bh:driverId "cursor-cloud" ;
    bh:trustTier bh:RuntimeSelfAnnounced ;
    bh:defaultPosture bh:ObserveOnly ;
    bh:supportsPosture bh:ObserveOnly ;
    bh:hasCapability bh:RepoFreePrompt ;
    bh:supportsFacet [
        a bh:CapabilityFacetContract ;
        bh:forCapability bh:MultiRepoMount ;
        bh:facetKey "extra_repos" ;
        bh:inputJsonSchema '{json.dumps({"type": "array"})}'
    ] .
"""
    with pytest.raises(CapabilityValidationError):
        parse_and_validate_capability_ttl(undeclared_facet_cap_ttl)


def test_legacy_beast_capabilities_lift_and_validate():
    lifted_ttl = lift_legacy_beast_capabilities_to_ttl("legacy-node-7", ["sql_audit", "summarize"])
    eps = parse_and_validate_capability_ttl(lifted_ttl)
    assert len(eps) == 1
    assert eps[0].endpoint_id == "legacy-node-7"
    assert set(eps[0].legacy_capability_tags) == {"sql_audit", "summarize"}
    assert eps[0].default_posture == ExecutionPosture.OBSERVE_ONLY


def test_facet_and_posture_validation_fail_closed():
    cursor_ep = parse_and_validate_capability_ttl(CURSOR_SAMPLE_TTL)[0]
    lean_ep = parse_and_validate_capability_ttl(DOMAIN_PEER_TTL)[0]

    # Valid facet passes
    validate_facets_for_endpoint(
        cursor_ep,
        {"extra_repos": ["https://github.com/nkllon/beast-agent"]},
        ExecutionPosture.OBSERVE_ONLY,
    )

    # Unknown facet fails closed (never silently dropped)
    with pytest.raises(CapabilityValidationError):
        validate_facets_for_endpoint(
            cursor_ep,
            {"nonexistent_facet": True},
            ExecutionPosture.OBSERVE_ONLY,
        )

    # Wrong JSON type for declared facet fails closed
    with pytest.raises(CapabilityValidationError):
        validate_facets_for_endpoint(
            cursor_ep,
            {"extra_repos": "not-an-array"},
            ExecutionPosture.OBSERVE_ONLY,
        )

    # Unsupported write posture on ObserveOnly endpoint fails closed
    with pytest.raises(PostureViolationError):
        validate_facets_for_endpoint(
            lean_ep,
            {},
            ExecutionPosture.ISOLATED_BRANCH_WRITE,
        )


def test_declarative_matching_and_ranking():
    cursor_ep = parse_and_validate_capability_ttl(CURSOR_SAMPLE_TTL)[0]
    lean_ep = parse_and_validate_capability_ttl(DOMAIN_PEER_TTL)[0]
    endpoints = [cursor_ep, lean_ep]

    # Exclude google model family + require SemanticCodebaseIndex -> selects cursor-cloud
    q1 = DiscoveryQuery(
        required_capabilities=("SemanticCodebaseIndex",),
        exclude_model_families=("google",),
    )
    matched1 = match_and_rank_endpoints(endpoints, q1)
    assert [e.endpoint_id for e in matched1] == ["cursor-cloud"]

    # Require Tier 2 Neo4jAdmitted -> selects lean-prover-1 only
    q2 = DiscoveryQuery(min_trust_tier=TrustTier.NEO4J_ADMITTED)
    matched2 = match_and_rank_endpoints(endpoints, q2)
    assert [e.endpoint_id for e in matched2] == ["lean-prover-1"]

    # Preferred capability ranks higher when both satisfy required_capabilities
    q3 = DiscoveryQuery(
        required_capabilities=("RepoFreePrompt",),
        preferred_capabilities=("https://nkllon.com/ontology/domain#LeanFormalProof",),
    )
    matched3 = match_and_rank_endpoints(endpoints, q3)
    assert [e.endpoint_id for e in matched3] == ["lean-prover-1", "cursor-cloud"]
