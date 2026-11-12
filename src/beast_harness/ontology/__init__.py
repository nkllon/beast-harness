"""Canonical Turtle + SHACL ontology assets for beast-harness."""

from importlib.resources import files


def read_capability_ontology_ttl() -> str:
    return (files(__package__) / "beast_harness_capability.ttl").read_text(encoding="utf-8")


def read_capability_shapes_ttl() -> str:
    return (files(__package__) / "beast_harness_capability.shacl.ttl").read_text(encoding="utf-8")
