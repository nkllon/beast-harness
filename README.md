# `beast-harness`

Capability-modeled multi-agent execution substrate, MCP server, and Beast Network bridge.

> **Intended Audience:** LLM calling agents (via MCP or Beast Network), SHACL/RDF capability routers and Casper closeout verifiers, and human operators / harness driver authors.

## Overview

`beast-harness` decouples heterogeneous agent environments (Cursor Cloud Agents & `cursor-sdk`, Google Antigravity SDK, Anthropic Managed Agents & `claude-agent-sdk`, OpenAI Codex App-Server JSON-RPC, and live Beast Network peers) behind a shared session lifecycle (`session_id`, `run_id`, `status`, `answer`, `cancel`) while preserving **every harness's unique native capabilities** via SHACL-validated Turtle capability graphs, cost/latency ranking profiles, and typed JSON-Schema facets.

## Installation

```bash
uv sync --extra all --extra dev
```

## Entrypoints

- `beast-harness-mcp` — Stdio/SSE Model Context Protocol server (`harness_discover`, `harness_start`, `harness_send`, `harness_status`, `harness_wait`, `harness_result`, `harness_cancel`, `harness_review`).
- `beast-harness-bridge` — Hosts any local/cloud `HarnessDriver` as a live capability-announcing node on the Beast Network (`beast-mailbox-core` + Redis Streams).
